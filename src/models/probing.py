import os
# OlmoEarth's FlexiPatchEmbed internally uses bicubic2d_aa which is not yet
# implemented on Apple Silicon MPS. This fallback allows that one op to run
# on CPU transparently while everything else stays on MPS.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn

class OlmoEarthProbingHead(nn.Module):
    """
    Segmentation head that attaches to the frozen OlmoEarth embeddings.
    Upsamples the patch representations (14x14) back to original image resolution (224x224).
    """
    def __init__(self, embed_dim=768, num_classes=1):
        super().__init__()
        
        # A lightweight CNN decoder to progressively upsample 14x14 patches -> 224x224
        # 224 / 14 = 16x upsampling ratio required.
        self.decoder = nn.Sequential(
            # Input: [B, 768, 14, 14]
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample 1: 14x14 -> 56x56 (4x scale)
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample 2: 56x56 -> 224x224 (4x scale)
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            
            # Final 1x1 conv to map to logits
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        x: Spatial Tokens from OlmoEarth.
           Expected shape: [Batch, H_patches, W_patches, Dim] -> [B, 14, 14, 768]
        """
        # Rearrange to PyTorch spatial standard [Batch, Channels, Height, Width]
        # .contiguous() securely anchors memory layouts during backward PyTorch gradient tracing! 
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Decode to semantic mask
        logits = self.decoder(x)
        return logits

class EndToEndOlmoSegmenter(nn.Module):
    """
    Wrapper module combining the frozen OlmoEarth foundation model and the probing head.
    Easily pluggable into standard PyTorch training loops.
    """
    def __init__(self, foundation_model, embed_dim=768, num_classes=1):
        super().__init__()

        # Pin encoder to training device (MPS/CUDA) alongside the probe head.
        # PYTORCH_ENABLE_MPS_FALLBACK=1 (set in trainer.py) handles the one
        # unsupported op (bicubic2d_aa) by silently falling back to CPU for it.
        self.encoder = foundation_model
        
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = OlmoEarthProbingHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x, patch_size=16):
        # x shape: [Batch, Time, Channels, NativeHeight, NativeWidth]
        batch_size, time_dim, channels, native_h, native_w = x.shape
        import torch.nn.functional as F

        # A1. DOWNSAMPLE 3m -> 224x224 (10m equivalent) on the training device
        x_squeezed    = x.squeeze(1)
        x_downsampled = F.interpolate(x_squeezed, size=(224, 224), mode='bilinear', align_corners=False)

        # A2. Transpose to OlmoEarth format [B, H, W, T, C]
        x_transposed = x_downsampled.unsqueeze(1).permute(0, 3, 4, 1, 2).contiguous()

        # Build Foundation Model input wrapper
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
        dummy_time = torch.tensor([[[15, 6, 2023]]], dtype=torch.long, device=x.device)
        timestamps = dummy_time.repeat(x.size(0), 1, 1)

        masked_olmo_sample = MaskedOlmoEarthSample(
            sentinel2_l2a=x_transposed,
            sentinel2_l2a_mask=torch.zeros_like(x_transposed),
            timestamps=timestamps
        )

        # A3. Run frozen ViT encoder
        self.encoder.eval()
        with torch.no_grad():
            output_dict = self.encoder.encoder(masked_olmo_sample, patch_size=patch_size)

        from olmoearth_pretrain.nn.latent_mim import unpack_encoder_output
        latent, _, _ = unpack_encoder_output(output_dict)
        s2_tokens      = latent.sentinel2_l2a
        spatial_tokens = s2_tokens.mean(dim=(3, 4))

        # A4. Project 14x14 tokens -> 224x224 logits via CNN probe head
        logits = self.head(spatial_tokens)

        # A5. UPSAMPLE logits back to canonical tile size for loss calculation
        logits_native = F.interpolate(logits, size=(native_h, native_w), mode='bilinear', align_corners=False)

        return logits_native
