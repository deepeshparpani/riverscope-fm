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
        self.encoder = foundation_model
        
        # Explicitly freeze the foundation model parameters to ensure we are only training the head
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.head = OlmoEarthProbingHead(embed_dim=embed_dim, num_classes=num_classes)
        
    def forward(self, x, patch_size=16):
        # x shape: [Batch, Time, Channels, NativeHeight, NativeWidth]
        batch_size, time_dim, channels, native_h, native_w = x.shape
        import torch.nn.functional as F
        
        # A1. DYNAMICALLY DOWNSAMPLE raw PlanetScope imagery to 10m/px Olmo/Alpha size (224x224)
        x_squeezed = x.squeeze(1) # Drop dummy time for spatial interpolation
        x_downsampled = F.interpolate(x_squeezed, size=(224, 224), mode='bilinear', align_corners=False)
        
        # A2. Transpose to OlmoEarth format: [Batch, Height, Width, Time, Channels]
        # Current layout: [Batch, Channels, 224, 224] -> [B, 1, 12, 224, 224] -> [B, 224, 224, 1, 12]
        x_transposed = x_downsampled.unsqueeze(1).permute(0, 3, 4, 1, 2).contiguous()

        # Build the Foundation Model Wrapper
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
        
        # Generate dummy timestamps (Batch, Time, D=3) mimicking Jun 15th, 2023
        dummy_time = torch.tensor([[[15, 6, 2023]]], dtype=torch.long, device=x.device)
        timestamps = dummy_time.repeat(x.size(0), 1, 1) 
        
        masked_olmo_sample = MaskedOlmoEarthSample(
            sentinel2_l2a=x_transposed,
            sentinel2_l2a_mask=torch.zeros_like(x_transposed),
            timestamps=timestamps
        )
        
        # Forward pass isolating ViT Encoder (Bypassing Decoder sequence)
        self.encoder.eval() 
        with torch.no_grad():
            output_dict = self.encoder.encoder(masked_olmo_sample, patch_size=patch_size)
            
        from olmoearth_pretrain.nn.latent_mim import unpack_encoder_output
        latent, _, _ = unpack_encoder_output(output_dict)
        s2_tokens = latent.sentinel2_l2a 
        spatial_tokens = s2_tokens.mean(dim=(3, 4))
        
        # Project to 224x224 segmentations
        logits = self.head(spatial_tokens)
        
        # A3. DYNAMICALLY UPSAMPLE FM logits back to original academic resolution map (NativeH, NativeW)
        logits_native = F.interpolate(logits, size=(native_h, native_w), mode='bilinear', align_corners=False)
        
        return logits_native
