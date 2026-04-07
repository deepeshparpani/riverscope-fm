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
        # x arrives from DataLoader as PyTorch standard [Batch, Time, Channels, Height, Width]
        # Example shape: [B, 1, 12, 224, 224]
        
        # 1. Transpose to OlmoEarth format: [Batch, Height, Width, Time, Channels]
        # Dims: 0=B, 1=T, 2=C, 3=H, 4=W ---> 0=B, 3=H, 4=W, 1=T, 2=C
        x_transposed = x.permute(0, 3, 4, 1, 2)
        
        # 2. Build the Foundation Model Wrapper
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
        
        # Generate dummy timestamps (Batch, Time, D=3) mimicking Jun 15th, 2023
        # PyTorch requires Embeddings indices to be integers!
        dummy_time = torch.tensor([[[15, 6, 2023]]], dtype=torch.long, device=x.device)
        timestamps = dummy_time.repeat(x.size(0), 1, 1) # Expand to match batch size
        
        masked_olmo_sample = MaskedOlmoEarthSample(
            sentinel2_l2a=x_transposed,
            sentinel2_l2a_mask=torch.zeros_like(x_transposed),
            timestamps=timestamps
        )
        
        # 3. Forward pass strictly isolating the ViT Encoder (Bypassing Decoder sequence to fix 0-length MPS shader crash)
        self.encoder.eval() # Ensure encoder behaves deterministically
        with torch.no_grad():
            output_dict = self.encoder.encoder(masked_olmo_sample, patch_size=patch_size)
            
        # 4. Restructure output embeddings locally
        from olmoearth_pretrain.nn.latent_mim import unpack_encoder_output
        latent, _, _ = unpack_encoder_output(output_dict)
        
        # 3. Target the Sentinel-2 branch mapping
        s2_tokens = latent.sentinel2_l2a # [Batch, 14, 14, Time, BandSets, 768]
        
        # 4. Average pooling across Time (dim 3) and Band_Sets (dim 4)
        # Results in layout: [Batch, 14, 14, 768]
        spatial_tokens = s2_tokens.mean(dim=(3, 4))
        
        # 5. Project to full image segmentations
        logits = self.head(spatial_tokens)
        
        return logits
