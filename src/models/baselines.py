import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class BaselineRiverSegmenter(nn.Module):
    """
    Standard PyTorch baselines (U-Net, FPN) for performance comparison against 
    Foundation Models. Powered by segmentation_models_pytorch.
    """
    def __init__(self, architecture='unet', encoder_name='resnet34', encoder_weights='imagenet', in_channels=12, classes=1):
        """
        Args:
            architecture (str): 'unet' or 'fpn'.
            encoder_name (str): Backbone model (e.g. 'resnet34', 'efficientnet-b0').
            encoder_weights (str or None): Pre-trained weights to use. Typically 'imagenet'.
            in_channels (int): Expected to be 12 for Sentinel-2 RiverScope imagery.
            classes (int): Prediction classes (1 for water mapping).
        """
        super().__init__()
        
        self.architecture = architecture.lower()
        
        if self.architecture == 'unet':
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes
            )
        elif self.architecture == 'fpn':
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes
            )
        else:
            raise ValueError(f"Architecture {architecture} not supported. Use 'unet' or 'fpn'.")

    def forward(self, x):
        """
        x: Expected [Batch, Channels, Height, Width]
        Note: The RiverScopeDataset outputs [Batch, Time, Channels, Height, Width] 
              for OlmoEarth (5D). We automatically squeeze the dummy time dimension here.
        """
        # If input has a dummy time dimension (e.g., [B, 1, 12, 224, 224]), squeeze it to [B, 12, 224, 224]
        if x.dim() == 5:
            x = x.squeeze(1) # Removes the T=1 dimension
            
        logits = self.model(x)
        return logits
