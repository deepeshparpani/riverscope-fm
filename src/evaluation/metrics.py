import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (IoU) / Jaccard Index for binary segmentation.
    Returns the average IoU over the batch.
    """
    # Apply sigmoid since model outputs raw logits
    preds = (torch.sigmoid(preds) > threshold).float()
    
    # Calculate intersection and union
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    
    # Add a small epsilon to avoid division by zero when both masks are empty
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def compute_extended_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Computes IoU, Precision, Recall, and F1 Score for binary segmentation.
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    
    # Calculate True Positives, False Positives, False Negatives
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    
    intersection = tp
    union = tp + fp + fn
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }

class BinaryDiceLoss(nn.Module):
    """
    Dice loss to combat severe class imbalance.
    Since water is only ~4.6% of pixels, normal BCE gets easily skewed by land pixels.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        
        # Flatten tensors for unified calculation using reshape to support safe gradient un-striding
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
