import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.evaluation.metrics import compute_iou, BinaryDiceLoss

class RiverScopeTrainer:
    """
    PyTorch training loop specialized for the RiverScope semantic segmentation task.
    Supports auto device-mapping (MPS for Mac M-series, CUDA, or CPU).
    """
    def __init__(self, model, train_loader, val_loader, device=None, lr=1e-4):
        # Hardware acceleration check (MPS is ideal on Mac M1/M2/M3)
        if device is None:
            self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        
        # Loss Composition:
        # 1. BCEWithLogitsLoss with pos_weight explicitly giving a 20x multiplier to the 4.6% minority 'water' class
        # 2. Additive DiceLoss handles regional overlapping overlap structures 
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(self.device))
        self.dice_loss = BinaryDiceLoss()
        
    def criterion(self, logits, targets):
        return self.bce_loss(logits, targets) + self.dice_loss(logits, targets)

    def train_epoch(self):
        self.model.train()
        total_loss, total_iou = 0.0, 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (Bypass MPS Flash Attention Bug using pure Math SDPA)
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                logits = self.model(images)
                
            loss = self.criterion(logits, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metric accumulation
            total_loss += loss.item()
            total_iou += compute_iou(logits, masks)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader), total_iou / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss, total_iou = 0.0, 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass (Bypass MPS Flash Attention Bug using pure Math SDPA)
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    logits = self.model(images)
                    
                loss = self.criterion(logits, masks)
                
                total_loss += loss.item()
                total_iou += compute_iou(logits, masks)
                
        return total_loss / len(self.val_loader), total_iou / len(self.val_loader)

    def fit(self, epochs=10, save_path='best_model.pth'):
        """
        Executes the main training loop and saves early-stopping checkpoints exclusively on Val-IoU peaks.
        """
        best_iou = 0.0
        print(f"Starting Training on device: {self.device}...")
        
        for epoch in range(epochs):
            print(f"\\n[Epoch {epoch+1}/{epochs}]")
            train_loss, train_iou = self.train_epoch()
            val_loss, val_iou = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
            
            # Save the optimal model state based on IoU fidelity
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ Checkpoint hit! New best model saved with IoU: {best_iou:.4f}")
                
        return best_iou
