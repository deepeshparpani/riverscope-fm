import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.evaluation.metrics import compute_iou, BinaryDiceLoss

class RiverScopeTrainer:
    """
    PyTorch training loop specialized for the RiverScope semantic segmentation task.
    Supports auto device-mapping (MPS for Mac M-series, CUDA, or CPU).

    Hyperparameter rationale (Phase 5):
      - lr=1e-4:       Matches the previous best hyperparameter config.
      - weight_decay:  0 (removed via AdamW default) — linear probes on frozen
                       embeddings can underfit if regularization is too strong.
      - Scheduler:     None (Flat LR). Reverting back to Phase 1 baseline.
      - pos_weight=20: Reverted back to 20 to heavily prioritize recall (Phase 1 baseline).
      - BCE Loss only: Dice loss removed to isolate the impact of pos_weight tuning.
    """
    def __init__(self, model, train_loader, val_loader, device=None, lr=1e-4, epochs=25):
        # Hardware acceleration check (MPS is ideal on Mac M-series)
        if device is None:
            self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        # Optimizer: lower LR (1e-4) matching previous best, NO weight decay (0)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # Scheduler: None (Flat LR for Phase 5)
        self.scheduler = None

        # Loss: BCE only
        # pos_weight=20.0 gives 20x gradient signal on the minority water class
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(self.device))

    def criterion(self, logits, targets):
        return self.bce_loss(logits, targets)

    def train_epoch(self):
        self.model.train()
        total_loss, total_iou = 0.0, 0.0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks, _ in pbar:  # geo_meta discarded — only needed for inference
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass — sdpa_kernel forces Math backend for OlmoEarth's
            # ViT attention layers which are incompatible with MPS Flash Attention.
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
            for images, masks, _ in tqdm(self.val_loader, desc="Validation"):  # geo_meta discarded
                images, masks = images.to(self.device), masks.to(self.device)

                # Forward pass — sdpa_kernel for OlmoEarth MPS attention compatibility
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    logits = self.model(images)

                loss = self.criterion(logits, masks)

                total_loss += loss.item()
                total_iou += compute_iou(logits, masks)

        return total_loss / len(self.val_loader), total_iou / len(self.val_loader)

    def fit(self, epochs=None, save_path='best_model.pth'):
        """
        Executes the main training loop with cosine LR annealing.
        Saves checkpoints exclusively on Val-IoU peaks (best model strategy).
        LR is stepped once per epoch after validation.
        """
        if epochs is None:
            epochs = self.epochs

        best_iou = 0.0
        print(f"Starting Training on device: {self.device}...")
        print(f"LR schedule: CosineAnnealing | Start LR: {self.optimizer.param_groups[0]['lr']:.1e} -> eta_min: 1e-6 over {epochs} epochs\n")

        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch+1}/{epochs}] | LR: {current_lr:.2e}")
            train_loss, train_iou = self.train_epoch()
            val_loss, val_iou = self.validate()

            print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")

            # No scheduler step in Phase 5 (Flat LR)

            # Save the optimal model state based on Val IoU
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ Checkpoint hit! New best model saved with IoU: {best_iou:.4f}")

        return best_iou
