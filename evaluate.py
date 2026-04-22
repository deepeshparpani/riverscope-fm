import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.dataset import RiverScopeDataset
from src.models.probing import EndToEndOlmoSegmenter
from src.evaluation.metrics import compute_extended_metrics
from huggingface_hub import snapshot_download
from olmoearth_pretrain.model_loader import load_model_from_id

def main(args):
    print(f"Loading dataset from: {args.data_root}")
    train_csv = os.path.join(args.data_root, 'train.csv')
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Could not find train.csv at {train_csv}.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using evaluation device: {device}")

    # 1. Load Dataset (Only evaluating on Validation set)
    BATCH_SIZE = 8
    full_dataset = RiverScopeDataset(train_csv, args.data_root)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # MUST match training seed
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f'✅ Validation Set Size: {len(val_dataset)} images')

    # 2. Load Model Architecture
    model_repo = 'allenai/OlmoEarth-v1-Base'
    print(f'Loading foundation model {model_repo}...')
    local_dir = snapshot_download(repo_id=model_repo)
    foundation_model = load_model_from_id(local_dir, load_weights=False)  # We just need the architecture

    model = EndToEndOlmoSegmenter(
        foundation_model=foundation_model,
        embed_dim=768,
        num_classes=1
    )

    # 3. Load Trained Weights
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Could not find trained weights at {args.weights_path}")
        
    print(f"Loading trained weights from {args.weights_path}...")
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 4. Evaluation Loop
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    print("\n🚀 Starting Evaluation...")
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            logits = model(images)
            
            # Compute batch metrics
            metrics = compute_extended_metrics(logits, masks, threshold=0.5)
            
            total_iou += metrics['iou']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            
    # Calculate averages
    num_batches = len(val_loader)
    avg_iou = total_iou / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches

    print("\n" + "="*40)
    print("🏆 FINAL EVALUATION METRICS")
    print("="*40)
    print(f"Mean IoU (mIoU):  {avg_iou:.4f}")
    print(f"F1 Score (Dice):  {avg_f1:.4f}")
    print(f"Precision:        {avg_precision:.4f}")
    print(f"Recall:           {avg_recall:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RiverScope Probing Model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to your dataset directory")
    parser.add_argument("--weights_path", type=str, default="best_olmo_cosine_probe.pth", help="Path to the trained .pth file")
    args = parser.parse_args()
    main(args)
