import torch
from huggingface_hub import snapshot_download
from olmoearth_pretrain.model_loader import load_model_from_id

# 1. Ensure model is loaded (using your successful logic)
model_repo = "allenai/OlmoEarth-v1-Base"
local_dir = snapshot_download(repo_id=model_repo)
model = load_model_from_id(local_dir, load_weights=True)
model.eval()

print("\n" + "="*30)
print("🚀 SUCCESS: OlmoEarth is Loaded!")
print("="*30)

try:
    # 2. Prepare dummy data
    # Shape: [Batch, Time, Channels, Height, Width]
    # OlmoEarth-v1-Base expects 12 spectral bands
    dummy_input = torch.randn(1, 1, 12, 224, 224) 

    with torch.no_grad():
        # THE FIX: Pass patch_size=16 explicitly as required by LatentMIM.forward()
        output = model(dummy_input, patch_size=16)
    
    # 3. Check the latent features
    # For OlmoEarth, the output is typically the hidden state tensor
    print(f"Feature Map Shape: {output.shape}")
    # Expected: [1, 197, 768] (1 batch, 196 patches + 1 CLS, 768 dims)

    # 4. Success for the Proposal!
    print("\n--- PROPOSAL DATA READY ---")
    print(f"Successfully extracted {output.shape[-1]}-dimensional embeddings.")
    print("Pipeline validation: COMPLETE.")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")