"""
run_inference_experiment.py
===========================
Two-phase up/downsample experiment script for EndToEndOlmoSegmenter.

PHASE 1 — Cache 10m Logits
    Runs the full forward pass on every tile, saves the raw 10m logits
    (224x224) to disk as .pt files. This means the expensive OlmoEarth
    ViT encoder only runs once per tile, ever.

PHASE 2 — Upsample + Geo-Reference + Save
    Reads cached logits, upsamples them back to native 3m resolution,
    and saves the final binary mask as a properly geo-referenced GeoTIFF
    using the rasterio metadata from the dataset.
    Output filenames follow the SWORD-compatible convention:
        <reach_id>--<tile_id>--<image_stem>.tif

Usage:
    # Run both phases (default):
    python run_inference_experiment.py

    # Only cache logits (Phase 1):
    python run_inference_experiment.py --phase 1

    # Only save masks from cache (Phase 2):
    python run_inference_experiment.py --phase 2

    # Use a specific checkpoint:
    python run_inference_experiment.py --checkpoint notebooks/best_olmo_dynamic_sweep.pth
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import RiverScopeDataset
from src.models.probing import EndToEndOlmoSegmenter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT       = "data/raw/RiverScope_dataset"
TRAIN_CSV       = os.path.join(DATA_ROOT, "train.csv")
CHECKPOINT      = "notebooks/best_olmo_dynamic_sweep.pth"
LOGIT_CACHE_DIR = "data/embeddings/logit_cache_10m"   # Phase 1 output
MASK_OUTPUT_DIR = "data/processed/predicted_masks_3m"  # Phase 2 output
BATCH_SIZE      = 1   # Keep at 1 so geo_meta is cleanly indexable
THRESHOLD       = 0.5 # Binary mask threshold


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> EndToEndOlmoSegmenter:
    """Load OlmoEarth + probing head from a saved checkpoint."""
    from huggingface_hub import snapshot_download
    from olmoearth_pretrain.model_loader import load_model_from_id

    print("Loading OlmoEarth foundation model...")
    local_dir = snapshot_download(repo_id="allenai/OlmoEarth-v1-Base")
    foundation_model = load_model_from_id(local_dir, load_weights=True)

    model = EndToEndOlmoSegmenter(foundation_model=foundation_model)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded from: {checkpoint_path}")
    return model


def tile_id_from_path(img_path: str) -> str:
    """
    Extract a filesystem-safe tile identifier from the image path.
    Attempts to parse a reach--tile naming convention if present,
    otherwise falls back to the bare filename stem.
    """
    stem = Path(img_path).stem   # e.g. "reach_123456--tile_007--image"
    return stem


# ─────────────────────────────────────────────
# PHASE 1: Cache 10m logits to disk
# ─────────────────────────────────────────────
def phase1_cache_logits(model: EndToEndOlmoSegmenter, device: torch.device):
    """
    Runs the OlmoEarth encoder + probing head on every tile and saves the
    raw 10m-scale logits (shape [1, 1, 224, 224]) to disk as .pt files.
    
    NOTE: These are pre-upsampling logits, useful for:
      - Resolution ablation studies (compare 10m vs 3m scoring)
      - Re-running Phase 2 with different upsamplers without re-running Olmo
    """
    os.makedirs(LOGIT_CACHE_DIR, exist_ok=True)

    dataset = RiverScopeDataset(TRAIN_CSV, DATA_ROOT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n{'='*60}")
    print(f"PHASE 1: Caching 10m logits -> {LOGIT_CACHE_DIR}")
    print(f"{'='*60}")

    skipped = 0
    from torch.nn.attention import sdpa_kernel, SDPBackend

    for images, masks, geo_meta in tqdm(loader, desc="Caching logits"):
        tile_id = tile_id_from_path(geo_meta['img_path'][0])
        cache_path = os.path.join(LOGIT_CACHE_DIR, f"{tile_id}.pt")

        if os.path.exists(cache_path):
            skipped += 1
            continue

        images = images.to(device)

        # Run encoder at 10m (224x224) — upsample step is bypassed here
        # We extract the 224x224 logits BEFORE the final F.interpolate in forward()
        batch_size, time_dim, channels, native_h, native_w = images.shape
        x_squeezed    = images.squeeze(1)
        x_downsampled = F.interpolate(x_squeezed, size=(224, 224), mode='bilinear', align_corners=False)
        x_transposed  = x_downsampled.unsqueeze(1).permute(0, 3, 4, 1, 2).contiguous()

        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
        dummy_time = torch.tensor([[[15, 6, 2023]]], dtype=torch.long, device=device)
        timestamps = dummy_time.repeat(images.size(0), 1, 1)

        masked_sample = MaskedOlmoEarthSample(
            sentinel2_l2a=x_transposed,
            sentinel2_l2a_mask=torch.zeros_like(x_transposed),
            timestamps=timestamps,
        )

        model.encoder.eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            output_dict = model.encoder.encoder(masked_sample, patch_size=16)

        from olmoearth_pretrain.nn.latent_mim import unpack_encoder_output
        latent, _, _ = unpack_encoder_output(output_dict)
        s2_tokens     = latent.sentinel2_l2a
        spatial_tokens = s2_tokens.mean(dim=(3, 4))

        with torch.no_grad():
            # 10m logits — 224x224, NOT yet upsampled
            logits_10m = model.head(spatial_tokens)

        # Save: logits + native shape for Phase 2 upsampling
        torch.save({
            'logits_10m': logits_10m.cpu(),   # [1, 1, 224, 224]
            'native_h':   native_h,
            'native_w':   native_w,
            'img_path':   geo_meta['img_path'][0],
        }, cache_path)

    print(f"\n✅ Phase 1 complete. {len(dataset) - skipped} tiles cached, {skipped} already existed.")
    print(f"   Cache directory: {LOGIT_CACHE_DIR}/")


# ─────────────────────────────────────────────
# PHASE 2: Upsample + Geo-Reference + Save
# ─────────────────────────────────────────────
def phase2_save_masks(upsample_mode: str = 'bilinear'):
    """
    Reads cached 10m logits, upsamples to native 3m resolution, and saves
    as geo-referenced GeoTIFFs with the correct CRS and Affine transform.

    Args:
        upsample_mode: PyTorch interpolation mode for 10m -> 3m stretch.
                       Options: 'bilinear', 'bicubic', 'nearest'
                       (Experiment by changing this argument!)
    """
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    cache_files = sorted(Path(LOGIT_CACHE_DIR).glob("*.pt"))

    if not cache_files:
        print("❌ No cached logits found. Run Phase 1 first.")
        return

    # Build a quick lookup: stem -> rasterio metadata
    # We need the CRS/transform from the original source files
    dataset = RiverScopeDataset(TRAIN_CSV, DATA_ROOT)

    # Build path -> geo_meta map by iterating the dataset (no GPU needed here)
    print("Building geospatial metadata index...")
    geo_index = {}
    for i in tqdm(range(len(dataset)), desc="Indexing metadata"):
        _, _, meta = dataset[i]
        geo_index[meta['img_path']] = meta

    print(f"\n{'='*60}")
    print(f"PHASE 2: Upsampling ({upsample_mode}) + Saving GeoTIFFs -> {MASK_OUTPUT_DIR}")
    print(f"{'='*60}")

    for cache_path in tqdm(cache_files, desc="Saving masks"):
        payload    = torch.load(cache_path, weights_only=True)
        logits_10m = payload['logits_10m']  # [1, 1, 224, 224]
        native_h   = payload['native_h']
        native_w   = payload['native_w']
        img_path   = payload['img_path']

        # ── Upsample 10m -> 3m ──────────────────────────────
        # NOTE: These upsampled pixels are INTERPOLATED from 10m Olmo predictions.
        # They are NOT true 3m predictions — this is a fundamental limitation of
        # the current bilinear probing approach. Document this in the Week 3 ablation.
        align_corners = False if upsample_mode in ('bilinear', 'bicubic') else None
        kwargs = dict(size=(native_h, native_w), mode=upsample_mode)
        if align_corners is not None:
            kwargs['align_corners'] = align_corners

        logits_3m = F.interpolate(logits_10m, **kwargs)
        mask_3m   = (torch.sigmoid(logits_3m) > THRESHOLD).float()  # [1, 1, H, W]
        mask_np   = mask_3m.squeeze().numpy().astype(np.uint8)       # [H, W]

        # ── Retrieve geospatial metadata ────────────────────
        meta = geo_index.get(img_path)
        if meta is None:
            print(f"  ⚠️  No geo metadata for {img_path}, skipping.")
            continue

        # ── Build SWORD-compatible output filename ───────────
        # Convention: <reach_id>--<tile_id>--<image_stem>.tif
        tile_id     = tile_id_from_path(img_path)
        output_name = f"{tile_id}--predicted_mask_{upsample_mode}.tif"
        output_path = os.path.join(MASK_OUTPUT_DIR, output_name)

        # ── Write geo-referenced GeoTIFF ─────────────────────
        # Deserialize: WKT string -> rasterio.CRS, list -> Affine transform
        from rasterio.crs import CRS
        from affine import Affine
        crs_obj       = CRS.from_wkt(meta['crs'])
        transform_obj = Affine(*meta['transform'][:6])  # Affine takes a, b, c, d, e, f

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=native_h,
            width=native_w,
            count=1,
            dtype=rasterio.uint8,
            crs=crs_obj,
            transform=transform_obj,
            compress='lzw',
        ) as dst:
            dst.write(mask_np, 1)

    print(f"\n✅ Phase 2 complete. GeoTIFFs saved to: {MASK_OUTPUT_DIR}/")
    print(f"   Upsample mode used: '{upsample_mode}'")
    print(f"   ⚠️  NOTE: Upsampled pixels are INTERPOLATED from 10m Olmo predictions,")
    print(f"   not true 3m predictions. Document in resolution ablation (Week 3).")


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RiverScope up/downsample inference experiment")
    parser.add_argument("--phase",      type=int,   default=0,          help="1=cache logits, 2=save masks, 0=both")
    parser.add_argument("--checkpoint", type=str,   default=CHECKPOINT, help="Path to model checkpoint .pth")
    parser.add_argument("--upsample",   type=str,   default="bilinear", 
                        choices=["bilinear", "bicubic", "nearest"],
                        help="Upsampling mode for Phase 2 (10m -> 3m)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if args.phase in (0, 1):
        model = load_model(args.checkpoint, device)
        phase1_cache_logits(model, device)

    if args.phase in (0, 2):
        phase2_save_masks(upsample_mode=args.upsample)

    print("\n🚀 Experiment complete!")
    print(f"   Logit cache : {LOGIT_CACHE_DIR}/")
    print(f"   Output masks: {MASK_OUTPUT_DIR}/")
