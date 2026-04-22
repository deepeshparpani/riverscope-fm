import os
import torch
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from torch.utils.data import Dataset

class RiverScopeDataset(Dataset):
    """
    PyTorch Dataset for loading Sentinel-2 and PlanetScope satellite imagery masks.
    Resamples all tiles to a consistent canonical size (default 512x512) so batches
    can be stacked, while still delivering much higher resolution than the model's
    internal 224x224 processing size.
    Also returns rasterio geospatial metadata for geo-referenced mask saving downstream.
    """
    
    def __init__(self, csv_file: str, data_root: str, tile_size: tuple = (512, 512)):
        """
        Args:
            csv_file (str):      Path to the train.csv file.
            data_root (str):     Root directory where images are stored.
            tile_size (tuple):   Canonical (H, W) to resample all tiles to.
                                 Ensures consistent batch stacking. Default 512x512.
                                 Must be larger than the model's internal 224x224.
        """
        super().__init__()
        
        self.tile_size = tile_size
        
        # Load mapping file
        self.df = pd.read_csv(csv_file)
        
        # Pre-compute absolute paths to avoid OS path resolution overhead during iterations
        self.image_paths = [os.path.join(data_root, row) for row in self.df['s2_path_reprojected'].values]
        self.mask_paths = [os.path.join(data_root, row) for row in self.df['label_path'].values]
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # ==========================================
        # 1. NATIVE LOAD — NO DOWNSAMPLING
        # The dataset is the single source of truth.
        # Any downsampling is the model's responsibility (inside probing.py).
        # NOTE: pixels returned here are interpolated-at-source 3m PlanetScope pixels.
        # ==========================================
        with rasterio.open(img_path) as src:
            total_channels = src.count
            read_channels = min(total_channels, 12)
            
            # Resample to canonical tile_size via rasterio for consistent batch stacking.
            # tile_size (e.g. 512x512) is still >> model's internal 224x224, preserving
            # the higher-resolution spatial detail the dynamic pipeline requires.
            img_data = src.read(
                indexes=tuple(range(1, read_channels + 1)),
                out_shape=(read_channels, self.tile_size[0], self.tile_size[1]),
                resampling=Resampling.bilinear
            )
            
            # Preserve geospatial metadata for downstream geo-referenced mask saving.
            # CRS and Affine are serialized to plain Python types so PyTorch's
            # default collate function can batch them without crashing.
            geo_meta = {
                'crs':       src.crs.to_wkt(),          # rasterio.CRS -> WKT string
                'transform': list(src.transform),        # Affine -> 9-element list
                'height':    src.height,
                'width':     src.width,
                'img_path':  img_path,
            }
            
        # ==========================================
        # 1B. PLANETSCOPE -> SENTINEL-2 BAND ALIGNMENT
        # ==========================================
        # If the input has exactly 4 channels, we assume it is PlanetScope (Blue, Green, Red, NIR).
        # We align it to OlmoEarth's expected Sentinel-2 index ordering.
        # Standard S2 12-band order: B01, B02(Blue), B03(Green), B04(Red), B05, B06, B07, B08(NIR), B8A, B09, B11, B12
        native_h, native_w = self.tile_size[0], self.tile_size[1]
        if read_channels == 4:
            s2_aligned = np.zeros((12, native_h, native_w), dtype=img_data.dtype)
            s2_aligned[1] = img_data[0]  # PS Blue  -> S2 B02 (Index 1)
            s2_aligned[2] = img_data[1]  # PS Green -> S2 B03 (Index 2)
            s2_aligned[3] = img_data[2]  # PS Red   -> S2 B04 (Index 3)
            s2_aligned[7] = img_data[3]  # PS NIR   -> S2 B08 (Index 7)
            img_data = s2_aligned
        elif read_channels < 12:
            # Fallback zero-padding for any other missing channel count
            padding = np.zeros((12 - read_channels, native_h, native_w), dtype=img_data.dtype)
            img_data = np.concatenate([img_data, padding], axis=0)
            
        # ==========================================
        # 2. NATIVE LOAD MASK — NO DOWNSAMPLING
        # ==========================================
        with rasterio.open(mask_path) as src:
            mask_data = src.read(
                1,
                out_shape=(self.tile_size[0], self.tile_size[1]),
                resampling=Resampling.bilinear
            )
            # Re-binarize cleanly to avoid jagged nearest-neighbor artifacts
            mask_data = (mask_data > 0.5).astype(np.uint8)
            
        # ==========================================
        # 3. TENSOR FORMATTING
        # ==========================================
        img_tensor = torch.from_numpy(img_data).float()
        mask_tensor = torch.from_numpy(mask_data).float()
        
        # Inject the dummy Time dimension for OlmoEarth compatibility
        # Shape: [12, H, W] -> [1, 12, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        
        # Shape: [H, W] -> [1, H, W]
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return img_tensor, mask_tensor, geo_meta
