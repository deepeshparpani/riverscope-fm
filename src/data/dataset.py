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
    Designed for memory efficiency and zero overhead iterations for OlmoEarth-v1-Base.
    """
    
    def __init__(self, csv_file: str, data_root: str):
        """
        Args:
            csv_file (str): Path to the train.csv file.
            data_root (str): Root directory where images are stored.
        """
        super().__init__()
        
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
        # 1. LAZY LOAD & RESIZE IMAGE 
        # ==========================================
        with rasterio.open(img_path) as src:
            total_channels = src.count
            read_channels = min(total_channels, 12)
            
            # Read into dynamically resized out_shape to avoid reading full-res arrays into RAM
            # Note: This directly accommodates the paper's 3m -> 10m downsampling!
            img_data = src.read(
                indexes=tuple(range(1, read_channels + 1)),
                out_shape=(read_channels, 224, 224),
                resampling=Resampling.bilinear
            )
            
        # ==========================================
        # 1B. PLANETSCOPE -> SENTINEL-2 BAND ALIGNMENT
        # ==========================================
        # If the input has exactly 4 channels, we assume it is PlanetScope (Blue, Green, Red, NIR).
        # We align it to OlmoEarth's expected Sentinel-2 index ordering to prevent spectral confusion!
        # Standard S2 12-band array: B01, B02(Bl), B03(Gr), B04(Red), B05, B06, B07, B08(NIR), B8A, B09, B11, B12.
        if read_channels == 4:
            s2_aligned = np.zeros((12, 224, 224), dtype=img_data.dtype)
            s2_aligned[1] = img_data[0] # PS Blue  -> S2 B02 (Array Index 1)
            s2_aligned[2] = img_data[1] # PS Green -> S2 B03 (Array Index 2)
            s2_aligned[3] = img_data[2] # PS Red   -> S2 B04 (Array Index 3)
            s2_aligned[7] = img_data[3] # PS NIR   -> S2 B08 (Array Index 7)
            img_data = s2_aligned
        elif read_channels < 12:
            # Fallback zero-padding for any other missing channel lengths
            padding = np.zeros((12 - read_channels, 224, 224), dtype=img_data.dtype)
            img_data = np.concatenate([img_data, padding], axis=0)
            
        # ==========================================
        # 2. LAZY LOAD & RESIZE MASK
        # ==========================================
        with rasterio.open(mask_path) as src:
            # Masks are assumed single-channel categorical masks
            mask_data = src.read(
                1,
                out_shape=(224, 224),
                resampling=Resampling.nearest
            )
            
        # ==========================================
        # 3. TENSOR FORMATTING
        # ==========================================
        img_tensor = torch.from_numpy(img_data).float()
        mask_tensor = torch.from_numpy(mask_data).float()
        
        # Inject the dummy Time dimension requested for OlmoEarth compatibility
        # Current shape: [12, 224, 224] -> [1, 12, 224, 224]
        img_tensor = img_tensor.unsqueeze(0)
        
        # Current shape: [224, 224] -> [1, 224, 224]
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return img_tensor, mask_tensor
