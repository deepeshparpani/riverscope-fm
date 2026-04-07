import os
import torch
import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import Dataset

class RiverScopeDataset(Dataset):
    """
    PyTorch Dataset for loading high-res Sentinel-2 and PlanetScope satellite imagery masks natively.
    """
    
    def __init__(self, csv_file: str, data_root: str):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(data_root, row) for row in self.df['s2_path_reprojected'].values]
        self.mask_paths = [os.path.join(data_root, row) for row in self.df['label_path'].values]
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # 1. READ NATIVE PLANETSCOPE
        with rasterio.open(img_path) as src:
            total_channels = src.count
            read_channels = min(total_channels, 12)
            img_data = src.read(indexes=tuple(range(1, read_channels + 1)))
            
        # Extract native bounds 
        _, h, w = img_data.shape
        
        # 1B. PLANETSCOPE -> SENTINEL-2 BAND ALIGNMENT
        if read_channels == 4:
            s2_aligned = np.zeros((12, h, w), dtype=img_data.dtype)
            s2_aligned[1] = img_data[0] # PS Blue
            s2_aligned[2] = img_data[1] # PS Green
            s2_aligned[3] = img_data[2] # PS Red
            s2_aligned[7] = img_data[3] # PS NIR
            img_data = s2_aligned
        elif read_channels < 12:
            padding = np.zeros((12 - read_channels, h, w), dtype=img_data.dtype)
            img_data = np.concatenate([img_data, padding], axis=0)
            
        # 2. READ NATIVE MASK
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
            
        # 3. TENSOR FORMATTING
        img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)   # [1, 12, NativeH, NativeW]
        mask_tensor = torch.from_numpy(mask_data).float().unsqueeze(0) # [1, NativeH, NativeW]
        
        return img_tensor, mask_tensor
