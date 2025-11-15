# dataset.py
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class NiftiSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_suffix=".nii.gz", mask_suffix="_SEG.nii.gz"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(image_suffix)
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)

        # load image
        image = nib.load(image_path).get_fdata().astype(np.float32)
        
        base = img_name.replace(self.image_suffix, "")
        mask_name = base + self.mask_suffix
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        # add channels: (C, H, W, D)
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
