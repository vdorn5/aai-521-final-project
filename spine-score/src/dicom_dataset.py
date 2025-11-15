# dicom_dataset.py
import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

from src.utils import load_dicoms_from_folder   # ⬅ uses your code
# or:
# from dicom_loader import load_dicoms_from_folder


class DICOMSegDataset(Dataset):
    """
    Dataset that loads:
        - A 3D MRI volume from a folder of DICOM slices
        - A 3D segmentation mask from a matching .nii.gz file
    """

    def __init__(self, image_root, mask_root,
                mask_suffix="_SEG.nii.gz",
                transform=None):
        
        self.image_root = image_root
        self.mask_root = mask_root
        self.mask_suffix = mask_suffix
        self.transform = transform

        # every subfolder inside images/ is one scan
        self.study_folders = sorted([
            f for f in os.listdir(image_root)
            if os.path.isdir(os.path.join(image_root, f))
        ])

    def __len__(self):
        return len(self.study_folders)

    def __getitem__(self, idx):
        study_id = self.study_folders[idx]

        # ---- Load DICOM Volume ----
        dicom_folder = os.path.join(self.image_root, study_id)
        spinal_scan = load_dicoms_from_folder(dicom_folder)

        volume = spinal_scan.volume.astype(np.float32)  # (H, W, D)

        # ---- Load segmentation mask ----
        mask_path = os.path.join(self.mask_root, study_id + self.mask_suffix)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)  # (H, W, D)

        # ---- add channel dimension ----
        volume = torch.tensor(volume).unsqueeze(0)     # (1, H, W, D)
        mask = torch.tensor(mask).unsqueeze(0)         # (1, H, W, D)

        sample = {"image": volume, "mask": mask}

        # ---- Apply MONAI transforms if provided ----
        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["mask"]
