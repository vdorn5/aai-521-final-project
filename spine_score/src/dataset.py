import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import pydicom
import nibabel as nib
from scipy.ndimage import zoom

def load_dicom_volume(dicom_folder):
    dicom_files = [pydicom.dcmread(f) for f in sorted(glob.glob(os.path.join(dicom_folder, "*.dcm")))]
    dicom_files.sort(key=lambda x: int(x.InstanceNumber))
    volume = np.stack([f.pixel_array for f in dicom_files], axis=-1)  # (H, W, D)
    return volume.astype(np.float32)

def load_mask(mask_file):
    mask = nib.load(mask_file).get_fdata()
    return mask.astype(np.uint8)

def get_dicom_spacing(dicom_folder):
    dicom_files = [pydicom.dcmread(f) for f in sorted(glob.glob(os.path.join(dicom_folder, "*.dcm")))]
    dicom_files.sort(key=lambda x: int(x.InstanceNumber))
    first_slice = dicom_files[0]
    row_spacing, col_spacing = map(float, first_slice.PixelSpacing)
    try:
        slice_thickness = float(first_slice.SliceThickness)
    except AttributeError:
        slice_thickness = float(first_slice.SpacingBetweenSlices)
    return (row_spacing, col_spacing, slice_thickness)  # (H, W, D)

def resample_volume_mask(volume, mask, orig_spacing, target_spacing=(1.0,1.0,1.0)):
    """
    Resample volume and mask to isotropic spacing.
    """
    zoom_factors = [1] + [o/t for o, t in zip(orig_spacing, target_spacing)]
    volume_resampled = zoom(volume, zoom_factors, order=1)
    mask_resampled   = zoom(mask,   zoom_factors, order=0)
    return volume_resampled, mask_resampled

def crop_or_pad(volume, target_shape):
    """
    Safely crop or pad a 4D volume (C,H,W,D) to target_shape (C,tH,tW,tD)
    """
    C, H, W, D = volume.shape
    _, tH, tW, tD = target_shape

    vol = np.zeros(target_shape, dtype=volume.dtype)

    # Height
    if H >= tH:
        h_start = (H - tH)//2
        h_end = h_start + tH
        h_slice_src = slice(h_start, h_end)
        h_slice_dst = slice(0, tH)
    else:
        pad_before = (tH - H)//2
        h_slice_src = slice(0, H)
        h_slice_dst = slice(pad_before, pad_before+H)

    # Width
    if W >= tW:
        w_start = (W - tW)//2
        w_end = w_start + tW
        w_slice_src = slice(w_start, w_end)
        w_slice_dst = slice(0, tW)
    else:
        pad_before = (tW - W)//2
        w_slice_src = slice(0, W)
        w_slice_dst = slice(pad_before, pad_before+W)

    # Depth
    if D >= tD:
        d_start = (D - tD)//2
        d_end = d_start + tD
        d_slice_src = slice(d_start, d_end)
        d_slice_dst = slice(0, tD)
    else:
        pad_before = (tD - D)//2
        d_slice_src = slice(0, D)
        d_slice_dst = slice(pad_before, pad_before+D)

    # Copy from source to destination
    vol[:, h_slice_dst, w_slice_dst, d_slice_dst] = volume[:, h_slice_src, w_slice_src, d_slice_src]

    return vol


class SpineSegDataset(Dataset):
    def __init__(self, dataset_entries, target_shape=(1,220,220,45), target_spacing=(1.0,1.0,1.0)):
        """
        dataset_entries: list of tuples (dicom_folder, mask_file)
        target_shape: final output shape (C,H,W,D)
        target_spacing: voxel spacing in mm (H,W,D)
        """
        self.entries = dataset_entries
        self.target_shape = target_shape
        self.target_spacing = target_spacing

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # Added patient_id (Edited)
        patient_id, dicom_path, mask_path = self.entries[idx]

        # Load volume and mask
        volume = load_dicom_volume(dicom_path)  # (H,W,D)
        mask   = load_mask(mask_path)           # (H,W,D)
        orig_spacing = get_dicom_spacing(dicom_path)

        # Align the mask to DICOM
        mask = np.rot90(mask, k=1)      # rotate 90 degrees
        mask = np.flip(mask, axis=2)    # flip along vertical axis

        # Add channel dimension
        volume = np.expand_dims(volume, axis=0)  # (1,H,W,D)
        mask   = np.expand_dims(mask, axis=0)

        # Resample to target spacing
        volume, mask = resample_volume_mask(volume, mask, orig_spacing, self.target_spacing)

        # Crop or pad to target shape
        volume = crop_or_pad(volume, self.target_shape)
        mask   = crop_or_pad(mask, self.target_shape)

        # Convert to torch tensors
        volume = torch.from_numpy(volume).float()
        mask   = torch.from_numpy(mask).long()

        # Return volume, mask, and patient ID           (Edited)
        from pathlib import Path                      # (Edited) (added patient_id)
        patient_id = Path(dicom_path).parents[1].name # (Edited)

        if idx == 0:
            print("---- DEBUG in __getitem__ ----")
            print("volume shape:", volume.shape, " dtype:", volume.dtype,
                " min/max:", volume.min(), volume.max())
            print("mask shape:", mask.shape, " dtype:", mask.dtype,
                " unique:", np.unique(mask))
        
        return volume, mask.squeeze(0), patient_id    # (Edited)
