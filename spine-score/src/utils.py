import os
import glob
import numpy as np
from typing import List, Union
from pydicom import dcmread, FileDataset, DataElement
from pydicom.tag import Tag


class SpinalScan:
    def __init__(self, volume, pixel_spacing, slice_thickness):
        self.volume = volume
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness


def load_dicoms(
    paths: List[Union[str, bytes, os.PathLike]],
    require_extensions: bool = True,
    metadata_overwrites: dict = {},
) -> SpinalScan:
    if require_extensions:
        assert all(p.endswith(".dcm") for p in paths), \
            "All paths must end with .dcm"

    dicom_files = [dcmread(path) for path in paths]
    dicom_files = [overwrite_tags(df, metadata_overwrites) for df in dicom_files]

    # verify tags and orientation
    for idx, df in enumerate(dicom_files):
        missing_tags = check_missing_tags(df)
        if len(missing_tags) > 0:
            raise ValueError(
                f"Missing tags in {paths[idx]}: {missing_tags}"
            )

        if not is_sagittal_dicom_slice(df):
            raise ValueError(
                f"File is not sagittal: {paths[idx]}"
            )

    # sort by slice index
    dicom_files = sorted(dicom_files, key=lambda df: df.InstanceNumber)

    # construct volume
    pixel_spacing = np.mean([np.array(df.PixelSpacing) for df in dicom_files], axis=0)
    slice_thickness = np.mean([df.SliceThickness for df in dicom_files])

    volume = np.stack([df.pixel_array for df in dicom_files], axis=-1)

    return SpinalScan(
        volume=volume,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness
    )


def load_dicoms_from_folder(
    path: Union[str, bytes, os.PathLike],
    require_extensions: bool = True,
    metadata_overwrites: dict = {},
) -> SpinalScan:
    """
    Load a DICOM series from a folder containing slices.
    """
    slices = [
        f for f in glob.glob(os.path.join(path, "*"))
        if is_dicom_file(f)
    ]

    return load_dicoms(slices, require_extensions, metadata_overwrites)


def is_sagittal_dicom_slice(dicom_file: FileDataset) -> bool:
    if Tag("ImageOrientationPatient") in dicom_file:
        io = np.array(dicom_file.ImageOrientationPatient).round()
        return (io[[0, 3]] == [0, 0]).all()
    else:
        raise ValueError("Missing ImageOrientationPatient tag")


def overwrite_tags(dicom_file: FileDataset, metadata_overwrites: dict) -> FileDataset:
    possible_overwrites = {
        "PixelSpacing": "DS",
        "SliceThickness": "DS",
        "ImageOrientationPatient": "DS",
    }

    for tag, value in metadata_overwrites.items():
        if tag not in possible_overwrites:
            raise NotImplementedError(f"Cannot overwrite tag {tag}")

        if Tag(tag) in dicom_file:
            dicom_file[Tag(tag)] = DataElement(Tag(tag),
                                            possible_overwrites[tag],
                                            value)
        else:
            dicom_file.add_new(Tag(tag),
                            possible_overwrites[tag],
                            value)
    return dicom_file


def check_missing_tags(dicom_file: FileDataset):
    required = ["PixelData", "PixelSpacing", "SliceThickness", "InstanceNumber"]
    return [t for t in required if Tag(t) not in dicom_file]


def is_dicom_file(path: Union[str, bytes, os.PathLike]) -> bool:
    try:
        dcmread(path)
        return True
    except:
        return False
