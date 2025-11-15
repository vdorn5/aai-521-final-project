import os
import zipfile
import pandas as pd
import shutil
from pathlib import Path
from typing import Optional

ZIP_PATH = "/workspace/data/DukeCSpineSeg_"    # Hard coded defualt for data
INTERIM_DIR = "/workspace/data/interim"        # Hard coded default

def extract_zip_files(input_file: str, output_path: str) -> None:
    """Extracts any zip files in the raw dataset into the interim directory.

    Args:
        raw_dir (str): Path to the raw dataset directory.
        interim_dir (str): Path to the interim directory where zip contents will be extracted.

    Returns:
        None
    """
    os.makedirs(output_path, exist_ok=True)
    
    with zipfile.ZipFile(input_file, "r") as zip:
        zip.extractall(output_path)
        
def extract_tsv_files(zip: str = ZIP_PATH, interim_dir: Path = INTERIM_DIR) -> pd.DataFrame:
    """Extracts and combines TSV metadata files from the raw dataset.

    Args:
        raw_dir (str): Path to the raw dataset directory.
        interim_dir (str): Path to the interim directory where combined metadata will be saved.

    Returns:
        Combined Dataframe
    """
    zip = zip + "structured.zip" # path to zip with tsv files
    
    extract_zip_files(zip, interim_dir)
    
    all_tsvs = []
    for root, _, files in os.walk(interim_dir):
        for f in files:
            if f.endswith(".tsv"):
                tsv_path = os.path.join(root, f)
                df = pd.read_csv(tsv_path, sep="\t")
                all_tsvs.append(df)

    
    combined_df = pd.concat(all_tsvs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    return combined_df


def extract_data(zip_path: str = ZIP_PATH, interim_dir: Path = INTERIM_DIR) -> None:
    """Copies MRI images, segmentation masks, and annotation files into interim folders.

    Args:
        raw_dir (str): Path to the raw dataset directory.
        interim_dir (str): Path to the interim directory where files will be copied.

    Returns:
        None
    """
    types = ["segmentation", "imaging_files", "annotation"]
    
    # loop through all zips and extract data 
    for t in types:
        zip_file = zip_path + t + ".zip"
        output_dir = interim_dir / t
        extract_zip_files(zip_file, output_dir)
        
        print(f"Files in {zip_file} extracted.")

    