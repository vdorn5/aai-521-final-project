<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# aai-521-final-project
=======
# spine-score
>>>>>>> 467bff1 (Initial commit)
=======
# spine-score
=======
# Lumbar Disk & Vertebrae Segmentation (3D Medical Imaging)

**Course:** USD Applied Computer Vision for AI (AAI-521-04)  
**Project:** Cervical Spine Disk Segmentation  
**Team Names:** Victoria Dorn, Jacob Tassos, Mostafa Zamaniturk, Puja Nandini   
**Group Number:** Group 3

This repository provides a full deep-learning pipeline for **3D spine segmentation**, including:

- DICOM loading  
- NIfTI mask handling  
- Resampling to isotropic voxel spacing  
- Preprocessing + dataset class  
- 3D U-Net model training  
- Evaluation metrics (Dice, Accuracy, Precision, Recall, F1)  
- GPU training using **Docker + Dev Containers**

This project is built to run inside a **VS Code Dev Container**, ensuring a reproducible and GPU-ready environment.
>>>>>>> 6b401ee (Updated model training pipeline, eda, and documentation)


---

# Quick Start

<<<<<<< HEAD
update to python 10
>>>>>>> 2ff17a8 (WIP load data working)
=======
## 1. Requirements
You need:

- **Docker** (required for GPU training)
- **Docker Desktop** or the NVIDIA-Docker runtime
- **Visual Studio Code**
- **Dev Containers extension**
- An **NVIDIA GPU** + CUDA drivers (for accelerated training)


## 2. Download Data

The training data is **not included** due to size and data rights.

Download the MIDRC dataset from: 

**https://data.midrc.org/discovery/H6K0-A61V**

Store the `.zip` files in the `data/` folder to work with current structure.

## 3. Open Development Environment (Dev Container)

This project includes a `.devcontainer/` directory and the example walkthrough here leverages that setup. 

Open this project in VS Code → it will prompt:

> **"Reopen in Container?" → Yes**

The Dev Container automatically sets up:

- Python 3  
- PyTorch with GPU support  
- All required dependencies  
- A consistent environment for running the notebook & scripts  


## 3. Run Notebook

The notebook (`disk_segmentation_pipeline.ipynb`)will walk through the steps to extract, transform, and load this data to use it with model training. It also will walk through an initial UNet modeling example.

### Model Weights

The `weights/` folder uses Git LFS to store large model files.

> Note: If you don't want to retrain the model just use the load model code at the bottom of the notebook file. 

---

# Repository Structure Overview
```
.
│
├── src/
│ ├── dataset.py # SpineSegDataset implementation
│ ├── extract.py # Handles unzipping and data prep for the dataloader
│ └── model.py # UNet3D, training loop, metrics
│
├── weights/ # Stored model weights (Git LFS)
│ └── unet3d_weights_vX.X.pth
│
├── data/ # Dataset folder (EMPTY initially)
│ └── # You will place downloaded files here
│
├── notebooks/
│ └── spine_segmentation_walkthrough.ipynb
│   # Full end-to-end tutorial:
│   # - DICOM loading
│   # - Preprocessing
│   # - Training
│   # - Evaluation
│   # - Visualization
│
├── .devcontainer/ # VS Code Dev Container config
└── README.md # This file
```

---

# Citation & References

This project makes use of the **Duke University Cervical Spine MRI Segmentation Dataset (CSpineSeg)** published via MIDRC.

### Dataset Citation (APA)
Zhou, L., Wiggins, W., Zhang, J., Colglazier, R., Willhite, J., Dixon, A., Malinzak, M., Gu, H., Mazurowski, M. A., & Calabrese, E. (2025).  
*The Duke University Cervical Spine MRI Segmentation Dataset (CSpineSeg)* [Data set].  
Medical Imaging and Data Resource Center (MIDRC).  
[https://doi.org/10.60701/H6K0-A61V](https://doi.org/10.60701/H6K0-A61V)

### Dataset Details
- **Title:** The Duke University Cervical Spine MRI Segmentation Dataset (CSpineSeg)  
- **Publisher:** Medical Imaging and Data Resource Center (MIDRC)  
- **Publication Year:** 2025  
- **Resource Type:** Dataset  
- **Files:** 1 data package  
- **Data Description:** 1255 cervical spine MRI series (without contrast) from 1232 patients, with 1255 corresponding segmentation masks.

### Access the Dataset
- Dataset page: [https://data.midrc.org/discovery/H6K0-A61V](https://data.midrc.org/discovery/H6K0-A61V)  
- DOI link: [https://doi.org/10.60701/H6K0-A61V](https://doi.org/10.60701/H6K0-A61V)  
- Contact: [https://www.midrc.org/midrc-contact](https://www.midrc.org/midrc-contact)

### BibTeX Entry
```bibtex
@dataset{CSpineSeg2025,
  author = {Zhou, Longfei and Wiggins, Walter and Zhang, Jikai and Colglazier, Roy and Willhite, Jay and Dixon, Austin and Malinzak, Michael and Gu, Hanxue and Mazurowski, Maciej A. and Calabrese, Evan},
  title = {The Duke University Cervical Spine MRI Segmentation Dataset (CSpineSeg)},
  year = {2025},
  publisher = {Medical Imaging and Data Resource Center (MIDRC)},
  doi = {10.60701/H6K0-A61V},
  url = {https://data.midrc.org/discovery/H6K0-A61V},
  note = {Dataset}
}
>>>>>>> 6b401ee (Updated model training pipeline, eda, and documentation)
