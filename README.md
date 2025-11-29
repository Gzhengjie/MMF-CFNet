# MMF-CFNet: An Automatic Glacier Extraction Network with Multi-attention for Dynamic Multimodal Feature Fusion

# 📘 Project Title

This repository contains the official implementation of our paper **"MMF-CFNet: An Automatic Glacier Extraction Network with Multi-attention for Dynamic Multimodal Feature Fusion"**.  
# 📄 Overview

This repository contains the official implementation of our paper:
“MMF-CFNet: An Automatic Glacier Extraction Network with Multi-attention for Dynamic Multimodal Feature Fusion”, submitted to Journal/Conference Name, Year.
Our work proposes a multi-level, multi-modal glacier segmentation framework, integrating optical images, SAR data, DEM, and band indices through hierarchical fusion and attention mechanisms. The model is designed to address challenges such as:
+ Multi-modal feature interference 
+ Complex spatial structures in high-resolution remote sensing imagery 
+ Spectral similarity among land cover types 
+ Multi-scale feature aggregation

# 🔧 Features

- Four-branch multi-modal feature extraction (Optical / SAR / Index / DEM)  
- A multi-level feature extraction module based on main-auxiliary structure
- A cross-layer linking module based on multi-attention mechanism
- A feature aggregation module based on W-MSA and SW-MSA 
- End-to-end training & inference
- Reproducible experiments on Tibetan Plateau glacier datasets
- Automatically extract and output glacier segmentation masks

# 📁 Repository Structure

```powershell
├── config/              
│   └── mmf-cfnet.yaml          # config fire
├── data/              # Data description or download scripts 
│   ├── gee_download_dem.js      # Google Earth Engine script for downloading DEM
│   └── gee_download_sentinel-1.js          # GEE script for Sentinel-1 SAR VV/VH data
│   └── gee_download_sentinel-2.js          # GEE script for Sentinel-2 SR optical data
├── models/            # Model architecture and components
│   ├── base/          # Basic blocks, utilities, and common layers    
│   ├── datasets/      # Dataset loader and preprocessing pipelines
│   ├── encoders/      # Multi-modal encoders (Optical, SAR, DEM, Indices)
│   └── mmf-cfnet      # Main network: multi-modal fusion + cross-layer fusion + W-MSA & SW-MSA aggregation
├── sample/            # Sample data and examples
│   ├── image          # Example input images
│   └── label          # Corresponding ground truth segmentation masks      
├── utils/             # Common utility functions 
│                      # (logging, metrics, visualization, losses, etc.)
├── scripts/           # Training and testing scripts
│   ├── train.py       # Training entry point
│   └── test.py        # Evaluation and inference
├── requirements.txt   # Requirements
└── README.md          # Project documentation
```
# 📦 Environment Setup
1. Clone the repository
```bash
  git clone https://github.com/Gzhengjie/MMF-CFNet.git
  cd MMF-CFNet
```
2. Install dependencies
```bash
  pip install -r requirements.txt
```
### Recommended Environment：

- Python 3.8+
- PyTorch ≥ 1.12
- CUDA 11.3 / 11.6
- GDAL  3.4.1
- mmcv / einops / timm

# 📊 Dataset Preparation
Our experiments use multi-modal data from the Tibetan Plateau:
<tr>

| Data Source                      | Description                   |
|:---------------------------------|:------------------------------|
| Sentinel-2 OPT                   | Optical (B2, B3, B4, B8, B11) |
| Sentinel-1 GRD                   | SAR (VV, VH)                  |
| optical-derived spectral indices | NDVI, NDSI, NDWI              |
| Copernicus DEM                   | Elevation data                |
All datasets used in this study were downloaded from Google Earth Engine (GEE), including Sentinel-1, Sentinel-2, Copernicus DEM, and derived spectral indices. 
The provided scripts gee_download_sentinel-1.js, gee_download_sentinel-2.js, and gee_download_dem.js can be used to reproduce the data acquisition process.
https://console.cloud.google.com/earth-engine

# 🚀 Training
```bash
python scripts/train.py --config configs/mmfnet.yaml
```

# 🧪 Testing
```bash
python scripts/test.py --checkpoint checkpoints/model_best.pth
```
# 📈 Results
| **Metric** | **Value** |
|-----------------|-------------|
| IoU             | 0.90        |
| F1 Score        | 0.94        |
| OA              | 0.98        |

### Download trained DAM-CGNet

- The trained model is available for [Google Drive](https://drive.usercontent.google.com/download?id=1O5_n1dK3Jsz8qnp74D5J5cRcpqMByoSg&export=download&authuser=0&confirm=t&uuid=652b3120-d5ed-4929-a91e-54de30357632&at=ALWLOp5jxYaIRMUlrERjRG4dAqUV:1764401341932). To ensure proper access and usage, please follow these steps:

  Click on the Google Drive link.

  Send a request for access by clicking the "Request" button.
  Once your access is granted, you can download the model file.
  Thank you for your understanding, and please feel free to reach out if you encounter any issues.