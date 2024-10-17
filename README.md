# ProCanFDL

## Overview
ProCanFDL is a federated deep learning (FDL) framework designed for cancer subtyping using mass spectrometry (MS)-based proteomic data. This repository contains the code used for training local, centralised, and federated models on the ProCan Compendium, as well as external validation datasets. The codebase also includes utilities for data preparation, model evaluation, and performance visualisation.

## Installation
1. To install ProCanFDL, clone the repository and install the required dependencies using the following commands:
```pip install -r requirements.txt```
2. Install PyTorch using the following command:
```pip install torch torchvision torchaudio``` 

## Run ProCanFDL with the ProCan Compendium
```python ProCanFDLMain_compendium.py```

## Run ProCanFDL with external validation datasets
```python ProCanFDLMain_external.py```

## Run SHAP analysis
```python RunSHAP.py```