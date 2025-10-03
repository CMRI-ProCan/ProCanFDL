# ProCanFDL

## Overview
ProCanFDL is a federated deep learning (FDL) framework designed for cancer subtyping using mass spectrometry (MS)-based proteomic data. This repository contains the code used for training local, centralised, and federated models on the ProCan Compendium (n = 7,525 samples from 30 cohorts, representing 19,930 replicate DIA-MS runs) and external validation datasets (n = 887 samples from 10 cohorts). The framework achieved a 43% performance gain on the hold-out test set (n = 625) compared with local models and matched centralised model performance. The codebase also includes utilities for data preparation, model evaluation, and performance visualisation.

## Features

- **Cancer Subtype Classification**: Classify 14 different cancer subtypes using proteomic data
- **Federated Learning**: Train models across distributed datasets whilst preserving data privacy
- **Pretrained Models**: Use our pretrained models for inference on your own data
- **Easy Configuration**: Centralised configuration management for easy customisation
- **Comprehensive Utilities**: Tools for data preparation, evaluation, and SHAP analysis

## Supported Cancer Types

The model can classify the following 14 cancer subtypes:
- Breast carcinoma
- Colorectal adenocarcinoma
- Cutaneous melanoma
- Cutaneous squamous cell carcinoma
- Head and neck squamous
- Hepatocellular carcinoma
- Leiomyosarcoma
- Liposarcoma
- Non-small cell lung adenocarcinoma
- Non-small cell lung squamous
- Oesophagus adenocarcinoma
- Pancreas neuroendocrine
- Pancreatic ductal adenocarcinoma
- Prostate adenocarcinoma

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ProCanFDL.git
cd ProCanFDL
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

### 4. Verify installation
```bash
cd ProCanFDL
python config.py
```

This will check if all required files are present and show the configuration summary.

## Quick Start: Running Inference on Your Data

### Step 1: Prepare Your Data

Your input data should be a CSV file with:
- First column: Sample IDs
- Remaining columns: Protein UniProt IDs as column names
- Values: Log2-transformed protein expression values

**Example format** (see `data/example_input_template.csv`):
```csv
sample_id,P37108,Q96JP5,Q8N697,P36578,O76031,...
sample_001,5.234,3.456,6.789,4.123,5.678,...
sample_002,4.987,3.654,5.432,3.876,4.321,...
```

**Important Notes:**
- Missing proteins will be automatically filled with zeros
- The model expects ~8000 protein features (see `data/data_format.csv` for complete list)
- Data should be log2-transformed and normalised
- Missing values should be handled before input (or will be filled with 0)

### Step 2: Download Pretrained Model

The pretrained ProCanFDL global model will be made available upon publication. Place the pretrained model file in the `pretrained/` directory:
```
pretrained/cancer_subtype_pretrained.pt
```

### Step 3: Run Inference

```bash
cd ProCanFDL
python inference.py --input /path/to/your/data.csv --output predictions.csv
```

**Options:**
```bash
# Basic usage
python inference.py --input my_data.csv --output predictions.csv

# Use a custom model
python inference.py --input my_data.csv --output predictions.csv --model path/to/custom_model.pt

# Skip feature alignment (if your data already has correct feature order)
python inference.py --input my_data.csv --output predictions.csv --no_align
```

### Step 4: Interpret Results

The output CSV file will contain:
- `sample_id`: Your sample identifier
- `predicted_class`: Predicted cancer subtype
- `predicted_class_id`: Numeric class ID
- `confidence`: Prediction confidence (max probability)
- `prob_[cancer_type]`: Probability for each cancer type

**Example output:**
```csv
sample_id,predicted_class,predicted_class_id,confidence,prob_Breast_carcinoma,...
sample_001,Breast_carcinoma,0,0.92,0.92,...
sample_002,Colorectal_adenocarcinoma,1,0.85,0.05,...
```

## Training Models

### Configuration

All paths and hyperparameters are centralised in `config.py`. Key settings based on the published model:

```python
# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Hyperparameters (optimised values from publication)
DEFAULT_HYPERPARAMETERS = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'hidden_dim': 256,
    'dropout': 0.2,
    'batch_size': 100,
    'epochs': 200,
}

# Cancer types (14 subtypes for ProCan Compendium)
DEFAULT_CANCER_TYPES = [...]  # List of 14 cancer types
```

You can modify these settings by editing `config.py` or by overriding them in your training scripts.

### Training on ProCan Compendium

This script trains models using the ProCan Compendium dataset (n = 7,525 samples from 30 cohorts) with federated learning:

```bash
cd ProCanFDL
python ProCanFDLMain_compendium.py
```

**What it does:**
- Simulates federated learning across 4 local sites
- Site 1: Contains cohort 1 (pan-cancer cohort with 14 cancer subtypes)
- Sites 2-4: Randomly distributed subsets of cohorts 2-30
- Trains local models on different cohorts
- Aggregates models using federated averaging (FedAvg) over 10 iterations
- Evaluates on held-out test set (10% of cohorts 2-30)
- Saves model checkpoints and performance metrics

**Output:**
- Models saved to: `models/Fed/[experiment_name]/`
- Results saved to: `results/[experiment_name]/`

### Training with External Validation Datasets

This script includes external datasets (DIA-MS and TMT from CPTAC) for validation, expanding to 16 cancer subtypes:

```bash
cd ProCanFDL
python ProCanFDLMain_external.py
```

**Additional features:**
- Integrates data from two distinct MS technologies (DIA-MS and TMT)
- Z-score normalisation per dataset batch for platform harmonisation
- Handles heterogeneous data sources across 6 simulated sites
- Extended validation across multiple cohorts (n = 887 external samples)
- Evaluates on 16 cancer subtypes (adds high-grade serous ovarian carcinoma and clear-cell renal cell carcinoma)

### Customising Training

To customise training, modify the parameters at the top of the training scripts:

```python
# Number of federated learning clients
N_clients = 3
N_included_clients = 3

# Number of training repeats and iterations
N_repeats = 10
N_iters = 10

# Override hyperparameters from config
hypers = config.DEFAULT_HYPERPARAMETERS.copy()
hypers['epochs'] = 300  # Increase training epochs
hypers['batch_size'] = 128  # Larger batch size
```

## SHAP Analysis

Run SHAP (SHapley Additive exPlanations) analysis to understand feature importance:

```bash
cd ProCanFDL
python RunSHAP.py
```

**What it does:**
- Loads trained model
- Computes SHAP values for all features
- Identifies important proteins for each prediction
- Saves SHAP values for further analysis

**Output:**
- SHAP values saved as pickle files in model directory
- Can be loaded for visualisation and analysis

## Project Structure

```
ProCanFDL/
├── ProCanFDL/
│   ├── config.py              # Configuration management
│   ├── inference.py           # Inference script for pretrained models
│   ├── FedModel.py           # Neural network architecture
│   ├── FedTrain.py           # Training and evaluation class
│   ├── FedAggregateWeights.py # Federated learning aggregation
│   ├── ProCanFDLMain_compendium.py  # Train on ProCan Compendium
│   ├── ProCanFDLMain_external.py    # Train with external data
│   ├── RunSHAP.py            # SHAP analysis
│   └── utils/
│       ├── ProtDataset.py    # PyTorch dataset class
│       ├── utils.py          # Utility functions
│       └── ...
├── data/
│   ├── example_input_template.csv  # Example input format
│   ├── data_format.csv       # Complete feature list
│   └── ...                   # Your data files
├── pretrained/
│   └── cancer_subtype_pretrained.pt  # Pretrained model
├── models/                   # Saved models
├── results/                  # Training results
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Data Requirements

### For Inference
- **Format**: CSV file with samples × proteins
- **Proteins**: UniProt IDs as column names
- **Values**: Log2-transformed protein expression
- **Sample handling**: Missing proteins filled with 0

### For Training
Required files (place in `data/` directory):
- `all_protein_list_mapping.csv`: Protein ID to gene name mapping
- `P10/E0008_P10_protein_averaged_log2_transformed_EB.csv`: ProCan training data
- `P10/replicate_corr_protein.csv`: Sample quality metrics
- `sample_info/sample_metadata_path_noHek_merged_replicates_Adel_EB_2.0_no_Mucinous.xlsx`: Sample metadata

**Data Availability:**
The raw DIA-MS data and processed data of cohort 1 (ProCan Compendium pan-cancer cohort) and the corresponding spectral library have been deposited in the Proteomics Identification database (PRIDE) under dataset identifier **PXD056810**.

Optional (for external validation):
- External DIA-MS datasets: PXD019549, PXD007810
- CPTAC TMT datasets available at the Proteomic Data Commons (PDC): https://proteomic.datacommons.cancer.gov/pdc/cptac-pancancer

## Troubleshooting

### Common Issues

**1. Missing data files**
```
FileNotFoundError: Protein mapping file not found
```
**Solution**: Ensure `data/all_protein_list_mapping.csv` exists. Run `python config.py` to check file status.

**2. Model not found**
```
FileNotFoundError: Model file not found: pretrained/cancer_subtype_pretrained.pt
```
**Solution**: Download the pretrained model and place it in `pretrained/` directory.

**3. Feature dimension mismatch**
```
RuntimeError: size mismatch
```
**Solution**: Ensure your input data is properly aligned with expected features. Don't use `--no_align` flag unless certain.

**4. GPU/CUDA issues**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `config.py` or training scripts, or run on CPU (automatic fallback).

### Getting Help

- Check the [Issues](https://github.com/yourusername/ProCanFDL/issues) page
- Review the example input format in `data/example_input_template.csv`
- Run `python config.py` to verify your setup
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Advanced Usage

### Custom Model Architecture

To modify the model architecture, edit `FedModel.py`:

```python
class FedProtNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(FedProtNet, self).__init__()
        # Modify architecture here
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Add layer
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
```

### Custom Loss Functions

To use different loss functions, modify `FedTrain.py`:

```python
# In TrainFedProtNet.__init__
self.criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss
# or
self.criterion = FocalLoss()  # Custom loss
```


## Performance Benchmarks

Performance on held-out test set from ProCan Compendium (14 cancer subtypes):

| Model Type | Macro-averaged AUROC | Accuracy |
|-----------|----------|----------|
| Centralised | 0.9999 | 0.990 |
| ProCanFDL (Global) | 0.9992 | 0.965 |
| Site 1 (Local) | 0.9805 | 0.847 |
| Site 2 (Local) | 0.9502 | - |
| Site 3 (Local) | 0.9680 | - |
| Site 4 (Local) | 0.9522 | - |

External validation (16 cancer subtypes including DIA-MS and TMT data):

| Model Type | Macro-averaged AUROC | Accuracy |
|-----------|----------|----------|
| Centralised | 0.9999 | - |
| ProCanFDL (Global) | 0.9987 | - |
| Local models | 0.5162-0.9133 | - |

ProCanFDL achieved a **43% performance improvement** over local models whilst matching centralised model performance and maintaining data privacy.

*Results from Cai et al., Cancer Discovery 2025

## Citation

If you use ProCanFDL in your research, please cite:

```bibtex
@article{cai2025procanfdl,
  title={Federated Deep Learning Enables Cancer Subtyping by Proteomics},
  author={Cai, Zhaoxiang and Boys, Emma L and Noor, Zainab and Aref, Adel T and Xavier, Dylan and Lucas, Natasha and Williams, Steven G and Koh, Jennifer MS and Poulos, Rebecca C and Wu, Yangxiu and Dausmann, Michael and MacKenzie, Karen L and others},
  journal={Cancer Discovery},
  year={2025},
  doi={10.1158/2159-8290.CD-24-1488},
  publisher={American Association for Cancer Research}
}
```

**Full reference:**
Cai Z, Boys EL, Noor Z, Aref AT, Xavier D, Lucas N, et al. Federated Deep Learning Enables Cancer Subtyping by Proteomics. Cancer Discovery. 2025. DOI: 10.1158/2159-8290.CD-24-1488

## License

This work is distributed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) licence.

©2025 The Authors; Published by the American Association for Cancer Research

## Contact

For questions and support:
- Create an issue on GitHub
- Email the corresponding authors:
  - Peter G. Hains: phains@cmri.org.au
  - Phillip J. Robinson: probinson@cmri.org.au
  - Qing Zhong: qzhong@cmri.org.au
  - Roger R. Reddel: rreddel@cmri.org.au

**ProCan**  
Children's Medical Research Institute  
Faculty of Medicine and Health, The University of Sydney  
214 Hawkesbury Road, Westmead 2145, Australia

## Acknowledgements

This work was supported by:
- Australian Cancer Research Foundation
- Cancer Institute NSW (2017/TPG001, REG171150, 15/TRC/1-01)
- NSW Ministry of Health (CMP-01)
- The University of Sydney
- Cancer Council NSW (IG 18-01)
- Ian Potter Foundation
- Medical Research Future Fund
- National Health and Medical Research Council (NHMRC) of Australia (GNT1170739, GNT1138536, GNT1137064, GNT2007839, GNT2018514)
- National Breast Cancer Foundation (IIRS-18-164)
- Sydney Cancer Partners Translational Partners Fellowship (Cancer Institute NSW Capacity Building Grant 2021/CBG0002)

ProCan is conducted under the auspices of a Memorandum of Understanding between Children's Medical Research Institute and the U.S. National Cancer Institute's International Cancer Proteogenomics Consortium, encouraging cooperation in proteogenomic cancer research.

**Biospecimen Contributors:**
The study includes samples from 20 research groups across eight countries: Australia, USA, Canada, Spain, Austria, Sweden, and Greece. We gratefully acknowledge all collaborating institutions and the patients who consented to participate in research.

For complete acknowledgements, please see the published paper: Cai et al., Cancer Discovery 2025.