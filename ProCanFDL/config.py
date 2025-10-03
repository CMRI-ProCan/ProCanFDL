"""
Configuration file for ProCanFDL
Centralized configuration for paths, hyperparameters, and settings
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ============================================================================
# Directory Paths
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"

# Create directories if they don't exist
for directory in [MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Files
# ============================================================================
# Protein mapping file
PROTEIN_MAPPING_FILE = DATA_DIR / "all_protein_list_mapping.csv"

# Main data files
PROCAN_DATA_FILE = (
    DATA_DIR / "P10" / "E0008_P10_protein_averaged_log2_transformed_EB.csv"
)
PROCAN_EXTERNAL_DATA_FILE = (
    DATA_DIR
    / "P10"
    / "external"
    / "DIA_datasets"
    / "E0008_P10_protein_averaged_log2_transformed_EB_cptac_dia_no_norm_20240820.csv"
)
REPLICATE_CORR_FILE = DATA_DIR / "P10" / "replicate_corr_protein.csv"
SAMPLE_METADATA_FILE = (
    DATA_DIR
    / "sample_info"
    / "sample_metadata_path_noHek_merged_replicates_Adel_EB_2.0_no_Mucinous.xlsx"
)

# Pretrained model
PRETRAINED_MODEL_PATH = PRETRAINED_DIR / "cancer_subtype_pretrained.pt"

# ============================================================================
# Model Hyperparameters
# ============================================================================
DEFAULT_HYPERPARAMETERS = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "hidden_dim": 256,
    "dropout": 0.2,
    "batch_size": 100,
    "epochs": 200,
}

# ============================================================================
# Cancer Type Classifications
# ============================================================================
# Default target and cancer types for the 14-class model
DEFAULT_TARGET = "cancer_type_2"
DEFAULT_CANCER_TYPES = [
    "Breast_carcinoma",
    "Colorectal_adenocarcinoma",
    "Cutaneous_melanoma",
    "Cutaneous_squamous_cell_carcinoma",
    "Head_and_neck_squamous",
    "Hepatocellular_carcinoma",
    "Leiomyosarcoma",
    "Liposarcoma",
    "Non_small_cell_lung_adeno",
    "Non_small_cell_lung_squamous",
    "Oesophagus_adenocarcinoma",
    "Pancreas_neuroendocrine",
    "Pancreatic_ductal_adenocarcinoma",
    "Prostate_adenocarcinoma",
]

# Extended cancer types (16-class model)
EXTENDED_CANCER_TYPES = DEFAULT_CANCER_TYPES + [
    "High_grade_serous_ovary",
    "Clear_cell_renal_cell_carcinoma",
]

# Alternative classifications
BROAD_CANCER_TYPES = [
    "Adenocarcinoma",
    "Sarcoma",
    "Squamous",
    "Melanoma",
    "Neuroendocrine",
]
TISSUE_TYPES = [
    "Lung",
    "Colorectal",
    "Breast",
    "Ovary",
    "Stomach/Oesophagus",
    "Prostate",
    "Pancreas",
    "Liver",
]

# ============================================================================
# Data Processing Parameters
# ============================================================================
META_COL_NUMS = 73  # Number of metadata columns before protein data
QUALITY_THRESHOLD = 0.9  # Pearson correlation threshold for sample quality

# ============================================================================
# Federated Learning Parameters
# ============================================================================
FED_DEFAULT_PARAMS = {
    "n_clients": 3,
    "n_included_clients": 3,
    "n_repeats": 10,
    "n_iters": 10,
}

# ============================================================================
# Random Seeds
# ============================================================================
RANDOM_SEED = 0


# ============================================================================
# Device Configuration
# ============================================================================
def get_device():
    """Get the available device (GPU if available, else CPU)"""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Helper Functions
# ============================================================================
def get_model_path(experiment_name):
    """Get path for saving/loading models"""
    path = MODEL_DIR / "Fed" / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_path(experiment_name=None):
    """Get path for saving results"""
    if experiment_name:
        path = RESULTS_DIR / experiment_name
    else:
        path = RESULTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_required_files():
    """
    Check if required files exist and provide helpful error messages
    Returns: tuple (bool, list) - (all_exist, missing_files)
    """
    required_files = [
        (PROTEIN_MAPPING_FILE, "Protein mapping file"),
    ]

    optional_files = [
        (PROCAN_DATA_FILE, "ProCan compendium data (required for training)"),
        (REPLICATE_CORR_FILE, "Replicate correlation file (required for training)"),
        (SAMPLE_METADATA_FILE, "Sample metadata file (required for training)"),
        (PRETRAINED_MODEL_PATH, "Pretrained model (required for inference)"),
    ]

    missing_required = []
    missing_optional = []

    for file_path, description in required_files:
        if not file_path.exists():
            missing_required.append(f"  - {description}: {file_path}")

    for file_path, description in optional_files:
        if not file_path.exists():
            missing_optional.append(f"  - {description}: {file_path}")

    return len(missing_required) == 0, missing_required, missing_optional


def print_config_summary():
    """Print a summary of the current configuration"""
    print("=" * 80)
    print("ProCanFDL Configuration Summary")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Pretrained Directory: {PRETRAINED_DIR}")
    print(f"\nDefault Target: {DEFAULT_TARGET}")
    print(f"Number of Cancer Types: {len(DEFAULT_CANCER_TYPES)}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 80)


if __name__ == "__main__":
    print_config_summary()
    all_exist, missing_required, missing_optional = check_required_files()

    if not all_exist:
        print("\n❌ Missing Required Files:")
        for item in missing_required:
            print(item)

    if missing_optional:
        print("\n⚠️  Missing Optional Files:")
        for item in missing_optional:
            print(item)

    if all_exist and not missing_optional:
        print("\n✅ All files present!")

