"""
Inference Script for ProCanFDL
Run the pretrained model on custom proteomic data
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import project modules
from FedModel import FedProtNet
from FedTrain import TrainFedProtNet
from utils.ProtDataset import ProtDataset
import config


def load_protein_mapping():
    """Load protein ID to name mapping"""
    if not config.PROTEIN_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Protein mapping file not found: {config.PROTEIN_MAPPING_FILE}\n"
            "Please ensure the data files are in the correct location."
        )

    mapping_df = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=0)
    prot_id_to_name = mapping_df.to_dict()["Gene"]
    prot_name_to_id = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=3).to_dict()[
        "UniProtID"
    ]

    return prot_id_to_name, prot_name_to_id


def load_pretrained_model(model_path=None):
    """
    Load the pretrained model

    Args:
        model_path: Path to model file (if None, uses default pretrained model)

    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = config.PRETRAINED_MODEL_PATH
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure the pretrained model is available at: {config.PRETRAINED_MODEL_PATH}\n"
            "You can download it from the project repository or train your own model."
        )

    # Get device
    device = config.get_device()

    # Load model state dict to determine architecture
    state_dict = torch.load(model_path, map_location=device)

    # Infer model dimensions from state dict
    input_dim = state_dict["fc1.weight"].shape[1]
    hidden_dim = state_dict["fc1.weight"].shape[0]
    num_classes = state_dict["fc2.weight"].shape[0]

    # Create model
    model = FedProtNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=config.DEFAULT_HYPERPARAMETERS["dropout"],
    ).to(device)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Model loaded successfully!")
    print(f"   Input dimension: {input_dim}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   Number of classes: {num_classes}")

    return model, device


def prepare_input_data(data_file, expected_features=None):
    """
    Prepare input data for inference

    Args:
        data_file: Path to CSV file with proteomic data
        expected_features: List of expected feature names (protein IDs) in order

    Returns:
        Prepared data as numpy array
    """
    # Load data
    data = pd.read_csv(data_file, index_col=0)

    print(f"\n📊 Input data shape: {data.shape}")
    print(f"   Samples: {data.shape[0]}")
    print(f"   Features: {data.shape[1]}")

    # If expected features are provided, align the data
    if expected_features is not None:
        print(f"\n🔄 Aligning features with model expectations...")

        # Check which features are present
        present_features = [f for f in expected_features if f in data.columns]
        missing_features = [f for f in expected_features if f not in data.columns]

        print(f"   Present features: {len(present_features)}/{len(expected_features)}")

        if missing_features:
            print(f"   ⚠️  Missing {len(missing_features)} features")
            if len(missing_features) <= 10:
                print(f"      Missing: {missing_features}")

        # Create aligned dataframe with zeros for missing features
        aligned_data = pd.DataFrame(0, index=data.index, columns=expected_features)

        # Fill in present features
        for feature in present_features:
            aligned_data[feature] = data[feature]

        data = aligned_data

    # Fill NaN values with 0
    data = data.fillna(0)

    return data.to_numpy(), data.index.tolist()


def get_feature_order():
    """
    Get the expected order of features from the data format file

    Returns:
        List of feature names in expected order
    """
    # Try to load feature order from data format file
    data_format_file = config.DATA_DIR / "data_format.csv"

    if data_format_file.exists():
        # Read the header to get protein IDs
        format_df = pd.read_csv(data_format_file, nrows=0)
        # Get protein IDs (columns after metadata)
        protein_cols = [
            col
            for col in format_df.columns
            if col
            not in [
                "prep_lims_id",
                "tracking_lims_id",
                "tracking_filename_sample_info",
                # Add other metadata column names...
            ]
        ]
        return protein_cols

    # If data format file doesn't exist, try to infer from training data
    if config.PROCAN_DATA_FILE.exists():
        print("⚠️  Using feature order from training data...")
        train_df = pd.read_csv(config.PROCAN_DATA_FILE, index_col=0, nrows=0)
        # Skip metadata columns
        protein_cols = list(train_df.columns[config.META_COL_NUMS :])
        return protein_cols

    return None


def run_inference(model, data, device, cancer_types=None):
    """
    Run inference on the data

    Args:
        model: Loaded PyTorch model
        data: numpy array of shape (n_samples, n_features)
        device: torch device
        cancer_types: List of cancer type names (optional)

    Returns:
        DataFrame with predictions and probabilities
    """
    # Convert to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(data_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    # Convert to numpy
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()

    # Create results dataframe
    results = pd.DataFrame()

    # Use cancer type names if provided
    if cancer_types is None:
        cancer_types = config.DEFAULT_CANCER_TYPES

    # Ensure we have the right number of cancer types
    if len(cancer_types) != probabilities.shape[1]:
        print(
            f"⚠️  Number of cancer types ({len(cancer_types)}) doesn't match model output ({probabilities.shape[1]})"
        )
        cancer_types = [f"Class_{i}" for i in range(probabilities.shape[1])]

    # Add predicted class
    results["predicted_class"] = [cancer_types[p] for p in predictions]
    results["predicted_class_id"] = predictions

    # Add probabilities for each class
    for i, cancer_type in enumerate(cancer_types):
        results[f"prob_{cancer_type}"] = probabilities[:, i]

    # Add confidence (max probability)
    results["confidence"] = probabilities.max(axis=1)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ProCanFDL inference on custom proteomic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with default pretrained model
  python inference.py --input my_data.csv --output predictions.csv
  
  # Run inference with custom model
  python inference.py --input my_data.csv --output predictions.csv --model path/to/model.pt
  
  # Run inference for 16-class model
  python inference.py --input my_data.csv --output predictions.csv --model_type extended

Input file format:
  - CSV file with samples as rows and proteins as columns
  - First column should be sample IDs
  - Column names should be UniProt IDs
  - Missing values will be filled with zeros
  - Example:
      sample_id,P37108,Q96JP5,Q8N697,...
      sample1,5.23,3.45,6.78,...
      sample2,4.12,3.98,5.67,...
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file with proteomic data"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output CSV file for predictions"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Path to model file (default: use pretrained model)",
    )
    parser.add_argument(
        "--model_type",
        choices=["default", "extended"],
        default="default",
        help="Model type: default (14 classes) or extended (16 classes)",
    )
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="Skip feature alignment (use if data already has correct feature order)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ProCanFDL Inference")
    print("=" * 80)

    # Check input file exists
    input_file = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load model
    print(f"\n📦 Loading model...")
    model, device = load_pretrained_model(args.model)

    # Get feature order
    feature_order = None if args.no_align else get_feature_order()
    if feature_order is not None:
        print(f"\n📋 Expected features: {len(feature_order)}")

    # Load and prepare data
    print(f"\n📂 Loading input data from: {input_file}")
    data, sample_ids = prepare_input_data(input_file, feature_order)

    # Select cancer types based on model type
    if args.model_type == "extended":
        cancer_types = config.EXTENDED_CANCER_TYPES
    else:
        cancer_types = config.DEFAULT_CANCER_TYPES

    # Run inference
    print(f"\n🔮 Running inference...")
    results = run_inference(model, data, device, cancer_types)

    # Add sample IDs
    results.insert(0, "sample_id", sample_ids)

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print(f"\n✅ Inference complete!")
    print(f"   Results saved to: {output_file}")
    print(f"\n📊 Prediction summary:")
    print(results["predicted_class"].value_counts())
    print(f"\nAverage confidence: {results['confidence'].mean():.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

