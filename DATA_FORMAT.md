# Data Format Guide for ProCanFDL

This guide explains the expected data format for using ProCanFDL, both for inference and training.

## Input Data Format for Inference

### Basic Structure

Your input CSV file should have the following structure:

```csv
sample_id,P37108,Q96JP5,Q8N697,P36578,O76031,...
sample_001,5.234,3.456,6.789,4.123,5.678,...
sample_002,4.987,3.654,5.432,3.876,4.321,...
sample_003,5.678,4.012,6.234,4.456,5.123,...
```

### Column Requirements

1. **First Column: Sample Identifiers**
   - Column name: `sample_id` (or any descriptive name)
   - Values: Unique identifiers for each sample
   - Format: String or numeric IDs

2. **Remaining Columns: Protein Features**
   - Column names: UniProt protein IDs (e.g., P37108, Q96JP5, etc.)
   - Values: Log2-transformed protein expression values
   - Format: Numeric (float)

### Data Requirements

#### Essential Requirements
- **File Format**: CSV (comma-separated values)
- **Encoding**: UTF-8
- **Decimal Separator**: Period (.)
- **Missing Values**: Can be left empty, will be filled with 0

#### Biological Requirements
- **Transformation**: Data should be log2-transformed
- **Normalization**: Recommended (e.g., quantile normalization, z-score)
- **Quality Control**: Remove low-quality samples before input
- **Technical Replicates**: Average replicates before input

#### Feature Alignment
The model expects specific protein features in a specific order (see `data/data_format.csv` for the complete list). The inference script will automatically:
- Match your protein columns to the expected features
- Add missing proteins with zero values
- Reorder features to match model expectations

### Example Data

See `data/example_input_template.csv` for a minimal example.

**Full example:**
```csv
sample_id,P37108,Q96JP5,Q8N697,P36578,O76031
patient_001,5.234,3.456,6.789,4.123,5.678
patient_002,4.987,3.654,5.432,3.876,4.321
patient_003,5.678,4.012,6.234,4.456,5.123
control_001,3.456,2.345,4.567,2.890,3.234
```

## Expected Protein Features

The model was trained on approximately 8,000 protein features. The complete list is available in `data/data_format.csv` (columns after the metadata columns).

### Common Proteins (subset)

Here are some of the key proteins the model expects (first 50 shown):

```
P37108, Q96JP5, Q8N697, P36578, O76031, A6NIH7, Q9BTD8, Q9P258,
P17542, Q68DK7, P55036, A1X283, P05154, Q9BQE4, Q8WWH5, O75602,
P18509, Q8TDZ2, Q96GQ5, P59768, Q6P2E9, Q9UNF1, Q9NZQ3, Q9NQP4,
...
```

**Note**: If your data is missing some proteins, they will be filled with zeros. However, having most of the expected proteins will improve prediction accuracy.

## Data Preprocessing Guidelines

### Step-by-Step Preprocessing

1. **Raw Data Quality Control**
   - Remove samples with >30% missing values
   - Remove proteins detected in <10% of samples
   - Check for technical outliers

2. **Missing Value Imputation**
   - For training: Use appropriate imputation (e.g., KNN, MinProb)
   - For inference: Missing proteins will be set to 0

3. **Normalization**
   - Quantile normalization (recommended)
   - Or median normalization
   - Or z-score normalization

4. **Log Transformation**
   - Apply log2 transformation
   - Ensure no zero/negative values before log

5. **Quality Metrics**
   - Check technical replicate correlations (if available)
   - Recommended: Pearson r > 0.9 for replicates

### Example Preprocessing Code

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load raw data
data = pd.read_csv('raw_data.csv', index_col=0)

# Remove samples with high missing rate
missing_rate = data.isnull().sum(axis=1) / data.shape[1]
data = data[missing_rate < 0.3]

# Remove proteins with low detection rate
detection_rate = data.notna().sum(axis=0) / data.shape[0]
data = data.loc[:, detection_rate > 0.1]

# Fill remaining missing values
data = data.fillna(data.median())

# Log2 transformation
data_log2 = np.log2(data + 1)  # Add 1 to avoid log(0)

# Quantile normalization (example)
from sklearn.preprocessing import quantile_transform
data_normalized = pd.DataFrame(
    quantile_transform(data_log2, axis=0),
    index=data_log2.index,
    columns=data_log2.columns
)

# Save processed data
data_normalized.to_csv('processed_data.csv')
```

## Training Data Format

### Full Dataset Structure

For training, the dataset includes both metadata and protein features:

```csv
prep_lims_id,tracking_lims_id,...[73 metadata columns],P37108,Q96JP5,...[~8000 proteins]
sample_001,track_001,...,5.234,3.456,...
sample_002,track_002,...,4.987,3.654,...
```

### Metadata Columns

The first 73 columns contain sample metadata including:
- Sample identifiers (LIMS IDs)
- Cohort information
- Tissue descriptors
- Cancer type labels
- Processing information

### Key Metadata Fields

**Essential for training:**
- `prep_cohort`: Cohort identifier (for federated learning)
- `cancer_type_2`: Target label for classification
- `sample_tissue_descriptor`: "Malignant" or "Normal"
- `subject_collaborator_patient_id`: Patient ID (for proper train/test split)

**Quality control:**
- `prep_lims_id`: Sample ID for correlation checks
- Replicate information (if available)

## Feature Order

The exact order of protein features is critical for model inference. The reference order is stored in `data/data_format.csv`.

### Extracting Feature Order

```python
import pandas as pd

# Read the header of the data format file
format_df = pd.read_csv('data/data_format.csv', nrows=0)

# Extract protein columns (after metadata)
META_COL_NUMS = 73
protein_columns = list(format_df.columns[META_COL_NUMS:])

print(f"Number of expected proteins: {len(protein_columns)}")
print(f"First 10 proteins: {protein_columns[:10]}")
```

## Common Issues and Solutions

### Issue 1: Wrong Protein IDs

**Problem**: Using gene symbols instead of UniProt IDs
```csv
sample_id,TP53,BRCA1,EGFR  # ❌ Wrong
```

**Solution**: Use UniProt IDs
```csv
sample_id,P04637,P38398,P00533  # ✓ Correct
```

### Issue 2: Not Log-Transformed

**Problem**: Raw intensity values (very large range)
```csv
sample_id,P37108,Q96JP5
sample_001,125000,89000  # ❌ Too large
```

**Solution**: Log2 transform
```csv
sample_id,P37108,Q96JP5
sample_001,16.93,16.44  # ✓ Correct range
```

### Issue 3: Wrong Separator

**Problem**: Using semicolon or tab
```csv
sample_id;P37108;Q96JP5  # ❌ Wrong separator
```

**Solution**: Use comma
```csv
sample_id,P37108,Q96JP5  # ✓ Correct
```

### Issue 4: Headers in Data Rows

**Problem**: Column names repeated in data
```csv
sample_id,P37108,Q96JP5
sample_id,P37108,Q96JP5  # ❌ Header repeated
sample_001,5.234,3.456
```

**Solution**: Single header row only
```csv
sample_id,P37108,Q96JP5  # ✓ Correct
sample_001,5.234,3.456
sample_002,4.987,3.654
```

## Validation Checklist

Before running inference, verify:

- [ ] File is CSV format with comma separator
- [ ] First column contains unique sample IDs
- [ ] Column names are UniProt IDs (e.g., P37108, Q96JP5)
- [ ] Values are numeric (floats)
- [ ] Data is log2-transformed (typical range: 0-20)
- [ ] No NaN values (or handled appropriately)
- [ ] File encoding is UTF-8
- [ ] No duplicate sample IDs
- [ ] No duplicate protein columns

## Quick Validation Script

```python
import pandas as pd
import numpy as np

def validate_input_data(file_path):
    """Validate input data format"""
    print("Validating input data...")
    
    # Load data
    data = pd.read_csv(file_path, index_col=0)
    print(f"✓ File loaded successfully")
    print(f"  Shape: {data.shape}")
    
    # Check for duplicates
    assert data.index.is_unique, "❌ Duplicate sample IDs found"
    print(f"✓ No duplicate sample IDs")
    
    # Check column names (should be mostly UniProt IDs)
    uniprot_pattern = r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$'
    uniprot_cols = data.columns.str.match(uniprot_pattern).sum()
    print(f"✓ {uniprot_cols}/{len(data.columns)} columns match UniProt ID pattern")
    
    # Check data types
    numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
    print(f"✓ {numeric_cols}/{len(data.columns)} columns are numeric")
    
    # Check value ranges (log2 data typically 0-20)
    value_min = data.min().min()
    value_max = data.max().max()
    print(f"✓ Value range: {value_min:.2f} to {value_max:.2f}")
    if value_max > 100:
        print("  ⚠️  Warning: Values seem large. Is data log-transformed?")
    
    # Check missing values
    missing_pct = (data.isnull().sum().sum() / data.size) * 100
    print(f"✓ Missing values: {missing_pct:.2f}%")
    
    print("\n✅ Validation complete!")

# Usage
validate_input_data('your_data.csv')
```

## Getting Help

If you encounter issues with data formatting:

1. Check this guide first
2. Review `data/example_input_template.csv`
3. Run the validation script above
4. Check the [Troubleshooting](README.md#troubleshooting) section in README
5. Open an issue on GitHub with:
   - Your data format (first few rows)
   - Error message
   - Steps you've tried

## Additional Resources

- **Example data**: `data/example_input_template.csv`
- **Feature list**: `data/data_format.csv`
- **Inference script**: `ProCanFDL/inference.py`
- **Main README**: `README.md`


