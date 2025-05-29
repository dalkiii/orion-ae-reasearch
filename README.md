# Acoustic Emission Classification - Usage Guide

## Installation

```bash
git clone <repository-url>
cd acoustic-emission-classification
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Complete Pipeline

### 1. Data Loading

Unzip your acoustic emission dataset:

```bash
# Extract zip file to data directory
python scripts/load_data.py path/to/your/dataset.zip

# Extract to custom directory
python scripts/load_data.py path/to/your/dataset.zip --output_dir custom_data/
```

**After extraction:**
```
data/
├── measurementSeries_B/
│   ├── 05cNm/
│   │   ├── file1.mat
│   │   └── file2.mat
│   └── ...
└── ...
```

### 2. Data Preprocessing

Convert raw .mat files to preprocessed .npy scalogram files:

```bash
# Basic preprocessing (uses default config)
python scripts/run_preprocessing.py

# With custom parameters
python scripts/run_preprocessing.py \
    --root-dir ./data \
    --output-dir ./data/processed \
    --sensors A B C \
    --verbose
```

**After preprocessing:**
```
data/processed/
├── measurementSeries_B/
│   ├── 05cNm/
│   │   ├── file1.mat_A_rev2.npy
│   │   ├── file1.mat_B_rev2.npy
│   │   └── file1.mat_C_rev2.npy
│   └── ...
└── ...
```

### 3. GAN Training (Optional)

Generate synthetic data to handle class imbalance:

```bash
# Train GAN with default settings
python scripts/run_gan_training.py

# Train with custom parameters
python scripts/run_gan_training.py \
    --data-root ./data/processed \
    --epochs 1000 \
    --batch-size 64

# Generate synthetic data using existing model
python scripts/run_gan_training.py \
    --generate-only \
    --samples-per-class 1000
```

**Generated data structure:**
```
data/generated/
├── 05cNm/
│   └── generated_05cNm_1000.npy
├── 10cNm/
│   └── generated_10cNm_1000.npy
└── ...
```

### 4. Classification Training

Run classification experiments:

```bash
# Run classification with preprocessed data
python experiments/run_classification.py

# Results will be saved to ./results/
```

## Quick Start

For a complete end-to-end run:

```bash
# 1. Extract data
python scripts/load_data.py your_dataset.zip

# 2. Preprocess
python scripts/run_preprocessing.py

# 3. Train GAN (optional, for class imbalance)
python scripts/run_gan_training.py --epochs 500

# 4. Train classifier
python experiments/run_classification.py
```