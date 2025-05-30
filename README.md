# IMB-SpecGAN

## Installation

```bash
git clone https://github.com/dalkiii/orion-ae-reasearch
cd orion-ae-reasearch
pipenv shell
pip install -r requirements.txt
```

## Complete Pipeline

### 1. Data Loading

Unzip your acoustic emission dataset:

```bash
# Extract zip file to data directory
python scripts/load_data.py path/to/your/dataset.zip
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
python scripts/run_preprocess.py
```

**After preprocessing:**
```
data/resized/
├── measurementSeries_B/
│   ├── 05cNm/
│   │   ├── file1.mat_A_resized.npy
│   │   ├── file1.mat_B_resized.npy
│   │   └── file1.mat_C_resized.npy
│   └── ...
└── ...
```

### 3. GAN Training

Generate synthetic data to handle class imbalance:

```bash
# Train GAN with default settings
python scripts/run_gan_train.py

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