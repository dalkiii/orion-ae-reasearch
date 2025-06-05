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

### 4. GAN Performance Evaluation

Evaluate GAN quality and performance:

```bash
# Comprehensive analysis of all checkpoints
python scripts/run_gan_evaluation.py

# Quick analysis with fewer samples
python scripts/run_gan_evaluation.py --quick

# Analyze specific epochs only
python scripts/run_gan_evaluation.py --epochs 300 600 900 1200

# Custom analysis parameters
python scripts/run_gan_evaluation.py \
    --checkpoint-dir ./checkpoints/gan \
    --results-dir ./results/gan_analysis \
    --samples-per-class 200 \
    --tsne-perplexity 50
```

### 5. Classification Training

Run classification experiments:

```bash
# Run classification with preprocessed data
python scripts/run_classification.py

# Results will be saved to ./results/
```

#### Single GAN Checkpoint Evaluation

```bash
# Use a specific GAN checkpoint for data augmentation
python scripts/run_classification.py \
    --gan-checkpoint ./checkpoints/netG_epoch_150.pth

# With custom augmentation strategy
python scripts/run_classification.py \
    --gan-checkpoint ./checkpoints/netG_epoch_150.pth \
    --augmentation-strategy balance

# Custom GAN parameters
python scripts/run_classification.py \
    --gan-checkpoint ./checkpoints/netG_epoch_150.pth \
    --gan-latent-dim 256 \
    --gan-d-model-dim 256 \
    --augmentation-strategy oversample
```

#### Multiple GAN Checkpoint Evaluation

Evaluate classification performance across different GAN training epochs to find the optimal checkpoint:

```bash
# Evaluate all checkpoints in a directory
python scripts/run_classification.py \
    --gan-checkpoint-dir ./checkpoints

# Custom evaluation settings
python scripts/run_classification.py \
    --gan-checkpoint-dir ./checkpoints \
    --checkpoint-interval 30 \
    --num-runs 3 \
    --results-dir ./results/gan_evaluation
```
