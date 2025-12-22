# Classification Training Pipeline

Multi-class Image Classification Training and Validation
==================================================

A training pipeline for classifying images into four categories:
bat, bird, human, and other.
The pipeline implements a ResNet architecture optimised for small input images.

Key Features
------------
- Stratified Group K-Fold cross-validation to prevent site-level data leakage
- Optional CIFAR-10 pre-training for backbone weight initialisation
- Class weighting to handle imbalanced datasets
- Flexible threshold calibration strategies for the target class (bat)

Architecture
------------
The model uses a CIFAR-style ResNet with:
- No aggressive downsampling in the stem (suitable for small inputs)
- Squeeze-and-Excitation blocks for channel attention (Hu et al., 2018)
- Global average pooling followed by dropout regularisation
- Single softmax classification head

References
----------
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hu, J., et al. (2018). Squeeze-and-Excitation Networks. CVPR.
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent
  with Warm Restarts. ICLR.

Usage
-----
With configuration file:
    $ python classification_training.py --config config.ini

With CLI arguments (overrides config file):
    $ python classification_training.py --data_dir /path/to/data --output_dir /path/to/output

Mixed usage (CLI arguments override config file values):
    $ python classification_training.py --config config.ini --epochs 50

Output
------
- Trained model weights (.keras format)
- Cross-validation metrics and confusion matrices
- Threshold calibration analysis plots
- Classification reports with standard deviations across folds

## Quick Start

### 1. Create Your Configuration File
At minimum, update these paths in the `[paths]` section:
```ini
[paths]
data_dir = /path/to/your/data
output_dir = /path/to/your/output
```

### 2. Run Training

```bash
python classification_training.py --config my_experiment.ini
```
## What You'll Get

After training completes, the output directory will contain:

- **Models:**
  - `holdout_model.keras` - Model trained on training sites for holdout evaluation
  - `production_model.keras` - Final model trained on all data (if enabled)
  - `fold{N}_best.keras` - Best model from each cross-validation fold

- **Metrics:**
  - `fold_metrics.csv` - Per-fold training summaries
  - `oof_classification_report.csv` - Out-of-fold evaluation metrics with standard deviations
  - `oof_threshold_report.csv` - Bat detection threshold calibration results

- **Visualisations:**
  - `oof_confusion_matrix.png` - Confusion matrix for all classes
  - `oof_confusion_bat.png` - Binary bat/not-bat confusion matrix
  - `oof_pr_curves.png` - Precision-recall curves for all classes
  - `oof_threshold_analysis.png` - Threshold selection diagnostics
  - `fold{N}_confusion_matrix.png` - Per-fold confusion matrices

- **Metadata:**
  - `data_split_log.csv` - Record of train/holdout site allocation
  - `class_weights.csv` - Computed class weights
  - `bat_threshold_oof.txt` - Calibrated threshold for bat detection
  - `optimal_epochs.txt` - Median optimal epochs across folds
  - `cifar_report.csv` - CIFAR-10 pre-training results (if enabled)

## Data Format

Your data directory should be organised as:
```
data_dir/
├── 1_bat/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── 2_bird/
│   ├── image001.png
│   └── ...
├── 3_human/
│   └── ...
└── 4_other/
    └── ...
```

Images should follow this naming convention for site-based splitting:
```
{frame}_{tracking_id}_{site}_{date}_{camera}.png
```
Example: `001_42_SiteA_20230615_cam1.png`

## Common Tasks

### Quick Testing

```bash
# Fast test run: no pre-training, no k-fold, few epochs
python classification_training.py \
    --config my_experiment.ini \
    --pretrain_on_cifar10 false \
    --k_fold_val false \
    --epochs 20
```

## Configuration Options

See [CONFIG_USAGE.md](CONFIG_USAGE.md) for complete documentation.

Key configuration sections:
- **[paths]** - Data and output directories
- **[image]** - Image preprocessing settings
- **[training]** - Batch size, epochs, class weighting
- **[cifar_pretrain]** - CIFAR-10 pre-training settings
- **[fine_tuning]** - Learning rates and warmup epochs
- **[model]** - Architecture parameters
- **[threshold]** - Threshold calibration strategy
- **[pipeline]** - Pipeline control flags

## Help

```bash
# Show all available command-line arguments
python classification_training.py --help
```

## Requirements

- Python 3.9+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

Install with:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn tqdm
```

## Architecture Details

The model uses a CIFAR-style ResNet optimized for small images:
- **Stem**: Single 3x3 convolution (no aggressive downsampling)
- **Blocks**: Basic residual blocks with optional SE attention
- **Pooling**: Global average pooling
- **Head**: Dropout + softmax classifier

Key features:
- Stratified Group K-Fold cross-validation (prevents site leakage)
- Optional CIFAR-10 pre-training for better initialisation
- Class weighting for imbalanced datasets
- Flexible threshold calibration for bat detection
