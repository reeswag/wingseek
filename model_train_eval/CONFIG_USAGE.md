# Configuration File Usage Guide

## Overview

The `classification_training.py` script supports configuration via:
1. **INI configuration files** - for persistent, version-controlled settings
2. **Command-line arguments** - for quick overrides and experimentation
3. **Mixed mode** - combine both, with CLI arguments taking precedence

## Quick Start

### Using a Configuration File

```bash
# Run training with your config
python classification_training.py --config my_config.ini
```

### Using CLI Arguments Only

```bash
python classification_training.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --epochs 30 \
    --batch_size 128
```

### Mixed Mode (CLI Overrides Config)

```bash
# Use config file as base, but override specific parameters
python classification_training.py \
    --config my_config.ini \
    --epochs 50 \
    --batch_size 128 \
    --fine_tune_lr 0.00005
```

## Configuration File Format

Configuration files use the INI format with the following sections:

### [paths]
- `data_dir` - Directory containing class subdirectories
- `output_dir` - Directory for saving results

### [image]
- `min_dimensions` - Minimum image size to include
- `img_height`, `img_width` - Target image dimensions
- `channels` - Number of channels (1=grayscale, 3=RGB)

### [training]
- `batch_size` - Training batch size
- `epochs` - Number of training epochs
- `k_folds` - Number of cross-validation folds
- `use_class_weights` - Enable class weighting (true/false)
- `bat_weight_multiplier` - Extra weight for bat class

### [cifar_pretrain]
- `enabled` - Enable CIFAR-10 pre-training (true/false)
- `epochs` - CIFAR-10 training epochs
- `batch_size` - CIFAR-10 batch size
- `val_split` - Validation split fraction
- `learning_rate` - CIFAR-10 learning rate

### [fine_tuning]
- `learning_rate` - Fine-tuning learning rate
- `head_start_epochs` - Head-only training epochs
- `head_start_lr` - Head-only learning rate

### [model]
- `initial_filters` - Initial conv layer filters
- `dropout_rate` - Dropout probability
- `use_se` - Enable SE blocks (true/false)
- `weight_decay` - L2 regularization coefficient

### [threshold]
- `strategy` - Threshold selection: fbeta, precision_constrained, recall_constrained
- `beta` - F-beta score parameter
- `target_precision` - Min precision (for precision_constrained)
- `target_recall` - Min recall (for recall_constrained)

### [pipeline]
- `visualise_transformations` - Generate augmentation visualizations (true/false)
- `k_fold_val` - Perform k-fold CV (true/false)
- `train_production_model` - Train final model on all data (true/false)
- `holdout_fraction` - Fraction of sites for holdout
- `random_seed` - Random seed for reproducibility

## Complete Command-Line Reference

```bash
python classification_training.py --help
```

### Available Arguments

**Configuration File:**
- `--config PATH` - Path to .ini configuration file

**Paths:**
- `--data_dir PATH` - Data directory
- `--output_dir PATH` - Output directory

**Image Parameters:**
- `--min_dimensions INT` - Minimum image dimensions
- `--img_height INT` - Target image height
- `--img_width INT` - Target image width
- `--channels INT` - Number of channels

**Training:**
- `--batch_size INT` - Batch size
- `--epochs INT` - Number of epochs
- `--k_folds INT` - Number of CV folds
- `--use_class_weights true/false` - Use class weighting
- `--bat_weight_multiplier FLOAT` - Bat class weight multiplier

**CIFAR-10 Pre-training:**
- `--pretrain_on_cifar10 true/false` - Enable pre-training
- `--cifar_pretrain_epochs INT` - Pre-training epochs
- `--cifar_pretrain_batch_size INT` - Pre-training batch size
- `--cifar_pretrain_val_split FLOAT` - Validation split
- `--cifar_pretrain_lr FLOAT` - Pre-training learning rate

**Fine-tuning:**
- `--fine_tune_lr FLOAT` - Fine-tuning learning rate
- `--head_start_epochs INT` - Head-only epochs
- `--head_start_lr FLOAT` - Head-only learning rate

**Model Architecture:**
- `--initial_filters INT` - Initial filters
- `--dropout_rate FLOAT` - Dropout rate
- `--use_se true/false` - Use SE blocks
- `--weight_decay FLOAT` - Weight decay

**Threshold Calibration:**
- `--threshold_strategy CHOICE` - Strategy: fbeta, precision_constrained, recall_constrained
- `--threshold_beta FLOAT` - Beta for F-beta
- `--target_precision FLOAT` - Target precision
- `--target_recall FLOAT` - Target recall

**Pipeline Control:**
- `--visualise_transformations true/false` - Visualize augmentations
- `--k_fold_val true/false` - Perform k-fold CV
- `--train_production_model true/false` - Train production model
- `--holdout_fraction FLOAT` - Holdout fraction
- `--random_seed INT` - Random seed

## Example Workflows

### Hyperparameter Sweep

```bash
# Base configuration
for lr in 0.0001 0.00005 0.00001; do
    python classification_training.py \
        --config base_config.ini \
        --fine_tune_lr $lr \
        --output_dir "results/lr_${lr}"
done
```

### Quick Experiment with Different Batch Sizes

```bash
for bs in 32 64 128; do
    python classification_training.py \
        --config my_config.ini \
        --batch_size $bs \
        --output_dir "results/batch_${bs}"
done
```

### Disable Pre-training for Faster Testing

```bash
python classification_training.py \
    --config my_config.ini \
    --pretrain_on_cifar10 false \
    --epochs 10 \
    --k_fold_val false
```

## Tips and Best Practices

1. **Version Control**: Keep configuration files in version control to track experimental settings

2. **Naming Convention**: Use descriptive config file names:
   - `baseline_config.ini`
   - `high_recall_config.ini`
   - `production_config.ini`

3. **Comments**: Add comments in config files to explain non-obvious choices

4. **Reproducibility**: Always set `random_seed` for reproducible results

5. **Incremental Changes**: Start with a working config and change one parameter at a time

6. **Output Organization**: Use different output directories for each experiment:
   ```bash
   python classification_training.py \
       --config base.ini \
       --output_dir "results/experiment_001"
   ```

## Troubleshooting

**Config file not found:**
```bash
ERROR: Configuration file not found: config.ini
```
Solution: Check the path to your config file

**Invalid boolean value:**
Boolean parameters must be `true` or `false` (lowercase) in config files, or `true`/`false` in CLI

**Missing required parameters:**
If neither config file nor CLI arguments provide required paths, the script will use placeholder values. Always specify:
- `data_dir`
- `output_dir`

## Default Values

If no configuration file or CLI arguments are provided, the script uses these defaults:

- Image size: 64x64, grayscale (1 channel)
- Batch size: 64, Epochs: 25, K-folds: 16
- CIFAR-10 pre-training: Enabled (15 epochs)
- Class weighting: Enabled (bat multiplier: 2.0)
- Fine-tune LR: 0.0001, Head-start LR: 0.0003
- Threshold strategy: F-beta (beta=2.0)
- Holdout fraction: 0.2, Random seed: 1234

See `example_config.ini` for the complete list of defaults.
