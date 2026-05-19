# Scaling bat roost surveys: automated event screening in near-infrared video

This folder contains the code used in the paper: Scaling bat roost surveys: automated event screening in near-infrared video

## Pipeline Overview

1. **Motion extraction** (`motion_extraction/`): scripts used to detect + track motion and output crops/logs and motion_grids.
2. **Dataset preparation** (`data_annotation/`): scripts to turn labels into class-organised training data.
3. **Training + validation** (`model_train_eval/`): scripts used to train and validate the deep-learning classifier, inclusive of results.
4. **Inference + review UI** (`predict/`): scripts used for inference and to generate predictions for holdout evaluation.
5. **Holdout evaluation** (`holdout_evaluation/`): data and  scripts needed to reproduce pipeline performance metrics from the paper.

## Requirements

### Python
- Python 3.9+

### Python packages
- tensorflow==2.15.1
- numpy==1.24.4
- opencv-python
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- Pillow
- imageio

### Optional
- OpenCV with CUDA for GPU-accelerated motion extraction (otherwise CPU is used)
