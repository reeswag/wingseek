# Vision-Based Bat Detection: A Hybrid Motion-Deep Learning Approach

This folder contains the end-to-end scripts used in the paper: Vision-Based Bat Detection: A Hybrid Motion-Deep Learning Approach

## Pipeline Overview

1. **Motion extraction** (`motion_extraction/`): scripts used to detect + track motion in long IR videos and output crops/logs and motion_grids.
2. **Dataset preparation** (`data_annotation/`): scripts to turn manual labels into class-organised training data and optionally extract annotated video clips.
3. **Training + validation** (`model_train_eval/`): scripts used to train the multi-class classifier, run cross-validation, and calibrate an operational threshold.
4. **Inference + review UI** (`predict/`): scripts to train and validate AI classifer models on crops and generate predictions for holdout evaluation (CSVs/plots/motion grid).
5. **Holdout evaluation** (`holdout_evaluation/`): data and  scripts needed to reproduce pipeline performance metrics.

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

### System tools
- ffmpeg + ffprobe (required for `data_annotation/annotation_to_clip.py`)

### Optional
- OpenCV with CUDA for GPU-accelerated motion extraction (otherwise CPU is used)