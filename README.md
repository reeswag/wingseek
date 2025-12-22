# Vision-Based Bat Detection: A Hybrid Motion-Deep Learning Approach

This folder contains the end-to-end scripts used in the paper: Vision-Based Bat Detection: A Hybrid Motion-Deep Learning Approach

## Pipeline Overview

1. **Motion extraction** (`motion_extraction/`): detect + track motion in long IR videos and write crops/GIFs/logs.
2. **Dataset preparation** (`data_annotation/`): turn manual labels into class-organised training data; optionally extract annotated video clips.
3. **Training + validation** (`model_train_eval/`): train the multi-class classifier, run cross-validation, and calibrate an operational “bat” threshold.
4. **Inference + review UI** (`predict/`): run a trained 4-class model (bat/bird/human/other) on crops and generate outputs (CSVs/plots/HTML motion grid).
5. **Holdout evaluation** (`holdout_evaluation/`): evaluate bat detection performance at a fixed threshold and quantify workload reduction + emergence recall.

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