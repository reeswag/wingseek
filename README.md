# Vision-Based Bat Detection: A Hybrid Motion-Deep Learning Approach

This folder contains the end-to-end scripts used in the paper: Vision-Based Bat Detection: A Hybrid Motion-Deep Learning Approach

## Pipeline Overview

1. **Motion extraction** (`motion_extraction/`): detect + track motion in long IR videos and write crops/GIFs/logs.
2. **Dataset preparation** (`data_annotation/`): turn manual labels into class-organised training data; optionally extract annotated video clips.
3. **Training + validation** (`model_train_eval/`): train the multi-class classifier, run cross-validation, and calibrate an operational “bat” threshold.
4. **Inference + review UI** (`predict/`): run a trained 4-class model (bat/bird/human/other) on crops and generate outputs (CSVs/plots/HTML motion grid).
5. **Holdout evaluation** (`holdout_evaluation/`): evaluate bat detection performance at a fixed threshold and quantify workload reduction + emergence recall.

## Repository Structure

### `motion_extraction/` — Motion detection & tracking

- `motion_extraction/motion_extraction.py`: MOG2 + tracking pipeline that generates motion events, crops, GIFs, and CSV logs.
- `motion_extraction/motion_extraction.ini`: INI template for reproducible runs (CLI overrides INI; INI overrides defaults).
- `motion_extraction/motion_extraction.sh`: Example runner wrapper for batch-style execution.
- `motion_extraction/notebook/motion_extraction.ipynb`: Notebook version for experimentation.

See `motion_extraction/README.md` for details and parameter reference.

### `data_annotation/` — Annotation processing & clip extraction

- `data_annotation/batch_annotator.py`: Takes `motion_labels.csv` from the motion grid and organises frames/crops/GIFs into class folders for training.
- `data_annotation/annotation_to_clip.py`: Extracts video clips around annotated events; can overlay batfinder bounding boxes and preserves audio.
- `data_annotation/extract_BB_info.py`: Bounding-box diagnostics (size distributions/plots) to help inform model input resolution.

See `data_annotation/README.md` for a typical annotation-to-training workflow.

### `model_train_eval/` — Training + threshold calibration

- `model_train_eval/classification_training.py`: Multi-class training pipeline (ResNet-style), with (optional) CIFAR-10 pre-training, stratified group K-fold CV, and bat-threshold calibration.
- `model_train_eval/classification_training.ini`: Example experiment configuration.
- `model_train_eval/CONFIG_USAGE.md`: Configuration reference and common patterns.
- `model_train_eval/outputs/`: Training outputs used for the paper (plots) plus model weights used for holdout evaluation and a separate “production” model trained on all data.

See `model_train_eval/README.md` for outputs and common tasks.

### `predict/` — Multi-class inference + HTML review grid

- `predict/predict.py`: Runs a trained 4-class model on motion crops; writes per-frame + per-sequence prediction CSVs, moving averages, plots, and (optionally) an interactive HTML grid for manual review/annotation.
- `predict/predict.ini`: Example configuration file for inference + visualisation options.
- `predict/predict.sh`: Example runner wrapper.
- `predict/dict.json`: Example label dictionary for the HTML grid annotation buttons.

See `predict/README.md` for usage and output formats.

### `holdout_evaluation/` — Holdout metrics & workload reduction

- `holdout_evaluation/holdout_evaluation.py`: Evaluates a holdout CSV at a provided operational threshold; reports metrics, workload reduction (events + duration), and emergence recall.
- `holdout_evaluation/holdout_evaluation.csv`: Included holdout dataset used to reproduce the final paper results.
- `holdout_evaluation/holdout_eval.ini`: Example config for repeatable evaluation runs.
- `holdout_evaluation/holdout_eval.sh`: Example sweep script to reproduce paper-style reporting across multiple score columns.

See `holdout_evaluation/README.md` for input requirements and produced plots/tables.

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