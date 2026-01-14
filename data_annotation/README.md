# Data Annotation and Preparation

This directory contains tools for processing manual annotations and organising training data.

## Overview

The annotation workflow consists of two main scripts that work together to prepare data for model training:

1. **[batch_annotator.py](#batch_annotatorpy)** - Organises manually annotated motion sequences into class-specific directories
2. **[extract_BB_info.py](#extract_bb_infopy)** - Analyses bounding box dimensions and generates diagnostic plots

---

## batch_annotator.py

Processes annotated motion sequences from the HTML grid interface and organises them into class-specific directories suitable for model training.

### Key Features

- Batch processing of multiple video directories
- Flexible annotation dictionary via JSON configuration
- Optional exclusion of default-labelled sequences
- Copies both frame crops and GIF sequences

### Quick Start

#### Basic Usage

```bash
python batch_annotator.py \
    --analysis_dir /path/to/annotated_videos \
    --output_dir /path/to/organised_dataset
```

#### With Custom Annotation Dictionary

```bash
python batch_annotator.py \
    --analysis_dir /path/to/annotated_videos \
    --output_dir /path/to/organised_dataset \
    --annotation_dict custom_labels.json \
    --default_label other \
    --exclude_default
```

### Input Directory Structure

```
analysis_dir/
    video_001/
        motion_labels.csv              # Manual annotations from motion grid
        motion_log_video_001.csv       # Frame-to-sequence mapping
        motion_frames/                 # Full motion frames
        motion_frames_unpadded/        # Cropped motion frames
        gifs/                          # Motion sequence GIFs
    video_002/
        ...
```

### Output Directory Structure

```
output_dir/
    motion_frames/
        bat/                           # Full frames of bats
        bird/                          # Full frames of birds
        human/                         # Full frames of humans
        other/                         # Full frames of other objects
    crops/
        bat/                           # Cropped frames of bats
        bird/                          # Cropped frames of birds
        ...
    gifs/
        bat/                           # GIF sequences of bats
        bird/                          # GIF sequences of birds
        ...
```

### Command-Line Arguments

**Required:**
- `--analysis_dir PATH` - Parent directory containing video subdirectories with annotations
- `--output_dir PATH` - Root directory for organised output structure

**Optional:**
- `--annotation_dict PATH` - Path to JSON file mapping label keys to display names (default: built-in dictionary)
- `--default_label STRING` - Label to assign to unlabelled sequences (default: `other`)
- `--exclude_default` - If set, exclude default-labelled items from output

### Annotation Dictionary Format

The annotation dictionary is a JSON file mapping internal label keys to human-readable display names:

```json
{
    "bat": "Bat",
    "bird": "Bird",
    "human": "Human",
    "other": "Other"
}
```

### Special Notes Handling

The script automatically handles special annotations in the `notes` field:

- **"unclear"** → Relabelled as `unclear_bat`
- **"shadow"** → Relabelled as `bat_shadow`

### Output Files

For each processed video directory:

- **Updated CSV:** `motion_log_<video_id>_updated.csv` with propagated labels
- **Organised frames:** Copied to `<output_dir>/motion_frames/<label>/`
- **Cropped frames:** Copied to `<output_dir>/crops/<label>/`
- **GIF sequences:** Copied to `<output_dir>/gifs/<label>/`

---

## extract_BB_info.py

A diagnostic tool for analysing bounding box dimensions in image datasets and generating visualisation plots. Useful for understanding dataset characteristics before training.

### Key Features
- Individual per-class dimension distribution heatmaps
- Combined bounding box area box plots across all classes

### Quick Start

#### Basic Analysis

```bash
python extract_BB_info.py \
    --input_dir /path/to/dataset
```

### Input Directory Structure

```
input_dir/
    1_bat/                             # Bat class images
        image001.png
        image002.png
        ...
    0_other/                           # Other class images
        image001.png
        image002.png
        ...
    2_bird/                            # Bird class images (optional)
        ...
```

**Note:** The script automatically discovers all subdirectories containing PNG files. Each subdirectory is treated as a separate class.

### Output Products

When `--plot_path` is specified, the script generates:

#### Individual Class Heatmaps
- **Filename:** `bounding_box_heatmap_<class_name>.png`
- **Content:** Hexbin density plot showing width vs height distribution

#### Combined Area Box Plot
- **Filename:** `bounding_box_area_boxplot_all_classes.png`
- **Content:** Box plot comparing bounding box areas across all classes

### Command-Line Arguments

**Required:**
- `--input_dir PATH` - Root directory containing class subdirectories (auto-discovered)

**Optional:**
- `--min_dimensions INT` - Minimum width and height threshold (images below this are excluded)
- `--plot_path PATH` - Directory for saving diagnostic plots (default: same as input_dir)
- `--output_csv PATH` - Path to save dimension analysis CSV (optional)

### Output CSV Format

When `--output_csv` is specified, generates a CSV with columns:

| Column | Description | Example |
|--------|-------------|---------|
| `filepath` | Absolute path to image file | `/data/1_bat/img001.png` |
| `label` | Class label string | `1_bat` |
| `width` | Image width in pixels | `128` |
| `height` | Image height in pixels | `96` |
| `label_int` | Integer class label | `0` |

### Dimension Filtering

The `--min_dimensions` parameter filters out images where **either** width or height is below the threshold:

### Understanding the Plots

#### Hexbin Heatmap

The hexbin heatmap shows the density distribution of bounding box dimensions:

- **X-axis:** Bounding box width (pixels)
- **Y-axis:** Bounding box height (pixels)
- **Colour:** log₁₀(count) - darker colours indicate more images
- **Annotation:** Percentage of images below a defined threshold

#### Area Box Plot

The box plot compares bounding box areas across classes:

- **Box:** Interquartile range (25th to 75th percentile)
- **Line:** Median area
- **Whiskers:** 1.5× IQR range
- **Red line:** Model input resolution (64×64 = 4096 px²)
  
---

## Common Workflows

### Workflow 1: Prepare Training Dataset

```bash
# Step 1: Organise annotations into class directories
python batch_annotator.py \
    --analysis_dir /data/raw_annotations \
    --output_dir /data/training_set

# Step 2: Validate dataset dimensions
python extract_BB_info.py \
    --input_dir /data/training_set/crops \
    --min_dimensions 64 \
    --plot_path /data/diagnostics \
    --output_csv /data/diagnostics/dimensions.csv
```
---

## Requirements

All scripts share the following dependencies:

```bash
pip install pandas tqdm pillow opencv-python matplotlib seaborn
```