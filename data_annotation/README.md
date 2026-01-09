# Data Annotation and Preparation

This directory contains tools for processing manual annotations, organising training datasets, and extracting annotated video clips from the bat detection pipeline.

## Overview

The annotation workflow consists of three main scripts that work together to prepare data for model training:

1. **[batch_annotator.py](#batch_annotatorpy)** - Organises manually annotated motion sequences into class-specific directories
2. **[annotation_to_clip.py](#annotation_to_clippy)** - Extracts video clips with optional bounding box overlays
3. **[extract_BB_info.py](#extract_bb_infopy)** - Analyses bounding box dimensions and generates diagnostic plots

## Typical Workflow

1. **Annotate motion sequences** using motion grid interface 
2. **Organise annotations** into class directories with `batch_annotator.py`
3. **Validate dataset** using dimension diagnostics from `extract_BB_info.py`
4. **Extract video clips** (optional) with `annotation_to_clip.py` for review

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

## annotation_to_clip.py

Extracts video clips from annotated motion events with optional bounding box overlays. Supports both fast stream-copy mode and annotated mode with OpenCV-rendered bounding boxes.

### Key Features

- Bounding box overlay with label-specific colour coding
- Audio is preserved
- Batch processing of multiple processed videos
- Outputs a csv file allowing integration into downstream reporting / analysis.

### Quick Start

#### Basic Usage (Fast Stream Copy) no annotations

```bash
python annotation_to_clip.py \
    --analysis-dir /path/to/annotations \
    --base-video-dir /path/to/videos
```

#### With Bounding Box Annotations

```bash
python annotation_to_clip.py \
    --analysis-dir /path/to/annotations \
    --base-video-dir /path/to/videos \
    --annotations True \
    --buffer 5.0
```

#### With CSV Video Mapping

```bash
python annotation_to_clip.py \
    --analysis-dir /path/to/annotations \
    --base-video-dir /path/to/videos \
    --video-mapping-csv mapping.csv \
    --annotations True
```

### Input Requirements

**Required Files in analysis_dir:**
- `motion_labels.csv` - Event annotations with columns: `file`, `label`, `notes` (optional)

**Optional Files (for bounding box overlay):**
- `motion_log_<video_id>.csv` - Frame-level bounding boxes with columns: `file`, `bb`, `label`

### Video Mapping CSV Format

The CSV should contain two columns:

| Column | Description | Example |
|--------|-------------|---------|
| `input` | Path to video file (absolute or relative to base-video-dir) | `/data/videos/site_001.mp4` |
| `output` | Video ID matching annotation files | `site_001` |

Example:
```csv
input,output
/videos/site_001.mp4,site_001
/videos/site_002.MP4,site_002
```

### Output Structure

```
output_dir/
    video_001/
        video_001_clip_0_5.mp4         # Clip starting at minute 5
        video_001_clip_1_12.mp4        # Clip starting at minute 12
        events_with_output_paths.csv   # Updated event list with clip paths
    video_002/
        ...
```

### Command-Line Arguments

**Required:**
- `--analysis-dir PATH` - Directory containing `motion_labels.csv`
- `--base-video-dir PATH` - Root directory for video files

**Optional:**
- `--video-mapping-csv PATH` - CSV with 'input' (video path) and 'output' (video ID) columns (default: None, use directory search)
- `--annotations BOOL` - Annotate clips with bounding boxes from `motion_log_{id}.csv` (default: False)
- `--annotation_dict PATH` - JSON file with annotations to extract as clips (default: None, extract all)
- `--buffer FLOAT` - Seconds of padding before and after each event (default: 5.0)
- `--output-dir PATH` - Where to write clips and output CSV (default: `<analysis_dir>/clips_annot`)
- `--exts LIST` - Video extensions to try (default: `.mp4 .avi .mov .mkv`)
- `--no-frame-index` - Do not draw frame index overlay

### Bounding Box Colour Coding

When `--annotations True` is specified, bounding boxes are colour-coded by label:

| Label | Colour (BGR) | Visual |
|-------|--------------|--------|
| Bat | Green | (0, 255, 0) |
| Bird | Blue | (255, 0, 0) |
| Human | Orange | (0, 165, 255) |
| Other | White | (255, 255, 255) |

### Event Window Merging

The script automatically merges overlapping event windows using the specified buffer:

1. For each event frame, create a time window: `[frame_time - buffer, frame_time + buffer]`
2. Sort all windows by start time
3. Merge overlapping or contiguous windows
4. Extract one clip per merged window

### Audio Handling

The script preserves original audio through a two-stage muxing process:

1. **Stream copy attempt:** Fast but only works with MP4-compatible audio codecs (AAC, MP3, ALAC, AC3)
2. **AAC re-encode fallback:** Slower but handles incompatible codecs
3. **Video-only fallback:** If both fail, outputs video without audio

### Output CSV

The script generates `events_with_output_paths.csv` for each video, containing all original columns plus:

| Column | Description | Example |
|--------|-------------|---------|
| `output_file_path` | Absolute path to extracted clip | `/data/clips/video_001/video_001_clip_0_5.mp4` |
| `frame_number` | Frame index of the event | `274` |
| `video_id` | Video identifier | `site_001` |

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

# Step 3: Review diagnostic plots and proceed to training
```

### Workflow 2: Extract Annotated Clips for Review

```bash
# Extract video clips with bounding box overlays
python annotation_to_clip.py \
    --analysis-dir /data/annotations/site_001 \
    --base-video-dir /data/videos \
    --video-mapping-csv /data/video_mapping.csv \
    --annotations True \
    --buffer 5.0 \
    --output-dir /data/review_clips
```
---

## Requirements

All three scripts share the following dependencies:

```bash
pip install pandas tqdm pillow opencv-python matplotlib seaborn
```

**External Dependencies:**
- `ffmpeg` - Required for video clip extraction (`annotation_to_clip.py`)
- `ffprobe` - Required for FPS detection (`annotation_to_clip.py`)

---

## File Format Reference

### motion_labels.csv

Generated by motion grid interface, contains sequence-level annotations:

| Column | Description | Example |
|--------|-------------|---------|
| `file` | Sequence identifier | `274_0_site_001` |
| `label` | Class label | `bat`, `bird`, `other`, `human` |
| `notes` | Optional text annotations | `unclear`, `shadow`, or empty |

### motion_log_{video_id}.csv

Generated by motion detection pipeline, contains frame-level bounding boxes:

| Column | Description | Example |
|--------|-------------|---------|
| `file` | Frame identifier | `274_0_site_001` |
| `gif_id` | Sequence identifier (optional) | `274_0_site_001` |
| `id` | Video ID (optional) | `site_001` |
| `bb` | Bounding box in YOLO format | `((80,465,12,11), (0.044,0.435,0.006,0.010))` |
| `label` | Class label (added by batch_annotator) | `bat` |

### Bounding Box Format

Bounding boxes in `motion_log` CSV use a tuple format:
```python
((x_pixel, y_pixel, w_pixel, h_pixel), (x_norm, y_norm, w_norm, h_norm))
```

Where:
- First tuple: Pixel coordinates (absolute)
- Second tuple: YOLO normalised coordinates (0-1 range)

The scripts use the normalised (YOLO) coordinates for overlay rendering.

---