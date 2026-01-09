# Inference Pipeline

A prediction pipeline for applying trained multi-class models to detect bats, birds, humans, and other objects in night vision video footage. The pipeline processes motion detection crops and aggregates predictions at the sequence level.

## Key Features

- Batch or single-folder inference mode
- Minimum dimension filtering to exclude low-quality crops
- Sequence-level aggregation with multiple statistical summaries
- Moving average smoothing for temporal patterns
- Interactive visualisation with probability-based filtering
- GIF preview functionality for high-confidence detections
- Configuration file support for reproducible inference

## Quick Start

### 1. Create Configuration File (Optional)

```bash
# Copy example configuration
cp example_predict_config.ini my_predict_config.ini

# Edit with your settings
nano my_predict_config.ini
```

### 2. Run Inference

#### With Configuration File

```bash
# Batch mode (process multiple videos)
python predict.py --config my_predict_config.ini \
    --model_path trained_model.keras \
    --batch_dir /path/to/videos

# Single video mode
python predict.py --config my_predict_config.ini \
    --model_path trained_model.keras \
    --input_dir /path/to/video/motion_frames_unpadded
```

## Outputs

### Per-Image Predictions

**`<video_id>_predictions.csv`**
- Frame-level predictions for all images
- Columns:
  - `filename`: Image filename
  - `predicted_class`: Argmax class index (0=bat, 1=bird, 2=human, 3=other)
  - `label`: Class name
  - `prob_bat`, `prob_bird`, `prob_human`, `prob_other`: Softmax probabilities
  - `gif_id`: Motion sequence identifier
  - `is_na`: Boolean flag for excluded frames

### Sequence-Level Predictions

**`<video_id>_sequence_predictions.csv`**
- Aggregated predictions for each motion sequence
- Columns:
  - `gif_id`: Sequence identifier
  - `time`: Time stamp in seconds
  - `predicted_class_avg`: Class based on average probabilities
  - `predicted_label_avg`: Class name
  - For each class (bat, bird, human, other):
    - `average_<class>_probability`: Mean across all frames
    - `median_<class>_probability`: Median across all frames
    - `min_<class>_probability`: Minimum frame probability
    - `max_<class>_probability`: Maximum frame probability
    - `top_frac_<class>_probability`: Mean of top 20% of frames

### Moving Average Statistics

**`<video_id>_sequence_predictions_ma.csv`**
- Same as sequence predictions with additional moving average columns
- Smooths temporal patterns using configurable window size
- Columns:
  - All sequence prediction columns plus:
  - `average_bat_probability_ma`: Moving average of mean bat probability
  - `median_bat_probability_ma`: Moving average of median bat probability
  - `max_bat_probability_ma`: Moving average of max bat probability
  - `top_frac_bat_probability_ma`: Moving average of top-fraction bat probability

### Visualisations

**Time-Series Plots:**
- `<video_id>_sequence_predictions_avg.png`: Average bat probability over time
- `<video_id>_sequence_predictions_med.png`: Median bat probability over time
- `<video_id>_sequence_predictions_max.png`: Maximum bat probability over time
- `<video_id>_sequence_predictions_topf.png`: Top-fraction bat probability over time

Each plot shows:
- Raw sequence probabilities (semi-transparent)
- Moving average smoothed curve
- Threshold reference line (red dashed)

**Interactive HTML Grid:**
- `motion_grid_<metric>_<video_id>.html`: Interactive visualisation
- Features:
  - Hover-to-animate GIF previews
  - Probability-based filtering (Bat ≥, Bird ≤, Human ≤)
  - Click-to-annotate interface
  - CSV export of annotations
  - Light/dark theme toggle
  - Keyboard navigation

## Configuration Options

### Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Inference batch size | 32 |
| `fps` | Video frame rate for time stamps | 60.0 |
| `ma_window` | Moving average window size | 3 |
| `threshold` | Bat probability threshold for plots | 0.5 |
| `metric` | Summary metric (average/median/max/top_frac or None) | None |
| `min_dimensions` | Minimum image size filter (pixels) | None |

### Path Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gifs_subdir` | GIF subdirectory name | gifs |
| `motion_subdir` | Motion frames subdirectory name | motion_frames_unpadded |
| `annotation_dict` | Path to annotation button JSON | None |

### Visualisation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `html_grid` | Generate HTML visualisation | false |
| `display_gifs` | Show OpenCV GIF previews | false |
| `auto_play` | Enable autoplay mode | false |
| `sort_grid` | Sort grid by class (bat/bird/human) | None |
| `skipped_files` | Show skipped files in HTML | false |

## Input Data Requirements

### Directory Structure

For batch mode, each video directory should contain:

```
video_directory/
├── motion_frames_unpadded/    # Cropped motion detection frames
│   ├── 001_42_site1.png
│   ├── 002_42_site1.png
│   └── ...
├── gifs/                       # Motion sequence GIFs
│   ├── 42.gif
│   ├── 43.gif
│   └── posters/               # Static preview images
│       ├── 42.png
│       └── 43.png
└── motion_log_<video_id>.csv  # Frame-to-sequence mapping
```

### Mapping CSV Format

The `motion_log_<video_id>.csv` file must contain:

| Column | Description |
|--------|-------------|
| `file` | Filename (without extension) |
| `gif_id` | Motion sequence identifier |

Example:
```csv
file,gif_id
001_42_site1,42
002_42_site1,42
003_43_site1,43
```

## Aggregation Statistics

### Summary Metrics

For each motion sequence, five statistics are computed for each class:

1. **Average**: Arithmetic mean of all frame probabilities

2. **Median**: Middle value of frame probabilities

3. **Minimum**: Lowest frame probability
   
4. **Maximum**: Highest frame probability

5. **Top-Fraction**: Mean of top 20% of frames

## Motion Grid Features

### Interactive Filtering

The motion grid provides three probability sliders:

- **Bat ≥**: Show only sequences with bat probability above threshold
- **Bird ≤**: Hide sequences with bird probability above threshold
- **Human ≤**: Hide sequences with human probability above threshold

### Validation Workflow

1. **Review**: Hover over tiles to play animations
2. **Annotate**: Click label buttons to record true labels
3. **Notes**: Add free-text notes for context
4. **Export**: Download annotations as CSV.

### Keyboard Shortcuts

- **Arrow Up/Down**: Scroll through grid
- **Spacebar**: Toggle autoplay mode (all GIFs loop continuously)
- **Theme Toggle Button**: Switch between light and dark modes

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- pandas
- matplotlib
- OpenCV (cv2)
- imageio
- tqdm

Install with:
```bash
pip install tensorflow numpy pandas matplotlib opencv-python imageio tqdm
```