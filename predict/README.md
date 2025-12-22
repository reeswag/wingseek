# Multi-class Image Classification Inference Pipeline

A prediction pipeline for applying trained multi-class models to detect bats, birds, humans, and other objects in night vision video footage. The pipeline processes motion detection crops and aggregates predictions at the sequence level.

## Key Features

- Batch or single-folder inference mode
- Minimum dimension filtering to exclude low-quality crops
- Sequence-level aggregation with multiple statistical summaries
- Moving average smoothing for temporal patterns
- Interactive HTML visualisation with probability-based filtering
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

#### With CLI Arguments Only

```bash
# Batch mode with custom threshold
python predict.py --model_path trained_model.keras \
    --batch_dir /path/to/videos \
    --threshold 0.4 \
    --html_grid \
    --metric average

# Single video with minimum dimension filter
python predict.py --model_path trained_model.keras \
    --input_dir /path/to/video/motion_frames_unpadded \
    --min_dimensions 10 \
    --threshold 0.5
```

#### Mixed Mode (CLI Overrides Config)

```bash
python predict.py --config my_predict_config.ini \
    --model_path trained_model.keras \
    --batch_dir /path/to/videos \
    --threshold 0.35 \
    --sort_grid bat
```

## Output Products

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
   - Best for balanced view of sequence confidence
   - Most commonly used metric

2. **Median**: Middle value of frame probabilities
   - Robust to outlier frames
   - Good when some frames may be poor quality

3. **Minimum**: Lowest frame probability
   - Most conservative estimate
   - Useful for ensuring consistent detection

4. **Maximum**: Highest frame probability
   - Peak detection confidence
   - Can identify clear bat presence even in noisy sequence

5. **Top-Fraction**: Mean of top 20% of frames
   - Reduces impact of poor quality frames
   - Balances sensitivity with noise reduction
   - Configurable via `top_frac_mean()` function

### Choosing a Metric

| Metric | Use Case |
|--------|----------|
| Average | General-purpose, balanced assessment |
| Median | Sequences with variable frame quality |
| Maximum | Identifying clear positives, high sensitivity |
| Minimum | Conservative threshold, high precision |
| Top-Fraction | Robust to noise whilst maintaining sensitivity |

## HTML Grid Features

### Interactive Filtering

The HTML grid provides three probability sliders:

- **Bat ≥**: Show only sequences with bat probability above threshold
- **Bird ≤**: Hide sequences with bird probability above threshold
- **Human ≤**: Hide sequences with human probability above threshold

This filtering strategy optimises for bat detection by:
1. Requiring high bat confidence
2. Excluding likely birds
3. Excluding likely human activity

### Annotation Workflow

1. **Review**: Hover over tiles to preview GIF animations
2. **Annotate**: Click label buttons to record ground truth
3. **Notes**: Add free-text notes for context
4. **Export**: Download annotations as CSV for validation

### Keyboard Shortcuts

- **Arrow Up/Down**: Scroll through grid
- **Spacebar**: Toggle autoplay mode (all GIFs loop continuously)
- **Theme Toggle Button**: Switch between light and dark modes

## Common Use Cases

### Standard Inference

Process videos with default settings:

```bash
python predict.py --model_path model.keras \
    --batch_dir /data/videos
```

### High-Quality Filtering

Exclude small/poor quality frames:

```bash
python predict.py --config predict_config.ini \
    --model_path model.keras \
    --batch_dir /data/videos \
    --min_dimensions 15
```

### Low-Threshold Screening

Use conservative threshold for high recall:

```bash
python predict.py --config predict_config.ini \
    --model_path model.keras \
    --batch_dir /data/videos \
    --threshold 0.2 \
    --metric top_frac
```

### Manual Review Workflow

Generate HTML grid for annotation:

```bash
python predict.py --config predict_config.ini \
    --model_path model.keras \
    --batch_dir /data/videos \
    --html_grid \
    --metric average \
    --sort_grid bat \
    --threshold 0.3
```

### OpenCV Preview Mode

Display high-confidence detections:

```bash
python predict.py --config predict_config.ini \
    --model_path model.keras \
    --input_dir /data/video/motion_frames_unpadded \
    --display_gifs \
    --threshold 0.7 \
    --metric max
```

## Troubleshooting

### "No images in directory"

**Problem:** No `.jpg`, `.jpeg`, or `.png` files found

**Solution:**
- Check `--motion_subdir` matches your directory structure
- Verify image file extensions are lowercase
- Ensure directory path is correct

### "Mapping CSV not found"

**Problem:** Cannot find `motion_log_<video_id>.csv`

**Solution:**
- Check CSV filename matches video directory name
- Verify CSV is in the same directory as motion_frames_unpadded/
- Run motion detection pipeline first if CSV is missing

### Out of Memory

**Problem:** GPU/CPU runs out of memory during inference

**Solution:**
- Reduce `batch_size` (try 16 or 8)
- Process videos one at a time (use `--input_dir` instead of `--batch_dir`)
- Close other applications using GPU memory

### HTML Grid Empty

**Problem:** Grid loads but shows no sequences

**Solution:**
- Check threshold sliders (reset to Bat ≥ 0.0, Bird ≤ 1.0, Human ≤ 1.0)
- Verify GIFs exist in `gifs/` subdirectory
- Check sequence predictions CSV has matching gif_ids

### Missing Predictions

**Problem:** Some frames show `is_na: True`

**Solution:**
- Check `min_dimensions` filter (NA frames are too small)
- Review error log (`log.txt` in parent directory)
- Verify images are not corrupted

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

## Performance Tips

1. **Batch Size**: Increase for faster inference (GPU permitting)
2. **Min Dimensions**: Filter out tiny crops to reduce processing time
3. **HTML Grid**: Generate only for specific metric if reviewing manually
4. **Moving Average**: Adjust window size based on sequence length
5. **Memory**: Process large batches one video at a time

## References

- **Top-Fraction Aggregation**: Custom metric for temporal sequence summarisation
- **HTML Grid Design**: Interactive visualisation with probability-based filtering
- **Moving Average**: Temporal smoothing for trend identification

## Contact

For questions or issues, please contact [Your Contact Info]
