Motion Detection and Tracking Pipeline
=======================================

Overview
--------
This module implements a comprehensive motion detection and tracking system for
identifying and monitoring rare motion events in long-duration infrared video files.
The pipeline combines classical computer vision techniques with Kalman filtering to
provide robust object detection and tracking capabilities specifically optimised for
bat detection in low-contrast nocturnal surveillance footage.

Key Algorithms
--------------
The system utilises three core algorithmic components:

1. **MOG2 Background Subtraction** (Zivkovic & van der Heijden, 2006):
   An adaptive Gaussian mixture model that efficiently learns background statistics
   and identifies foreground objects through pixel-wise density estimation.

2. **Kalman Filtering**:
   A recursive state estimation framework that maintains a two-dimensional velocity
   state (vx, vy) for each tracked object, enabling prediction of object positions
   in subsequent frames under a constant velocity motion model.

3. **Hungarian Algorithm Matching**:
   An optimal assignment method (via `scipy.optimize.linear_sum_assignment`) that
   associates predicted track positions with current detections by minimising a cost
   function combining spatial proximity and object area similarity.

Key Features
------------
- Adaptive background modelling resilient to gradual illumination changes
- Multi-object tracking with occlusion handling and track persistence
- Area-based cost function for stable tracking in 3D scenes without depth information
  (Bjerge et al., 2022)
- Region of Interest (ROI) masking to exclude areas of repetitive background motion
- Interactive parameter tuning interface for optimisation across different environments
- Standardised output format compatible with deep learning annotation pipelines

Pipeline Architecture
---------------------
The processing pipeline operates as follows:

1. **Frame Acquisition**: Sequential reading of video frames
2. **Background Subtraction**: MOG2 generates foreground mask highlighting moving objects
3. **Morphological Filtering**: Opening and closing operations remove sensor noise
4. **Blob Detection**: Connected component analysis identifies distinct foreground regions
5. **Centroid Extraction**: Bounding box centroids serve as detection coordinates
6. **Kalman Prediction**: Each active track predicts its position for the current frame
7. **Hungarian Matching**: Optimal assignment between predictions and detections
8. **State Update**: Matched tracks update their Kalman state; unmatched detections
   initialise new tracks
9. **Track Management**: Tracks unmatched for N consecutive frames are terminated
10. **Output Generation**: Tracked objects are cropped and saved with metadata

For detailed methodology, refer to the Methods section of the accompanying paper.

Library Dependencies
--------------------
This pipeline utilises the following core libraries:

- **OpenCV (cv2)**: Provides the MOG2 background subtraction algorithm, Kalman
  filtering implementation, and morphological operations. CUDA acceleration is
  supported for GPU-enabled systems to enhance processing throughput.
- **PIL (Python Imaging Library)**: Used for image format conversion and manipulation
  during output generation.
- **NumPy**: Facilitates efficient array operations for mask processing, state vector
  manipulation, and cost matrix computation.
- **SciPy**: The `scipy.optimize.linear_sum_assignment` function implements the
  Hungarian algorithm for optimal track-to-detection assignment, minimising the sum
  of assignment costs across all matches.
- **Pandas**: Manages metadata and detection logs in tabular format with standardised
  YOLO bounding box coordinates.
- **Matplotlib**: Generates heat maps for interactive Region of Interest (ROI)
  selection, visualising motion density across the frame.


## Quick Start

The `motion_extraction.py` pipeline supports three methods for configuring parameters:

1. **Command-line arguments** (highest priority)
2. **INI configuration file** (medium priority)
3. **Default values** in the script (lowest priority)


### Method 1: Command-Line Arguments

Run the pipeline with specific parameters:

```bash
python motion_extraction.py \
    --video-dir /path/to/videos \
    --analysis-dir /path/to/output \
    --mog-history 100 \
    --var-threshold 15
```

### Method 2: INI Configuration File

Create or edit `motion_extraction.ini` with your parameters, then:

```bash
python motion_extraction.py --config motion_extraction.ini
```

### Method 3: Combined Approach

Use an INI file for base configuration and override specific parameters via CLI:

```bash
python motion_extraction.py \
    --config motion_extraction.ini \
    --mog-history 100 \
    --tracking-buffer 0.2
```

## Getting Help

View all available command-line options:

```bash
python motion_extraction.py --help
```

## Configuration File Format

The `motion_extraction.ini` file uses standard INI format with sections for each parameter category:

```ini
[InputOutput]
video_dir = /path/to/videos
analysis_dir = /path/to/output
csv_path = /path/to/batch_list.csv
allowed_exts = mp4,avi,mts
ARCHIVE_OUTPUTS = False

[MOG2]
mog_history = 50
var_threshold = 10
detect_shadows = 0
kernel_size = 5
motion_threshold_min = 10
motion_threshold_max = 50000

[Tracking]
min_distance_ratio = 0.01
trigger_threshold = 15
tracking_buffer = 0.15
tracking_distance = 0.15
area_weight = 0.01
min_event_length = 3

# See motion_extraction.ini for complete template with all parameters
```

## Parameter Reference

### Input/Output Configuration

| Parameter | CLI Argument | Type | Default | Description |
|-----------|--------------|------|---------|-------------|
| `video_dir` | `--video-dir` | path | `path_to_input_directory` | Root directory containing input videos |
| `analysis_dir` | `--analysis-dir` | path | `path_to_output_directory` | Output directory for analysis results |
| `csv_path` | `--csv-path` | path | None | CSV file with 'input' and 'output' columns |
| `allowed_exts` | `--allowed-exts` | string | `mp4,avi,mts` | Comma-separated video extensions |
| `ARCHIVE_OUTPUTS` | `--archive-outputs` / `--no-archive-outputs` | boolean | False | Archive outputs to tar files |

### MOG2 Background Subtraction

| Parameter | CLI Argument | Type | Default | Description |
|-----------|--------------|------|---------|-------------|
| `mog_history` | `--mog-history` | int | 50 | Number of frames for background model |
| `var_threshold` | `--var-threshold` | int | 10 | Variance threshold for pixel classification |
| `detect_shadows` | `--detect-shadows` | int | 0 | Shadow detection (0=disabled, 1=enabled) |
| `kernel_size` | `--kernel-size` | int | 5 | Morphological kernel size in pixels |
| `motion_threshold_min` | `--motion-threshold-min` | int | 10 | Minimum blob area in pixels² |
| `motion_threshold_max` | `--motion-threshold-max` | int | 50000 | Maximum blob area in pixels² |

### Motion Tracking

| Parameter | CLI Argument | Type | Default | Description |
|-----------|--------------|------|---------|-------------|
| `min_distance_ratio` | `--min-distance-ratio` | float | 0.01 | Bounding box merge threshold |
| `trigger_threshold` | `--trigger-threshold` | int | 15 | Max detections per frame before skipping |
| `tracking_buffer` | `--tracking-buffer` | float | 0.15 | Max seconds track persists without detection |
| `tracking_distance` | `--tracking-distance` | float | 0.15 | Max normalised distance for association |
| `area_weight` | `--area-weight` | float | 0.01 | Area similarity weight in cost function |
| `min_event_length` | `--min-event-length` | int | 3 | Minimum frames for GIF generation |

### ROI and Heatmap

| Parameter | CLI Argument | Type | Default | Description |
|-----------|--------------|------|---------|-------------|
| `mask_tuner_flag` | `--mask-tuner` | boolean | False | Enable interactive ROI mask tuner |
| `motion_tuner_flag` | `--motion-tuner` | boolean | False | Enable interactive parameter tuner |
| `mask_starting_point` | `--mask-starting-point` | int | 2000 | Seconds to skip before heatmap sampling |
| `heatmap_samples` | `--heatmap-samples` | int | 20 | Number of temporal samples for heatmap |
| `ADJACENT_TOL` | `--adjacent-tol` | int | 0 | Event mask interval merge tolerance |

### Output Generation

| Parameter | CLI Argument | Type | Default | Description |
|-----------|--------------|------|---------|-------------|
| `merge_bb` | `--merge-bb` / `--no-merge-bb` | boolean | True | Enable bounding box merging |
| `save_masks` | `--save-masks` | boolean | False | Save binary foreground masks |
| `save_fg` | `--save-fg` | boolean | False | Save foreground cutouts |

## Examples

### Example 1: Basic Usage with Directories

```bash
python motion_extraction.py \
    --video-dir /data/recordings \
    --analysis-dir /data/output
```

### Example 2: Adjust MOG2 Sensitivity

```bash
python motion_extraction.py \
    --config motion_extraction.ini \
    --var-threshold 5 \
    --motion-threshold-min 5
```

### Example 3: Enable Interactive Tuners

```bash
python motion_extraction.py \
    --config motion_extraction.ini \
    --mask-tuner \
    --motion-tuner
```

### Example 4: Batch Processing with CSV

```bash
python motion_extraction.py \
    --video-dir /data/root \
    --analysis-dir /data/output \
    --csv-path /data/batch_list.csv
```

### Example 5: Archive Outputs

```bash
python motion_extraction.py \
    --config motion_extraction.ini \
    --archive-outputs
```

## Configuration Priority

When the same parameter is specified in multiple places, the priority order is:

1. **Command-line argument** (highest)
2. **INI file value**
3. **Default value** (lowest)

Example:
```bash
# motion_extraction.ini contains: mog_history = 50
# This command will use mog_history = 100 (CLI overrides INI)
python motion_extraction.py --config motion_extraction.ini --mog-history 100
```

## Verifying Active Configuration

The pipeline prints an "ACTIVE CONFIGURATION" summary at startup showing all parameter values in use:

```
================================================================================
ACTIVE CONFIGURATION
================================================================================
Video Directory:          /path/to/videos
Analysis Directory:       /path/to/output
CSV Path:                 None
Allowed Extensions:       avi, mp4, mts
Archive Outputs:          False

MOG2 Parameters:
  History:                50
  Var Threshold:          10
  Detect Shadows:         0
  ...
================================================================================
```
