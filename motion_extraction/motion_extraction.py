#!/usr/bin/env python3
"""
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

Configuration
-------------
All parameters can be configured via:
    - Command-line arguments (highest priority)
    - INI configuration file (--config)
    - Default values in script (lowest priority)

Run `python motion_extraction.py --help` for all configuration options.
"""

import cv2
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
from scipy.optimize import linear_sum_assignment
from bisect import bisect_right
import csv
import tarfile
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import argparse
import configparser
from typing import Dict, Any

# =============================================================================
# Configuration Management Functions
# =============================================================================

def load_config_from_ini(ini_path: str) -> Dict[str, Any]:
    """
    Load configuration parameters from an INI file.

    Args:
        ini_path: Path to the motion_extraction.ini configuration file

    Returns:
        Dictionary containing all configuration parameters with appropriate type conversions
    """
    config = configparser.ConfigParser()
    config.read(ini_path)

    params = {}

    # Input/Output Configuration
    if config.has_section('InputOutput'):
        io = config['InputOutput']
        params['video_dir'] = io.get('video_dir', 'path_to_input_directory')
        params['analysis_dir'] = io.get('analysis_dir', 'path_to_output_directory')
        params['csv_path'] = io.get('csv_path', None) if io.get('csv_path', '').strip() else None
        params['allowed_exts'] = set(io.get('allowed_exts', 'mp4,avi,mts').split(','))
        params['ARCHIVE_OUTPUTS'] = io.getboolean('ARCHIVE_OUTPUTS', False)

    # MOG2 Background Subtraction Parameters
    if config.has_section('MOG2'):
        mog2 = config['MOG2']
        params['mog_history'] = mog2.getint('mog_history', 50)
        params['var_threshold'] = mog2.getint('var_threshold', 10)
        params['detect_shadows'] = mog2.getint('detect_shadows', 0)
        params['kernel_size'] = mog2.getint('kernel_size', 5)
        params['motion_threshold_min'] = mog2.getint('motion_threshold_min', 10)
        params['motion_threshold_max'] = mog2.getint('motion_threshold_max', 50000)

    # Rapid Scanning Parameters
    if config.has_section('RapidScan'):
        rapid = config['RapidScan']
        params['rapid_scan_seconds'] = rapid.getint('rapid_scan_seconds', 0)
        params['skip_factor'] = rapid.getint('skip_factor', 2)

    # Motion Tracking Parameters
    if config.has_section('Tracking'):
        track = config['Tracking']
        params['min_distance_ratio'] = track.getfloat('min_distance_ratio', 0.01)
        params['trigger_threshold'] = track.getint('trigger_threshold', 15)
        params['tracking_buffer'] = track.getfloat('tracking_buffer', 0.15)
        params['tracking_distance'] = track.getfloat('tracking_distance', 0.15)
        params['area_weight'] = track.getfloat('area_weight', 0.01)
        params['min_event_length'] = track.getint('min_event_length', 3)

    # ROI and Heatmap Parameters
    if config.has_section('ROI'):
        roi = config['ROI']
        params['mask_tuner_flag'] = roi.getboolean('mask_tuner_flag', False)
        params['motion_tuner_flag'] = roi.getboolean('motion_tuner_flag', False)
        params['mask_starting_point'] = roi.getint('mask_starting_point', 2000)
        params['heatmap_samples'] = roi.getint('heatmap_samples', 20)
        params['ADJACENT_TOL'] = roi.getint('ADJACENT_TOL', 0)

    # Output Generation Flags
    if config.has_section('Output'):
        output = config['Output']
        params['merge_bb'] = output.getboolean('merge_bb', True)
        params['save_masks'] = output.getboolean('save_masks', False)
        params['save_fg'] = output.getboolean('save_fg', False)

    # Parameter Tuning Utilities
    if config.has_section('Tuning'):
        tuning = config['Tuning']
        params['FLOAT_SCALE'] = tuning.getint('FLOAT_SCALE', 1000)
        params['playback_speed'] = tuning.getfloat('playback_speed', 1.0)

    # Annotation Dictionary
    if config.has_section('Annotations'):
        annot = config['Annotations']
        params['annotation_dictionary'] = dict(annot.items())

    return params


def parse_command_line_args() -> argparse.Namespace:
    """
    Parse command-line arguments for all configurable parameters.

    Returns:
        Namespace object containing all CLI arguments
    """
    parser = argparse.ArgumentParser(
        description='Motion Detection and Tracking Pipeline for Bat Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with default parameters (edit script directly)
  python motion_extraction.py

  # Run with custom input/output directories
  python motion_extraction.py --video-dir /path/to/videos --analysis-dir /path/to/output

  # Run with INI configuration file
  python motion_extraction.py --config motion_extraction.ini

  # Run with INI file and CLI overrides (CLI takes precedence)
  python motion_extraction.py --config motion_extraction.ini --mog-history 100 --var-threshold 15
        '''
    )

    # Configuration file
    parser.add_argument('--config', type=str, metavar='PATH',
                        help='Path to INI configuration file (overrides defaults, overridden by CLI args)')

    # Input/Output Configuration
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument('--video-dir', type=str, metavar='PATH',
                          help='Root directory containing input videos')
    io_group.add_argument('--analysis-dir', type=str, metavar='PATH',
                          help='Output directory for analysis results')
    io_group.add_argument('--csv-path', type=str, metavar='PATH',
                          help='CSV file with "input" and "output" columns for batch processing')
    io_group.add_argument('--allowed-exts', type=str, metavar='EXT1,EXT2,...',
                          help='Comma-separated video file extensions (default: mp4,avi,mts)')
    io_group.add_argument('--archive-outputs', action='store_true', default=None,
                          help='Archive processed outputs to tar files')
    io_group.add_argument('--no-archive-outputs', action='store_false', dest='archive_outputs', default=None,
                          help='Do not archive outputs (default)')

    # MOG2 Background Subtraction Parameters
    mog2_group = parser.add_argument_group('MOG2 Background Subtraction')
    mog2_group.add_argument('--mog-history', type=int, metavar='N',
                            help='Number of frames for background model (default: 50)')
    mog2_group.add_argument('--var-threshold', type=int, metavar='N',
                            help='Variance threshold for pixel classification (default: 10)')
    mog2_group.add_argument('--detect-shadows', type=int, choices=[0, 1], metavar='0|1',
                            help='Enable shadow detection (0=disabled, 1=enabled, default: 0)')
    mog2_group.add_argument('--kernel-size', type=int, metavar='N',
                            help='Morphological kernel size in pixels (default: 5)')
    mog2_group.add_argument('--motion-threshold-min', type=int, metavar='N',
                            help='Minimum blob area in pixels² (default: 10)')
    mog2_group.add_argument('--motion-threshold-max', type=int, metavar='N',
                            help='Maximum blob area in pixels² (default: 50000)')

    # Rapid Scanning Parameters
    scan_group = parser.add_argument_group('Rapid Scanning')
    scan_group.add_argument('--rapid-scan-seconds', type=int, metavar='N',
                            help='Duration for rapid scan mode in seconds (0=disabled, default: 0)')
    scan_group.add_argument('--skip-factor', type=int, metavar='N',
                            help='Frame subsampling rate for rapid scan (default: 2)')

    # Motion Tracking Parameters
    track_group = parser.add_argument_group('Motion Tracking')
    track_group.add_argument('--min-distance-ratio', type=float, metavar='RATIO',
                             help='Bounding box merge threshold as fraction of frame diagonal (default: 0.01)')
    track_group.add_argument('--trigger-threshold', type=int, metavar='N',
                             help='Max detections per frame before skipping (default: 15)')
    track_group.add_argument('--tracking-buffer', type=float, metavar='SECS',
                             help='Max seconds track persists without detection (default: 0.15)')
    track_group.add_argument('--tracking-distance', type=float, metavar='RATIO',
                             help='Max normalized distance for track-detection association (default: 0.15)')
    track_group.add_argument('--area-weight', type=float, metavar='WEIGHT',
                             help='Area similarity weight in cost function (default: 0.01)')
    track_group.add_argument('--min-event-length', type=int, metavar='N',
                             help='Minimum frames for GIF generation (default: 3)')

    # ROI and Heatmap Parameters
    roi_group = parser.add_argument_group('ROI and Heatmap')
    roi_group.add_argument('--mask-tuner', action='store_true', dest='mask_tuner_flag', default=None,
                           help='Enable interactive ROI mask tuner')
    roi_group.add_argument('--motion-tuner', action='store_true', dest='motion_tuner_flag', default=None,
                           help='Enable interactive motion parameter tuner')
    roi_group.add_argument('--mask-starting-point', type=int, metavar='SECS',
                           help='Seconds to skip before heatmap sampling (default: 2000)')
    roi_group.add_argument('--heatmap-samples', type=int, metavar='N',
                           help='Number of temporal samples for heatmap (default: 20)')
    roi_group.add_argument('--adjacent-tol', type=int, metavar='N',
                           help='Tolerance for merging adjacent event mask intervals (default: 0)')

    # Output Generation Flags
    output_group = parser.add_argument_group('Output Generation')
    output_group.add_argument('--merge-bb', action='store_true', dest='merge_bb', default=None,
                              help='Enable bounding box merging (default)')
    output_group.add_argument('--no-merge-bb', action='store_false', dest='merge_bb', default=None,
                              help='Disable bounding box merging')
    output_group.add_argument('--save-masks', action='store_true', default=None,
                              help='Save binary foreground masks')
    output_group.add_argument('--save-fg', action='store_true', default=None,
                              help='Save foreground cutouts')

    # Parameter Tuning Utilities
    tuning_group = parser.add_argument_group('Parameter Tuning')
    tuning_group.add_argument('--float-scale', type=int, metavar='N',
                              help='Float trackbar scaling factor (default: 1000)')
    tuning_group.add_argument('--playback-speed', type=float, metavar='SPEED',
                              help='Playback speed multiplier for tuner (default: 1.0)')

    return parser.parse_args()


def merge_configurations(defaults: Dict[str, Any], ini_params: Dict[str, Any],
                         cli_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration from multiple sources with proper precedence.

    Precedence order (highest to lowest):
        1. Command-line arguments
        2. INI file parameters
        3. Default values

    Args:
        defaults: Default parameter dictionary
        ini_params: Parameters loaded from INI file
        cli_args: Parsed command-line arguments

    Returns:
        Merged configuration dictionary
    """
    config = defaults.copy()

    # Apply INI file parameters (override defaults)
    config.update(ini_params)

    # Apply CLI arguments (override both defaults and INI)
    # Only apply if explicitly set (not None)
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if value is not None and key != 'config':
            # Convert CLI arg names (with hyphens) to variable names (with underscores)
            param_name = key.replace('-', '_')

            # Handle special case for allowed_exts (comma-separated string to set)
            if param_name == 'allowed_exts':
                config[param_name] = set(value.split(','))
            else:
                config[param_name] = value

    return config


# =============================================================================
# CPU vs GPU processing
# =============================================================================
"""
This block checks for a CUDA-enabled version of OpenCV and, if found,
enables GPU processing by default.
"""

print("OpenCV version:", cv2.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2,'cuda') else 'no cuda')

if hasattr(cv2,'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
    use_cuda = True
    print("GPU Processing Activated")
else:
    use_cuda = False
    print("CPU Processing Activated")
    
# =============================================================================
# Configuration Initialisation
# =============================================================================
"""
This section initialises all configurable parameters from three sources with the following precedence:
    1. Command-line arguments (highest priority)
    2. INI configuration file (if provided via --config)
    3. Default values (lowest priority)

Parameters can be supplied via:
    - Editing the defaults dictionary below directly in the script
    - Providing a motion_extraction.ini file via --config argument
    - Passing individual parameters via command-line arguments

See `python motion_extraction.py --help` for all available CLI options.
"""

# Define default parameter values
defaults = {
    # Input/Output Configuration
    'video_dir': "path_to_input_directory",
    'analysis_dir': "path_to_output_directory",
    'csv_path': None,
    'allowed_exts': {"mp4", "avi", "mts"},
    'ARCHIVE_OUTPUTS': False,

    # MOG2 Background Subtraction Parameters
    'mog_history': 50,
    'var_threshold': 10,
    'detect_shadows': 0,
    'kernel_size': 5,
    'motion_threshold_min': 10,
    'motion_threshold_max': 50000,

    # Rapid Scanning Parameters
    'rapid_scan_seconds': 0,
    'skip_factor': 2,

    # Motion Tracking Parameters
    'min_distance_ratio': 0.01,
    'trigger_threshold': 15,
    'tracking_buffer': 0.15,
    'tracking_distance': 0.15,
    'area_weight': 0.01,
    'min_event_length': 3,

    # ROI and Heatmap Parameters
    'mask_tuner_flag': False,
    'motion_tuner_flag': False,
    'mask_starting_point': 2000,
    'heatmap_samples': 20,
    'ADJACENT_TOL': 0,

    # Output Generation Flags
    'merge_bb': True,
    'save_masks': False,
    'save_fg': False,

    # Parameter Tuning Utilities
    'FLOAT_SCALE': 1000,
    'playback_speed': 1.0,

    # Annotation Dictionary
    'annotation_dictionary': {
        'bat': 'Bat',
        'bird': 'Bird',
        'other': 'Other',
        'human': 'Human'
    },
    'default_annotation': {'NA': 'NA'}
}

# Parse command-line arguments
args = parse_command_line_args()

# Load INI file if provided
ini_params = {}
if args.config:
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    ini_params = load_config_from_ini(args.config)
    print(f"Loaded configuration from: {args.config}")

# Merge configurations with proper precedence (CLI > INI > defaults)
config = merge_configurations(defaults, ini_params, args)

# Extract parameters from merged configuration
video_dir = config['video_dir']
analysis_dir = config['analysis_dir']
csv_path = config['csv_path']
allowed_exts = config['allowed_exts']
ARCHIVE_OUTPUTS = config['ARCHIVE_OUTPUTS']

mog_history = config['mog_history']
var_threshold = config['var_threshold']
detect_shadows = config['detect_shadows']
kernel_size = config['kernel_size']
motion_threshold_min = config['motion_threshold_min']
motion_threshold_max = config['motion_threshold_max']

rapid_scan_seconds = config['rapid_scan_seconds']
skip_factor = config['skip_factor']

min_distance_ratio = config['min_distance_ratio']
trigger_threshold = config['trigger_threshold']
tracking_buffer = config['tracking_buffer']
tracking_distance = config['tracking_distance']
area_weight = config['area_weight']
min_event_length = config['min_event_length']

mask_tuner_flag = config['mask_tuner_flag']
motion_tuner_flag = config['motion_tuner_flag']
mask_starting_point = config['mask_starting_point']
heatmap_samples = config['heatmap_samples']
ADJACENT_TOL = config.get('ADJACENT_TOL', 0)

merge_bb = config['merge_bb']
save_masks = config['save_masks']
save_fg = config['save_fg']

FLOAT_SCALE = config['FLOAT_SCALE']
playback_speed = config['playback_speed']

annotation_dictionary = config['annotation_dictionary']
default_annotation = config['default_annotation']

# Print active configuration summary
print("\n" + "="*80)
print("ACTIVE CONFIGURATION")
print("="*80)
print(f"Video Directory:          {video_dir}")
print(f"Analysis Directory:       {analysis_dir}")
print(f"CSV Path:                 {csv_path if csv_path else 'None'}")
print(f"Allowed Extensions:       {', '.join(sorted(allowed_exts))}")
print(f"Archive Outputs:          {ARCHIVE_OUTPUTS}")
print(f"\nMOG2 Parameters:")
print(f"  History:                {mog_history}")
print(f"  Var Threshold:          {var_threshold}")
print(f"  Detect Shadows:         {detect_shadows}")
print(f"  Kernel Size:            {kernel_size}")
print(f"  Motion Threshold Min:   {motion_threshold_min}")
print(f"  Motion Threshold Max:   {motion_threshold_max}")
print(f"\nTracking Parameters:")
print(f"  Min Distance Ratio:     {min_distance_ratio}")
print(f"  Trigger Threshold:      {trigger_threshold}")
print(f"  Tracking Buffer:        {tracking_buffer}")
print(f"  Tracking Distance:      {tracking_distance}")
print(f"  Area Weight:            {area_weight}")
print(f"  Min Event Length:       {min_event_length}")
print(f"\nInteractive Tuners:")
print(f"  Mask Tuner:             {mask_tuner_flag}")
print(f"  Motion Tuner:           {motion_tuner_flag}")
print("="*80 + "\n")

# =============================================================================
# Input and Output Configuration
# =============================================================================
"""
This section sets how batch runs find videos and where outputs are written:
- `video_dir`: Root directory scanned for videos with extensions in `allowed_exts`;
  relative paths from a CSV are resolved against this directory.
- `analysis_dir`: Per-video subdirectories are created here for crops, GIFs, logs,
  and metrics; the path is printed so runs are traceable.
- `csv_path` (optional): Expects `input` and `output` columns; `input` may be
  relative or absolute, `output` names the per-video folder/ID to keep filenames
  paper-ready.
- `ARCHIVE_OUTPUTS`: Toggle to compress processed outputs once verification is
  complete, useful for long-term storage or submissions.
- `allowed_exts` and the derived default IDs keep ingestion consistent so the
  manuscript results can be reproduced from the same file list.
"""

input_videos = []   # absolute paths to input videos
video_ids = []      # corresponding video IDs (aligned with input_videos)

# Ensure the analysis directory exists
os.makedirs(analysis_dir, exist_ok=True)

if csv_path and os.path.isfile(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'input', 'output'}.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain 'input' and 'output' columns.")

        for row in reader:
            raw_path = (row.get('input') or '').strip()
            vid_id   = (row.get('output') or '').strip()

            if not raw_path:
                continue

            # Allow relative paths in CSV (relative to video_dir)
            in_path = raw_path if os.path.isabs(raw_path) else os.path.join(video_dir, raw_path)

            # Filter by extension and existence
            ext = os.path.splitext(in_path)[1].lower().lstrip('.')
            if ext not in allowed_exts:
                print(f"Skipping (unsupported extension): {in_path}")
                continue
            if not os.path.isfile(in_path):
                print(f"Warning: file not found, skipping: {in_path}")
                continue

            input_videos.append(os.path.abspath(in_path))
            video_ids.append(vid_id)

else:
    for file in os.listdir(video_dir):
        if file.split(".")[-1].lower() in allowed_exts:
            input_videos.append(os.path.join(video_dir, file))

    # Keep original sorting behavior
    input_videos = sorted(input_videos)

    # Derive default IDs from filenames (without extension)
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in input_videos]

print(f"Number of input videos for batch analysis: {len(input_videos)}")

# =============================================================================
# Event Masking for Exclusions
# =============================================================================
"""
If an `event_masks.csv` is provided, the pipeline will screen out specified frame windows before processing. This was used to omit the initial 30‑second buffered event windows from secondary data mining.
- **Input format**: `event_masks.csv` with `video_id`, `start_frame`, and `end_frame` (or `start_time_s`, `end_time_s` if converted to frames using FPS). Multiple rows per video are allowed.
- **Behaviour**: the intervals are merged (respecting any tolerance) so overlapping or adjacent windows become a single exclusion range.
- **Effect**: frames inside the merged exclusion windows are skipped for detection/tracking, preventing buffered segments from being re-analysed or double-counted.
"""

def merge_intervals(intervals, tol=0): 
    """
    intervals: list of (start, end) inclusive. tol merges if next.start <= prev.end + tol.
    """ 
    if not intervals:
        return [] 
    intervals = sorted(intervals, key=lambda x: x[0]) 
    merged = [list(intervals[0])] 
    for s, e in intervals[1:]: 
        last_s, last_e = merged[-1] 
        if s <= last_e + tol: 
            merged[-1][1] = max(last_e, e) 
        else:
            merged.append([s, e]) 
    return merged

def make_interval_checker(merged_intervals):
    """
    Create efficient frame mask checker using binary search.

    Constructs a callable that performs O(log n) lookup to determine if
    a given frame index falls within any masked interval. Used to skip
    pre-annotated or problematic frames during motion extraction.

    Args:
        merged_intervals: List of non-overlapping [start, end] intervals from merge_intervals

    Returns:
        callable: Function accepting frame_idx (int) returning bool (True if masked)
    """
    if not merged_intervals:
        # Always False: no masked frames for this video
        print("No masks")
        return lambda frame: False
    # Ensure sorted, non-overlapping (your merge_intervals already guarantees this)
    starts = [s for s, _ in merged_intervals]
    intervals = merged_intervals  # [[s,e], [s,e], ...]

    def _check(frame: int) -> bool:
        i = bisect_right(starts, frame) - 1
        return i >= 0 and intervals[i][0] <= frame <= intervals[i][1]
    return _check

if "event_masks.csv" in os.listdir(video_dir):
    events_masks = pd.read_csv(os.path.join(video_dir,"event_masks.csv"))
    START_COL = "StartFrame"
    END_COL = "EndFrame"
    KEY_COL = "file"
    # ADJACENT_TOL now configured via CLI/INI/defaults above
    
    events_masks[START_COL] = events_masks[START_COL].astype(int)
    events_masks[END_COL] = events_masks[END_COL].astype(int) 
    
    masks = {} 
    for file, grp in events_masks.groupby(KEY_COL, sort=False): 
        intervals = list(zip(grp[START_COL], grp[END_COL])) 
        merged = merge_intervals(intervals, tol=ADJACENT_TOL) 
        masks[file] = merged
    
    mask_checkers = {fp: make_interval_checker(iv) for fp, iv in masks.items()}
    
    print("Event Masks have been applied to video inputs.")
else:
    mask_checkers = {}
    print("No event masks supplied.")
    
# =============================================================================
# MOG2 Background Subtraction Parameters
# =============================================================================
"""
Algorithm Overview
------------------
The Mixture of Gaussians 2 (MOG2) algorithm (Zivkovic & van der Heijden, 2006)
performs adaptive background subtraction by modelling each pixel as a mixture of
Gaussian distributions. The algorithm automatically learns which distributions
represent background and which represent foreground, updating these models
incrementally as new frames are processed. This adaptive approach enables the
system to handle gradual illumination changes, swaying vegetation, and other
dynamic background elements common in outdoor video monitoring scenarios.

Parameter Descriptions
----------------------
The following parameters control the MOG2 background model behaviour:

`mog_history` (default: 50 frames):
    Number of frames utilised to build the initial background model. A shorter
    history enables faster adaptation to changing conditions but may be less
    robust to transient foreground objects. This value was optimised for bat
    detection in nocturnal infrared footage.

`var_threshold` (default: 10):
    Variance threshold for pixel classification. Lower values increase sensitivity
    to subtle motion but may trigger false positives from sensor noise. This
    threshold determines how many standard deviations a pixel must deviate from
    the background model to be classified as foreground.

`detect_shadows` (default: 0):
    Shadow detection flag. Disabled (0) for infrared footage where shadows are
    not present. When enabled, shadows are detected and marked separately from
    foreground objects.

Morphological Filtering Parameters
-----------------------------------
After MOG2 generates the binary foreground mask, morphological operations remove
sensor noise:

`kernel_size` (default: 5×5 pixels):
    Structuring element size for morphological operations. The pipeline applies:

    - Opening operation: Erosion followed by dilation, removing small noise blobs
      from the foreground mask
    - Closing operation: Dilation followed by erosion, filling small holes within
      foreground objects

    These operations are essential for cleaning sensor noise in low-light infrared
    imagery whilst preserving object boundaries.

Blob Area Thresholds
--------------------
Connected components in the foreground mask are filtered by area to remove
residual noise:

`motion_threshold_min` (default: 10 pixels²):
    Minimum blob area. Smaller regions are discarded as sensor noise.

`motion_threshold_max` (default: 50,000 pixels²):
    Maximum blob area. Larger regions typically represent background motion or
    frame-wide illumination changes.

These thresholds were optimised using annotated bat events from ecological
consultants across the training dataset to achieve sufficient sensitivity whilst
minimising false positives.

Scanning Parameters
-------------------
`rapid_scan_seconds`:
    Enables fast initial scanning of video files. The feature was disabled for
    purposes of the paper analysis.

`skip_factor`:
    Frame subsampling rate for rapid scanning mode.

Note
----
All parameters above are now configured via CLI arguments, INI file, or the
defaults dictionary in the Configuration Initialisation section.
"""

# Create MOG2 background subtractor with configured parameters
if use_cuda:
    mog2Subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
else:
    mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    
# =============================================================================
# Motion Tracking Parameters
# =============================================================================
"""
Kalman Filter and Hungarian Algorithm Configuration
----------------------------------------------------
The following parameters control the multi-object tracking system:

`tracking_buffer` (seconds):
    Maximum duration a track persists without matching detections before
    termination. This buffer accommodates temporary occlusions and missed
    detections. Converted to frames internally via multiplication by frame rate.

`tracking_distance` (pixels):
    Maximum spatial distance threshold for associating predictions with detections.
    Predicted positions beyond this threshold from any detection are considered
    unmatched.

`area_weight` (default: 0.1):
    Weighting factor for area similarity in the cost function. The cost function
    combines spatial proximity with normalised area difference:
    cost = spatial_distance + area_weight × |area_difference|/mean_area
    This area-based approach stabilises tracking in 3D scenes without depth
    information (Bjerge et al., 2022).

`mask_tuner_flag`:
    Enables interactive parameter tuning interface for optimisation across
    different deployment environments.

These parameters were optimised using annotated bat events to balance detection
sensitivity with tracking stability.

Note
----
All parameters above are now configured via CLI arguments, INI file, or the
defaults dictionary in the Configuration Initialisation section.
"""

# =============================================================================
# Helper Functions
# =============================================================================
"""
Key Utilities Shared Across the Pipeline
-----------------------------------------

`merge_intervals(intervals, tol)`:
    Coalesces overlapping or nearby frame intervals to report contiguous events
    cleanly.

`video_loader(path)`:
    Opens the capture, returns `capture`, `fps`, and `total_frames`, normalising
    fractional frame rates so downstream timing is reliable.

`merge_bounding_boxes(...)`:
    Merges overlapping or nearby detections using buffered IoU and union-find to
    avoid duplicate tracks when blobs touch.

`xywh_to_yolo(...)`:
    Converts pixel-space boxes to normalised YOLO `[x_c, y_c, w, h]` format used
    in the output CSVs.

`_jsonable(...)`:
    Converts NumPy-heavy structures into serialisable lists/dicts for saving masks
    and parameter logs.
"""

# Video loader is a function that loads video files and returns the fps and length (in terms of frame count).
def video_loader(path):
    """
    Load video file and return capture object with metadata.

    Args:
        path (str): Absolute path to an input video.

    Returns:
        tuple: A tuple containing:
            - capture (cv2.VideoCapture): OpenCV VideoCapture object to read video frames.
            - fps (int): Frames per second (FPS) of the video.
            - total_frames (int): Total number of frames in the video.
    """
    capture = cv2.VideoCapture(path)
    # Account for framerates like 59.955555
    fps = int(math.ceil(capture.get(cv2.CAP_PROP_FPS)))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return capture, fps, total_frames

def merge_bounding_boxes(bounding_boxes, min_distance_ratio, frame_height, frame_width):
    """
    Efficiently merge nearby or overlapping bounding boxes using Intersection over Union (IoU).
    
    Prevents duplicate tracking of the same object when multiple overlapping foreground
    blobs are detected. The algorithm buffers each bounding box by a minimum distance
    threshold, identifies overlapping regions using a sweep-line approach with union-find
    clustering, then merges the original (unbuffered) boxes within each cluster.
    
    Strategy:
        1. Buffer each box by min_distance pixels to define proximity neighbourhood
        2. Sort buffered boxes by x-coordinate and sweep once
        3. Use active set and union-find to cluster overlapping buffered boxes
        4. For each cluster, merge original (unbuffered) boxes and clip to frame bounds
    
    This approach ensures that objects detected as multiple fragments are consolidated
    into single tracks, preventing the MultiObjectTracker from initiating redundant
    tracks for the same physical object.
    
    Args:
        bounding_boxes: List of (x, y, w, h) tuples in pixel coordinates
        min_distance_ratio: Buffering distance as fraction of frame diagonal
        frame_height: Video frame height (pixels)
        frame_width: Video frame width (pixels)
    
    Returns:
        list: Merged bounding boxes as (x, y, w, h) tuples in original coordinates
    """
    if not bounding_boxes:
        return []

    # 1) Buffer distance in pixels (based on smaller frame dim)
    min_distance = float(min_distance_ratio) * float(min(frame_width, frame_height))

    # Prepare arrays for fast access
    # original boxes
    xs  = []
    ys  = []
    ws  = []
    hs  = []
    # buffered boxes (x1, y1, x2, y2)
    bx1 = []
    by1 = []
    bx2 = []
    by2 = []

    # precompute buffered boxes, clipped to frame
    for (x, y, w, h) in bounding_boxes:
        xs.append(x); ys.append(y); ws.append(w); hs.append(h)
        x1 = max(0.0, x - min_distance)
        y1 = max(0.0, y - min_distance)
        x2 = min(float(frame_width),  x + w + min_distance)
        y2 = min(float(frame_height), y + h + min_distance)
        bx1.append(x1); by1.append(y1); bx2.append(x2); by2.append(y2)

    n = len(bounding_boxes)
    idx = list(range(n))

    # 2) Sort indices by buffered x1 (left edge)
    idx.sort(key=lambda i: bx1[i])

    # 3) Union-Find (Disjoint Set) for clustering overlapping boxes
    # Each box starts in its own cluster; overlaps merge clusters
    parent = list(range(n))
    rank   = [0]*n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Sweep-line algorithm: process boxes left-to-right, maintaining active set
    # Active set contains boxes whose right edge extends beyond current box's left edge
    # This reduces overlap checks from O(n²) to O(n log n) for typical scenes
    active = []  # list of indices (into 0..n-1) that are still overlapping in x
    from collections import deque
    active = deque()  # allows O(1) pops from left

    for k in idx:
        # drop boxes that are strictly left of current box (no x-overlap)
        while active and bx2[active[0]] < bx1[k]:
            active.popleft()

        # check y-overlap with active set; if also y-overlap, union
        y1_k, y2_k = by1[k], by2[k]
        for j in active:
            # y overlap?
            if not (y2_k <= by1[j] or y1_k >= by2[j]):
                union(k, j)

        # add current to active
        active.append(k)

    # 4) Collect clusters and merge ORIGINAL boxes within each cluster
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    merged = []
    for root, inds in clusters.items():
        # merge original boxes (not buffered), then clip
        min_x = min(xs[i] for i in inds)
        min_y = min(ys[i] for i in inds)
        max_x = max(xs[i] + ws[i] for i in inds)
        max_y = max(ys[i] + hs[i] for i in inds)

        # clip to frame
        min_x = max(0, int(round(min_x)))
        min_y = max(0, int(round(min_y)))
        max_x = min(frame_width,  int(round(max_x)))
        max_y = min(frame_height, int(round(max_y)))

        w = max_x - min_x
        h = max_y - min_y
        if w > 0 and h > 0:
            merged.append((min_x, min_y, w, h))

    return merged

# =============================================================================
# Motion Tracking Implementation
# =============================================================================
"""
Overview
--------
The motion tracking system employs Kalman filtering with Hungarian algorithm
matching to maintain consistent object identities across frames. This approach
enables robust tracking through temporary occlusions, missed detections, and
overlapping trajectories.

KalmanTracker Class
-------------------
Each tracked object is represented by a `KalmanTracker` instance that maintains
a six-dimensional state vector and predicts object positions in subsequent frames:

State Vector: [x, y, area, vx, vy, d_area]
    - x, y: Object centroid position (pixels)
    - area: Bounding box area (pixels²)
    - vx, vy: Two-dimensional velocity state (pixels/frame)
    - d_area: Rate of area change (pixels²/frame)

Motion Model:
    The system employs a constant velocity motion model, assuming objects maintain
    their current velocity between frames. The transition matrix predicts the next
    state as::

        x(t+1) = x(t) + vx × Δt
        y(t+1) = y(t) + vy × Δt
        area(t+1) = area(t) + d_area × Δt
        vx(t+1) = vx(t)  [constant velocity assumption]
        vy(t+1) = vy(t)  [constant velocity assumption]

Noise Parameters:
    Process Noise Covariance:
        Models uncertainty in the motion model (default: 0.03 × Identity)

    Measurement Noise Covariance:
        Models sensor noise in detections (default: 0.1 × Identity)

    These parameters control the filter's responsiveness to new measurements versus
    reliance on predictions. Higher process noise increases responsiveness to erratic
    motion, whilst higher measurement noise increases reliance on the motion model.

MultiObjectTracker Class
------------------------
The `MultiObjectTracker` manages multiple `KalmanTracker` instances and performs
optimal assignment between predictions and detections.

Cost Function:
    Combines spatial proximity with area similarity (implemented in
    `_calculate_distance`)::

        cost = spatial_distance + area_weight × (|pred_area - det_area| / mean_area)

    This formulation stabilises tracking in 3D scenes where objects at different
    depths exhibit different apparent sizes. Without depth information, incorporating
    area similarity prevents track identity switches when objects pass near each
    other at different distances from the camera.

Hungarian Algorithm Matching:
    The `update` method constructs a cost matrix C[i,j] representing the cost of
    assigning tracker i to detection j. The Hungarian algorithm (via
    `scipy.optimize.linear_sum_assignment`) finds the assignment minimising the
    total cost across all matches. This optimisation guarantees globally optimal
    assignment, preventing greedy matching errors.

Track Management:
    1. Matched Detections:
       Update corresponding Kalman filter with new measurement, reset unmatched
       counter

    2. Unmatched Detections:
       Initialise new tracks with zero initial velocity

    3. Unmatched Tracks:
       Increment unmatched counter; terminate tracks exceeding threshold

This approach handles track births (new objects entering the scene), deaths
(objects exiting), and temporary occlusions (missed detections).
"""

class KalmanTracker:
    """
    Kalman filter tracker for a single object.
    
    Maintains a six-dimensional state vector [x, y, area, vx, vy, d_area] with
    a two-dimensional velocity state (vx, vy) for predicting object positions in
    subsequent frames. Employs a constant velocity motion model suitable for
    tracking objects with approximately linear motion between frames.
    
    State Vector:
        x, y: Object centroid position (pixels)
        area: Bounding box area (pixels²)
        vx, vy: Two-dimensional velocity state (pixels/frame)
        d_area: Rate of area change (pixels²/frame)
    
    Measurement Vector:
        [x, y, area] - Directly observed from blob detection
    
    The velocity components (vx, vy) are estimated by the Kalman filter from
    successive position measurements, enabling prediction of future positions
    and robust tracking through temporary occlusions.
    
    Noise Parameters:
        processNoiseCov: Motion model uncertainty (default: 0.03)
        measurementNoiseCov: Sensor noise in detections (default: 0.1)
    """
    def __init__(self):
        """
        Initialise Kalman filter with 6-dimensional state and 3-dimensional measurement.
        """
        # State: [x, y, area, vx, vy, d_area], Measurement: [x, y, area]
        self.kf = cv2.KalmanFilter(6, 3)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],   # x
            [0, 1, 0, 0, 0, 0],   # y
            [0, 0, 1, 0, 0, 0]    # area
        ], dtype=np.float32)

        # Constant velocity motion model: position += velocity × dt
        # Assumes linear motion between frames (appropriate for short time steps)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],   # x  = x + vx*dt
            [0, 1, 0, 0, 1, 0],   # y  = y + vy*dt
            [0, 0, 1, 0, 0, 1],   # a  = a + da*dt
            [0, 0, 0, 1, 0, 0],   # vx = vx (constant velocity)
            [0, 0, 0, 0, 1, 0],   # vy = vy (constant velocity)
            [0, 0, 0, 0, 0, 1],   # da = da (constant area change)
        ], dtype=np.float32)

        # Tunables (can be adjusted per scene)
        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)

        self.initialised = False

    def set_dt(self, dt: float):
        """
        Set time step Δt in the transition model.
        
        Scales velocities to be per-second when dt = 1/fps. Typically set to
        1/fps for frame-by-frame processing or k/fps for subsampled processing.
        
        Args:
            dt: Time step in seconds (typically 1/fps)
        """
        F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=np.float32)
        self.kf.transitionMatrix = F

    def initialise(self, x: float, y: float, area: float):
        """
        Initialise the filter with first detection.
        
        Sets initial position and area from detection with zero initial velocities.
        The velocity state will be estimated from subsequent measurements.
        
        Args:
            x: Initial x-coordinate (pixels)
            y: Initial y-coordinate (pixels)
            area: Initial bounding box area (pixels²)
        """
        self.kf.statePost = np.array([[x], [y], [area], [0.0], [0.0], [0.0]], dtype=np.float32)
        self.initialised = True

    def predict(self):
        """
        Predict object state for current frame using motion model.
        
        Returns predicted measurement-space state (x, y, area) based on previous
        state and constant velocity assumption. Used for track-to-detection matching.
        
        Returns:
            np.ndarray: Predicted (x, y, area) as 3-element array
        """
        if not self.initialised:
            return self.kf.statePost[:3, 0]
        pred = self.kf.predict()
        return pred[:3, 0]

    def update(self, measurement):
        """
        Update filter with new measurement.
        
        Corrects predicted state using measurement, refining position and velocity
        estimates. The Kalman gain balances prediction uncertainty against
        measurement noise to produce optimal state estimate.
        
        Args:
            measurement: Detection as (x, y, area) tuple or array
        """
        self.kf.correct(np.array(measurement, dtype=np.float32))

    def get_state(self):
        """
        Return current state vector.
        
        Returns:
            tuple: (x, y, area, vx, vy, d_area) representing full state
        """
        s = self.kf.statePost.reshape(-1)
        return float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])

# ----------------------------
# Multi-object tracker
# ----------------------------
class MultiObjectTracker:
    """
    Multi-object tracker using Kalman filtering and Hungarian algorithm matching.
    
    Manages multiple KalmanTracker instances, performing optimal assignment between
    predicted track positions and current detections. The Hungarian algorithm minimises
    the sum of assignment costs across all matches, guaranteeing globally optimal
    association and preventing track identity switches.
    
    Cost Function:
        Combines spatial proximity with similarity in object area:
        cost = spatial_distance + area_weight × (|pred_area - det_area| / mean_area)
    
        This formulation stabilises tracking in 3D scenes without depth information
        (Bjerge et al., 2022). Objects at different depths exhibit different apparent
        sizes; incorporating area similarity prevents identity switches when objects
        pass near each other at different distances from the camera.
    
    Track Management:
        - Matched detections: Update corresponding Kalman filter, reset unmatched counter
        - Unmatched detections: Initialise new tracks with zero velocity
        - Unmatched tracks: Increment counter; terminate after threshold exceeded
    
    This approach handles track births, deaths, and temporary occlusions whilst
    maintaining consistent object identities throughout video sequences.
    
    Args:
        max_unmatched_frames: Frames before terminating unmatched tracks
        max_distance: Maximum spatial distance for valid associations (pixels)
        area_weight: Weighting factor for area similarity in cost function
    
    Reference:
        Bjerge, K., et al. (2022). Real-time insect tracking and monitoring with
        computer vision and deep learning. Remote Sensing in Ecology and
        Conservation, 9(3), 315-327.
    """
    def __init__(self, max_unmatched_frames=10, max_distance=0.1, area_weight=0.1):
        """
        Initialise multi-object tracker with specified parameters.

        Args:
            max_unmatched_frames: Maximum frames a track persists without detection
            max_distance: Maximum spatial distance for valid track-detection associations
            area_weight: Weighting factor for area similarity in cost calculation
        """
        self.trackers = []            # List[KalmanTracker]
        self.track_ids = []           # Parallel list of IDs
        self.unmatched_counts = []    # Parallel list of unmatched frame counters
        self.next_id = 0
        self.max_unmatched_frames = max_unmatched_frames
        self.max_distance = max_distance
        self.area_weight = area_weight
        self.tracker_log = 0
        self.last_kinematics = {}     # for optional external readback

    def set_dt_all(self, dt: float):
        """
        Set time step Δt in the transition model.
        
        Scales velocities to be per-second when dt = 1/fps. Typically set to
        1/fps for frame-by-frame processing or k/fps for subsampled processing.
        
        Args:
            dt: Time step in seconds (typically 1/fps)
        """
        for trk in self.trackers:
            trk.set_dt(dt)

    def _calculate_distance(self, prediction, detection):
        """
        Calculate cost for associating prediction with detection.
        
        Combines spatial proximity with similarity in object area. The area term
        is normalised by mean area to provide scale-invariant weighting. This
        formulation stabilises tracking in 3D scenes where objects at different
        depths exhibit different apparent sizes (Bjerge et al., 2022).
        
        Args:
            prediction: Predicted (x, y, area) from Kalman filter
            detection: Observed (x, y, area) from blob detection
        
        Returns:
            float: Association cost (lower is better match)
        """
        spatial_distance = np.linalg.norm(np.array(prediction[:2]) - np.array(detection[:2]))
        avg_area = ((prediction[2] + detection[2]) / 2.0) + 1e-6
        rel_area_diff = abs(prediction[2] - detection[2]) / avg_area
        return spatial_distance + self.area_weight * rel_area_diff

    def update(self, detections, logging=False):
        """
        Update filter with new measurement.
        
        Corrects predicted state using measurement, refining position and velocity
        estimates. The Kalman gain balances prediction uncertainty against
        measurement noise to produce optimal state estimate.
        
        Args:
            measurement: Detection as (x, y, area) tuple or array
        """
        """
        Update with list of detections (x, y, area).
        Returns:
            detection_to_tracker_id: {det_idx: track_id}
            detection_kinematics:    {det_idx: {"vx":..., "vy":..., "speed":...}}
        """
        # 1) Predict all trackers
        predicted_states = [trk.predict() for trk in self.trackers]

        detection_to_tracker_id = {}
        detection_kinematics = {}

        num_trackers = len(predicted_states)
        num_detections = len(detections)

        if num_trackers == 0 or num_detections == 0:
            matched_pairs = []
            unmatched_trackers = set(range(num_trackers))
            unmatched_detections = set(range(num_detections))
        else:
            unmatched_trackers = set(range(num_trackers))
            unmatched_detections = set(range(num_detections))

            # Build cost matrix: rows=tracks, cols=detections
            # Infinite cost (1e10) prevents invalid associations beyond max_distance
            cost = np.zeros((num_trackers, num_detections), dtype=np.float32)
            for i, pred in enumerate(predicted_states):
                for j, det in enumerate(detections):
                    d = self._calculate_distance(pred, det)
                    cost[i, j] = d if d <= self.max_distance else 1e10

            # Hungarian algorithm finds optimal assignment minimising total cost
            # Guarantees globally optimal solution (prevents greedy mismatches)
            matched_pairs = []
            valid_row = np.any(cost != 1e10, axis=1)
            valid_col = np.any(cost != 1e10, axis=0)
            if np.any(valid_row) and np.any(valid_col):
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] < 1e10:
                        matched_pairs.append((r, c))

            for trk_idx, det_idx in matched_pairs:
                if trk_idx in unmatched_trackers:
                    unmatched_trackers.remove(trk_idx)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)

        # 3) Update matched trackers
        for trk_idx, det_idx in matched_pairs:
            self.trackers[trk_idx].update(detections[det_idx])
            self.unmatched_counts[trk_idx] = 0

            tid = self.track_ids[trk_idx]
            detection_to_tracker_id[det_idx] = tid

            # Read velocities from state
            x, y, a, vx, vy, da = self.trackers[trk_idx].get_state()
            speed = float(np.hypot(vx, vy))
            detection_kinematics[det_idx] = {"vx": float(vx), "vy": float(vy), "speed": speed, "track_id": tid}

        # 4) Handle unmatched trackers
        to_remove = []
        for idx in unmatched_trackers:
            self.unmatched_counts[idx] += 1
            if self.unmatched_counts[idx] > self.max_unmatched_frames:
                to_remove.append(idx)

        for i in reversed(to_remove):
            self.trackers.pop(i)
            self.track_ids.pop(i)
            self.unmatched_counts.pop(i)
            if logging:
                print(f"Tracker Removed: index={i}")

        # 5) Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            new_trk = KalmanTracker()
            self.tracker_log += 1
            x, y, a = detections[det_idx]
            new_trk.initialise(x, y, a)
            self.trackers.append(new_trk)
            self.unmatched_counts.append(0)

            new_id = self.next_id
            self.next_id += 1
            self.track_ids.append(new_id)

            detection_to_tracker_id[det_idx] = new_id

            # Fresh trackers have zero velocity
            detection_kinematics[det_idx] = {"vx": 0.0, "vy": 0.0, "speed": 0.0, "track_id": new_id}

        if logging and len(self.unmatched_counts) > 0:
            print(f"Active: {len(self.track_ids)} | Map: {detection_to_tracker_id}")

        self.last_kinematics = detection_kinematics
        return detection_to_tracker_id, detection_kinematics

    def get_tracked_objects(self):
        """
        Return current state of all active tracks.
        
        Returns:
            List of (x, y, area) arrays for each tracked object
        """
        return [trk.kf.statePost[:3, 0] for trk in self.trackers]

# =============================================================================
# Parameter Tuning UI
# =============================================================================
"""
Interactive Parameter Optimisation
-----------------------------------
The pipeline includes an interactive tuning interface enabling real-time
visualisation of detection performance across different parameter configurations.
This interface was utilised to optimise the MOG2 background subtraction and
tracking parameters for bat detection in low-contrast infrared footage.

Tuning Methodology
------------------
Parameters for the background subtraction and Kalman filter were tuned using the
annotated bat events provided by ecological consultants across the training data.
The objective was to achieve sufficient sensitivity to detect all bat events
identified within the human annotations whilst minimising false positives from
sensor noise, vegetation motion, and other background artefacts.

Interface Components
--------------------
The tuning interface provides:

1. Heat Map Generation:
   Visualises motion density across the frame, revealing areas of repetitive
   background motion that should be excluded via ROI masking.

2. Real-time Parameter Adjustment:
   Sliders enable dynamic modification of:

   - MOG2 history length (background model temporal window)
   - Variance threshold (foreground classification sensitivity)
   - Blob area thresholds (noise filtering bounds)
   - Tracking distance and buffer parameters

3. Visual Feedback:
   Side-by-side comparison of original frame and foreground mask with detected
   bounding boxes overlaid, facilitating immediate assessment of parameter changes.

Parameter Selection Strategy
----------------------------
The tuning process prioritised:

Sensitivity:
    Ensuring detection of all annotated bat events (minimising false negatives)

Specificity:
    Reducing false positives from noise and background motion

Stability:
    Maintaining consistent track identities through temporary occlusions

Generalisation:
    Performance across diverse environmental conditions and deployment sites

Parameters were validated against held-out test sequences to ensure the optimised
values generalise beyond the training dataset.
"""

def reset_tracker():
    """
    Reinitialise the global multi-object tracker.

    Creates a new MultiObjectTracker instance with current parameter values,
    resetting all active tracks. Called when motion detection parameters change
    during interactive tuning to ensure consistent tracking behaviour.
    """
    global tracker, fps
    tracker = MultiObjectTracker(
        max_unmatched_frames=int(tracking_buffer * fps),
        max_distance=tracking_distance,
        area_weight=area_weight
    )

def reset_subtractor():
    """
    Reinitialise the MOG2 background subtractor.

    Resets the background model when parameters change or when seeking to
    non-contiguous video sections. The MOG2 algorithm maintains adaptive
    background statistics; reinitialisation prevents contamination from
    previous video content when parameters or playback position changes.
    """    
    global mog2Subtractor
    if use_cuda:
        mog2Subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    else:
        mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    reset_tracker()

# The following functions allow MOG2 parameters and noise reduction to be updated via interactive trackbars.
def update_mog2_history(val):
    """
    Update MOG2 history length parameter.

    Args:
        val: Number of frames in background history (larger values = slower adaptation)
    """
    global mog2Subtractor, mog_history, var_threshold, detect_shadows
    mog_history = val
    if use_cuda:
        mog2Subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    else:
        mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    reset_tracker()
    
def update_mog2_var(val):
    """
    Update MOG2 variance threshold parameter.

    Args:
        val: Threshold for squared Mahalanobis distance (larger values = less sensitive)
    """
    global mog2Subtractor, mog_history, var_threshold, detect_shadows
    var_threshold = val
    if use_cuda:
        mog2Subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    else:
        mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    reset_tracker()
    
def update_mog2_shadows(val):
    """
    Toggle MOG2 shadow detection.

    Args:
        val: Boolean flag (0=disabled, 1=enabled)
    """
    global mog2Subtractor, mog_history, var_threshold, detect_shadows
    detect_shadows = val
    if use_cuda:
        mog2Subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    else:
        mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
    reset_tracker()
    
def update_motion_threshold_min(val):
    """
    Update minimum blob area threshold.

    Args:
        val: Minimum contour area in pixels (filters noise)
    """
    global motion_threshold_min
    motion_threshold_min = val
    reset_tracker()
    
def update_motion_threshold_max(val):
    """
    Update maximum blob area threshold.

    Args:
        val: Maximum contour area in pixels (filters frame-wide motion)
    """
    global motion_threshold_min, motion_threshold_max
    motion_threshold_max = max(motion_threshold_min,val)
    reset_tracker()
    
def update_trigger_threshold(val):
    """
    Update trigger threshold for event recording.

    Args:
        val: Maximum number of detections to record (exceeding this skips frame)
    """
    global trigger_threshold
    trigger_threshold = val
    reset_tracker()

def update_kernel_size(val):
    """
    Update morphological operation kernel size.

    Args:
        val: Kernel dimension in pixels (must be odd; 0 disables morphology)
    """
    global kernel_size
    if val <= 2:
        kernel_size = 0
        cv2.setTrackbarPos('Noise Reduction', 'Motion Masks', kernel_size)
    elif (val%2) == 0:
        kernel_size = val + 1
        cv2.setTrackbarPos('Noise Reduction', 'Motion Masks', kernel_size)
    else:
        kernel_size = val
    reset_tracker()
        
def update_min_distance_ratio(val):
    """
    Update bounding box merge distance threshold.

    Args:
        val: Distance ratio (scaled by FLOAT_SCALE) as fraction of frame dimension
    """
    global min_distance_ratio, FLOAT_SCALE
    new_ratio = val / FLOAT_SCALE
    min_distance_ratio = new_ratio
    reset_tracker()
    
def update_tracking_buffer(val):
    """
    Update tracking persistence duration.

    Args:
        val: Maximum frames (scaled by FLOAT_SCALE) to persist tracks without detection
    """
    global tracking_buffer, FLOAT_SCALE
    tracking_buffer = val/FLOAT_SCALE
    reset_tracker()

def update_tracking_distance(val):
    """
    Update maximum association distance for tracking.

    Args:
        val: Maximum spatial distance (scaled by FLOAT_SCALE) for track-detection matching
    """
    global tracking_distance, FLOAT_SCALE
    tracking_distance = val / FLOAT_SCALE
    reset_tracker()

def update_area_weight(val):
    """
    Update area similarity weight in cost function.

    Args:
        val: Weighting factor (scaled by FLOAT_SCALE) for area difference term
    """
    global area_weight, FLOAT_SCALE
    area_weight = val / FLOAT_SCALE
    reset_tracker()
    

def update_playback_speed(val):
    """
    Update interactive tuner playback speed.

    Args:
        val: Speed multiplier (scaled by FLOAT_SCALE); 0 defaults to 0.1
    """
    global playback_speed, FLOAT_SCALE
    # Avoid division by zero: if val is zero, use a minimal speed value (e.g., 0.1)
    playback_speed = val / FLOAT_SCALE if val > 0 else 0.1

# bb_gen calculating the area for each contour and draws bounding boxes around objects meeting criteria for minimum and maximum detection areas.     
def bb_gen(mask, calibration = False):
    """
    Generate bounding boxes from foreground mask.
    
    Extracts distinct foreground blobs from the background-subtracted mask using
    connected component analysis. Returns bounding boxes for objects within the
    specified area range, filtering out sensor noise and frame-wide illumination
    changes. The centroid of each bounding box is subsequently used as the
    detection coordinate for Kalman filter tracking.
    
    Morphological Processing:
        Applies dilation with elliptical structuring element (3×3 pixels, 2 iterations)
        to connect nearby foreground components and fill small holes within objects.
        This operation enhances blob cohesion whilst preserving object boundaries.
    
    Area Filtering:
        Blobs are filtered by motion_threshold_min and motion_threshold_max to remove:
        - Small regions: Residual sensor noise from the MOG2 background subtraction
        - Large regions: Background motion or frame-wide illumination changes
    
    These thresholds were optimised using annotated bat events provided by ecological
    consultants to achieve sufficient sensitivity whilst minimising false positives.
    
    Args:
        mask: Binary foreground mask from MOG2 background subtraction (uint8 array)
        calibration: If True, prints area information for all contours regardless
                     of threshold (used during interactive parameter tuning)
    
    Returns:
        list: Bounding boxes as (x, y, width, height) tuples in pixel coordinates
    
    Notes:
        The function utilises morphological dilation to enhance blob detection,
        particularly beneficial for small, fast-moving objects in low-contrast
        infrared footage where motion blur may fragment foreground regions.
    """
    global motion_threshold_min, motion_threshold_max
    bounding_boxes = []
    # Morphological dilation with elliptical kernel connects nearby foreground regions
    # Two iterations bridge small gaps caused by motion blur in infrared footage
    dilated = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if motion_threshold_min <= area <= motion_threshold_max:
            if calibration:
                print(f"Motion Detected | Area: {area}")
            (x,y,w,h) = cv2.boundingRect(contour)
            bounding_boxes.append((x,y,w,h))
    return bounding_boxes        

# motion_comparison generates combined frame with the motion mask above the input frame. Bounding boxes are dynamically drawn over each image using bb_gen if motion is detected. 
def motion_comparison(frame, current_frame, total_frames, mask=None, merge_bb=False, trigger_threshold=trigger_threshold):
    """
    Generate side-by-side comparison of input frame and motion mask.

    Creates visualisation showing MOG2 foreground mask above grayscale input,
    with bounding boxes overlaid on detected objects. Used during interactive
    parameter tuning to assess detection quality in real-time.

    Args:
        frame: Input video frame (BGR)
        current_frame: Current frame number for display
        total_frames: Total frame count (unused but kept for signature consistency)
        mask: Optional ROI mask to constrain processing region
        merge_bb: Enable IoU-based bounding box merging
        trigger_threshold: Maximum detections before marking frame as outlier

    Returns:
        np.ndarray: Combined visualisation frame (vertically stacked, resized)
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        frame_to_process = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        frame_to_process = frame
    
    if use_cuda:
        if 'g' not in locals():
            g = cv2.cuda_GpuMat()
            fgmask = cv2.cuda_GpuMat()
            stream = cv2.cuda_Stream()

        # async upload on the same stream
        g.upload(frame_to_process, stream)

        # ensure fgmask has correct size/type (8UC1)
        if fgmask.empty() or fgmask.size() != g.size():
            fgmask.create(g.size(), cv2.CV_8UC1)

        # CUDA MOG2: image, learningRate, stream, [fgmask]
        mog2Subtractor.apply(g, float(-1.0), stream, fgmask)

        # wait before downloading (sync point)
        stream.waitForCompletion()
        mog2_mask = fgmask.download()
    else:
        mog2_mask = mog2Subtractor.apply(frame_to_process, learningRate=-1.0)
        
    if kernel_size:
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_OPEN, kernel)
        mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_CLOSE, kernel)
    
    bounding_boxes = bb_gen(mog2_mask, calibration = True)
    if merge_bb:
        frame_height, frame_width=frame_to_process.shape[:2]
        bounding_boxes=merge_bounding_boxes(bounding_boxes, min_distance_ratio = min_distance_ratio, frame_width= frame_width, frame_height = frame_height)
        
    if bounding_boxes:
        print(f"Number of Objects: {len(bounding_boxes)}")
        for (x,y,w,h) in bounding_boxes:
            cv2.rectangle(mog2_mask, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(gray_frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(gray_frame,(10,10), (270,40), (255,255,255), -1)
        if len(bounding_boxes) <= trigger_threshold:
            cv2.putText(gray_frame, "Motion detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.putText(gray_frame, "M_THRESHOLD", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        
    combined_frame = cv2.resize(cv2.vconcat([mog2_mask, gray_frame]),(0, 0),fx=0.7, fy=0.5)
    cv2.rectangle(combined_frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(combined_frame, f"{current_frame}", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  
    return combined_frame

tracker_colors = {}

def motion_comparison_tracking(frame, current_frame, total_frames, tracker=None, mask=None, merge_bb=False, trigger_threshold=trigger_threshold):
    """
    Generate tracking visualisation with coloured bounding boxes per track ID.

    Similar to motion_comparison but includes Kalman tracking, drawing bounding
    boxes in unique colours per track ID to visualise trajectory consistency.
    Displays track IDs and updates tracker state for real-time assessment.

    Args:
        frame: Input video frame (BGR)
        current_frame: Current frame number for display
        total_frames: Total frame count (unused but kept for signature consistency)
        tracker: MultiObjectTracker instance for state management
        mask: Optional ROI mask to constrain processing region
        merge_bb: Enable IoU-based bounding box merging
        trigger_threshold: Maximum detections before marking frame as outlier

    Returns:
        np.ndarray: Combined visualisation frame with coloured tracking overlays
    """
    # Get a grayscale version (for the subtractor) but convert later for colored drawing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        frame_to_process = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        frame_to_process = frame
    
    if use_cuda:
        # Create once (outside your frame loop if possible)
        if 'g' not in locals():
            g = cv2.cuda_GpuMat()
            #fgmask = cv2.cuda_GpuMat()
            stream = cv2.cuda_Stream()

        # Async upload on the SAME stream
        g.upload(frame_to_process, stream)

        # Ensure output buffer exists and matches size/type (MOG2 mask is 8UC1)
        #if fgmask.empty() or fgmask.size() != g.size():
        #    fgmask.create(g.size(), cv2.CV_8UC1)

        # Correct arg order: (image, learningRate, stream[, fgmask])
        cuda_mask = mog2Subtractor.apply(g, float(-1.0), stream)

        # Sync before download (only needed if you use the result immediately)
        stream.waitForCompletion()
        mog2_mask = cuda_mask.download()
    else:
        mog2_mask = mog2Subtractor.apply(frame_to_process, learningRate=-1.0)
        
    if kernel_size:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_OPEN, kernel)
        mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_CLOSE, kernel)

    bounding_boxes = bb_gen(mog2_mask, calibration=True)
    if merge_bb:
        frame_height, frame_width = frame_to_process.shape[:2]
        bounding_boxes = merge_bounding_boxes(bounding_boxes, min_distance_ratio=min_distance_ratio, frame_width=frame_width, frame_height=frame_height)

    # Convert the grayscale images to BGR so that colored drawings are visible.
    mog2_mask_color = cv2.cvtColor(mog2_mask, cv2.COLOR_GRAY2BGR)
    gray_color = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Update tracker and draw bounding boxes with tracking ids in unique colors
    if bounding_boxes:
        print(f"Number of Objects: {len(bounding_boxes)}")
        frame_height, frame_width = frame.shape[:2]
        detections = []
        for (x, y, w, h) in bounding_boxes:
            center_x = (x + w // 2) / frame_width
            center_y = (y + h // 2) / frame_height
            area = (w * h) / (frame_width * frame_height)
            detections.append((center_x, center_y, area))
        
        # Update tracker with the current detections and get mapping: detection index -> track id
        
        if len(bounding_boxes) <= trigger_threshold:
            detection_map, det_kin = tracker.update(detections)

            for idx, (x, y, w, h) in enumerate(bounding_boxes):
                if idx in detection_map:
                    track_id = detection_map[idx]
                    # Assign a unique color for this track id if not already set.
                    if track_id not in tracker_colors:
                        tracker_colors[track_id] = tuple(np.random.randint(0, 256, 3).tolist())
                    color = tracker_colors[track_id]
                else:
                    color = (255, 0, 0)  # Fallback color if no id is available
                
                # Draw the bounding box on both images using the unique color
                cv2.rectangle(mog2_mask_color, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(gray_color, (x, y), (x+w, y+h), color, 2)
                cv2.putText(mog2_mask_color, f"ID: {detection_map.get(idx, 'N/A')}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.rectangle(gray_color, (10, 10), (270, 40), (255, 255, 255), -1)
            
            cv2.putText(gray_color, "Motion detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            
        else:
            cv2.putText(gray_color, "M_THRESHOLD", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            tracker.update([])
    else:
        # No bounding boxes; update the tracker with an empty list.
        tracker.update([])

    # Combine the two frames vertically.
    combined_frame = cv2.resize(cv2.vconcat([mog2_mask_color, gray_color]), (0, 0), fx=0.7, fy=0.5)
    cv2.rectangle(combined_frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(combined_frame, f"{current_frame}", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return combined_frame


def motion_tuner(capture, fps, total_frames, mask, merge_bb=False):
    """
    Interactive parameter tuning interface for motion detection.

    Displays real-time visualisation with interactive trackbars for adjusting
    MOG2 parameters, morphological operations, and tracking thresholds. Keyboard
    controls enable frame navigation and playback control for efficient parameter
    optimisation across diverse video content.

    Keyboard Controls:
        Space: Pause/resume playback
        z/x: Seek backward/forward 1 second
        a/s: Seek backward/forward 1 minute (resets background model)
        q: Quit tuning interface

    Args:
        capture: OpenCV VideoCapture object
        fps: Video frame rate
        total_frames: Total frame count
        mask: Optional ROI mask to constrain processing region
        merge_bb: Enable IoU-based bounding box merging

    Notes:
        Parameter values are printed to console upon exit for recording
        optimal settings. Background model is automatically reset when
        seeking to non-contiguous video sections.
    """
    if not capture.isOpened():
        print("Error: Video capture is not open.")
        return

    paused = False
    current_frame = 0

    while True:
        if not paused:
            ret, frame = capture.read()
            if not ret:
                print("End of video, restarting...")
                reset_subtractor()
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = 0
                continue

            # Pass the tracker instance to the motion_comparison_tracking function so it can update detections and draw IDs.
            cv2.imshow('Motion Mask', motion_comparison_tracking(frame, current_frame, total_frames, tracker, mask, merge_bb, trigger_threshold))
            current_frame += 1

        # Use the playback_speed to control the delay in waitKey
        wait_time = int(25 / playback_speed) if playback_speed > 0 else 25
        key = cv2.waitKey(wait_time)

        if key & 0xFF == ord('q'):
            break
        if paused:
            if key & 0xFF == ord(' '):
                paused = False
        else:
            if key & 0xFF == ord(' '):
                paused = True
            elif key & 0xFF == ord('z'):
                current_frame = max(0, current_frame - fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key & 0xFF == ord('x'):
                current_frame = min(total_frames, current_frame + fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key & 0xFF == ord('a'):
                current_frame = max(0, current_frame - 60 * fps)
                reset_subtractor()
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key & 0xFF == ord('s'):
                current_frame = min(total_frames, current_frame + 60 * fps)
                reset_subtractor()
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(
        f"MogHistory: {mog_history} | MogThreshold: {var_threshold} | Detect Shadows: {detect_shadows} | "
        f"Motion Threshold Min: {motion_threshold_min} | Motion Threshold Max: {motion_threshold_max} | Kernel Size: {kernel_size} | "
        f"Min Distance Ratio: {min_distance_ratio} | Trigger Threshold: {trigger_threshold} | "
        f"Tracking Buffer: {tracking_buffer} | Tracking Distance: {tracking_distance} | "
        f"Area Weight: {area_weight} | Min Event Length: {min_event_length}"
    )
    
# =============================================================================
# Region of Interest (ROI) Selection
# =============================================================================
"""
ROI Selection for Background Motion Exclusion
----------------------------------------------
Whilst the MOG2 algorithm is able to dynamically learn repetitive background motion,
through experimentation it was found to be triggered by frequent but non-continuous
repetitive motion, for example vegetation moving in gusts of wind.

To account for this, the pipeline includes an optional Region of Interest (ROI)
selection tool. The tool generates a heat map of background motion from samples
within the video and allows the user to define ROI polygons within the image that
exclude hot spots of background motion.
"""

def heatmap_generator(
    capture, fps, total_frames, n, kernel_size,
    seconds=2, skip_seconds=0,
    use_cuda=False, mog_history=mog_history, var_threshold=var_threshold, detect_shadows=detect_shadows,
    learning_rate=-1.0, reset_between_samples=True
):
    """
    Generate motion heatmap from sampled video windows.

    Processes evenly-spaced temporal samples across video duration, accumulating
    foreground detections to visualise spatial distribution of motion. Applies
    logarithmic compression, percentile stretching, and CLAHE enhancement for
    optimal visualisation of both frequent and infrequent motion patterns.

    The heatmap identifies regions of persistent motion (e.g., vegetation) for
    ROI mask creation, enabling exclusion of repetitive non-target motion from
    subsequent detection processing.

    Processing Strategy:
        1. Sample n evenly-spaced temporal windows of 'seconds' duration
        2. Apply MOG2 background subtraction with optional morphological filtering
        3. Accumulate binary foreground masks across all samples
        4. Apply visualisation enhancements (log compression, stretch, CLAHE)
        5. Overlay colourised heatmap on base frame for spatial context

    Args:
        capture: OpenCV VideoCapture object
        fps: Video frame rate
        total_frames: Total frame count
        n: Number of temporal sampling points
        kernel_size: Morphological operation kernel size (0=disabled)
        seconds: Duration of each sampling window (default: 2)
        skip_seconds: Initial video offset before sampling (default: 0)
        use_cuda: Enable CUDA acceleration if available (default: False)
        mog_history: MOG2 background history length
        var_threshold: MOG2 variance threshold
        detect_shadows: Enable MOG2 shadow detection
        learning_rate: MOG2 learning rate (-1.0=automatic)
        reset_between_samples: Reinitialise MOG2 between samples (default: True)

    Returns:
        np.ndarray: Colourised heatmap overlay on base frame (None if insufficient frames)

    Notes:
        Visualisation enhancements (log compression, CLAHE) are applied only to
        the returned image; underlying motion accumulation remains unmodified to
        preserve detection fidelity. Turbo colourmap is preferred for perceptual
        uniformity; falls back to HOT if unavailable.
    """
    if not capture or not capture.isOpened():
        print("Error: Video capture is not open.")
        return None

    # --- pick backend deterministically ---
    if use_cuda and (not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() < 1):
        print("CUDA requested but not available; falling back to CPU.")
        use_cuda = False

    # (Re)create subtractor inside to avoid passing a CPU one by accident.
    if use_cuda:
        mog2 = cv2.cuda.createBackgroundSubtractorMOG2(
            history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        g      = cv2.cuda_GpuMat()
        fgmask = cv2.cuda_GpuMat()
        stream = cv2.cuda_Stream()
    else:
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows
        )

    # --- plan sampling ---
    skip_frames = int(max(0, fps * skip_seconds))
    if skip_frames >= total_frames:
        print("Warning: skip_seconds >= video length; starting from 0.")
        skip_frames = 0
    frame_indices = np.linspace(skip_frames, total_frames - 1, num=max(1, int(n)), dtype=int)

    base_image = None
    heat_layer = None

    # reuse morph kernel
    kernel = None
    if kernel_size and kernel_size > 0:
        kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)

    # --- iterate sampled windows ---
    for idx in tqdm(frame_indices, desc="Processing sample points", total=len(frame_indices)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = capture.read()
        if not ok:
            continue

        if base_image is None:
            base_image = frame.copy()
            h, w = frame.shape[:2]
            heat_layer = np.zeros((h, w), np.float32)
            if use_cuda:
                fgmask.create(h, w, cv2.CV_8UC1)

        steps = int(max(1, round(fps * seconds)))
        for _ in range(steps):
            ok, frame = capture.read()
            if not ok:
                break

            if use_cuda:
                if 'g' not in locals():
                    g = cv2.cuda_GpuMat()
                    stream = cv2.cuda_Stream()
                if fgmask.empty() or fgmask.size() != g.size():
                    fgmask.create(g.size(), cv2.CV_8UC1)

                g.upload(frame, stream)
                cuda_mask = mog2.apply(g, float(learning_rate), stream)
                stream.waitForCompletion()
                mog2_mask = cuda_mask.download()
            else:
                mog2_mask = mog2.apply(frame, learningRate=float(learning_rate))

            if kernel is not None:
                mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_OPEN, kernel)

            heat_layer += (mog2_mask > 0).astype(np.float32)

        if reset_between_samples:
            if use_cuda:
                mog2 = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=mog2.getHistory(),
                    varThreshold=mog2.getVarThreshold(),
                    detectShadows=mog2.getDetectShadows()
                )
            else:
                mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=mog2.getHistory(),
                    varThreshold=mog2.getVarThreshold(),
                    detectShadows=mog2.getDetectShadows()
                )

    if heat_layer is None:
        return None
    
    # Visualisation Enhancements # 
    # Logarithmic compression lifts faint motion (infrequent events) whilst
    # preventing saturation of hotspots (frequent motion like vegetation)
    heat_vis = np.log1p(heat_layer)

    # Percentile stretching maps 2nd-98th percentile to full dynamic range
    # Ignores extreme outliers for better contrast across typical motion intensities
    lo, hi = np.percentile(heat_vis, (2, 98))
    if hi <= lo:  # degenerate case guard
        lo, hi = float(heat_vis.min()), float(heat_vis.max())
    heat_vis = np.clip(heat_vis, lo, hi)
    heat_norm = cv2.normalize(heat_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE (Contrast Limited Adaptive Histogram Equalisation) enhances local contrast
    # Separates similar intensities in different spatial regions (e.g., multiple vegetation patches)
    heat_norm = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(heat_norm)

    # 4) Motion mask (visual-only) to colorise just the moving areas
    thr = 40  # ↑ for fewer/stronger blobs, ↓ for more/fainter
    motion_mask = (heat_norm > thr).astype(np.uint8)
    if kernel is not None:
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
    motion_mask = motion_mask.astype(bool)

    # 5) High-contrast colormap
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_HOT)
    col_heat_layer = cv2.applyColorMap(heat_norm, cmap)

    # 6) Stronger, masked blend over slightly dimmed background
    bg = cv2.convertScaleAbs(base_image, alpha=0.72, beta=0)   # dim background a bit
    overlay = bg.copy()
    cv2.addWeighted(overlay, 0.0, col_heat_layer, 0.95, 0, dst=overlay)  # bold color

    heat_map = bg.copy()
    heat_map[motion_mask] = overlay[motion_mask]

    return heat_map

# draw_mask allows users to manually draw Region of interest polygons over a heat map. This allows repetative motion to be excluded from the motion detection process. 
points = []          # collected points (clicks and rectangle corners)
_dragging = False    # are we currently dragging a rect?
_start = None        # rect start (x, y)
_current = None    

def draw_mask(event, x, y, flags, param):
    """
    OpenCV mouse callback for interactive ROI polygon drawing.

    Supports two interaction modes:
    - Click: Add individual polygon vertices
    - Drag: Add rectangle (4 corner points automatically)
    - Right-click: Remove last point

    Args:
        event: OpenCV mouse event type
        x: Mouse x-coordinate
        y: Mouse y-coordinate
        flags: OpenCV mouse event flags
        param: Optional callback parameter (unused)

    Notes:
        Modifies global 'points' list. Small drags (<3 pixels) are treated
        as single clicks to prevent accidental rectangles from jittery input.
    """
    global points, _dragging, _start, _current

    if event == cv2.EVENT_LBUTTONDOWN:
        _dragging = True
        _start = (x, y)
        _current = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and _dragging and (flags & cv2.EVENT_FLAG_LBUTTON):
        _current = (x, y)  # for preview only

    elif event == cv2.EVENT_LBUTTONUP and _dragging:
        _dragging = False
        x1, y1 = _start
        x2, y2 = x, y
        _current = None

        # Normalize rectangle
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

        # If the drag was tiny, treat it as a single click
        if abs(x_max - x_min) < 3 and abs(y_max - y_min) < 3:
            points.append((x, y))
        else:
            # Add only the four corners (TL, TR, BR, BL)
            corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            points.extend(corners)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
        else:
            print("No points to clear...")

def find_polygons(polys):
    """
    Compute bounding rectangle encompassing all ROI polygons.

    Determines the minimal rectangular region containing all polygon vertices,
    enabling video cropping to the region of interest. This reduces computational
    overhead by processing only relevant frame regions.

    Args:
        polys: List of polygon arrays (each polygon as Nx2 point array)

    Returns:
        tuple: (x, y, w, h) bounding rectangle in pixel coordinates, or None if empty

    Notes:
        Processing separate polygons in parallel would improve efficiency but
        complicates extraction of final crops relative to original video coordinates.
        Current implementation uses a single bounding rectangle for simplicity.
    """
    if polys:
        all_points = np.concatenate(polys)
        x,y,w,h = cv2.boundingRect(all_points)
        return x,y,w,h
    else:
        print("No ROI selected")

def xywh_to_yolo(xywh, image_width, image_height):
    """
    Convert bounding box from pixel coordinates to normalised YOLO format.

    Transforms (x, y, width, height) pixel coordinates to normalised centre-based
    representation (center_x, center_y, width, height) in range [0, 1]. This
    format is compatible with YOLO annotation tools and deep learning frameworks.

    Args:
        xywh: Tuple of (x, y, width, height) in pixel coordinates
        image_width: Frame width in pixels
        image_height: Frame height in pixels

    Returns:
        tuple: (center_x, center_y, width, height) normalised to [0, 1], rounded to 6 decimals
    """
    x, y, w, h = xywh
    center_x = round((x + w / 2) / image_width, 6)
    center_y = round((y + h / 2) / image_height, 6)
    width = round(w / image_width, 6)
    height = round(h / image_height, 6)
    return (center_x, center_y, width, height)

def _jsonable(x):
    """
    Recursively convert NumPy types to JSON-serialisable Python types.

    Args:
        x: Object to convert (ndarray, numeric, list, dict, or primitive)

    Returns:
        JSON-serialisable equivalent (list, float, dict, or primitive)
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x

def save_jsons(mask_array, polys_array, motion_params_array, out_dir=video_dir):
    """
    Persist ROI masks, polygons, and motion parameters to JSON files.

    Saves interactive tuning results for reproducibility, enabling consistent
    processing across multiple runs without manual re-configuration.

    Args:
        mask_array: Binary ROI mask array
        polys_array: List of polygon vertex arrays
        motion_params_array: Dictionary of motion detection parameters
        out_dir: Output directory path (default: video_dir)

    Output Files:
        mask_array.json: Serialised binary mask
        polys_array.json: Serialised polygon coordinates
        motion_params_array.json: Serialised parameter dictionary
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mask_array.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(mask_array), f)
    with open(os.path.join(out_dir, "polys_array.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(polys_array), f)
    with open(os.path.join(out_dir, "motion_params_array.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(motion_params_array), f)
        
# =============================================================================
# Motion Extractor
# =============================================================================
"""
Frame-by-Frame Motion Detection and Extraction
-----------------------------------------------
Once detection parameters have been defined, all frames within the input video are
processed in sequence. Where objects of interest are detected, bounding boxes are
defined around each object, which are then reformatted to user-defined minimum
dimensions.

By standardising minimum bounding box dimensions it is possible to retain some
contextual information around the object of interest which facilitates the annotation
process. Minimum bounding box dimensions of 224 × 224 pixels were used in this
instance based on the resolution of the input videos.

The `motion_extractor_kalman` routine returns `(motion_logs, decode_error)`, where:
    - `motion_logs`: List of per-frame detections (filename, timestamp, bounding box
      tuples, speed, and speed_area_normalised)
    - `decode_error`: Flags a frame index if decoding halted early

Side effects include writing cropped frames (padded and unpadded), optional
masks/foreground cut-outs, YOLO-formatted bounding boxes, and kinematic descriptors
used downstream for GIF creation and metrics.
"""


def motion_extractor_kalman(
        video_id,
        capture,
        fps,
        output_dir,
        total_frames,
        merge_bb=True,
        min_distance_ratio=0.01,
        trigger_threshold=5,
        tracking_buffer=30,
        tracking_distance=200,
        area_weight=0.1,
        rapid_scan_seconds: float = 0.0,
        skip_factor: int = 1,
        roi_mask=None,
        roi_crop=None
    ):
    """
    Main motion detection and tracking pipeline.
    
    Processes video frames sequentially to detect and track moving objects using MOG2
    background subtraction, morphological filtering, blob detection, and Kalman filter
    tracking with Hungarian algorithm matching. Outputs cropped frames containing
    detected objects with standardised minimum dimensions, facilitating efficient
    annotation or downstream processing via deep learning models.
    
    Processing Pipeline:
        1. Frame Acquisition: Sequential reading from video capture object
        2. Background Subtraction: MOG2 generates binary foreground mask
        3. Morphological Filtering: Opening and closing operations remove noise
        4. Blob Detection: Connected component analysis via bb_gen function
        5. Bounding Box Merging: IoU-based consolidation prevents duplicate tracks
        6. Centroid Extraction: Bounding box centres serve as detection coordinates
        7. Kalman Prediction: Each active track predicts position for current frame
        8. Hungarian Matching: Optimal assignment between predictions and detections
        9. State Update: Matched tracks update; unmatched detections initialise new tracks
        10. Track Management: Terminate tracks exceeding unmatched threshold
        11. Output Generation: Crop and save frames containing tracked objects
    
    The pipeline maintains a standardised minimum bounding box dimension (default: 224×224
    pixels) to retain contextual information around detected objects, facilitating manual
    annotation and improving deep learning model performance.
    
    Args:
        video_id: Unique identifier for output file naming
        capture: OpenCV VideoCapture object
        fps: Video frame rate (frames per second)
        output_dir: Directory for saving cropped frames and metadata
        total_frames: Total frame count in video
        merge_bb: Enable IoU-based bounding box merging (default: True)
        min_distance_ratio: Merging distance threshold as fraction of frame diagonal
        trigger_threshold: Minimum detections to trigger event recording
        tracking_buffer: Maximum frames to persist tracks without detections (seconds)
        tracking_distance: Maximum spatial distance for track-detection association (pixels)
        area_weight: Weighting factor for area similarity in cost function
        rapid_scan_seconds: Duration for fast initial scanning (0 = disabled)
        skip_factor: Frame subsampling rate for rapid scanning
        roi_mask: Binary mask defining region of interest (None = full frame)
        roi_crop: Crop bounds for region of interest (None = full frame)
        save_detections: Enable saving of cropped detection frames
    
    Returns:
        tuple: (intervals, metadata)
            intervals: List of (start_frame, end_frame) tuples for detected events
            metadata: Pandas DataFrame with detection coordinates, track IDs, and kinematics
    
    Output Format:
        Cropped frames saved as: {output_dir}/{video_id}_frame_{frame_num}_track_{track_id}.jpg
        Metadata CSV columns: frame, x, y, width, height, track_id, vx, vy, speed
        Bounding boxes stored in normalised YOLO format for compatibility with annotation tools
    
    Notes:
        All parameters were optimised using annotated bat events to achieve sufficient
        sensitivity whilst minimising false positives. The pipeline utilises CUDA
        acceleration when available to enhance processing throughput.
    """

    global motion_threshold_min
    global save_masks
    global save_fg

    # ----------------- helpers -----------------
    def is_valid_frame(frm):
        """
        Validate frame integrity for processing.

        Args:
            frm: Frame array to validate

        Returns:
            bool: True if frame is valid for processing
        """
        if frm is None:
            return False
        if not isinstance(frm, np.ndarray):
            return False
        if frm.ndim not in (2, 3):
            return False
        if 0 in frm.shape:
            return False
        if frm.dtype != np.uint8:
            try:
                _ = frm.astype(np.uint8)
            except Exception:
                return False
        return True

    # list to track frames with decode/validation errors
    decode_error_frames = []

    # safe wrappers that also know if it was an error vs EOF
    def safe_grab(cap, vid, frame_idx):
        """
        Safely attempt frame grab with error handling.

        Wraps cv2.VideoCapture.grab() with exception handling to distinguish
        between decode errors and end-of-stream conditions.

        Args:
            cap: OpenCV VideoCapture object
            vid: Video identifier for logging
            frame_idx: Current frame index for error messages

        Returns:
            tuple: (grabbed: bool, is_error: bool)
                grabbed: True if frame was grabbed successfully
                is_error: True if exception occurred (decode/IO issue)
        """
        try:
            grabbed = cap.grab()
            return grabbed, False
        except cv2.error as e:
            print(f"[WARN] cv2.error in grab() at frame {frame_idx} in {vid}: {e}")
            return False, True
        except Exception as e:
            print(f"[WARN] Unexpected error in grab() at frame {frame_idx} in {vid}: {e}")
            return False, True

    def safe_retrieve(cap, vid, frame_idx):
        """
        Safely retrieve frame buffer with error handling.

        Wraps cv2.VideoCapture.retrieve() with exception handling for robust
        processing of videos with corrupt or malformed frames.

        Args:
            cap: OpenCV VideoCapture object
            vid: Video identifier for logging
            frame_idx: Current frame index for error messages

        Returns:
            tuple: (ret: bool, frame: np.ndarray, is_error: bool)
                ret: True if frame retrieved successfully
                frame: Decoded frame array (None if failed)
                is_error: True if exception occurred
        """
        try:
            ret, frm = cap.retrieve()
            return ret, frm, False
        except cv2.error as e:
            print(f"[WARN] cv2.error in retrieve() at frame {frame_idx} in {vid}: {e}")
            return False, None, True
        except Exception as e:
            print(f"[WARN] Unexpected error in retrieve() at frame {frame_idx} in {vid}: {e}")
            return False, None, True

    # ----------------- init state -----------------
    current_frame   = 0
    motion_logs     = []
    skipped_frames  = []
    roi_x, roi_y, roi_w, roi_h = (0, 0, 0, 0)
    close_points    = 0

    frame_height = None
    frame_width  = None

    # --- Rapid Scan ---
    use_rapid = (rapid_scan_seconds > 0) and (skip_factor > 1)
    if use_rapid:
        rapid_frame_threshold = int(rapid_scan_seconds * fps)
    else:
        rapid_frame_threshold = total_frames + 1

    # --- ROI cropping polygon ---
    if roi_crop is not None:
        roi_x, roi_y, roi_w, roi_h = find_polygons(roi_crop)

    # --- Tracker ---
    tracker = MultiObjectTracker(
        max_unmatched_frames=int(tracking_buffer * fps),
        max_distance=tracking_distance,
        area_weight=area_weight
    )
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if use_cuda:
        g = cv2.cuda_GpuMat()
        stream = cv2.cuda_Stream()

    last_detection_frame = 0
    rapid_mode           = False

    checker = mask_checkers.get(video_id, lambda frame: False)


    with tqdm(total=total_frames) as pbar:

        # Helper to advance tracker state when skipping frame (for any reason)
        def _advance_without_frame():
            """
            Advance tracker time step without processing frame.

            Updates tracker state with predict-only cycle and increments progress bar.
            Used when skipping frames due to masking, rapid scan, or decode errors.
            """
            frames_advanced = 1
            tracker.set_dt_all(frames_advanced / fps)
            tracker.update([])   # predict-only / unmatched update
            pbar.update(1)
            pbar.set_postfix({
                "Events": tracker.tracker_log,
                "RapidScan": rapid_mode,
                "Skipped Frames": len(skipped_frames),
                "LastDetFrame": last_detection_frame,
                "Close_events": close_points,
                "Decode_Errors": len(decode_error_frames)
            })

        # NEW: make sure we don't loop forever; respect total_frames
        while current_frame < total_frames:
            current_frame += 1

            # ------------- SAFE GRAB -------------
            grabbed, grab_error = safe_grab(capture, video_id, current_frame)
            if not grabbed:
                if grab_error:
                    # Treat as a decode/IO error for this frame
                    decode_error_frames.append(current_frame)
                    _advance_without_frame()
                    continue
                else:
                    # Likely end-of-stream
                    break

            # ------------- RAPID SCAN LOGIC -------------
            # Enter rapid scan mode after prolonged inactivity (no detections for rapid_frame_threshold frames)
            # Subsamples frames (processes every skip_factor-th frame) to accelerate scanning of empty video sections
            if use_rapid and (current_frame - last_detection_frame) >= rapid_frame_threshold:
                rapid_mode = True

            must_decode = (current_frame == 1) or (current_frame == int(total_frames * 0.75))

            is_skip_frame = (rapid_mode and (current_frame % skip_factor != 0)) or checker(current_frame)
            need_decode   = must_decode or (not is_skip_frame)

            # ------------- SKIP FRAME (no decode) -------------
            if not need_decode:
                _advance_without_frame()
                continue

            # ------------- SAFE RETRIEVE & VALIDATE -------------
            ret, frame, retrieve_error = safe_retrieve(capture, video_id, current_frame)

            if (not ret) or (not is_valid_frame(frame)):
                # Log this frame as problematic
                decode_error_frames.append(current_frame)

                print(f"[WARN] Bad/invalid frame at {current_frame} in {video_id}. "
                      f"ret={ret}, retrieve_error={retrieve_error}")
                _advance_without_frame()
                continue

            # At this point we have a valid decoded frame

            if frame_height is None or frame_width is None:
                frame_height, frame_width = frame.shape[:2]

            # ------------- SAVE KEYFRAMES -------------
            if current_frame == 1:
                file_path = os.path.join(output_dir, f"{video_id}_start.png")
                cv2.imwrite(file_path, frame)

            if current_frame == int(total_frames * 0.75):
                if roi_mask is not None:
                    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                    file_path = os.path.join(output_dir, f"{video_id}_mask.png")
                    cv2.imwrite(file_path, masked_frame)

                file_path = os.path.join(output_dir, f"{video_id}_dark_point.png")
                cv2.imwrite(file_path, frame)

            # ------------- MAIN PROCESSING -------------
            frames_advanced = 1
            tracker.set_dt_all(frames_advanced / fps)

            if roi_mask is not None:
                output_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] if roi_crop else frame
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                frame_to_process = masked_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] if roi_crop else masked_frame
            else:
                frame_to_process = frame
                output_frame = frame

            # 2) Apply MOG2
            if use_cuda:
                g.upload(frame_to_process, stream)
                cuda_mask = mog2Subtractor.apply(g, float(-1.0), stream)
                mog2_mask = cuda_mask.download()
            else:
                mog2_mask = mog2Subtractor.apply(frame_to_process, learningRate=-1.0)

            # 3) Motion / bounding boxes
            if cv2.countNonZero(mog2_mask) > motion_threshold_min:
                if kernel_size:
                    mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_OPEN, kernel)
                    mog2_mask = cv2.morphologyEx(mog2_mask, cv2.MORPH_CLOSE, kernel)

                bounding_boxes = bb_gen(mog2_mask)

                initial_bb_count = len(bounding_boxes)
                if merge_bb:
                    frame__to_process_height, frame_to_process_width = frame_to_process.shape[:2]
                    bounding_boxes = merge_bounding_boxes(
                        bounding_boxes,
                        min_distance_ratio=min_distance_ratio,
                        frame_width=frame_to_process_width,
                        frame_height=frame__to_process_height
                    )
                    close_points += initial_bb_count - len(bounding_boxes)
            else:
                bounding_boxes = []

            if bounding_boxes:
                # Reset rapid scan
                last_detection_frame = current_frame
                rapid_mode = False

                # Too many detections -> skip saving but still keep tracking consistent
                if len(bounding_boxes) > trigger_threshold:
                    file_path = os.path.join(output_dir, "skipped_frames", f"{current_frame}.png")
                    cv2.imwrite(file_path, frame)
                    skipped_frames.append((f"{current_frame}", current_frame // fps))
                    tracker.update([])
                else:
                    # Convert bounding boxes to normalised coordinates for Kalman tracking
                    # Normalisation provides scale invariance and simplifies velocity interpretation
                    detections = []
                    for (x, y, w, h) in bounding_boxes:
                        center_x = (x + w // 2) / frame_width
                        center_y = (y + h // 2) / frame_height
                        area     = (w * h) / (frame_width * frame_height)
                        detections.append((center_x, center_y, area))

                    # 8) Update Kalman multi-object tracker
                    detection_map, det_kin = tracker.update(detections)

                    # 9) Crops + logging
                    event_time_seconds = current_frame // fps
                    for det_idx, (x, y, w, h) in enumerate(bounding_boxes):
                        i = detection_map[det_idx]

                        # Adjust for ROI offsets
                        if roi_crop is not None:
                            x_out = x + roi_x
                            y_out = y + roi_y
                        else:
                            x_out, y_out = x, y

                        # Enforce minimum 224×224 crop size to retain contextual information
                        # Standard size facilitates annotation and improves deep learning model performance
                        crop_w = max(w, 224)
                        crop_h = max(h, 224)
                        center_x_px = x + w // 2
                        center_y_px = y + h // 2

                        crop_x = max(0, min(center_x_px - crop_w // 2, frame_to_process.shape[1] - crop_w))
                        crop_y = max(0, min(center_y_px - crop_h // 2, frame_to_process.shape[0] - crop_h))

                        cropped_image = output_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        out_path = os.path.join(output_dir, "motion_frames", f"{current_frame}_{i}_{video_id}.png")
                        cv2.imwrite(out_path, cropped_image)

                        if save_masks:
                            cropped_mask = mog2_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                            mask_path = os.path.join(output_dir, "masks", f"{current_frame}_{i}_{video_id}.png")
                            cv2.imwrite(mask_path, cropped_mask)

                        if save_fg:
                            fg_img = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
                            fg_path = os.path.join(output_dir, "fg", f"{current_frame}_{i}_{video_id}.png")
                            cv2.imwrite(fg_path, fg_img)

                        # Unpadded crops
                        unpadded_image = output_frame[y:y+h, x:x+w]
                        unpadded_path = os.path.join(
                            output_dir, "motion_frames_unpadded", f"{current_frame}_{i}_{video_id}.png"
                        )
                        cv2.imwrite(unpadded_path, unpadded_image)

                        if save_masks:
                            unpadded_mask = mog2_mask[y:y+h, x:x+w]
                            umask_path = os.path.join(
                                output_dir, "masks_unpadded", f"{current_frame}_{i}_{video_id}.png"
                            )
                            cv2.imwrite(umask_path, unpadded_mask)

                        if save_fg:
                            unpadded_fg = cv2.bitwise_and(unpadded_image, unpadded_image, mask=unpadded_mask)
                            ufg_path = os.path.join(
                                output_dir, "fg_unpadded", f"{current_frame}_{i}_{video_id}.png"
                            )
                            cv2.imwrite(ufg_path, unpadded_fg)

                        # Log YOLO-style bbox + kinematics
                        yolo_bb = xywh_to_yolo(
                            (x_out, y_out, w, h),
                            frame.shape[1],
                            frame.shape[0]
                        )

                        kin = det_kin.get(det_idx, {"vx": np.nan, "vy": np.nan, "speed": np.nan})
                        speed = kin["speed"]

                        eps = 1e-6
                        area_norm = (w * h) / (frame_width * frame_height)
                        speed_area_norm = float(speed / max(np.sqrt(area_norm), eps))

                        motion_logs.append((
                            f"{current_frame}_{i}_{video_id}",
                            event_time_seconds,
                            ((x_out, y_out, w, h), yolo_bb),
                            float(speed),
                            speed_area_norm
                        ))
            else:
                # No detections; let tracker handle unmatched tracks
                tracker.update([])

            # ------------- UPDATE PROGRESS -------------
            pbar.update(1)
            pbar.set_postfix({
                "Events": tracker.tracker_log,
                "RapidScan": rapid_mode,
                "Skipped Frames": len(skipped_frames),
                "LastDetFrame": last_detection_frame,
                "Close_events": close_points,
                "Decode_Errors": len(decode_error_frames)
            })
    
    if current_frame != total_frames:
        print("Warning, not all frames fully Processed")
        decode_error_frame = current_frame
    else:
        decode_error_frame = None

    # return both motion logs and the list of problematic frames
    return motion_logs, decode_error_frame

directory_structure = [
    "motion_frames",
    "motion_frames_unpadded",
    "skipped_frames",
    "gifs"
]

if save_masks:
    for i in ["masks","masks_unpadded"]:
        directory_structure.append(i)

if save_fg:
    for i in ["fg","fg_unpadded"]:
        directory_structure.append(i)
        
# =============================================================================
# Annotation Helper Functions
# =============================================================================
"""
Detection Packaging for Annotation Workflow
--------------------------------------------
These utilities package detections into annotation-friendly artefacts:

`file_indexer`:
    Builds a lookup from stored crops to their frame metadata, enabling parallel
    GIF creation.

`_process_event` and `gif_generator_parallel`:
    Create buffered GIF snippets per event using the indexed crops, assigning
    `gif_id` values that propagate into the CSV outputs.

`create_html_from_gifs`:
    Renders simple grids for qualitative review; `pack_tar` can compress a video's
    outputs when archiving.

Output Files:
    The batch step below writes:

    - `motion_log_<video_id>.csv`: Frame-level detections with YOLO boxes, timings,
      and kinematics
    - `<video_id>_event_metrics.csv`: Event-level durations and speed statistics
      referenced in the manuscript tables
"""
def file_indexer(df):
    """
    Group motion log filenames by track index.

    Parses filenames of format "{frame}_{index}_{video_id}" to group all frames
    belonging to the same track, enabling event-based GIF generation.

    Args:
        df: Pandas DataFrame with 'file' column containing frame filenames

    Returns:
        dict: Mapping from track index to list of associated filenames
    """
    indexed_files = {}
    for file_name in df['file']:
        index = int(file_name.split('_')[1])
        
        if index not in indexed_files:
            indexed_files[index] = []
        
        indexed_files[index].append(file_name) 
    return indexed_files

# --- Pillow compatibility shims (older/newer versions) ---
_HAS_ENUMS = hasattr(Image, "Resampling")
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if _HAS_ENUMS else getattr(Image, "LANCZOS", Image.BICUBIC)
RESAMPLE_BILINEAR = Image.Resampling.BILINEAR if _HAS_ENUMS else Image.BILINEAR
DITHER_FS = Image.Dither.FLOYDSTEINBERG if _HAS_ENUMS else getattr(Image, "FLOYDSTEINBERG", 1)

if hasattr(Image, "Quantize"):
    MEDIANCUT = Image.Quantize.MEDIANCUT
    LIBIMAGEQUANT = getattr(Image.Quantize, "LIBIMAGEQUANT", None)  # may be None if not built-in
else:
    MEDIANCUT = getattr(Image, "MEDIANCUT", 0)
    LIBIMAGEQUANT = None

def files_to_gif(input_directory, output_directory, files, gif_id,
                 fps=15, target_size=(224, 224), bg=(0, 0, 0)):
    """
    Generate high-quality GIF from frame sequence.

    Produces optimised animated GIF using Pillow with global palette quantisation,
    Floyd-Steinberg dithering, and proper disposal mode to prevent inter-frame
    artefacts. Alpha channels are composited to solid background as GIF format
    does not support transparency.

    Processing Pipeline:
        1. Load frames and composite alpha to solid background
        2. Resize to target dimensions using Lanczos interpolation
        3. Build global 256-colour palette from sampled frames
        4. Quantise all frames with shared palette and dithering
        5. Save animated GIF with disposal=2 to prevent trails

    Args:
        input_directory: Directory containing input PNG frames
        output_directory: Directory for output GIF and poster
        files: List of frame filenames (without extension)
        gif_id: Identifier for output files
        fps: Playback frame rate (default: 15)
        target_size: Output dimensions (width, height) (default: 224×224)
        bg: Background RGB colour for alpha compositing (default: black)

    Output Files:
        {output_directory}/{gif_id}.gif: Animated GIF
        {output_directory}/posters/{gif_id}.png: Static poster (first frame)

    Notes:
        Uses libimagequant for palette generation if available (superior quality);
        falls back to median cut algorithm otherwise. Samples up to 120 frames
        for palette generation to balance quality and memory usage.
    """
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(os.path.join(output_directory, "posters"), exist_ok=True)

    gif_filename = os.path.join(output_directory, f"{gif_id}.gif")
    poster_filename = os.path.join(output_directory, "posters", f"{gif_id}.png")

    if os.path.exists(gif_filename) and os.path.exists(poster_filename):
        return

    # 1) Load → composite alpha → RGB → resize (Lanczos)
    frames_rgb = []
    for name in files:
        p = os.path.join(input_directory, f"{name}.png")
        im = Image.open(p).convert("RGBA")
        base = Image.new("RGBA", im.size, bg + (255,))
        im = Image.alpha_composite(base, im).convert("RGB")
        if im.size != target_size:
            im = im.resize(target_size, RESAMPLE_LANCZOS)
        frames_rgb.append(im)

    if not frames_rgb:
        return

    # Save poster from full-colour RGB (avoid paletted artefacts)
    frames_rgb[0].save(poster_filename, format="PNG")

    # Global palette strategy: build single 256-colour palette from sampled frames
    # This ensures temporal consistency (prevents colour flickering between frames)
    # libimagequant uses perceptually-weighted quantisation for superior quality
    if len(frames_rgb) > 120:
        step = max(1, len(frames_rgb) // 120)
        sample = frames_rgb[::step]
    else:
        sample = frames_rgb

    # Create "sprite sheet" from sampled frames for palette generation
    # Downsampling frames reduces memory whilst preserving colour distribution
    tile_w, tile_h = 64, 64
    sprite = Image.new("RGB", (tile_w * len(sample), tile_h))
    for i, fr in enumerate(sample):
        sprite.paste(fr.resize((tile_w, tile_h), RESAMPLE_BILINEAR), (i * tile_w, 0))

    quant_method = LIBIMAGEQUANT if LIBIMAGEQUANT is not None else MEDIANCUT
    global_palette_img = sprite.quantize(colors=256, method=quant_method)

    # 3) Quantize every frame using the SAME palette + good dither
    frames_p = [fr.quantize(palette=global_palette_img, dither=DITHER_FS) for fr in frames_rgb]

    # 4) Save animated GIF with stable palette & correct disposal (no trails)
    duration_ms = int(round(1000 / fps))
    frames_p[0].save(
        gif_filename,
        save_all=True,
        append_images=frames_p[1:],
        loop=0,
        duration=duration_ms,
        disposal=2,      # restore to background between frames
        optimize=False   # keep palette order stable
    )

def motion_log_updater(motion_log_df, updates, chunk_size=50_000, show_progress=True):
    """
    Bulk update motion log DataFrame with GIF IDs.

    Efficiently applies GIF identifier mapping to motion log using chunked
    updates to manage memory consumption for large datasets.

    Args:
        motion_log_df: Pandas DataFrame with 'file' column
        updates: Dict mapping filename to GIF ID (or "N/A" if skipped)
        chunk_size: Number of updates per batch (default: 50,000)
        show_progress: Display tqdm progress bar (default: True)

    Returns:
        pd.DataFrame: Updated motion log with 'gif_id' column populated
    """
    if show_progress:
        try:
            from tqdm import tqdm
            progress = lambda it, total: tqdm(it, total=total, desc="Updating motion log", unit="batch")
        except Exception:
            progress = lambda it, total: it
    else:
        progress = lambda it, total: it

    if not updates:
        return motion_log_df

    df = motion_log_df.set_index('file', drop=False)

    keys = list(updates.keys())
    n = len(keys)

    for start in progress(range(0, n, chunk_size), total=(n + chunk_size - 1) // chunk_size):
        end = min(start + chunk_size, n)
        s = pd.Series({k: updates[k] for k in keys[start:end]}, name='gif_id')
        df.loc[s.index, 'gif_id'] = s

    return df.reset_index(drop=True)

def _process_event(task):
    """
    Worker function for parallel GIF generation.

    Processes a single motion event (continuous sequence of tracked frames),
    generating animated GIF if event duration exceeds minimum threshold.
    Short events are skipped and copied to separate directory.

    Args:
        task: Tuple of (video_id, index, ev_start_frame, files,
                        input_dir, output_dir, frame_buffer, min_event_length)

    Returns:
        dict: Mapping from filename to GIF ID (or "N/A" if event skipped)

    Notes:
        Designed for concurrent execution via ProcessPoolExecutor. Events
        shorter than min_event_length are considered noise and excluded.
    """
    (video_id, index, ev_start_frame, files, 
     input_dir, output_dir, frame_buffer, min_event_length) = task

    skip_dir = os.path.join(output_dir, "skipped_frames")
    os.makedirs(skip_dir, exist_ok=True)

    log = {}
    # if too short, skip
    if len(files) < min_event_length:
        for fname in files:
            log[fname] = "N/A"
            src = os.path.join(input_dir, f"{fname}.png")
            dst = os.path.join(skip_dir, f"{fname}.png")
            shutil.copy(src, dst)
    else:
        gif_id = f"{ev_start_frame}_{index}_{video_id}"
        for fname in files:
            log[fname] = gif_id
        files_to_gif(input_dir, output_dir, files, gif_id)

    return log

def gif_generator_parallel(
    video_id,
    motion_log_df,
    input_directory,
    output_directory,
    indexed_files,
    frame_buffer=5,
    min_event_length=2,
    max_workers=None
):
    """
    Generate GIFs for all motion events using parallel processing.

    Segments tracked frames into discrete motion events based on temporal
    continuity, then generates animated GIFs for each event using process pool.
    Events are delimited by gaps exceeding frame_buffer threshold, matching
    the tracker's unmatched frame tolerance.

    Processing Strategy:
        1. Group frames by track index
        2. Identify event boundaries (gaps >= frame_buffer)
        3. Dispatch event processing to worker pool
        4. Collect GIF IDs and update motion log DataFrame

    Args:
        video_id: Video identifier for output naming
        motion_log_df: Pandas DataFrame with motion detection metadata
        input_directory: Directory containing cropped frame images
        output_directory: Directory for output GIFs and posters
        indexed_files: Dict mapping track index to filename lists
        frame_buffer: Frame gap threshold for event segmentation (default: 5)
        min_event_length: Minimum frames required for GIF generation (default: 2)
        max_workers: Maximum parallel processes (default: None = CPU count)

    Returns:
        pd.DataFrame: Updated motion log with 'gif_id' column populated

    Notes:
        Frame buffer threshold should match tracking_buffer used during motion
        extraction to ensure consistent event segmentation. Short events below
        min_event_length are skipped and copied to 'skipped_frames' directory.
    """
    # 1) build task list
    tasks = []
    for index, files in indexed_files.items():
        # parse & sort
        frames = sorted(
          ((int(f.split('_',1)[0]), f) for f in files),
          key=lambda x: x[0]
        )
        nums, names = zip(*frames) if frames else ([], [])
        # find boundaries
        bnds = [0]
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] >= frame_buffer: # matches the logic in the tracker, where frame_buffer is used as counts.
                bnds.append(i)
        bnds.append(len(nums))

        # extract events
        for start, end in zip(bnds, bnds[1:]):
            event_files = list(names[start:end])
            if not event_files:
                continue
            ev_start = nums[start]
            tasks.append((
                video_id,
                index,
                ev_start,
                event_files,
                input_directory,
                output_directory,
                frame_buffer,
                min_event_length
            ))

    # 2) dispatch pool
    all_logs = {}
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        future_to_task = {exe.submit(_process_event, t): t for t in tasks}
        for future in as_completed(future_to_task):
            log = future.result()
            all_logs.update(log)

    # 3) single DataFrame update
    print("GIF Generation Completed, updating motion log")
    motion_log_df = motion_log_updater(motion_log_df, all_logs)
    return motion_log_df

def create_html_from_gifs(base_directory, video_id, annotation_dictionary=annotation_dictionary, sorted_flag=True, skipped_files=True, auto_play=False):
    """
    Generate interactive HTML annotation interface for motion event review.

    Creates self-contained HTML page displaying motion events as motion grid with
    interactive annotation controls, keyboard navigation, and CSV export. Supports
    light/dark themes with system preference detection and manual override.

    Features:
        - Hover-to-play GIF animation (or Space to toggle all)
        - Customisable annotation buttons from dictionary
        - Text notes per event
        - Persistent annotation table with CSV export
        - Keyboard navigation (Arrow keys scroll, Space toggles playback)
        - Responsive dark/light theme

    Args:
        base_directory: Base directory containing 'gifs' subdirectory
        video_id: Video identifier for page title and output filename
        annotation_dictionary: Dict mapping button keys to display labels
        sorted_flag: Sort GIFs by file size (largest first) (default: True)
        skipped_files: Include skipped frames section (default: True)
        auto_play: Enable automatic GIF playback (default: False)

    Returns:
        str: Path to generated HTML file

    Output Files:
        motion_grid_{video_id}.html: Standard chronological grid
        motion_grid_sorted_{video_id}.html: Size-sorted grid (if sorted_flag=True)

    Notes:
        All annotations are stored in browser memory; users must export to CSV
        before closing or refreshing the page. The interface is optimised for
        rapid annotation of large event datasets by ecological consultants.
    """
    # Get list of GIF files in the directory
    static_gif_directory = os.path.join(base_directory, 'gifs')
    skipped_frames_directory = os.path.join(static_gif_directory, 'skipped_frames')
    gif_files = sorted([file for file in os.listdir(static_gif_directory) if file.endswith('.gif')], key=lambda x: int(x.split('_')[0]))
    
    if sorted_flag:
        gif_files = sorted(gif_files, key=lambda x: os.path.getsize(os.path.join(static_gif_directory, x)), reverse=True)
        
    static_img_directory = os.path.join(".", 'gifs', 'posters')
    static_gif_directory = os.path.join(".", 'gifs')
    
    annotation_block = '<div class="label-container">\n'
    annotation_block += '    <input type="text" placeholder="Notes">\n'

    for key, label in annotation_dictionary.items():
        annotation_block += f'    <button onclick="submitLabel(this, \'{key}\')">{label}</button>\n'

    annotation_block += '</div>'

    # Generate HTML content dynamically
    html_content = f'''
        <!DOCTYPE html>
        <html>
        <style>
            :root {{
                color-scheme: light dark;
                --bg: #ffffff; --text: #111827; --muted: #6b7280;
                --surface: #f8fafc; --surface-2: #ffffff; --border: #e5e7eb;
                --shadow: rgba(0, 0, 0, 0.08);
                --primary: #3b82f6; --primary-contrast: #ffffff;
                --danger: #ef4444; --danger-contrast: #ffffff;
                --table-header-bg: #f3f4f6; --table-row-border: #e5e7eb;
                --hover-overlay: rgba(255, 255, 255, 0.75); --hover-text: #111827;
            }}

            /* System preference dark mode */
            @media (prefers-color-scheme: dark) {{
                :root:not(.light) {{
                    --bg: #0b0d10; --text: #e5e7eb; --muted: #9ca3af;
                    --surface: #0f141a; --surface-2: #111827; --border: #30363d;
                    --shadow: rgba(0, 0, 0, 0.3);
                    --primary: #60a5fa; --primary-contrast: #0b0d10;
                    --danger: #f87171; --danger-contrast: #0b0d10;
                    --table-header-bg: #1f2937; --table-row-border: #30363d;
                    --hover-overlay: rgba(0, 0, 0, 0.55); --hover-text: #e5e7eb;
                }}
            }}

            /* Manual dark mode override */
            :root.dark {{
                --bg: #0b0d10; --text: #e5e7eb; --muted: #9ca3af;
                --surface: #0f141a; --surface-2: #111827; --border: #30363d;
                --shadow: rgba(0, 0, 0, 0.3);
                --primary: #60a5fa; --primary-contrast: #0b0d10;
                --danger: #f87171; --danger-contrast: #0b0d10;
                --table-header-bg: #1f2937; --table-row-border: #30363d;
                --hover-overlay: rgba(0, 0, 0, 0.55); --hover-text: #e5e7eb;
            }}
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; font-family: Arial, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", "Noto Sans";
                   padding-bottom: 72px; background: var(--bg); color: var(--text); }}
            .header {{ text-align: center; padding: 10px; }}
            .sub_header {{ text-align: center; color: var(--muted); }}

            .column {{
                float: left; width: 25%; padding: 10px; position: relative;
                background: var(--surface-2); border: 1px solid var(--border);
                border-radius: 8px; box-shadow: 0 2px 6px var(--shadow);
            }}
            .static, .active {{ padding: 5px; width: 95%; }}
            .active {{ display: none; }}
            .column:hover .static {{ display: none; }}
            .column:hover .active {{ display: inline; }}

            .column:hover::after {{
                content: attr(data-name);
                position: absolute; bottom: 15%; left: 0; width: 95%;
                background: var(--hover-overlay); color: var(--hover-text);
                padding: 5px; text-align: center; font-size: 12px;
            }}

            .row:after {{ content: ""; display: table; clear: both; }}

            .label-container {{ text-align: center; margin-top: 10px; color: var(--text); }}
            .label-container input[type="text"] {{
                width: 150px; padding: 6px 8px; margin-right: 10px;
                background: var(--surface); color: var(--text);
                border: 1px solid var(--border); border-radius: 6px;
            }}
            .label-container button {{
                padding: 6px 12px; background-color: var(--primary);
                color: var(--primary-contrast); border: none; cursor: pointer; border-radius: 6px;
            }}
            .label-container button:active {{ background-color: var(--danger); color: var(--danger-contrast); }}
            .label-container button.permanent-red {{ background-color: var(--danger); color: var(--danger-contrast); }}

            table {{ width: 90%; border-collapse: collapse; margin: 20px auto;
                    background: var(--surface-2); color: var(--text); }}
            th, td {{ border: 1px solid var(--table-row-border); padding: 8px; text-align: left; }}
            th {{ background-color: var(--table-header-bg); color: var(--text); }}

            .delete {{ background-color: var(--primary); padding: 5px 10px; color: var(--primary-contrast);
                       border: none; cursor: pointer; border-radius: 4px; }}
            .delete:active {{ background-color: var(--danger); color: var(--danger-contrast); }}

            .event-count {{ text-align: center; margin: 15px 0; font-size: 14px; color: var(--text); }}

            .button-row {{
                position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
                display: flex; gap: 8px; align-items: center; justify-content: center;
                padding: 8px 16px; padding-bottom: calc(8px + env(safe-area-inset-bottom));
                background: var(--surface-2); border-top: 1px solid var(--border);
                box-shadow: 0 -2px 10px var(--shadow); overflow-x: auto;
            }}
            .button-row .export-button {{ display: inline-flex; flex: 0 0 auto; margin: 0;
                background-color: var(--primary); color: var(--primary-contrast);
                cursor: pointer; border: 0; padding: 8px 12px; border-radius: 4px; }}
            .button-row .export-button:active {{ background-color: var(--danger); color: var(--danger-contrast); }}

            .theme-toggle {{ font-size: 12px; background: transparent; border: 1px solid var(--border);
                             color: var(--text); padding: 6px 10px; border-radius: 999px; cursor: pointer; }}
        </style>
        <body>
        
        <div class="header"><h1>Motion Grid: {video_id}</h1></div>
        <div class="sub_header">
            All motion events are shown below as gifs. Videos have been cropped to the object triggering the motion and will play on hover.<br>
            Press "Space" to toggle playback of all events. <br>
            Labels are recorded at the bottom of the window.<br>
        </div>
        <div class="sub_header" style="margin-top: 10px;">
            <strong style="color: var(--danger);">All labels are cleared once the browser is refreshed. Make sure to copy labels to a separate recording form or save as a csv using the export button.</strong>
        </div>
        <div class="event-count">
            <strong>Total Events:</strong> <span id="totalCount">0</span>
        </div>
        
        <div class="row">
    '''

    # Insert GIF elements into HTML content
    for file in gif_files:
        id = file.split('.')[0]
        
        if auto_play:
            html_content += f'''
                <div class="column" data-name="{id}">
                    <img class="static" src="{os.path.join(static_gif_directory, f"{id}.gif")}">
                    {annotation_block}
                </div>
            '''
        else:
            html_content += f'''
                <div class="column" data-name="{id}">
                    <img class="static" src="{os.path.join(static_img_directory, f"{id}.png")}">
                    <img class="active" src="{os.path.join(static_gif_directory, f"{id}.gif")}">
                    {annotation_block}
                </div>
            '''

    if skipped_files:
        skipped_frames = []
        if os.path.isdir(skipped_frames_directory):
            skipped_frames = [file for file in os.listdir(skipped_frames_directory) if file.endswith('.png')]

        if skipped_frames:
            html_content += '''
                </div>
                <div class="sub_header">
                    <h2>Skipped Files</h2>
                    <p class="sub_header">
                    The following files were excluded from GIF generation and moved to "./gifs/skipped_frames".
                    </p>    
            '''
            for file in skipped_frames:
                html_content += f'{file}<br>\n'
            html_content += '</div>'

    # Close HTML body
    html_content += '''
        </div>
        <div class="sub_header"><h2>Labels</h2></div>
        <table id="labelTable">
            <tr>
                <th>Motion Event</th>
                <th>Label</th>               
                <th>Notes</th>
                <th></th>
            </tr>
        </table>

        <div class="button-row">
            <button class="theme-toggle" id="themeToggle" title="Light/Dark">Toggle theme</button>
            <button class="export-button" onclick="labelDefaults()">Label Defaults</button>
            <button class="export-button" onclick="clearTable()">Clear Table</button>
            <button class="export-button" onclick="exportToCSV()">Export to CSV</button>
        </div>

        <script>
            (function () {
                const html = document.documentElement;
                const btn = document.getElementById('themeToggle');

                const saved = localStorage.getItem('theme');
                if (saved === 'dark') html.classList.add('dark');
                if (saved === 'light') html.classList.add('light');

                btn.addEventListener('click', () => {
                    if (html.classList.contains('dark')) {
                        html.classList.remove('dark'); html.classList.add('light');
                        localStorage.setItem('theme', 'light');
                    } else if (html.classList.contains('light')) {
                        html.classList.remove('light'); html.classList.add('dark');
                        localStorage.setItem('theme', 'dark');
                    } else {
                        html.classList.add('dark');
                        localStorage.setItem('theme', 'dark');
                    }
                });

                const totalEvents = document.querySelectorAll('.column').length;
                document.getElementById('totalCount').textContent = totalEvents;
            })();
        
            function submitLabel(button, label) {
                var columnDiv = button.parentNode.parentNode;
                var gifName = columnDiv.getAttribute("data-name");

                var notesInput = columnDiv.querySelector("input[type='text']");
                var notes = notesInput ? notesInput.value : '';

                var table = document.getElementById("labelTable");
                var row = table.insertRow(-1);
                var cell1 = row.insertCell(0);
                var cell2 = row.insertCell(1);
                var cell3 = row.insertCell(2);
                var cell4 = row.insertCell(3);              
                cell1.textContent = gifName;
                cell2.textContent = label;               
                cell3.textContent = notes;
                cell4.innerHTML = '<input type="button" class="delete" value="Delete" onclick="deleteRow(this)">';

                if (notesInput) notesInput.value = '';
                button.classList.add('permanent-red');
            }

            function labelDefaults() {
                var table = document.getElementById("labelTable");
                var labeled = new Set(
                    Array.from(table.querySelectorAll('tr td:first-child'))
                        .map(td => td.textContent.trim())
                        .filter(Boolean)
                );

                document.querySelectorAll('.column').forEach(function(col) {
                    var name = col.getAttribute('data-name');
                    if (!name || labeled.has(name)) return;

                    var row = table.insertRow(-1);
                    var c1 = row.insertCell(0);
                    var c2 = row.insertCell(1);
                    var c3 = row.insertCell(2);
                    var c4 = row.insertCell(3);
                    c1.textContent = name;
                    c2.textContent = 'default';
                    c3.textContent = '';
                    c4.innerHTML = '<input type="button" class="delete" value="Delete" onclick="deleteRow(this)">';
                    labeled.add(name);
                });
            }

            function clearTable() {
                var table = document.getElementById("labelTable");
                var rowsToDelete = table.rows.length - 1;
                if (rowsToDelete <= 0) {
                    alert('The table is already empty.');
                    return;
                }
                var ok = confirm(
                    'This will remove ' + rowsToDelete + ' row' + (rowsToDelete > 1 ? 's' : '') +
                    ' from the annotation table. This cannot be undone.\\n\\nProceed?'
                );
                if (!ok) return;

                for (var i = table.rows.length - 1; i > 0; i--) {
                    table.deleteRow(i);
                }
            }
            
            function deleteRow(row) {
                var i = row.parentNode.parentNode.rowIndex;
                document.getElementById("labelTable").deleteRow(i);
            }
            
            function exportToCSV() {
                var table = document.querySelector("#labelTable");
                var rows = table.querySelectorAll("tr");
                var csvContent = "data:text/csv;charset=utf-8,";
                csvContent += "file,label,notes\\n";
                
                rows.forEach(function(row) {
                    var cells = row.querySelectorAll("td");
                    if (cells.length === 0) return;
                    var rowData = [];
                    for (var i = 0; i < cells.length - 1; i++) {
                        var text = cells[i].innerText.replace(/"/g, '""');
                        if (/[",\\n]/.test(text)) text = '"' + text + '"';
                        rowData.push(text);
                    }
                    csvContent += rowData.join(",") + "\\n";
                });
                
                var encodedUri = encodeURI(csvContent);
                var link = document.createElement("a");
                link.setAttribute("href", encodedUri);
                link.setAttribute("download", "motion_labels.csv");
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            
            let autoplayMode = false;

            document.addEventListener('keydown', function(e) {
                if (e.code === 'Space') {
                    e.preventDefault();
                    autoplayMode = !autoplayMode;
                    document.querySelectorAll('.column').forEach(col => {
                        const staticImg = col.querySelector('img.static');
                        const activeImg = col.querySelector('img.active');
                        if (autoplayMode) {
                            staticImg.style.display = 'none';
                            activeImg.style.display = 'inline';
                        } else {
                            staticImg.style.display = '';
                            activeImg.style.display = '';
                        }
                    });
                }
            });

            document.addEventListener('keydown', function(e) {
                const tag = document.activeElement && document.activeElement.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA') return;

                const firstTile = document.querySelector('.column');
                const tileHeight = firstTile ? firstTile.offsetHeight : 0;
                if (!tileHeight) return;

                if (e.key === "ArrowDown") {
                    e.preventDefault();
                    window.scrollBy({ top: tileHeight * 2, left: 0 });
                }
                if (e.key === "ArrowUp") {
                    e.preventDefault();
                    window.scrollBy({ top: -tileHeight * 2, left: 0 });
                }
            });
        
        </script>
    </body>
    </html>
    '''
    
    if sorted_flag:
        output_html = os.path.join(base_directory, f"motion_grid_sorted_{video_id}.html")
    else:
        output_html = os.path.join(base_directory, f"motion_grid_{video_id}.html")
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    return f"Motion Grid: {output_html}"

def pack_tar(base_dir, video_id, compress="gz", remove_original=False):
    """
    Archive high-cardinality motion detection outputs.

    Consolidates motion frames, masks, and foreground images into compressed
    archive to reduce inode pressure on storage systems. Particularly beneficial
    for datasets with thousands of frames per video.

    Args:
        base_dir: Base directory containing video subdirectories
        video_id: Video identifier (subdirectory name)
        compress: Compression mode - "none" (.tar) or "gz" (.tar.gz) (default: "gz")
        remove_original: Delete source directories after archiving (default: False)

    Returns:
        str: Path to created archive file

    Raises:
        ValueError: If compress mode is invalid
        FileNotFoundError: If no target directories exist

    Archived Directories:
        - motion_frames / motion_frames_unpadded
        - masks / masks_unpadded
        - fg / fg_unpadded

    Notes:
        Only existing directories are included in archive. Missing directories
        are silently skipped rather than raising errors.
    """
    modes = {
        "none": ("w",    ".tar"),
        "gz":   ("w:gz", ".tar.gz"),
    }
    if compress not in modes:
        raise ValueError("compress must be 'none' or 'gz'")

    mode, ext = modes[compress]

    pack_dirs = [
        "motion_frames", "motion_frames_unpadded",
        "masks", "masks_unpadded",
        "fg", "fg_unpadded"
    ]

    out_dir = os.path.join(base_dir, str(video_id))
    os.makedirs(out_dir, exist_ok=True)

    # Only include dirs that actually exist
    include = [d for d in pack_dirs if os.path.isdir(os.path.join(out_dir, d))]
    if not include:
        raise FileNotFoundError("Nothing to pack: no target directories exist.")

    archive = os.path.join(out_dir, f"{video_id}_outputs{ext}")

    # Write archive
    with tarfile.open(archive, mode) as tar:
        for d in include:
            tar.add(os.path.join(out_dir, d), arcname=d)

    # Optional: remove originals after successful write
    if remove_original:
        for d in include:
            shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)

    return archive

def archive_one(base_dir, video_id, **pack_opts):
    """
    Archive single video outputs.

    Convenience wrapper for pack_tar enabling uniform interface with archive_all.

    Args:
        base_dir: Base directory containing video subdirectories
        video_id: Video identifier to archive
        **pack_opts: Additional arguments passed to pack_tar

    Returns:
        str: Path to created archive file
    """
    return pack_tar(base_dir, video_id, **pack_opts)

def archive_all(video_ids, base_dir, max_workers=2, **pack_opts):
    """
    Archive multiple video outputs in parallel.

    Processes multiple videos concurrently using process pool, displaying progress
    bar with success/error counts. Gracefully handles per-video failures without
    terminating batch operation.

    Args:
        video_ids: List of video identifiers to archive
        base_dir: Base directory containing video subdirectories
        max_workers: Maximum concurrent archiving processes (default: 2)
        **pack_opts: Additional arguments passed to pack_tar for each video

    Returns:
        dict: Mapping from video_id to archive path (None if failed)

    Notes:
        Conservative max_workers default (2) prevents I/O saturation on storage
        systems with limited IOPS. Increase for faster local storage (SSD/NVMe).
    """
    results = {}
    n = len(video_ids)
    ok = err = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(archive_one, base_dir, vid, **pack_opts): vid for vid in video_ids}
        with tqdm(total=n, desc="Archiving", unit="video") as pbar:
            for fut in as_completed(futs):
                vid = futs[fut]
                try:
                    results[vid] = fut.result()
                    ok += 1
                except Exception as e:
                    results[vid] = None
                    err += 1
                    # optional: log the error per item
                    tqdm.write(f"[ERR] {vid}: {e}")
                pbar.set_postfix(done=ok, errors=err, refresh=False)
                pbar.update(1)
    return results

# =============================================================================
# Processing Pipeline
# =============================================================================
"""
Complete Batch Processing Workflow
-----------------------------------
The pipeline processes videos through the following stages:

Data Input:
    Videos are discovered in `video_dir` (filtered by extension) or provided via
    `csv_path` with paired output IDs.

ROI and Noise Exclusion:
    Optional heat-map-driven ROI selection removes zones of repetitive background
    motion; masks and polygon vertices are stored in `masks_array`/`polys_array`
    and persisted with `save_jsons` for reproducibility. Please note that ROI
    selection was not used for purposes of the paper.

Parameter Tuning:
    When `mask_tuner_flag` or `motion_tuner_flag` are enabled, interactive trackbars
    visualise MOG2 behaviour and tracking overlap, allowing scene-specific
    optimisation. The tuned values are collected in `motion_params_array` and written
    to disk alongside the masks.

Motion Extraction:
    `motion_extractor_kalman` runs per video, applying the tuned parameters, ROI
    mask, and Kalman-Hungarian tracking. Bounding boxes are merged where necessary
    and standardised to a minimum crop size (224 px) so contextual pixels are
    retained for annotation. Crops, optional masks, and foreground cut-outs are
    saved per frame with kinematic descriptors.

Event Packaging:
    Frame crops are collated into GIFs via `gif_generator_parallel`;
    `motion_log_<video_id>.csv` stores per-frame detections (pixel and normalised
    YOLO coordinates) with timing and speed estimates; `<video_id>_event_metrics.csv`
    aggregates event-level duration and velocity statistics for reporting. HTML motion
    grids are generated for annotation.

Quality Control and Archiving:
    Processing metadata and any decode warnings are appended to `log.txt` (per video)
    and `processing_log.csv` (batch-level). Optional `archive_all` compresses outputs
    for storage purposes.

Per-Video Loop Order:
    ROI mask (if enabled) → optional motion tuner → extraction and logging →
    GIF/HTML creation → metric tables → optional archiving; console messages and
    `log.txt` are the first place to check for issues.
"""

# ===== Optional ROI selection and Motion Paramater Tuning ===== 
# This interactive tuning interface allows users to optimise motion detection parameters
# for their specific videos, as described in the paper's methodology ("tuned using annotated bat events").
# The interface provides: 1) ROI selection via polygon drawing to exclude unwanted regions
#                         2) Real-time parameter adjustment with live preview of motion detection results

# Clean up any existing OpenCV windows from previous runs
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)
except:
    print("No window to close")

# Initialise storage arrays for batch processing - these will hold tuned parameters for each video
masks_array = []          
polys_array = []           
motion_params_array = []  

# Process each video individually to allow video-specific parameter tuning
for i in range(len(input_videos)):
    video_path = input_videos[i]
    video_id   = video_ids[i]
    output_dir = os.path.join(analysis_dir, str(video_id))

    print(video_id)
    # Load video metadata (fps and frame count needed for parameter calculations)
    capture, fps, total_frames = video_loader(video_path)

    # ===== ROI SELECTION INTERFACE =====
    # Allows manual exclusion of problematic regions (e.g., camera housing, static vegetation)
    # by drawing polygons on a motion heatmap
    if mask_tuner_flag == True:
        # Initialise MOG2 background subtractor for generating the heatmap
        # MOG2 learns the background model to highlight areas with frequent motion
        if use_cuda:
            mog2Subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)
        else:
            mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=mog_history, varThreshold=var_threshold, detectShadows=detect_shadows)

        # Generate a motion heatmap by sampling frames across the video
        # This visualises where motion typically occurs, helping users identify areas to mask
        heat_map = heatmap_generator(
            capture, fps, total_frames, n=heatmap_samples, kernel_size=kernel_size,
            seconds=2, skip_seconds=mask_starting_point, use_cuda=use_cuda, reset_between_samples=True, learning_rate=-1.0
        )

        # Set up interactive window for polygon drawing
        cv2.namedWindow('Mask Generator')
        cv2.setMouseCallback('Mask Generator', draw_mask)  # Mouse callback handles click events for polygon vertices

        # Initialise blank mask (0 = excluded, 255 = included in analysis)
        mask = np.zeros_like(heat_map[:, :, 0])

        polys = [] 
        active_poly = 0 

        # Interactive polygon drawing loop
        while True:
            temp_frame = heat_map.copy()

            # Display live preview of current rectangle/polygon being drawn
            if _dragging and _start is not None and _current is not None:
                x1, y1 = _start
                x2, y2 = _current
                cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Visualise individual vertices as they're placed
            if points:
                for point in points:
                    cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)

            # Display all completed polygons overlaid on the heatmap
            if polys:
                for poly in polys:
                    cv2.polylines(temp_frame, [poly], isClosed=True, color=255)

            cv2.imshow('Mask Generator', temp_frame)

            # Keyboard controls for polygon editing:
            key = cv2.waitKey(25)
            if key & 0xFF == ord('q'):
                break  # Exit ROI selection

            elif key & 0xFF == ord(' '):
                # Spacebar: Finalise current polygon (minimum 3 vertices required)
                if len(points) >= 3:
                    points_array = np.array(points)
                    polys.append(points_array)
                    points = []  # Clear vertex list for next polygon

            elif key & 0xFF == ord('c'):
                # 'c': Remove the most recently completed polygon (undo)
                if polys:
                    polys.pop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Convert polygon coordinates into a binary mask for motion detection
        if polys:
            # Fill all polygon regions with 255 (these areas WILL be analysed)
            for poly in polys:
                cv2.fillPoly(mask, [poly], 255)

            # Apply mask to heatmap for visualisation
            masked = cv2.bitwise_and(heat_map, heat_map, mask=mask)

            # Find bounding box encompassing all polygons and crop to ROI
            x,y,w,h = find_polygons(polys)

            cropped_mask = masked[y:y+h, x:x+w]
            height, width = cropped_mask.shape[:2]

            # Display final masked region for user confirmation
            cv2.imshow("Cropped Mask",cropped_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        else:
            # No ROI selected - analyse entire frame
            print("No ROI Selected")
            mask[:] = 255  # Full frame mask (all pixels included)
            polys = None
            height, width = mask.shape[:2]

        capture.release()

    else:
        # ROI selection disabled - use full frame dimensions
        print("Crop Tuner De-activated")
        ret, frame = capture.read()
        if ret:
            height, width = frame.shape[:2]
            mask = np.full((height, width), 255, dtype=np.uint8)  # Include all pixels
        polys = None

    # Store mask and polygon data for this video (used in batch processing)
    masks_array.append(mask)
    polys_array.append(polys)

    # ===== MOTION PARAMETER TUNING INTERFACE =====
    # Interactive sliders allow real-time adjustment of motion detection parameters
    # while viewing live results on the video feed
    if motion_tuner_flag == True:
        # Initialise Kalman-based multi-object tracker
        # Tracking parameters are set based on current slider values (converted to frame counts where needed)
        tracker = MultiObjectTracker(
            max_unmatched_frames=int(tracking_buffer * fps),  
            max_distance=tracking_distance,                    
            area_weight=area_weight                            
            )

        # Reload video for motion tuning (capture was released after ROI selection)
        capture, fps, total_frames = video_loader(video_path)

        print("Motion Tuner Activated")
        cv2.namedWindow('Motion Masks', cv2.WINDOW_NORMAL)

        # Display current parameter values before tuning
        print(f"MogHistory: {mog_history} | MogThreshold: {var_threshold}| Detect Shadows: {detect_shadows}| Motion Threshold Min: {motion_threshold_min}| Motion Threshold Max: {motion_threshold_max}| Kernel Size: {kernel_size}")

        # Create interactive sliders for each parameter
        # Slider ranges are dynamically set to accommodate current values (ensuring they're visible)
        # Each slider has a callback function that updates the global parameter when moved

        # MOG2 Background Subtractor Parameters:
        mog_history_slider = cv2.createTrackbar('Hist', 'Motion Masks',  max(mog_history,0), max(1000,mog_history), update_mog2_history)          
        var_threshold_slider = cv2.createTrackbar('Var', 'Motion Masks', max(var_threshold,0), max(1000,var_threshold), update_mog2_var)        
        detect_shadows_toggle = cv2.createTrackbar('Shadows', 'Motion Masks', max(detect_shadows,0), max(detect_shadows,1), update_mog2_shadows) 

        # Motion Filtering Parameters:
        motion_threshold_min_slider = cv2.createTrackbar('MMin', 'Motion Masks', max(motion_threshold_min,0), max(4000, motion_threshold_min), update_motion_threshold_min) 
        motion_threshold_max_slider = cv2.createTrackbar('MMax', 'Motion Masks', max(motion_threshold_max,0), max(4000, motion_threshold_max), update_motion_threshold_max) 
        noise_reduction_toggle = cv2.createTrackbar('Kernel', 'Motion Masks', max(kernel_size,0), max(kernel_size,7), update_kernel_size)                                   

        # Bounding Box Merging and Triggering:
        close_points_slider = cv2.createTrackbar('CP_Merger', 'Motion Masks', max(int(min_distance_ratio * FLOAT_SCALE),0), max(100, int(min_distance_ratio * FLOAT_SCALE)), update_min_distance_ratio) 
        trigger_threshold_slider = cv2.createTrackbar('Trigger_Th', 'Motion Masks', max(trigger_threshold,1), max(1000, trigger_threshold), update_trigger_threshold)                                

        # Object Tracking Parameters (Kalman filter):
        tracking_buffer_slider = cv2.createTrackbar('Tracking Buffer', 'Motion Masks', max(int(tracking_buffer * FLOAT_SCALE), 0), max(1000, int(tracking_buffer * FLOAT_SCALE)), update_tracking_buffer)       
        tracking_distance_slider = cv2.createTrackbar('Tracking Distance', 'Motion Masks', max(int(tracking_distance * FLOAT_SCALE), 0), max(1000, int(tracking_distance * FLOAT_SCALE)), update_tracking_distance) 
        area_weight_slider = cv2.createTrackbar('Area Weight', 'Motion Masks', max(int(area_weight * FLOAT_SCALE), 0), max(1000, int(area_weight * FLOAT_SCALE)), update_area_weight)                     

        # Visualisation Control:
        playback_speed_slider = cv2.createTrackbar('Playback Speed', 'Motion Masks', max(int(playback_speed * FLOAT_SCALE), 0), max(300, int(playback_speed * FLOAT_SCALE)), update_playback_speed) 

        # Start the interactive tuning loop - displays video with motion detection overlay
        # Users adjust sliders and immediately see the effect on detection results
        motion_tuner(capture, fps, total_frames, mask = mask, merge_bb=merge_bb)
        capture.release()
    else:
        print("Motion Tuner Deactivated. Default motion parameters applied.")

    # Store the finalised parameters for this video
    # These values will be used in the batch motion extraction process
    tuned_params = {
        'mog_history':           mog_history,
        'var_threshold':         var_threshold,
        'detect_shadows':        detect_shadows,
        'motion_threshold_min':  motion_threshold_min,
        'motion_threshold_max':  motion_threshold_max,
        'kernel_size':           kernel_size,
        'min_distance_ratio':    min_distance_ratio,
        'trigger_threshold':     trigger_threshold,
        'tracking_buffer':       tracking_buffer,
        'tracking_distance':     tracking_distance,
        'area_weight':           area_weight
    }

    # Add tuned parameters to array for batch processing
    motion_params_array.append(tuned_params)

    # Log final parameter values for this video
    print(
        f"MogHistory: {mog_history} | MogThreshold: {var_threshold} | Detect Shadows: {detect_shadows} "
        f"| Motion Threshold Min: {motion_threshold_min} | Motion Threshold Max: {motion_threshold_max} "
        f"| Kernel Size: {kernel_size} | Close Point Merge: {min_distance_ratio} "
        f"| Trigger Threshold: {trigger_threshold} | Tracking Buffer: {tracking_buffer} "
        f"| Tracking Distance: {tracking_distance} | Area Weight: {area_weight}"
    )

# Save all masks, polygons, and optimised parameters to JSON files for logging purposes. 
save_jsons(masks_array, polys_array, motion_params_array, out_dir=video_dir)

# ===== Main Processing Pipeline ===== 

# Initialise storage for output file paths and error tracking
output_gif_grids = []
decode_log = []

# Process each input video sequentially
for i in range(len(input_videos)):
    # Retrieve video-specific configuration
    mask   = masks_array[i]
    polys  = polys_array[i]
    params = motion_params_array[i]

    video_path = input_videos[i]
    video_id   = video_ids[i]
    output_dir = os.path.join(analysis_dir, str(video_id))

    print(f"Processing input video: {video_id}")

    # Create output directory structure for this video
    for subdir in directory_structure:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            #print(f"Created directory: {subdir_path}")
        else:
            print(f"Directory already exists: {subdir_path}")


    # Extract Motion Events
    capture, fps, total_frames = video_loader(video_path)

    # Initialise MOG2 background subtractor (GPU-accelerated if available)
    if use_cuda:
        mog2Subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
            history     = params['mog_history'],
            varThreshold= params['var_threshold'],
            detectShadows=params['detect_shadows']
        )
    else:
        mog2Subtractor = cv2.createBackgroundSubtractorMOG2(
            history     = params['mog_history'],
            varThreshold= params['var_threshold'],
            detectShadows=params['detect_shadows']
        )

    if polys == None:
        print("No ROI Selected")
    else:
        print("Warning: ROI mask has been applied to input video.")

    # Log all processing parameters for reproducibility
    log_line = (
        f"MogHistory: {params['mog_history']} | MogThreshold: {params['var_threshold']} "
        f"| Detect Shadows: {params['detect_shadows']} | Motion Threshold Min: {params['motion_threshold_min']} "
        f"| Motion Threshold Max: {params['motion_threshold_max']} | Kernel Size: {params['kernel_size']} "
        f"| Close Point Merge: {params['min_distance_ratio']} | Trigger Threshold: {params['trigger_threshold']} "
        f"| Tracking Buffer: {params['tracking_buffer']} | Tracking Distance: {params['tracking_distance']} "
        f"| Area Weight: {params['area_weight']}"
    )

    with open(os.path.join(output_dir,'log.txt'), 'a') as logf:
        print(log_line, file=logf)

    # Core motion extraction with Kalman tracking - returns frame-level motion data
    motion_log, decode_error = motion_extractor_kalman(
        video_id,
        capture,
        fps,
        output_dir,
        total_frames,
        merge_bb=True,
        min_distance_ratio=params['min_distance_ratio'],
        trigger_threshold   = params['trigger_threshold'],
        tracking_buffer     = params['tracking_buffer'],
        tracking_distance   = params['tracking_distance'],
        area_weight         = params['area_weight'],
        rapid_scan_seconds=rapid_scan_seconds,
        skip_factor=skip_factor,
        roi_mask            = mask,
        roi_crop            = polys
    )

    capture.release()

    # Track video decode errors for troubleshooting
    if decode_error:
        with open(os.path.join(output_dir,'log.txt'), 'a') as logf:
            error = f"Warning: Decode Error at frame {decode_error}"
            print(error,file=logf)
        decode_log.append((video_id,decode_error))

    # Convert motion log to DataFrame and add metadata columns
    motion_log_df = pd.DataFrame(motion_log, columns = ['file','seconds','bb','speed','speed_area_normalised'])
    motion_log_df['label'] = next(iter(default_annotation.keys())) # Default label for faster annotation
    motion_log_df['id'] = video_id
    motion_log_df['gif_id'] = "N/A"  # Will be populated during GIF generation

    # Generate gifs

    # Index files by base name to group frames into motion events
    indexed_files = file_indexer(motion_log_df)

    # Set up directories for original motion frames and corresponding GIFs
    input_directories = [os.path.join(output_dir,'motion_frames')]
    output_directories = [os.path.join(output_dir,'gifs')]

    # Optionally include foreground masks and binary masks if enabled
    if save_fg:
        input_directories.append(os.path.join(output_dir,'fg'))
        output_directories.append(os.path.join(output_dir,'fg_gifs'))

    if save_masks:
        input_directories.append(os.path.join(output_dir,'masks'))
        output_directories.append(os.path.join(output_dir,'mask_gifs'))

    # Generate GIFs for each output type (original, foreground, masks)
    for input_directory, output_directory in zip(input_directories,output_directories):
        print(f"Input: {input_directory} | Output: {output_directory}")
        motion_log_df=gif_generator_parallel(video_id, motion_log_df, input_directory, output_directory, indexed_files, frame_buffer = int(tracking_buffer * fps), min_event_length=min_event_length,max_workers=4)

    # Save frame-level motion log
    csv_path = os.path.join(output_dir, f"motion_log_{video_id}.csv")
    motion_log_df.to_csv(csv_path, index=False)

    # Generate event level metrics
    # Extract frame number from filename for temporal analysis
    first_token = motion_log_df["file"].astype(str).str.split("_", n=1).str[0]
    motion_log_df["frame_num"] = pd.to_numeric(first_token, errors="coerce")

    # Ensure speed metrics are numeric
    for c in ["speed", "speed_area_normalised"]:
        motion_log_df[c] = pd.to_numeric(motion_log_df[c], errors="coerce")

    # Filter to only frames that belong to valid GIF events
    df_events = motion_log_df.loc[
        motion_log_df["gif_id"].notna() & (motion_log_df["gif_id"] != "N/A") & motion_log_df["frame_num"].notna()
    ].copy()

    # Aggregate frame-level metrics into event-level summaries
    event_metrics = (
        df_events
        .groupby(["id", "gif_id"], as_index=False)
        .agg(
            frames       = ("file", "count"),
            f_start      = ("frame_num", "min"),
            f_end        = ("frame_num", "max"),
            speed_median = ("speed", "median"),
            speed_avg    = ("speed", "mean"),
            speed_max    = ("speed", "max"),
            san_median   = ("speed_area_normalised", "median"),
            san_avg      = ("speed_area_normalised", "mean"),
            san_max      = ("speed_area_normalised", "max"),
        )
    )

    # Calculate temporal metrics for each event
    event_metrics["framespan"]  = (event_metrics["f_end"] - event_metrics["f_start"] + 1).astype(float)
    event_metrics["duration_s"] = event_metrics["framespan"] / fps

    # Helper function to format duration as HH:MM:SS.mmm for readability
    def fmt_hms_ms(secs: float) -> str:
        # HH:MM:SS.mmm
        secs = float(secs)
        h, rem = divmod(secs, 3600)
        m, s  = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    event_metrics["duration_hms"] = event_metrics["duration_s"].apply(fmt_hms_ms)

    # Reorder columns for logical presentation
    event_metrics = event_metrics[[
        "id", "gif_id", "frames", "duration_s", "duration_hms",
        "speed_median", "speed_avg", "speed_max",
        "san_median", "san_avg", "san_max"
    ]]

    # Save event-level metrics CSV
    event_csv_path = os.path.join(output_dir, f"{video_id}_event_metrics.csv")
    event_metrics.to_csv(event_csv_path, index=False)
    print(f"Saved event metrics → {event_csv_path}")

    # Generate HTML grids for visual review (sorted and unsorted versions)
    gif_grid = create_html_from_gifs(output_dir, video_id=video_id,  sorted_flag=True, skipped_files=False)
    gif_grid = create_html_from_gifs(output_dir, video_id=video_id, sorted_flag=False, skipped_files=False)
    output_gif_grids.append(gif_grid)
    print(f"Processing complete")
    print(output_gif_grids[-1])
print("Batch Motion Extraction Completed")

# Log any video decode errors encountered during processing
if decode_log:
    processing_log = os.path.join(analysis_dir, 'processing_log.csv')
    print("Warning processing errors detected! See processing log file for further details.")
    print(processing_log)

    file_exists = os.path.isfile(processing_log)

    with open(processing_log, 'a', newline='') as logf:
        writer = csv.writer(logf)

        # Write header only if file is new/empty
        if not file_exists or os.stat(processing_log).st_size == 0:
            writer.writerow(["video_id", "error_frame"])

        for video_id, error_frame in decode_log:
            writer.writerow([video_id, error_frame])

# Optionally archive the crop outputs to a tar ball. 
if ARCHIVE_OUTPUTS:
    archive_all(
        video_ids,
        analysis_dir,
        max_workers=1,          # try 2 for HDDs, 2–3 for NVMe
        compress="none",
        remove_original=True
    )

for i in output_gif_grids:
    print(i)