#!/usr/bin/env python3
"""
Multi-class Image Classification Inference Pipeline
====================================================

A prediction pipeline for applying trained multi-class models to detect
bats, birds, humans, and other objects in night vision video footage.
The pipeline processes motion detection crops and aggregates predictions
at the sequence level.

Key Features
------------
- Batch or single-folder inference mode
- Minimum dimension filtering to exclude low-quality crops
- Sequence-level aggregation with multiple statistical summaries
- Moving average smoothing for temporal patterns
- Interactive HTML visualisation with probability-based filtering
- GIF preview functionality for high-confidence detections

Output Products
---------------
- Per-image predictions (CSV with probabilities for all classes)
- Sequence-level aggregated predictions with time stamps
- Moving average probability curves
- Diagnostic plots with threshold indicators
- Interactive HTML grid for manual review and annotation

Workflow
--------
1. Load trained multi-class model (4 softmax outputs)
2. Batch process cropped images from motion detection
3. Aggregate frame-level predictions by sequence (gif_id)
4. Compute summary statistics (mean, median, max, top-fraction)
5. Generate visualisation and diagnostic outputs

Date: 17/12/2025
"""

import os
import sys
import argparse
import configparser
import copy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imageio
from tensorflow.keras import backend as K
import json
from tqdm import tqdm

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Class definitions - must match training pipeline order
CLASS_NAMES = ['bat', 'bird', 'human', 'other']   # indices: 0,1,2,3
CLASS_LABELS = list(range(len(CLASS_NAMES)))      # Integer labels: 0, 1, 2, 3
NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
SHORT_NAMES = ['bat', 'bird', 'human', 'other']   # Abbreviated names for plots
BAT_ID  = NAME_TO_ID['bat']    # Primary class of interest
BIRD_ID = NAME_TO_ID['bird']   # Bird class identifier


# =============================================================================
# FILE PARSING UTILITIES
# =============================================================================

def extract_numeric_key(filename, num=10):
    """
    Extract numeric components from filename for natural sorting.

    Parses filename to extract up to 'num' numeric tokens separated by
    underscores, enabling natural sorting (e.g., frame_2 before frame_10).

    Parameters
    ----------
    filename : str
        Image filename with underscore-separated components.
    num : int, optional
        Maximum number of numeric components to extract. Default is 10.

    Returns
    -------
    tuple
        Tuple of integers extracted from filename. Non-numeric components
        are skipped.

    Examples
    --------
    >>> extract_numeric_key("001_42_SiteA_20230615_cam1.png")
    (1, 42, 20230615, 1)
    """
    name, _ = os.path.splitext(filename)
    parts = name.split('_')
    numeric_parts = []
    for part in parts:
        try:
            numeric_parts.append(int(part))
        except ValueError:
            # Skip non-numeric components
            pass
    return tuple(numeric_parts[:num])

def extract_frame_number(filename):
    """
    Extract frame number from filename for temporal ordering.

    Assumes the first numeric token in the filename corresponds to the
    frame number within the video sequence.

    Parameters
    ----------
    filename : str
        Image filename with frame number as first numeric component.

    Returns
    -------
    int
        Frame number, or 0 if no numeric components found.

    Examples
    --------
    >>> extract_frame_number("0042_123_site1.png")
    42
    """
    nums = extract_numeric_key(filename)
    return nums[0] if nums else 0


# =============================================================================
# ERROR LOGGING
# =============================================================================

def _append_log(image_path: str, msg: str):
    """
    Append error messages to a centralised log file.

    Records image-specific errors to log.txt in the parent directory of
    the crops folder. Designed to be failure-tolerant to avoid crashing
    the inference pipeline on logging errors.

    Parameters
    ----------
    image_path : str
        Path to the problematic image file.
    msg : str
        Error message or description of the issue.

    Notes
    -----
    Log format: <image_path><TAB><message>
    Log location: <parent_of_crops_dir>/log.txt
    """
    try:
        # Navigate to parent directory (one level up from crops)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(image_path), os.pardir))
        log_path = os.path.join(parent_dir, "log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{image_path}\t{msg}\n")
    except Exception as log_err:
        # Last-resort fallback: print to console if logging fails
        # Don't raise exception to avoid crashing inference pipeline
        print(f"[log.txt write failed] {image_path}: {log_err}")


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_path, target_height=64, target_width=64):
    """
    Load and preprocess a single image for model inference.

    Implements the same preprocessing pipeline as training: grayscale
    decoding, normalisation to [0, 1], and resizing with padding to
    preserve aspect ratio.

    Parameters
    ----------
    image_path : str
        Path to the image file (PNG format expected).
    target_height : int, optional
        Desired output height in pixels. Default is 64.
    target_width : int, optional
        Desired output width in pixels. Default is 64.

    Returns
    -------
    tf.Tensor
        Preprocessed image tensor of shape (target_height, target_width, 1)
        with float32 dtype and values in [0, 1]. Returns zero-filled tensor
        if loading fails.

    Notes
    -----
    If image loading fails, logs the error and returns a black image to
    allow the pipeline to continue processing other images. The prediction
    for failed images will be meaningless but identifiable via log.txt.
    """
    try:
        # Read raw bytes from disk
        raw = tf.io.read_file(image_path)
        # Decode as single-channel (grayscale) PNG
        image = tf.io.decode_png(raw, channels=1)
        # Convert to float32 with range [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize with padding to preserve aspect ratio
        image = tf.image.resize_with_pad(image, target_height, target_width)
        return image
    except Exception as e:
        # Log error and return zero-filled image to avoid pipeline crash
        err_msg = f"{type(e).__name__}: {e}"
        print(f"Error processing: {image_path} -> using zeros. ({err_msg})")
        _append_log(image_path, err_msg)
        zero_img = tf.zeros([target_height, target_width, 1], dtype=tf.float32)
        return zero_img

def get_image_dimensions(image_path):
    """
    Extract image dimensions without full preprocessing.

    Used for minimum dimension filtering to exclude low-quality crops
    before inference.

    Parameters
    ----------
    image_path : str
        Path to image file.

    Returns
    -------
    tuple
        (height, width) in pixels, or (None, None) if file cannot be opened.

    Notes
    -----
    Uses TensorFlow's decode_image which auto-detects format. This is
    faster than full preprocessing when only dimensions are needed.
    """
    try:
        raw = tf.io.read_file(image_path)
        # Auto-detect format and decode as grayscale
        img = tf.io.decode_image(raw, channels=1, expand_animations=False)
        shape = tf.shape(img)
        height = int(shape[0].numpy())
        width = int(shape[1].numpy())
        return height, width
    except Exception as e:
        tqdm.write(f"[WARN] Could not open {image_path} for dimension check: {e}")
        return None, None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mapping(mapping_csv):
    """
    Load frame-to-sequence mapping from CSV file.

    Reads the mapping file that associates individual frame crops with
    their parent motion sequence (gif_id). This enables aggregation of
    frame-level predictions to sequence-level summaries.

    Parameters
    ----------
    mapping_csv : str
        Path to CSV file with 'file' and 'gif_id' columns.

    Returns
    -------
    dict
        Dictionary mapping base filename (without extension) to gif_id.
        NaN gif_ids are converted to None.

    Notes
    -----
    The mapping file is typically generated by the motion detection
    pipeline and named motion_log_<video_id>.csv.
    """
    mapping_df = pd.read_csv(mapping_csv)
    # Convert NaN gif_ids to None for consistent handling
    mapping_df['gif_id'] = mapping_df['gif_id'].where(mapping_df['gif_id'].notna(), None)
    # Create lookup dictionary keyed by filename without extension
    mapping_dict = dict(zip(mapping_df["file"].astype(str), mapping_df["gif_id"]))
    return mapping_dict


# =============================================================================
# PREDICTION AGGREGATION
# =============================================================================

def top_frac_mean(probs, frac=0.20, min_frames=3):
    """
    Compute mean of top fraction of probabilities.

    Aggregates sequence predictions by averaging only the highest-scoring
    frames. This reduces the impact of low-quality frames whilst maintaining
    sensitivity to peak detection confidence.

    Parameters
    ----------
    probs : array-like
        Per-frame probability values in range [0, 1].
    frac : float, optional
        Fraction of frames to include in mean (e.g., 0.20 = top 20%).
        Default is 0.20.
    min_frames : int, optional
        Minimum number of frames to include regardless of frac. Ensures
        meaningful statistics for short sequences. Default is 3.

    Returns
    -------
    float
        Mean probability of the top-k frames, or 0.0 if probs is empty.

    Notes
    -----
    If frac * N < min_frames, uses min_frames instead. This prevents
    over-optimistic scores for very short sequences.
    """
    N = len(probs)
    # Calculate number of frames to include: max(min_frames, frac * N)
    k = max(min_frames, int(np.floor(frac * max(N, 1))))
    if N == 0:
        return 0.0
    # Sort descending and take top k
    topk = sorted(probs, reverse=True)[:k]
    return float(np.mean(topk)) if topk else 0.0

def aggregate_sequence_predictions(predictions, fps=50):
    """
    Aggregate frame-level predictions to sequence-level summaries.

    Collects per-frame probabilities for each motion sequence (gif_id) and
    computes multiple statistical summaries (mean, median, min, max, top-frac)
    for each class. Also assigns a sequence-level predicted class based on
    average probabilities.

    Parameters
    ----------
    predictions : list of dict
        Per-frame prediction dictionaries containing:
        - filename: Image filename
        - gif_id: Motion sequence identifier
        - prob_bat, prob_bird, prob_human, prob_other: Class probabilities
        - is_na: Boolean flag for excluded frames (optional)
    fps : float, optional
        Video frame rate for time stamp calculation. Default is 50.

    Returns
    -------
    list of dict
        Sequence-level summaries with keys:
        - gif_id: Sequence identifier
        - time: Time in seconds (average frame number / fps)
        - predicted_class_avg: Argmax of average probabilities
        - predicted_label_avg: Class name corresponding to predicted_class_avg
        - average_{class}_probability: Mean probability for each class
        - median_{class}_probability: Median probability for each class
        - min_{class}_probability: Minimum probability for each class
        - max_{class}_probability: Maximum probability for each class
        - top_frac_{class}_probability: Top 20% mean for each class

    Notes
    -----
    Sequences with all frames marked as NA receive NaN for all metrics.
    Frame-level NA flags (e.g., from dimension filtering) are excluded
    from aggregation but still counted for time calculation.
    """
    # Initialise per-sequence accumulators
    seq_probs = {}   # gif_id -> { 'bat':[], 'bird':[], 'human':[], 'other':[] }
    time_dict = {}   # gif_id -> [frame_numbers]

    # Collect probabilities grouped by sequence
    for pred in predictions:
        gid = pred.get("gif_id", None)
        # Skip frames without sequence assignment
        if gid is None or pd.isna(gid):
            continue
        # Initialise accumulators for new sequences
        if gid not in seq_probs:
            seq_probs[gid] = {c: [] for c in CLASS_NAMES}
            time_dict[gid] = []
        # Track frame number for all frames (including NA) for time calculation
        time_dict[gid].append(extract_frame_number(pred["filename"]))
        # Skip NA frames (e.g., failed dimension filter) from probability aggregation
        if pred.get("is_na"):
            continue
        # Accumulate per-class probabilities for valid frames
        seq_probs[gid]['bat'].append(pred['prob_bat'])
        seq_probs[gid]['bird'].append(pred['prob_bird'])
        seq_probs[gid]['human'].append(pred['prob_human'])
        seq_probs[gid]['other'].append(pred['prob_other'])

    # Compute sequence-level statistics
    aggregated = []
    for gid, class_lists in seq_probs.items():
        has_valid = any(len(v) for v in class_lists.values())
        # Handle sequences with no valid predictions (all NA)
        if not has_valid:
            # Create NA record with all metrics set to NaN
            na_summary = {
                f'average_{cls}_probability': np.nan for cls in CLASS_NAMES
            }
            na_summary.update({
                f'median_{cls}_probability': np.nan for cls in CLASS_NAMES
            })
            na_summary.update({
                f'min_{cls}_probability': np.nan for cls in CLASS_NAMES
            })
            na_summary.update({
                f'max_{cls}_probability': np.nan for cls in CLASS_NAMES
            })
            na_summary.update({
                f'top_frac_{cls}_probability': np.nan for cls in CLASS_NAMES
            })

            # Calculate time stamp from average frame number
            avg_frame = np.mean(time_dict.get(gid, [])) if time_dict.get(gid, []) else 0.0
            time_sec = float(avg_frame) / float(fps) if fps > 0 else 0.0

            aggregated.append({
                "gif_id": gid,
                "time": round(time_sec, 4),
                "predicted_class_avg": np.nan,
                "predicted_label_avg": 'na',
                **na_summary
            })
            continue

        # Compute statistics for each class
        summary = {}
        for cls in CLASS_NAMES:
            # Convert to numpy array and remove any NaN values
            arr = np.array(class_lists[cls], dtype=float)
            arr = arr[~np.isnan(arr)]

            if arr.size == 0:
                # No valid predictions for this class in this sequence
                avg = med = minv = maxv = topf = 0.0
            else:
                # Calculate five summary statistics
                avg = float(arr.mean())       # Arithmetic mean
                med = float(np.median(arr))   # Median (robust to outliers)
                minv = float(arr.min())       # Minimum (most conservative estimate)
                maxv = float(arr.max())       # Maximum (peak confidence)
                topf = top_frac_mean(arr)     # Mean of top 20% (reduces noise)

            summary[f'average_{cls}_probability']   = round(avg, 4)
            summary[f'median_{cls}_probability']    = round(med, 4)
            summary[f'min_{cls}_probability']       = round(minv, 4)
            summary[f'max_{cls}_probability']       = round(maxv, 4)
            summary[f'top_frac_{cls}_probability']  = round(topf, 4)

        # Assign sequence-level class based on highest average probability
        avg_vec = np.array([summary[f'average_{c}_probability'] for c in CLASS_NAMES], dtype=float)
        pred_idx = int(np.argmax(avg_vec))
        pred_label = CLASS_NAMES[pred_idx]

        # Calculate sequence time stamp
        avg_frame = np.mean(time_dict[gid]) if time_dict[gid] else 0.0
        time_sec = float(avg_frame) / float(fps) if fps > 0 else 0.0

        aggregated.append({
            "gif_id": gid,
            "time": round(time_sec, 4),
            "predicted_class_avg": pred_idx,
            "predicted_label_avg": pred_label,
            **summary
        })
    return aggregated


# =============================================================================
# VISUALISATION UTILITIES
# =============================================================================

def gif_loader(gif_path, double_size=True):
    """
    Load GIF frames for OpenCV display.

    Reads all frames from a GIF file and converts them to BGR format
    for OpenCV rendering. Optionally upscales for easier viewing.

    Parameters
    ----------
    gif_path : str
        Path to GIF file.
    double_size : bool, optional
        If True, double the dimensions of each frame. Default is True.

    Returns
    -------
    list of np.ndarray
        List of BGR frames ready for cv2.imshow, or empty list on error.

    Notes
    -----
    Uses imageio for GIF decoding (RGB output) then converts to BGR
    for OpenCV compatibility.
    """
    try:
        # Read all frames (imageio returns RGB)
        gif = imageio.mimread(gif_path, memtest=False)
    except Exception as e:
        print(f"Error loading gif {gif_path}: {e}")
        return []

    frames = []
    for frame in gif:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if double_size:
            # Upscale for visibility
            h, w = frame_bgr.shape[:2]
            frame_bgr = cv2.resize(frame_bgr, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        frames.append(frame_bgr)
    return frames

def display_gif_with_text(frames, text, window_name="Gif"):
    """
    Display looping GIF with overlaid probability text.

    Shows motion sequence frames in an infinite loop with prediction
    information overlaid. Used for manual review of high-confidence
    detections.

    Parameters
    ----------
    frames : list of np.ndarray
        BGR frames from gif_loader().
    text : str
        Text to overlay (typically probability score).
    window_name : str, optional
        OpenCV window title. Default is "Gif".

    Notes
    -----
    Press 'q' to close window and advance to next sequence.
    Frame rate is ~10 fps (100ms delay per frame).
    """
    while True:
        for frame in frames:
            # Copy frame to avoid modifying original
            frame_copy = frame.copy()
            # Overlay text in red at top-left
            cv2.putText(frame_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame_copy)
            # 100ms delay = ~10 fps playback; check for 'q' keypress
            if cv2.waitKey(100) & 0xFF == ord('q'):
                cv2.destroyWindow(window_name)
                return


# =============================================================================
# HTML GRID GENERATION
# =============================================================================

def create_html_from_gifs(base_directory,
                          sequence_csv,
                          metric,
                          annotation_dictionary=None,
                          human_map=None,           # kept for API compatibility; ignored (we use model human)
                          sorted_flag=None,
                          skipped_files=False):
    """
    Generate interactive HTML visualisation grid for manual review.

    Creates a web-based interface displaying motion sequence GIFs with
    dynamic filtering based on bat, bird, and human probabilities. Includes
    annotation tools for ground-truth labelling and CSV export.

    Parameters
    ----------
    base_directory : str
        Root directory containing gifs subfolder and sequence predictions.
    sequence_csv : str
        Path to sequence-level predictions CSV.
    metric : str
        Which probability summary to use for sorting/filtering.
        One of: 'average', 'median', 'max', 'top_frac'.
    annotation_dictionary : dict, optional
        Mapping of annotation keys to button labels for manual labelling.
        Default provides: {"bat": "Bat", "other": "Other", "bird": "Bird",
        "human": "Human"}.
    human_map : dict, optional
        Deprecated parameter kept for backwards compatibility. Human
        probabilities are now read from model predictions.
    sorted_flag : str, optional
        If specified, sort grid by this class's probability ('bat', 'bird',
        or 'human'). Default is None (filename order).
    skipped_files : bool, optional
        If True, include section listing frames excluded from GIF generation.
        Default is False.

    Returns
    -------
    None
        Writes HTML file to <base_directory>/motion_grid_<metric>_<video_id>.html

    Notes
    -----
    The generated HTML includes:
    - Interactive sliders for probability filtering (Bat ≥, Bird ≤, Human ≤)
    - Hover-to-animate GIF previews
    - Click-to-annotate interface with CSV export
    - Light/dark theme toggle
    - Keyboard navigation (arrows, spacebar)
    - Real-time visible event counter

    The filtering strategy is optimised for bat detection:
    - Show sequences with high bat probability (≥ threshold)
    - AND low bird probability (≤ threshold)
    - AND low human probability (≤ threshold)
    """
    # Extract video identifier from directory name
    video_id = os.path.basename(base_directory)
    df = pd.read_csv(sequence_csv)

    # Construct column names based on chosen metric
    # e.g., metric='average' -> 'average_bat_probability', etc.
    bat_col   = f"{metric}_bat_probability"
    bird_col  = f"{metric}_bird_probability"
    human_col = f"{metric}_human_probability"

    # Set up directory paths
    gif_dir_abs = os.path.join(base_directory, 'gifs')
    posters_abs = os.path.join(gif_dir_abs, 'posters')
    skipped_frames_directory = os.path.join(gif_dir_abs, 'skipped_frames')

    # Find all GIF files to display
    gif_files = [f for f in os.listdir(gif_dir_abs) if f.endswith('.gif')]

    # Create lookup dictionaries for probability values by gif_id
    # These are used for filtering and sorting
    def _to_str(x): return str(x)
    bat_vals   = df[bat_col]   if bat_col   in df.columns else pd.Series(np.nan, index=df.index)
    bird_vals  = df[bird_col]  if bird_col  in df.columns else pd.Series(np.nan, index=df.index)
    human_vals = df[human_col] if human_col in df.columns else pd.Series(np.nan, index=df.index)
    id_to_bat   = dict(zip(df['gif_id'].astype(str), bat_vals))
    id_to_bird  = dict(zip(df['gif_id'].astype(str), bird_vals))
    id_to_human = dict(zip(df['gif_id'].astype(str), human_vals))

    # Apply sorting if requested
    sort_description = "filename order"
    sorting_applied = False
    if sorted_flag in {"bat", "bird", "human"}:
        sort_col_map = {"bat": bat_col, "bird": bird_col, "human": human_col}
        sort_col = sort_col_map[sorted_flag]
        if sort_col in df.columns:
            df_sorted = df.assign(
                _sort_key=pd.to_numeric(df[sort_col], errors='coerce')
            ).sort_values("_sort_key", ascending=False, na_position='last')
            order_index = {gid: idx for idx, gid in enumerate(df_sorted['gif_id'].astype(str))}
            gif_files.sort(
                key=lambda fn: (order_index.get(fn[:-4], len(order_index)), extract_numeric_key(fn))
            )
            sort_description = f"{metric} {sorted_flag} metric"
            sorting_applied = True
        else:
            print(f"[WARN] Sort column {sort_col} missing; using filename order instead.")
    if not sorting_applied:
        gif_files.sort(key=extract_numeric_key)

    # Use relative paths in HTML
    static_img_directory = os.path.join(".", 'gifs', 'posters')
    gif_dir_rel = os.path.join(".", 'gifs')

    annotations = annotation_dictionary or {}

    # --------------- HTML ---------------
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
                content: attr(data-name) " — Bat:" attr(data-bat) " Bird:" attr(data-bird) " Human:" attr(data-human);
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

            .threshold-control {{ text-align: center; margin: 15px 0; font-size: 14px; color: var(--text); }}
            .threshold-control input[type="range"] {{ width: 60%; vertical-align: middle; }}

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
          All motion events below. Adjust sliders to filter using <strong>Bat ≥</strong> and <strong>Bird/Human ≤</strong> thresholds.
          Sorting uses <strong>{sort_description}</strong>.
        </div>      
        <div class="sub_header threshold-control">
            <label for="batSlider">Bat ≥ <span id="batValue">0.50</span></label>
            <input id="batSlider" type="range" min="0" max="1" step="0.01" value="0.50">
            <br>
            <label for="birdSlider">Bird ≤ <span id="birdValue">1.00</span></label>
            <input id="birdSlider" type="range" min="0" max="1" step="0.01" value="1.00">
            <br>
            <label for="humSlider">Human ≤ <span id="humValue">1.00</span></label>
            <input id="humSlider" type="range" min="0" max="1" step="0.01" value="1.00">
        </div>
        
        <div class="row">
    '''

    # Annotation buttons
    annotation_block = '<div class="label-container"><input type="text" placeholder="Notes">\n'
    for key,label in annotations.items():
        annotation_block += f'<button onclick="submitLabel(this,\'{key}\')">{label}</button>\n'
    annotation_block += '</div>'

    # Tiles
    for file in gif_files:
        gid = file[:-4]
        b  = float(id_to_bat.get(gid, 0.0))
        bd = float(id_to_bird.get(gid, 0.0))
        h  = float(id_to_human.get(gid, 0.0))
        html_content += f'''
            <div class="column" data-name="{gid}" data-bat="{b:.2f}" data-bird="{bd:.2f}" data-human="{h:.2f}">
                <img class="static" src="{os.path.join(static_img_directory,f"{gid}.png")}">
                <img class="active" src="{os.path.join(gif_dir_rel,f"{gid}.gif")}">
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
                    // if no explicit class yet, flip based on system assumption
                    html.classList.add('dark');
                    localStorage.setItem('theme', 'dark');
                }
                });
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

            // Add all visible, unlabeled events with default label "default"
            function labelDefaults() {
                var table = document.getElementById("labelTable");
                var labeled = new Set(
                    Array.from(table.querySelectorAll('tr td:first-child'))
                        .map(td => td.textContent.trim())
                        .filter(Boolean)
                );

                var visibleCols = Array.from(document.querySelectorAll('.column'))
                    .filter(el => el.offsetParent !== null);

                visibleCols.forEach(function(col) {
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

            // NEW: Clear the table (with confirmation), preserving header row
            function clearTable() {
                var table = document.getElementById("labelTable");
                var rowsToDelete = table.rows.length - 1; // exclude header
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
                    if (cells.length === 0) return; // skip header row
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

            const batS  = document.getElementById('batSlider'),
                  birdS = document.getElementById('birdSlider'),
                  humS  = document.getElementById('humSlider'),
                  batV  = document.getElementById('batValue'),
                  birdV = document.getElementById('birdValue'),
                  humV  = document.getElementById('humValue');

            function updateFilters() {
                const bt = +batS.value, bdt = +birdS.value, ht = +humS.value;
                batV.textContent  = bt.toFixed(2);
                birdV.textContent = bdt.toFixed(2);
                humV.textContent  = ht.toFixed(2);
                document.querySelectorAll('.column').forEach(col => {{
                    const b  = +col.getAttribute('data-bat');
                    const bd = +col.getAttribute('data-bird');
                    const h  = +col.getAttribute('data-human');
                    col.style.display = (b >= bt && bd <= bdt && h <= ht) ? '' : 'none';
                }});
           }

            (function () {
                const counterWrap = document.createElement('div');
                counterWrap.className = 'sub_header';
                counterWrap.style.margin = '8px 0';
                counterWrap.innerHTML = '<strong>Visible Events:</strong> <span id="visibleCount">0</span> / <span id="totalCount">0</span>';
                const control = document.querySelector('.threshold-control');
                if (control) control.after(counterWrap);

                const totalEvents = document.querySelectorAll('.column').length;
                document.getElementById('totalCount').textContent = totalEvents;

                function updateVisibleCount() {
                    const visible = Array.from(document.querySelectorAll('.column'))
                        .filter(el => el.offsetParent !== null).length;
                    document.getElementById('visibleCount').textContent = visible;
                }

                const _updateFilters = updateFilters;
                window.updateFilters = function () {
                    _updateFilters();
                    updateVisibleCount();
                };

                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', () => {
                        updateFilters();
                    });
                } else {
                    updateFilters();
                }
            })();

            batS.addEventListener('input', updateFilters);
            birdS.addEventListener('input', updateFilters);
            humS.addEventListener('input', updateFilters);

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

                const firstVisibleTile = Array.from(document.querySelectorAll('.column'))
                    .find(el => el.offsetParent !== null);

                const tileHeight = firstVisibleTile ? firstVisibleTile.offsetHeight : 0;
                if (!tileHeight) return;

                const rowHeight = tileHeight;

                if (e.key === "ArrowDown") {
                    e.preventDefault();
                    window.scrollBy({ top: rowHeight * 2, left: 0 });
                }
                if (e.key === "ArrowUp") {
                    e.preventDefault();
                    window.scrollBy({ top: -rowHeight * 2, left: 0 });
                }
            });
        
        </script>
    </body>
    </html>
    '''

    suffix = f"{metric}_{sorted_flag}_sorted" if sorted_flag else f"{metric}"
    outname = f"motion_grid_{suffix}_{video_id}.html"
    outpath = os.path.join(base_directory, outname)
    with open(outpath, 'w') as f:
        f.write(html_content)
    print(f"Motion grid written to {outpath}")


# =============================================================================
# MAIN INFERENCE PIPELINE
# =============================================================================

def process_folder(args):
    """
    Execute complete inference pipeline for one video directory.

    Orchestrates the full prediction workflow: loading model, processing
    images in batches, aggregating to sequences, computing statistics,
    generating plots, and optionally creating HTML grids and displaying
    high-confidence detections.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object with attributes:
        - crops_dir : str
            Directory containing cropped motion frames
        - model_path : str
            Path to trained .keras model file
        - mapping_csv : str
            Path to frame-to-sequence mapping CSV
        - gifs_dir : str
            Directory containing sequence GIFs
        - output_csv : str
            Output path for per-frame predictions
        - sequence_csv : str
            Output path for sequence-level predictions
        - ma_csv : str
            Output path for moving-average statistics
        - batch_size : int
            Inference batch size
        - fps : float
            Video frame rate for time calculation
        - ma_window : int
            Moving average window size
        - threshold : float
            Bat probability threshold for plots/display
        - metric : str
            Summary metric for sorting ('average', 'median', 'max', 'top_frac')
        - min_dimensions : int or None
            Minimum image dimension filter
        - html_grid : bool
            Whether to generate HTML visualisation
        - display_gifs : bool
            Whether to show high-confidence GIFs in OpenCV
        - annotation_dict : str or None
            Path to JSON file with annotation button config
        - sort_grid : str or None
            Class to sort HTML grid by
        - skipped_files : bool
            Whether to show skipped files in HTML

    Returns
    -------
    None
        Writes multiple output files to disk.

    Notes
    -----
    This function is designed to be failure-tolerant: missing directories
    are skipped with a warning rather than raising exceptions.
    """
    # Validate input directory exists
    if not os.path.isdir(args.crops_dir):
        print(f"Skipping {args.crops_dir} (not found)")
        return

    print(f"\n=== Processing {args.crops_dir} ===")
    print(f"Loading model from {args.model_path}")
    # Load model without compiling (we only need forward pass)
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Load frame-to-sequence mapping
    mapping = load_mapping(args.mapping_csv)
    print(f"Loaded mapping ({len(mapping)} entries) from {args.mapping_csv}")

    # Collect and sort image files
    files = sorted(
        [f for f in os.listdir(args.crops_dir)
         if f.lower().endswith(('.jpg','.jpeg','.png'))],
        key=extract_numeric_key
    )
    if not files:
        print(f"No images in {args.crops_dir}, skipping.")
        return

    # -------------------------------------------------------------------------
    # BATCH INFERENCE
    # -------------------------------------------------------------------------
    preds = []
    with tqdm(total=len(files), desc="Detecting Bats (multiclass)", unit="img") as pbar:
        for i in range(0, len(files), args.batch_size):
            batch = files[i:i+args.batch_size]
            valid_items = []

            # Process each file in batch
            for fn in batch:
                filepath = os.path.join(args.crops_dir, fn)
                base = os.path.splitext(fn)[0]
                # Look up sequence ID from mapping
                gid = mapping.get(base, None)
                if pd.isna(gid):
                    gid = None

                # Apply minimum dimension filter if requested
                if args.min_dimensions is not None:
                    height, width = get_image_dimensions(filepath)
                    # Mark as NA if image is too small or unreadable
                    if height is None or width is None or width < args.min_dimensions or height < args.min_dimensions:
                        # Create NA record for filtered-out image
                        preds.append({
                            "filename": fn,
                            "predicted_class": "na",
                            "label": "na",
                            "prob_bat": "na",
                            "prob_bird": "na",
                            "prob_human": "na",
                            "prob_other": "na",
                            "gif_id": gid,
                            "is_na": True
                        })
                        pbar.update(1)
                        continue

                # Preprocess image and add to batch
                img_tensor = preprocess_image(filepath)
                valid_items.append((fn, gid, img_tensor))

            # Skip batch if all images were filtered out
            if not valid_items:
                continue

            # Stack into batch tensor and run inference
            tensor = tf.stack([item[2] for item in valid_items], axis=0)
            probs = model.predict(tensor, verbose=0)  # shape: [B, 4] => [bat,bird,human,other]

            # Parse predictions and create records
            for (fn, gid, _), arr in zip(valid_items, probs):
                # Defensive handling: ensure model output has 4 values
                if len(arr) < 4:
                    # Fallback for binary model: treat as [bat, other]
                    p_bat = float(arr[0])
                    p_vec = np.array([p_bat, 0.0, 0.0, 1.0 - p_bat], dtype=float)
                else:
                    p_vec = np.asarray(arr, dtype=float)

                # Normalise to ensure probabilities sum to 1.0
                s = p_vec.sum()
                if s > 0:
                    p_vec = p_vec / s

                # Extract individual class probabilities
                p_bat, p_bird, p_human, p_other = [float(x) for x in p_vec]
                # Determine predicted class (argmax)
                cls_idx = int(np.argmax(p_vec))
                cls_label = CLASS_NAMES[cls_idx]

                preds.append({
                    "filename": fn,
                    "predicted_class": cls_idx,
                    "label": cls_label,
                    "prob_bat": round(p_bat, 4),
                    "prob_bird": round(p_bird, 4),
                    "prob_human": round(p_human, 4),
                    "prob_other": round(p_other, 4),
                    "gif_id": gid
                })
                pbar.update(1)

    # -------------------------------------------------------------------------
    # SAVE PER-FRAME PREDICTIONS
    # -------------------------------------------------------------------------
    df = pd.DataFrame(preds)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved individual predictions to {args.output_csv}")

    # -------------------------------------------------------------------------
    # AGGREGATE TO SEQUENCE LEVEL
    # -------------------------------------------------------------------------
    seq = aggregate_sequence_predictions(preds, fps=args.fps)
    df_seq = pd.DataFrame(seq)
    df_seq.to_csv(args.sequence_csv, index=False)
    print(f"Saved sequence predictions to {args.sequence_csv}")

    # -------------------------------------------------------------------------
    # MOVING AVERAGES FOR TEMPORAL SMOOTHING
    # -------------------------------------------------------------------------
    df_sorted = df_seq.sort_values('time')
    w = args.ma_window

    # Compute moving averages for bat metrics only
    for metric_key, col in [
        ('avg',  'average_bat_probability'),
        ('med',  'median_bat_probability'),
        ('max',  'max_bat_probability'),
        ('topf', 'top_frac_bat_probability'),
    ]:
        ma_col = f'{col}_ma'
        if col in df_sorted.columns:
            df_sorted[ma_col] = df_sorted[col].rolling(window=w, center=True).mean()
    df_sorted.to_csv(args.ma_csv, index=False)
    print(f"Saved moving-average CSV to {args.ma_csv}")

    # -------------------------------------------------------------------------
    # GENERATE DIAGNOSTIC PLOTS
    # -------------------------------------------------------------------------
    times = df_seq.get('time', pd.Series(dtype=float)).tolist()
    base = os.path.splitext(args.sequence_csv)[0]

    # Create plots for each bat probability metric
    plot_specs = [
        ("avg",   'average_bat_probability',     'Average Bat Probability'),
        ("med",   'median_bat_probability',      'Median Bat Probability'),
        ("max",   'max_bat_probability',         'Max Bat Probability'),
        ("topf",  'top_frac_bat_probability',    'Top Fraction Bat Probability'),
    ]
    for tag, col, ylabel in plot_specs:
        if col not in df_seq.columns: 
            continue
        raw = df_seq[col].tolist()
        ma_col = f'{col}_ma'
        out_png = f"{base}_{tag}.png"

        # Create time-series plot with raw and smoothed curves
        plt.figure(figsize=(10,6))
        # Plot raw probabilities (semi-transparent)
        plt.plot(times, raw, marker='o', linestyle='-', alpha=0.3, label=f'Raw {tag}')
        # Plot moving average if available
        if ma_col in df_sorted.columns:
            plt.plot(df_sorted['time'], df_sorted[ma_col], marker='o', linestyle='-', label=f'MA {tag}')
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Over Time ({os.path.basename(base)})")
        # Add horizontal threshold line for reference
        plt.axhline(args.threshold, color='r', linestyle='--', label=f"Threshold ({args.threshold})")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_png)
        plt.close()
        print(f"Saved plot to {out_png}")

    # -------------------------------------------------------------------------
    # OPTIONAL: DISPLAY HIGH-CONFIDENCE GIFs
    # -------------------------------------------------------------------------
    if args.display_gifs:
        print("Displaying predicted-bat GIFs (press 'q' to advance)...")
        if os.path.isdir(args.gifs_dir):
            # Determine which metric to use for thresholding
            metric_key = args.metric or 'top_frac'
            metric_col = f"{metric_key}_bat_probability"
            if metric_col not in df_seq.columns:
                print(f"Metric column {metric_col} not found; skipping GIF display.")
            else:
                # Iterate through sequences and display those above threshold
                for _, row in df_seq.iterrows():
                    val = row.get(metric_col, None)
                    # Show GIF if bat probability exceeds threshold
                    if val is not None and float(val) > float(args.threshold):
                        gif_path = os.path.join(args.gifs_dir, f"{row['gif_id']}.gif")
                        if os.path.exists(gif_path):
                            txt = f"{metric_key.capitalize()}(bat)={float(val):.2f}"
                            print(f" Showing {gif_path} [{txt}]")
                            frames = gif_loader(gif_path, double_size=True)
                            if frames:
                                display_gif_with_text(frames, txt, window_name=f"GIF {row['gif_id']}")
                        else:
                            print(f"  GIF not found: {gif_path}")
        else:
            print(f"Gifs directory not found: {args.gifs_dir}")

        # Clean up OpenCV windows
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # -------------------------------------------------------------------------
    # OPTIONAL: GENERATE HTML GRID
    # -------------------------------------------------------------------------
    if args.html_grid:
        # Load annotation dictionary (or use defaults)
        ann = {"bat":"Bat","other":"Other","bird":"Bird","human":"Human"}
        if args.annotation_dict:
            with open(args.annotation_dict) as f:
                ann = json.load(f)

        # Generate HTML grid(s)
        if args.metric:
            # Single grid for specified metric
            create_html_from_gifs(
                base_directory=os.path.dirname(args.crops_dir),
                sequence_csv=args.sequence_csv,
                metric=args.metric,
                annotation_dictionary=ann,
                human_map=None,  # Deprecated: using model predictions
                sorted_flag=args.sort_grid,
                skipped_files=args.skipped_files
            )
        else:
            # Generate grids for all four metrics
            for metric_opt in ['average','median','max','top_frac']:
                create_html_from_gifs(
                    base_directory=os.path.dirname(args.crops_dir),
                    sequence_csv=args.sequence_csv,
                    metric=metric_opt,
                    annotation_dictionary=ann,
                    human_map=None,
                    sorted_flag=args.sort_grid,
                    skipped_files=args.skipped_files
                )


# =============================================================================
# CONFIGURATION LOADING AND PARSING
# =============================================================================

def load_config_from_ini(config_path):
    """
    Load configuration from .ini file.

    Parameters
    ----------
    config_path : str
        Path to .ini configuration file.

    Returns
    -------
    dict
        Dictionary containing all configuration parameters.
    """
    config = configparser.ConfigParser()
    if config_path:
        config.read(config_path)

    # Define default values
    defaults = {
        'batch_size': 32,
        'fps': 50.0,
        'ma_window': 3,
        'threshold': 0.404,
        'metric': None,
        'min_dimensions': None,
        'gifs_subdir': 'gifs',
        'motion_subdir': 'motion_frames_unpadded',
        'html_grid': True,
        'display_gifs': False,
        'annotation_dict': None,
        'auto_play': False,
        'sort_grid': None,
        'skipped_files': False,
    }

    cfg = {}

    # Inference section
    if config.has_section('inference'):
        cfg['batch_size'] = config.getint('inference', 'batch_size', fallback=defaults['batch_size'])
        cfg['fps'] = config.getfloat('inference', 'fps', fallback=defaults['fps'])
        cfg['ma_window'] = config.getint('inference', 'ma_window', fallback=defaults['ma_window'])
        cfg['threshold'] = config.getfloat('inference', 'threshold', fallback=defaults['threshold'])
        cfg['metric'] = config.get('inference', 'metric', fallback=defaults['metric'])
        if cfg['metric'] == 'None' or cfg['metric'] == '':
            cfg['metric'] = None
        cfg['min_dimensions'] = config.getint('inference', 'min_dimensions', fallback=defaults['min_dimensions']) if config.get('inference', 'min_dimensions', fallback=None) else None
    else:
        cfg['batch_size'] = defaults['batch_size']
        cfg['fps'] = defaults['fps']
        cfg['ma_window'] = defaults['ma_window']
        cfg['threshold'] = defaults['threshold']
        cfg['metric'] = defaults['metric']
        cfg['min_dimensions'] = defaults['min_dimensions']

    # Paths section
    if config.has_section('paths'):
        cfg['gifs_subdir'] = config.get('paths', 'gifs_subdir', fallback=defaults['gifs_subdir'])
        cfg['motion_subdir'] = config.get('paths', 'motion_subdir', fallback=defaults['motion_subdir'])
        cfg['annotation_dict'] = config.get('paths', 'annotation_dict', fallback=defaults['annotation_dict'])
        if cfg['annotation_dict'] == 'None' or cfg['annotation_dict'] == '':
            cfg['annotation_dict'] = None
    else:
        cfg['gifs_subdir'] = defaults['gifs_subdir']
        cfg['motion_subdir'] = defaults['motion_subdir']
        cfg['annotation_dict'] = defaults['annotation_dict']

    # Visualisation section
    if config.has_section('visualisation'):
        cfg['html_grid'] = config.getboolean('visualisation', 'html_grid', fallback=defaults['html_grid'])
        cfg['display_gifs'] = config.getboolean('visualisation', 'display_gifs', fallback=defaults['display_gifs'])
        cfg['auto_play'] = config.getboolean('visualisation', 'auto_play', fallback=defaults['auto_play'])
        cfg['sort_grid'] = config.get('visualisation', 'sort_grid', fallback=defaults['sort_grid'])
        if cfg['sort_grid'] == 'None' or cfg['sort_grid'] == '':
            cfg['sort_grid'] = None
        elif cfg['sort_grid']:
            cfg['sort_grid'] = cfg['sort_grid'].lower()
        cfg['skipped_files'] = config.getboolean('visualisation', 'skipped_files', fallback=defaults['skipped_files'])
    else:
        cfg['html_grid'] = defaults['html_grid']
        cfg['display_gifs'] = defaults['display_gifs']
        cfg['auto_play'] = defaults['auto_play']
        cfg['sort_grid'] = defaults['sort_grid']
        cfg['skipped_files'] = defaults['skipped_files']

    return cfg


def parse_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multi-class bat detection inference pipeline with HTML visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using config file
  python predict.py --config predict_config.ini --model_path model.keras --batch_dir /data

  # Using CLI arguments only
  python predict.py --model_path model.keras --input_dir /video/motion_frames_unpadded

  # Mixed (CLI overrides config)
  python predict.py --config predict_config.ini --model_path model.keras --batch_dir /data --threshold 0.4
        """
    )

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to .ini configuration file')

    # Mode selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_dir', help="Single-folder mode: path to motion_frames_unpadded")
    group.add_argument('--batch_dir',  help="Batch mode: parent directory")

    # Model
    parser.add_argument('--model_path', required=True, help="Path to .keras model (softmax 4-class: bat,bird,human,other)")

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=None,
                       help="Inference batch size")
    parser.add_argument('--fps', type=float, default=None,
                       help="Video frame rate for time calculation")
    parser.add_argument('--ma_window', type=int, default=None,
                       help="Moving average window size")
    parser.add_argument('--threshold', type=float, default=None,
                       help="Bat probability threshold for plots/display")
    parser.add_argument('--metric', choices=['average','median','max','top_frac'], default=None,
                       help="Metric for GIF sorting/display")
    parser.add_argument('--min_dimensions', type=int, default=None,
                       help="Minimum height/width (pixels). Frames below this are marked NA")

    # Paths
    parser.add_argument('--gifs_subdir', type=str, default=None,
                       help="Subdirectory name for GIFs")
    parser.add_argument('--motion_subdir', type=str, default=None,
                       help="Subdirectory name for motion frames")
    parser.add_argument('--annotation_dict', type=str, default=None,
                       help="Optional JSON file with annotation dictionary")

    # Visualisation
    parser.add_argument('--html_grid', action='store_true', default=None,
                       help="Generate an HTML grid sorted by bat metric, with bird/human filters")
    parser.add_argument('--display_gifs', action='store_true', default=None,
                       help="Show predicted-bat GIFs in OpenCV")
    parser.add_argument('--auto_play', action='store_true', default=None,
                       help="Spacebar toggles autoplay in HTML")
    parser.add_argument('--sort_grid', type=str, choices=['bat', 'bird', 'human'], default=None,
                       help="Sort HTML grid by this class")
    parser.add_argument('--skipped_files', type=lambda x: x.lower() == 'true', default=None,
                       help="Show skipped files in HTML (true/false)")
    return parser.parse_args()


def merge_config(config_dict, args):
    """
    Merge configuration from file and command-line arguments.

    Command-line arguments take precedence over config file values.

    Parameters
    ----------
    config_dict : dict
        Configuration loaded from .ini file.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    # Start with config file values (or defaults if no config file)
    cfg = config_dict.copy()

    # Override with CLI arguments where provided
    for key, value in vars(args).items():
        if key not in ['config', 'input_dir', 'batch_dir', 'model_path'] and value is not None:
            cfg[key] = value

    return cfg


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Main entry point for the prediction pipeline.

    Parses command-line arguments and executes inference in either single-folder
    mode (--input_dir) or batch mode (--batch_dir) for processing multiple videos.

    Command-Line Usage
    ------------------
    Single-folder mode:
        python predict.py --input_dir /path/to/video/motion_frames_unpadded \\
                         --model_path model.keras

    Batch mode:
        python predict.py --batch_dir /path/to/parent \\
                         --model_path model.keras \\
                         --html_grid --metric average

    Key Arguments
    -------------
    --input_dir : str
        Single-folder mode: path to motion_frames_unpadded directory.
    --batch_dir : str
        Batch mode: parent directory containing multiple video folders.
    --model_path : str (required)
        Path to trained .keras model (4-class softmax output).
    --batch_size : int
        Inference batch size. Default is 32.
    --threshold : float
        Bat probability threshold for plots/display. Default is 0.5.
    --metric : str
        Probability metric for sorting/display: 'average', 'median', 'max',
        or 'top_frac'. Default is None (generates all four).
    --min_dimensions : int
        Minimum image dimension filter. Frames smaller than this are marked
        as NA. Default is None (no filtering).
    --html_grid : flag
        Generate interactive HTML visualisation grid.
    --display_gifs : flag
        Show high-confidence bat detections in OpenCV windows.
    --sort_grid : str
        Sort HTML grid by class probability: 'bat', 'bird', or 'human'.

    Notes
    -----
    In batch mode, the script processes all subdirectories, expecting each
    to contain: motion_frames_unpadded/, gifs/, and motion_log_*.csv.
    """
    # -------------------------------------------------------------------------
    # CONFIGURATION LOADING
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration from file if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"ERROR: Configuration file not found: {args.config}")
            sys.exit(1)
        print(f"Loading configuration from: {args.config}")
        config_dict = load_config_from_ini(args.config)
    else:
        # Use defaults if no config file provided
        print("No configuration file provided. Using defaults and CLI arguments.")
        config_dict = load_config_from_ini(None)  # Will use all defaults

    # Merge config file with CLI arguments (CLI takes precedence)
    cfg = merge_config(config_dict, args)

    # Create namespace-like object with merged config
    class ConfigNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # Rebuild args with merged configuration
    merged_args = ConfigNamespace(
        input_dir=args.input_dir,
        batch_dir=args.batch_dir,
        model_path=args.model_path,
        batch_size=cfg['batch_size'],
        fps=cfg['fps'],
        ma_window=cfg['ma_window'],
        threshold=cfg['threshold'],
        metric=cfg['metric'],
        min_dimensions=cfg['min_dimensions'],
        gifs_subdir=cfg['gifs_subdir'],
        motion_subdir=cfg['motion_subdir'],
        html_grid=cfg['html_grid'],
        display_gifs=cfg['display_gifs'],
        annotation_dict=cfg['annotation_dict'],
        auto_play=cfg['auto_play'],
        sort_grid=cfg['sort_grid'],
        skipped_files=cfg['skipped_files']
    )

    args = merged_args

    # Print configuration summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Batch size: {args.batch_size}, FPS: {args.fps}")
    print(f"Threshold: {args.threshold}, Metric: {args.metric}")
    print(f"Min dimensions: {args.min_dimensions}")
    print(f"HTML grid: {args.html_grid}, Display GIFs: {args.display_gifs}")
    if args.sort_grid:
        print(f"Sort grid by: {args.sort_grid}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # BATCH OR SINGLE-FOLDER PROCESSING
    # -------------------------------------------------------------------------
    if args.batch_dir:
        # Batch mode: iterate over all subdirectories
        for sub in sorted(os.listdir(args.batch_dir)):
            exp = os.path.join(args.batch_dir, sub)
            # Skip non-directories
            if not os.path.isdir(exp):
                continue

            # Create folder-specific configuration
            a = copy.deepcopy(args)
            a.crops_dir    = os.path.join(exp, args.motion_subdir)
            a.mapping_csv  = os.path.join(exp, f'motion_log_{sub}.csv')
            a.gifs_dir     = os.path.join(exp, args.gifs_subdir)
            a.output_csv   = os.path.join(exp, f'{sub}_predictions.csv')
            a.sequence_csv = os.path.join(exp, f'{sub}_sequence_predictions.csv')
            a.ma_csv       = os.path.join(exp, f'{sub}_sequence_predictions_ma.csv')

            # Process this folder
            process_folder(a)
            # Clear TensorFlow session to free memory between videos
            K.clear_session()
    else:
        # Single-folder mode: process one video directory
        exp = os.path.abspath(os.path.dirname(args.input_dir))
        sub = os.path.basename(exp)

        # Create folder-specific configuration
        a = copy.deepcopy(args)
        a.crops_dir    = os.path.join(exp, args.motion_subdir)
        a.mapping_csv  = os.path.join(exp, f'motion_log_{sub}.csv')
        a.gifs_dir     = os.path.join(exp, args.gifs_subdir)
        a.output_csv   = os.path.join(exp, f'{sub}_predictions.csv')
        a.sequence_csv = os.path.join(exp, f'{sub}_sequence_predictions.csv')
        a.ma_csv       = os.path.join(exp, f'{sub}_sequence_predictions_ma.csv')

        # Process this folder
        process_folder(a)
        # Clear TensorFlow session
        K.clear_session()


if __name__ == '__main__':
    main()