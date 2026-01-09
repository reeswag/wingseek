#!/usr/bin/env python3
"""
Event Clip Extraction with Bounding Box Annotation
===================================================

A pipeline for extracting video clips from annotated motion events and
overlaying bounding box annotations. Supports both fast stream-copy mode
and annotated mode with OpenCV-rendered bounding boxes.

Key Features
------------
- CSV-based video path mapping for flexible source file organisation
- Event window merging with configurable time buffer
- Bounding box overlay with label-specific colour coding
- Audio preservation through ffmpeg muxing
- Fallback audio transcoding for compatibility
- Batch processing of multiple video IDs
- Per-video output organisation

Workflow
--------
1. Load motion_labels.csv with annotated events
2. Map video IDs to source files via CSV or directory search
3. Merge overlapping event windows with time buffer
4. For each merged window:
   - Extract frames via OpenCV (if annotating)
   - Draw bounding boxes from motion_log CSV
   - Mux original audio segment
   - Save clip to video-specific output directory
5. Update events CSV with output clip paths

Directory Structure
-------------------
Input (analysis_dir):
    motion_labels.csv              # Event annotations
    motion_log_<video_id>.csv       # Frame-level bounding boxes (optional)

Output (output_dir):
    <video_id>/
        <video_id>_clip_0_<minutes>.mp4
        <video_id>_clip_1_<minutes>.mp4
        events_with_output_paths.csv

Date: 17/12/2025
"""

import argparse
import ast
import collections
import csv
import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import shlex
import json

import cv2
import pandas as pd


# =============================================================================
# UTILITIES
# =============================================================================

def str2bool(v):
    """
    Convert string or boolean to boolean value.

    Parameters
    ----------
    v : str or bool
        Value to convert.

    Returns
    -------
    bool
        True if v is '1', 'true', 't', 'yes', or 'y' (case-insensitive),
        False otherwise.
    """
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1","true","t","yes","y")


def load_video_mapping_from_csv(csv_path: str, video_dir: str, allowed_exts: List[str]) -> Tuple[List[str], List[str]]:
    """
    Load video paths and IDs from a CSV file.

    Reads a CSV with 'input' and 'output' columns where 'input' contains
    video file paths (absolute or relative to video_dir) and 'output'
    contains the corresponding video IDs.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with 'input' and 'output' columns.
    video_dir : str
        Base directory for resolving relative paths in CSV.
    allowed_exts : List[str]
        List of allowed video extensions (e.g., ['mp4', 'avi', 'mov']).

    Returns
    -------
    tuple
        (input_videos, video_ids) - Lists of absolute video paths and their IDs.

    Raises
    ------
    ValueError
        If CSV doesn't contain required 'input' and 'output' columns.

    Notes
    -----
    Relative paths in the CSV are resolved relative to video_dir.
    Entries with unsupported extensions or missing files are skipped
    with warnings printed to console.
    """
    input_videos = []
    video_ids = []

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

    return input_videos, video_ids


def update_annotation_df(annotation_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite annotation_df['label'] using event_df labels via a key match:
    annotation_df['gif_id'] (preferred) or ['file'] == event_df['file'].
    """
    ann = annotation_df.copy()
    if event_df.empty:
        return ann
    mapping = dict(zip(event_df['file'].astype(str), event_df['label'].astype(str)))
    key_col = 'gif_id' if 'gif_id' in ann.columns else ('file' if 'file' in ann.columns else None)
    if key_col is None:
        # nothing we can map on; return as-is
        return ann
    ann[key_col] = ann[key_col].astype(str)
    if 'label' not in ann.columns:
        ann['label'] = ""
    ann['label'] = ann[key_col].map(mapping).fillna(ann['label'])
    return ann


def get_fps(path: str) -> float:
    """Use ffprobe to get the video's frame rate as a float."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    out = subprocess.check_output(cmd).decode().strip()
    if "/" in out:
        num, den = out.split('/')
        return float(num) / float(den)
    return float(out)


def yolo_to_pixel(box, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """Convert YOLO (xc, yc, w, h) to pixel (x_min, y_min, x_max, y_max)."""
    x_center, y_center, width, height = box
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return x_min, y_min, x_max, y_max


def bb_parser(bb_string: str):
    """
    Parse the 'bb' cell safely. Expected shape like:
      "((80, 465, 12, 11), (0.044792, 0.435648, 0.00625, 0.010185))"
    """
    try:
        tup = ast.literal_eval(bb_string)
        # If it's a pair, second is YOLO
        if isinstance(tup, (list, tuple)) and len(tup) == 2:
            return tup[1]
        # If already YOLO 4-tuple
        if isinstance(tup, (list, tuple)) and len(tup) == 4:
            return tuple(tup)
    except Exception:
        pass
    return None


def color_for_label(label: str) -> Tuple[int, int, int]:
    """BGR colour mapping for labels."""
    if label == "Bat":
        return (0, 255, 0)      # green
    if label == "Bird":
        return (255, 0, 0)      # blue
    if label == "Human":
        return (0, 165, 255)    # orange
    return (255, 255, 255)      # white default


def derive_video_id_from_file_field(raw_file_field: str) -> Tuple[int, str]:
    """
    motion_labels.csv 'file' typically looks like: "274_0_<video_id...>"
    Return (frame_number, video_id_string).
    """
    parts = str(raw_file_field).split('_')
    if len(parts) < 3:
        raise ValueError(f"Bad file field: {raw_file_field}")
    frame_number = int(parts[0])
    video_id = "_".join(parts[2:])
    return frame_number, video_id


def find_video_matches(base_video_dir: str, video_id: str, exts: List[str]) -> List[str]:
    matches = []
    for ext in exts:
        for variant in (ext.lower(), ext.upper()):
            pattern = os.path.join(base_video_dir, f"{video_id}*{variant}")
            matches.extend(glob.glob(pattern))
    return matches


def find_annotation_csv(analysis_dir: str, video_id: str) -> Optional[str]:
    """
    Try to find motion_log_{id}.csv where {id} is the folder's ID.
    Fallback to any motion_log_* that contains the base name.
    """
    exact = os.path.join(analysis_dir, f"motion_log_{video_id}.csv")
    if os.path.isfile(exact):
        return exact
    # As a last resort, accept a single motion_log_*.csv in the folder
    any_bf = glob.glob(os.path.join(analysis_dir, "motion_log_*.csv"))
    if len(any_bf) == 1:
        return any_bf[0]
    return None


def build_annotation_index(
    ann_df: pd.DataFrame,
    video_id: str,
    annotation_dict: dict = None
) -> Dict[int, List[dict]]:
    """
    Build a mapping: frame_index -> list of rows ({'label', 'bb'} ...).
    Uses 'file' like "274_<...>" to extract the frame index. If an 'id' column exists,
    filter by id == video_id.
    """
    df = ann_df
    if "id" in df.columns:
        df = df[df["id"] == video_id].copy()
        if df.empty:
            return {}

    index: Dict[int, List[dict]] = collections.defaultdict(list)
    for _, row in df.iterrows():
        file_field = str(row.get("file", ""))
        if "_" not in file_field:
            continue
        try:
            frame_idx = int(file_field.split("_", 1)[0])
        except Exception:
            continue

        bb_string = row.get("bb", None)
        label = str(row.get("label", ""))
        if annotation_dict is not None and label not in annotation_dict.keys():  # skip drawing logic..
            continue
        if bb_string is None:
            continue
        yolo_bb = bb_parser(str(bb_string))
        if yolo_bb is None:
            continue
        
        mapped_label = annotation_dict[label] if annotation_dict else label
        index[frame_idx].append({"label": mapped_label, "bb": yolo_bb})
    return index

def run_ffmpeg_args(args, quiet=True):
    base = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error' if quiet else 'info']
    cmd = base + args
    try:
        subprocess.run(cmd, check=True)
        return True, ""
    except subprocess.CalledProcessError:
        # return a shell-copyable command for debugging
        return False, ' '.join(shlex.quote(a) for a in cmd)

def mux_with_source_audio(src_video: str, annotated_video_only: str,
                        start_s: float, end_s: float, out_path: str, quiet=True):
    """
    1) Stream-copy attempt: OK only when audio is MP4-friendly (aac/mp3/ALAC/AC3).
    2) Fallback: re-encode audio to AAC (works for PCM/Opus/Vorbis).
    If both fail, keep video-only so you still get a playable result.
    """
    duration = max(0.0, end_s - start_s)
    if duration <= 0.0005:
        # nothing to do—just move the annotated video
        os.replace(annotated_video_only, out_path)
        return

    # --- Attempt 1: stream copy (works if audio already MP4-compatible) ---
    """
    ok, msg = run_ffmpeg_args([
        # trim the AUDIO input only: put -ss/-t BEFORE -i for input 0
        '-ss', f"{start_s:.3f}", '-t', f"{duration:.3f}", '-i', src_video,
        # annotated video as input 1 (no trimming here)
        '-i', annotated_video_only,
        # map video from input 1, audio from input 0 (optional)
        '-map', '1:v:0', '-map', '0:a:0?',
        '-c:v', 'copy', '-c:a', 'copy',
        '-shortest', '-movflags', '+faststart',
        out_path
    ], quiet=quiet)
    if ok:
        try: os.remove(annotated_video_only)
        except OSError: pass
        return
    """

    #print("[mux] copy failed, retrying with AAC re-encode…")
    # --- Attempt 2: re-encode audio to AAC (robust for PCM/etc.) ---
    ok2, msg2 = run_ffmpeg_args([
        # Accurate trim on the AUDIO input only
        '-ss', f"{start_s:.3f}", '-to', f"{end_s:.3f}", '-i', src_video,
        '-i', annotated_video_only,
        '-map', '1:v:0', '-map', '0:a:0?',
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest', '-movflags', '+faststart',
        out_path
    ], quiet=quiet)
    if ok2:
        try: os.remove(annotated_video_only)
        except OSError: pass
        return

    print("[mux] AAC re-encode also failed; keeping video-only.")
    #if msg:  print("  copy cmd:", msg)
    if msg2: print("  aac  cmd:", msg2)
    # Fallback: ensure you at least get the annotated video
    try:
        os.replace(annotated_video_only, out_path)
    except OSError:
        with open(annotated_video_only, 'rb') as fin, open(out_path, 'wb') as fout:
            fout.write(fin.read())
        try: os.remove(annotated_video_only)
        except OSError: pass

def annotate_and_write_clip(
    video_path: str,
    out_path: str,
    start_s: float,
    end_s: float,
    fps: float,
    ann_index: Optional[Dict[int, List[dict]]] = None,
    draw_frame_index: bool = True
):
    """
    Write an annotated clip that preserves the original audio:
    1) Write a video-only annotated segment to a temp file via OpenCV.
    2) Mux the corresponding audio segment from the source into the final out_path.
    """
    # temp video-only path (same folder)
    temp_video_only = out_path + ".video_only.tmp.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # keep widely-supported codec for temp
    out = cv2.VideoWriter(temp_video_only, fourcc, fps, (width, height))

    start_f = max(0, int(round(start_s * fps)))
    end_f = max(start_f, int(round(end_s * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    current_f = start_f

    while current_f < end_f:
        ret, frame = cap.read()
        if not ret:
            break

        if draw_frame_index:
            cv2.putText(frame, f"Frame: {current_f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if ann_index:
            # If your annotation frames are off-by-one vs. capture index, adjust here:
            rows = ann_index.get(current_f + 1, [])
            for r in rows:
                x_min, y_min, x_max, y_max = yolo_to_pixel(r["bb"], width, height)
                x_min = max(0, min(x_min, width - 1))
                y_min = max(0, min(y_min, height - 1))
                x_max = max(0, min(x_max, width - 1))
                y_max = max(0, min(y_max, height - 1))
                if x_max > x_min and y_max > y_min:
                    color = color_for_label(r["label"])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, r["label"], (x_min, max(0, y_min - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out.write(frame)
        current_f += 1

    out.release()
    cap.release()

    # Mux original audio segment with the annotated video-only clip
    mux_with_source_audio(video_path, temp_video_only, start_s, end_s, out_path)


def ffmpeg_copy_clip(video_path: str, out_path: str, start_s: float, end_s: float):
    duration = max(0.0, end_s - start_s)
    cmd = [
        'ffmpeg', '-y',
        '-ss', f"{start_s:.3f}",
        '-i', video_path,
        '-t', f"{duration:.3f}",
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '128k',
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def merge_event_windows_for_video(subdf: pd.DataFrame, fps: float, buffer_s: float) -> List[dict]:
    """
    subdf: motion rows for a single video_id (must have 'frame_number' and index)
    Returns: list of dicts [{start, end, indices:[row_idx,...]}] with merged windows in seconds.
    """
    windows = []
    for ridx, r in subdf.iterrows():
        t_event = r['frame_number'] / fps
        s = max(0.0, t_event - buffer_s)
        e = t_event + buffer_s
        windows.append((s, e, ridx))
    if not windows:
        return []

    windows.sort(key=lambda x: x[0])  # by start
    merged = []
    cur_s, cur_e, cur_idx = windows[0][0], windows[0][1], [windows[0][2]]
    for s, e, ridx in windows[1:]:
        if s <= cur_e:  # overlap/contiguous
            if e > cur_e:
                cur_e = e
            cur_idx.append(ridx)
        else:
            merged.append({"start": cur_s, "end": cur_e, "indices": cur_idx})
            cur_s, cur_e, cur_idx = s, e, [ridx]
    merged.append({"start": cur_s, "end": cur_e, "indices": cur_idx})
    return merged


# ---------------------------- Main ---------------------------- #

def main():
    """
    Main entry point for the clip extraction pipeline.

    Command-Line Usage
    ------------------
    Basic usage with directory search:
        python annotation_to_clip.py \\
            --analysis-dir /path/to/annotations \\
            --base-video-dir /path/to/videos \\
            --annotations True

    With CSV-based video mapping:
        python annotation_to_clip.py \\
            --analysis-dir /path/to/annotations \\
            --base-video-dir /path/to/videos \\
            --video-mapping-csv mapping.csv \\
            --annotations True

    With label filtering:
        python annotation_to_clip.py \\
            --analysis-dir /path/to/annotations \\
            --base-video-dir /path/to/videos \\
            --annotation_dict labels.json \\
            --annotations True

    Notes
    -----
    If --video-mapping-csv is provided, video paths are read from the CSV
    rather than searched in base-video-dir. The CSV should have 'input'
    (video path) and 'output' (video ID) columns.
    """
    p = argparse.ArgumentParser(
        description="Create/merge event clips from motion_labels.csv and overlay bounding boxes from motion_log_{id}.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with directory search
  python annotation_to_clip.py \\
      --analysis-dir /data/annotations \\
      --base-video-dir /data/videos \\
      --annotations True

  # With CSV mapping
  python annotation_to_clip.py \\
      --analysis-dir /data/annotations \\
      --base-video-dir /data/videos \\
      --video-mapping-csv videos.csv \\
      --annotations True
        """
    )
    p.add_argument('--analysis-dir', required=True,
                   help="Directory containing motion_labels.csv")
    p.add_argument('--base-video-dir', required=True,
                   help="Root dir for video files (used for CSV relative paths or directory search)")
    p.add_argument('--video-mapping-csv', default=None,
                   help="Optional CSV with 'input' (video path) and 'output' (video ID) columns")
    p.add_argument('--annotations', type=str2bool, default=False,
                   help="Annotate clips with bounding boxes from motion_log_{id}.csv (True/False)")
    p.add_argument('--exts', nargs='+',
                   default=['.mp4', '.avi', '.mov', '.mkv'],
                   help="Video extensions to try")
    p.add_argument('--buffer', type=float, default=5.0,
                   help="Seconds of padding before AND after each event")
    p.add_argument('--output-dir', default=None,
                   help="Where to write clips & output CSV (defaults to <analysis_dir>/clips_annot)")
    p.add_argument('--no-frame-index', action='store_true',
                   help="Do not draw frame index overlay")
    p.add_argument('--annotation_dict', default=None,
                   help="JSON file with annotations to extract as clips")
    args = p.parse_args()

    # Load annotation dictionary if provided
    if args.annotation_dict:
        with open(args.annotation_dict) as f:
            ann = json.load(f)
    else:
        ann = None

    analysis_dir = args.analysis_dir
    events_csv_path = os.path.join(analysis_dir, 'motion_labels.csv')
    base_video_dir = args.base_video_dir
    video_exts = [ext.lstrip('.') for ext in args.exts]  # Normalise extensions
    buf = args.buffer
    output_dir = args.output_dir or os.path.join(analysis_dir, 'clips_annot')

    if not os.path.isfile(events_csv_path):
        print(f"ERROR: No motion_labels.csv in {analysis_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # LOAD VIDEO MAPPING
    # -------------------------------------------------------------------------
    video_path_map: Dict[str, str] = {}  # video_id -> absolute_path

    if args.video_mapping_csv and os.path.isfile(args.video_mapping_csv):
        print(f"Loading video mapping from CSV: {args.video_mapping_csv}")
        try:
            input_videos, video_ids = load_video_mapping_from_csv(
                args.video_mapping_csv, base_video_dir, video_exts
            )
            video_path_map = dict(zip(video_ids, input_videos))
            print(f"Loaded {len(video_path_map)} video paths from CSV")
        except Exception as e:
            print(f"ERROR loading video mapping CSV: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if args.video_mapping_csv:
            print(f"Warning: CSV not found: {args.video_mapping_csv}, falling back to directory search")

    # -------------------------------------------------------------------------
    # LOAD AND PARSE EVENTS
    # -------------------------------------------------------------------------
    df = pd.read_csv(events_csv_path, dtype={'file': str, 'label': str})
    df['output_file_path'] = ''

    try:
        parsed = df['file'].apply(derive_video_id_from_file_field)
        df['frame_number'] = parsed.apply(lambda x: x[0])
        df['video_id'] = parsed.apply(lambda x: x[1])
    except Exception as e:
        print(f"ERROR parsing 'file' column: {e}", file=sys.stderr)
        sys.exit(1)

    # Sort by video + frame
    df = df.sort_values(['video_id', 'frame_number']).reset_index(drop=True)

    # Caches
    fps_cache: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # PROCESS EACH VIDEO
    # -------------------------------------------------------------------------
    for video_id, sub in df.groupby('video_id', sort=False):
        # Locate video file
        if video_path_map and video_id in video_path_map:
            # Use CSV mapping
            video_path = video_path_map[video_id]
            print(f"[{video_id}] Using CSV-mapped video: {video_path}")
        else:
            # Fallback to directory search
            matches = find_video_matches(base_video_dir, video_id, [f'.{ext}' for ext in video_exts])
            if not matches:
                df.loc[sub.index, 'output_file_path'] = f"Error: {video_id} not found ({','.join(video_exts)} ± case)"
                print(f"Error: {video_id} not found within {base_video_dir}")
                continue
            video_path = matches[0]
            print(f"[{video_id}] Found video via search: {video_path}")

        video_base = os.path.splitext(os.path.basename(video_path))[0]

        # FPS
        try:
            fps = fps_cache.setdefault(video_path, get_fps(video_path))
        except subprocess.CalledProcessError:
            df.loc[sub.index, 'output_file_path'] = 'Error: ffprobe'
            continue

        # Prepare output dir
        clip_dir = os.path.join(output_dir, f"{video_id}")
        os.makedirs(clip_dir, exist_ok=True)

        # Annotations
        ann_index = None
        if args.annotations:
            ann_csv = find_annotation_csv(analysis_dir, video_id)
            if ann_csv and os.path.isfile(ann_csv):
                print("Successfully found bounding box info...")
                try:
                    ann_df = pd.read_csv(ann_csv)
                    # Override labels using only this video's events
                    ann_df = update_annotation_df(ann_df, sub[['file','label']])
                    ann_index = build_annotation_index(ann_df, video_id, annotation_dict=ann)
                    if ann_index:
                        total_boxes = sum(len(v) for v in ann_index.values())
                        print(f"[{video_base}] Loaded {total_boxes} boxes from {os.path.basename(ann_csv)}")
                    else:
                        print(f"[{video_base}] No usable annotations after override; will stream-copy clips.")
                except Exception as e:
                    print(f"[{video_base}] Failed to load/override annotations: {e}")
                    ann_index = None
            else:
                print("No bounding box info found...") 
            # only keep desired labels.
            if ann:
                sub = sub.loc[sub['label'].isin(ann)].copy()

        # 1) Merge event windows
        windows = merge_event_windows_for_video(sub, fps, buf)

        # 2) Process each merged window and assign its path to all relevant events.
        for nclip, win in enumerate(windows):
            start_s, end_s = win['start'], win['end']
            out_name = f"{video_id}_clip_{nclip}_{int(start_s/60)}.mp4"
            out_path = os.path.join(clip_dir, out_name)

            if ann_index:
                annotate_and_write_clip(
                    video_path, out_path, start_s, end_s,
                    fps=fps, ann_index=ann_index, draw_frame_index=not args.no_frame_index
                )
            else:
                ffmpeg_copy_clip(video_path, out_path, start_s, end_s)

            # assign output path to all relevant events.
            df.loc[win['indices'], 'output_file_path'] = out_path
            print(f"[{video_id}] wrote {out_name}  ({end_s - start_s:.3f}s)")
            
        # 3) Save updated CSV
        out_csv = os.path.join(clip_dir, 'events_with_output_paths.csv')
        df.to_csv(out_csv, index=False)
        print(f"\nDone — events + clip paths written to: {out_csv}")

if __name__ == '__main__':
    main()