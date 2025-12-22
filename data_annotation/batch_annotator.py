#!/usr/bin/env python3
"""
Batch Motion Sequence Annotation and Dataset Preparation
=========================================================

A pipeline for processing manually annotated motion sequences from the HTML
grid interface and organising them into class-specific directories for model
training. The script reads annotation CSVs, applies label mapping rules, and
copies frames and GIFs to a structured output directory.

Key Features
------------
- Batch processing of multiple video directories
- Flexible annotation dictionary via JSON configuration
- Special handling for 'unclear' and 'shadow' notes
- Optional exclusion of default-labelled sequences
- Progress tracking with tqdm
- Automatic directory structure creation
- Both frame crops and GIF sequences copied to class folders

Workflow
--------
1. Load motion_labels.csv from each video directory
2. Apply special note-based label transformations (unclear, shadow)
3. Propagate labels from sequences to individual frames
4. Copy annotated frames to class-specific subdirectories
5. Copy annotated GIFs to class-specific subdirectories
6. Generate summary statistics per video

Directory Structure
-------------------
Input (analysis_dir):
    video_id/
        motion_labels.csv              # Manual annotations from HTML grid
        motion_log_<video_id>.csv       # Frame-to-sequence mapping
        motion_frames/                 # Full motion frames
        motion_frames_unpadded/        # Cropped motion frames
        gifs/                          # Motion sequence GIFs

Output (output_dir):
    motion_frames/<label>/             # Full frames organised by class
    crops/<label>/                     # Cropped frames organised by class
    gifs/<label>/                      # GIF sequences organised by class

Author: William Rees
Date: 17/12/2025
"""

import os
import sys
import shutil
import argparse
import json
import pandas as pd
from tqdm import tqdm


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Default annotation dictionary mapping internal keys to display names
DEFAULT_ANNOTATION_DICT = {
    'bat': 'Bat',
    'bat_shadow': 'Shadow',
    'unclear_bat': 'Unclear',
    'bird': 'Bird',
    'other': 'Other',
    'human': 'Human'
}

# Default pairs of (source_directory, destination_directory) for frame copying
DEFAULT_FRAME_PAIRS = [
    ("motion_frames", "motion_frames"),
    ("motion_frames_unpadded", "crops")
]


# =============================================================================
# LABEL PROCESSING
# =============================================================================

def update_annotations(df, identifier, category_gifs, annotation_dict):
    """
    Assign labels to DataFrame rows based on category membership.

    Takes a DataFrame and updates its 'label' column based on which
    sequences (gif_ids) appear in each category. Prints a summary of
    label distribution using human-readable names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'gif_id' and 'label' columns to update.
    identifier : str
        Descriptive name for this dataset (e.g., "GIFs" or "Frames")
        used in printed summary.
    category_gifs : dict
        Mapping from label_key to DataFrame containing sequences
        with that label. Structure: {label_key: DataFrame with 'file' column}.
    annotation_dict : dict
        Mapping from internal label keys to human-readable display names
        for summary output.

    Returns
    -------
    None
        Modifies df in place.

    Notes
    -----
    Labels are assigned by checking if each gif_id appears in the
    category's file list. If a gif_id appears in multiple categories,
    the last one processed will take precedence.
    """
    # Change label column to string type
    df["label"] = df["label"].astype("string")
    
    # Assign labels per category
    for label_key in annotation_dict.keys():
        files = category_gifs.get(label_key, pd.DataFrame()).get('file', [])
        df.loc[df['gif_id'].isin(files), 'label'] = label_key

    # Print summary statistics
    print(f"\nSummary of Annotations for: {identifier}")
    print("-" * 40)
    for label_key, human_name in annotation_dict.items():
        count = df[df['label'] == label_key].shape[0]
        if count > 0:  # Only show non-zero categories
            print(f"  {human_name}s: {count}")


# =============================================================================
# FILE COPYING UTILITIES
# =============================================================================

def copy_annotated_frames(motion_log_df, input_dir, output_dir, pairs,
                         default_label, exclude_default):
    """
    Copy annotated frames to class-specific output directories.

    Iterates through labelled frames and copies them from source directories
    to destination directories organised by class label.

    Parameters
    ----------
    motion_log_df : pd.DataFrame
        DataFrame with 'file', 'label' columns indicating frame annotations.
    input_dir : str
        Root directory containing source frame directories.
    output_dir : str
        Root directory for organised output structure.
    pairs : list of tuple
        List of (source_subdir, dest_subdir) pairs indicating which
        directories to copy from and to. For example:
        [("motion_frames", "motion_frames"), ("motion_frames_unpadded", "crops")]
    default_label : str
        The default label assigned to unlabelled sequences.
    exclude_default : bool
        If True, skip copying frames with the default label.

    Returns
    -------
    None
        Copies files to disk and prints progress via tqdm.

    Notes
    -----
    Creates destination directories automatically as needed.
    Skips files that don't exist or can't be copied, logging warnings
    via tqdm.write().
    """
    # Filter to labelled frames
    rows = motion_log_df.dropna(subset=['label'])
    if exclude_default:
        rows = rows[rows['label'] != default_label]

    total_ops = len(rows) * len(pairs)

    with tqdm(total=total_ops, desc="Copying labelled frames", unit="file") as pbar:
        for _, row in rows.iterrows():
            file = row['file']
            label = row['label']

            for src_base, dst_base in pairs:
                src_path = os.path.join(input_dir, src_base, f"{file}.png")
                dst_path = os.path.join(output_dir, dst_base, label, f"{file}.png")

                # Create destination directory if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    tqdm.write(f"[WARN] Skipping {src_path} -> {dst_path}: {e}")
                finally:
                    pbar.update(1)


def copy_annotated_gifs(gif_df, input_dir, output_dir, default_label, exclude_default):
    """
    Copy annotated GIF sequences to class-specific output directories.

    Parameters
    ----------
    gif_df : pd.DataFrame
        DataFrame with 'gif_id' and 'label' columns.
    input_dir : str
        Root directory containing 'gifs' subdirectory.
    output_dir : str
        Root directory for organised output structure.
    default_label : str
        The default label assigned to unlabelled sequences.
    exclude_default : bool
        If True, skip copying GIFs with the default label.

    Returns
    -------
    None
        Copies GIF files to disk.

    Notes
    -----
    Currently only copies from 'gifs' subdirectory. Additional source
    directories (fg_gifs, mask_gifs) can be enabled by uncommenting
    the relevant pairs in the source code.
    """
    for _, row in gif_df.iterrows():
        file = row['gif_id']
        label = row['label']

        # Skip default label if requested
        if exclude_default and label == default_label:
            continue

        # Define source/destination pairs for GIFs
        # Uncomment additional pairs if needed
        gif_pairs = [
            ("gifs", "gifs"),
            # ("fg_gifs", "fg_gifs"),
            # ("mask_gifs", "mask_gifs"),
        ]

        for src_base, dst_base in gif_pairs:
            src_path = os.path.join(input_dir, src_base, f"{file}.gif")
            dst_path = os.path.join(output_dir, dst_base, label, f"{file}.gif")

            # Create destination directory if needed
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"[WARN] Could not copy GIF {src_path}: {e}")


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_video(video_id, analysis_dir, output_dir, annotation_dict,
                 default_label, exclude_default, frame_pairs):
    """
    Process annotations for a single video directory.

    Orchestrates the complete workflow for one video: loading annotations,
    applying label transformations, updating frame labels, and copying
    organised files to output directories.

    Parameters
    ----------
    video_id : str
        Name of the video directory to process.
    analysis_dir : str
        Parent directory containing video subdirectories.
    output_dir : str
        Root directory for organised output structure.
    annotation_dict : dict
        Mapping from label keys to human-readable names.
    default_label : str
        Default label for unlabelled sequences.
    exclude_default : bool
        Whether to exclude default-labelled items from output.
    frame_pairs : list of tuple
        List of (source_dir, dest_dir) pairs for frame copying.

    Returns
    -------
    None
        Processes annotations and copies files to disk.

    Raises
    ------
    FileNotFoundError
        If required CSV files are missing.
    ValueError
        If annotation CSV has unexpected format.

    Notes
    -----
    Creates an updated CSV file (motion_log_<video_id>_updated.csv) with
    propagated labels in the input directory.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {video_id}")
    print(f"{'='*60}")

    input_dir = os.path.join(analysis_dir, video_id)

    # -------------------------------------------------------------------------
    # LOAD AND TRANSFORM ANNOTATIONS
    # -------------------------------------------------------------------------
    # Load motion_labels.csv (contains 'file' and 'label' from HTML grid)
    annotations_path = os.path.join(input_dir, "motion_labels.csv")
    if not os.path.exists(annotations_path):
        print(f"[WARN] Missing motion_labels.csv in {input_dir}, skipping.")
        return

    annotations = pd.read_csv(annotations_path)

    # Handle special notes: 'unclear' -> 'unclear_bat', 'shadow' -> 'bat_shadow'
    unclear_mask = annotations['notes'].fillna('').str.strip().str.casefold().eq('unclear')
    shadow_mask = annotations['notes'].fillna('').str.strip().str.casefold().eq('shadow')
    annotations.loc[unclear_mask, 'label'] = 'unclear_bat'
    annotations.loc[shadow_mask, 'label'] = 'bat_shadow'

    # Build category-to-DataFrame mapping for label propagation
    category_gifs = {
        label_key: annotations[annotations['label'] == label_key]
        for label_key in annotation_dict.keys()
    }

    # -------------------------------------------------------------------------
    # PROCESS GIF LABELS
    # -------------------------------------------------------------------------
    gif_dir = os.path.join(input_dir, "gifs")
    if not os.path.isdir(gif_dir):
        print(f"[WARN] Missing gifs directory in {input_dir}, skipping GIF processing.")
        gif_files = []
    else:
        # Sort GIFs by first numeric component for consistent ordering
        gif_files = sorted(
            [f.split('.')[0] for f in os.listdir(gif_dir) if f.endswith('.gif')],
            key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0
        )

    # Create DataFrame with default labels
    gif_df = pd.DataFrame(gif_files, columns=['gif_id'])
    gif_df['label'] = default_label

    # Update with manual annotations
    update_annotations(gif_df, "GIFs", category_gifs, annotation_dict)

    # -------------------------------------------------------------------------
    # PROCESS FRAME ANNOTATIONS
    # -------------------------------------------------------------------------
    motion_csv = os.path.join(input_dir, f"motion_log_{video_id}.csv")
    if not os.path.exists(motion_csv):
        print(f"[WARN] Missing motion_log CSV in {input_dir}, skipping frame processing.")
        return

    motion_log_df = pd.read_csv(motion_csv)

    # Propagate labels from sequences to individual frames
    update_annotations(motion_log_df, "Frames", category_gifs, annotation_dict)

    # Save updated frame annotations
    output_motion_csv = os.path.join(input_dir, f"motion_log_{video_id}_updated.csv")
    motion_log_df.to_csv(output_motion_csv, index=False)
    print(f"\nSaved updated frame labels to: {output_motion_csv}")

    # -------------------------------------------------------------------------
    # COPY ANNOTATED FILES TO OUTPUT
    # -------------------------------------------------------------------------
    # Copy frames to class-specific directories
    copy_annotated_frames(
        motion_log_df, input_dir, output_dir, frame_pairs,
        default_label, exclude_default
    )

    # Copy GIFs to class-specific directories
    copy_annotated_gifs(
        gif_df, input_dir, output_dir, default_label, exclude_default
    )

    print(f"\nCompleted processing: {video_id}")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Main entry point for the batch annotation pipeline.

    Parses command-line arguments and processes all video directories in
    the specified analysis directory.

    Command-Line Usage
    ------------------
    Basic usage with defaults:
        python batch_annotator.py \\
            --analysis_dir /path/to/annotated_videos \\
            --output_dir /path/to/organised_dataset

    With custom annotation dictionary:
        python batch_annotator.py \\
            --analysis_dir /path/to/annotated_videos \\
            --output_dir /path/to/organised_dataset \\
            --annotation_dict custom_labels.json \\
            --default_label moth \\
            --exclude_default

    Key Arguments
    -------------
    --analysis_dir : str (required)
        Parent directory containing video subdirectories with annotations.
    --output_dir : str (required)
        Root directory for organised output structure.
    --annotation_dict : str, optional
        Path to JSON file mapping label keys to display names.
        Default uses built-in dictionary.
    --default_label : str
        Label to assign to unlabelled sequences. Default is 'moth'.
    --exclude_default : flag
        If set, exclude default-labelled items from output.

    Annotation Dictionary Format
    ----------------------------
    JSON file should contain a mapping from internal label keys to
    human-readable display names:

    {
        "bat": "Bat",
        "bird": "Bird",
        "human": "Human",
        "other": "Other"
    }

    Notes
    -----
    The script processes all subdirectories in analysis_dir, expecting
    each to contain:
    - motion_labels.csv (from HTML grid export)
    - motion_log_<video_id>.csv (frame-to-sequence mapping)
    - motion_frames/ and motion_frames_unpadded/ (frame directories)
    - gifs/ (GIF sequence directory)
    """
    parser = argparse.ArgumentParser(
        description="Batch annotation processor for motion sequence datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python batch_annotator.py \\
      --analysis_dir /data/annotations \\
      --output_dir /data/training_set

  # Exclude default label and use custom dictionary
  python batch_annotator.py \\
      --analysis_dir /data/annotations \\
      --output_dir /data/training_set \\
      --annotation_dict labels.json \\
      --exclude_default
        """
    )

    # Required arguments
    parser.add_argument(
        '--analysis_dir',
        required=True,
        help="Parent directory containing video subdirectories with annotations"
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Root directory for organised output structure"
    )

    # Optional arguments
    parser.add_argument(
        '--annotation_dict',
        default=None,
        help="Optional JSON file with annotation dictionary mapping label keys to display names"
    )
    parser.add_argument(
        '--default_label',
        default='other',
        help="Label to assign to unlabelled sequences (default: other)"
    )
    parser.add_argument(
        '--exclude_default',
        action='store_true',
        help="If set, exclude default-labelled items from output"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # LOAD CONFIGURATION
    # -------------------------------------------------------------------------
    # Load annotation dictionary from JSON if provided, otherwise use default
    if args.annotation_dict:
        if not os.path.exists(args.annotation_dict):
            print(f"[ERROR] Annotation dictionary not found: {args.annotation_dict}")
            sys.exit(1)

        try:
            with open(args.annotation_dict, 'r') as f:
                annotation_dict = json.load(f)
            print(f"Loaded annotation dictionary from: {args.annotation_dict}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in annotation dictionary: {e}")
            sys.exit(1)
    else:
        annotation_dict = DEFAULT_ANNOTATION_DICT
        print("Using default annotation dictionary")

    # Display configuration
    print(f"\n{'='*60}")
    print("BATCH ANNOTATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Analysis directory: {args.analysis_dir}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Default label:      {args.default_label}")
    print(f"Exclude default:    {args.exclude_default}")
    print(f"\nAnnotation labels:")
    for key, name in annotation_dict.items():
        print(f"  {key:15s} -> {name}")

    # Validate input directory exists
    if not os.path.isdir(args.analysis_dir):
        print(f"\n[ERROR] Analysis directory does not exist: {args.analysis_dir}")
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # BATCH PROCESSING
    # -------------------------------------------------------------------------
    # Find all video subdirectories
    video_ids = [
        d for d in os.listdir(args.analysis_dir)
        if os.path.isdir(os.path.join(args.analysis_dir, d))
    ]

    if not video_ids:
        print(f"\n[WARN] No subdirectories found in {args.analysis_dir}")
        return

    print(f"\nFound {len(video_ids)} video directories to process")

    # Process each video directory
    for video_id in sorted(video_ids):
        try:
            process_video(
                video_id=video_id,
                analysis_dir=args.analysis_dir,
                output_dir=args.output_dir,
                annotation_dict=annotation_dict,
                default_label=args.default_label,
                exclude_default=args.exclude_default,
                frame_pairs=DEFAULT_FRAME_PAIRS
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to process {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # -------------------------------------------------------------------------
    # COMPLETION SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed videos from: {args.analysis_dir}")
    print(f"Output written to:     {args.output_dir}")
    print("\nOrganised structure:")
    print("  motion_frames/<label>/  - Full motion frames by class")
    print("  crops/<label>/          - Cropped frames by class")
    print("  gifs/<label>/           - GIF sequences by class")


if __name__ == '__main__':
    main()
