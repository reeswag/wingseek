#!/usr/bin/env python3
"""
Bounding Box Dimension Diagnostic Plotting
========================================================

A diagnostic tool for analysing bounding box dimensions in image datasets.
Useful for understanding dataset characteristics before training.

Outputs
---------------
- CSV with image dimensions and metadata
- Hexbin heatmaps for each class showing width/height distributions
- Combined box plot comparing bounding box areas across all classes

Workflow
--------
1. Automatically discover class subdirectories containing PNG images
2. Extract dimensions using PIL for all images in each class
3. Apply optional minimum dimension filter
4. Generate individual diagnostic plots for each class
5. Generate combined box plot comparing all classes
6. Report statistics on excluded images

Directory Structure
-------------------
Input (data_dir):
    1_bat/              # Bat class images
        image1.png
        image2.png
    0_other/            # Other class images
        image1.png
        image2.png

Output (plot_path):
    bounding_box_heatmap_1_bat.png
    bounding_box_heatmap_0_other.png
    bounding_box_area_boxplot_all_classes.png
"""

import os
import sys
import argparse
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# DIRECTORY UTILITIES
# =============================================================================

def discover_class_directories(data_dir):
    """
    Scans the input directory for subdirectories containing PNG images,
    treating each subdirectory as a separate class.

    Parameters
    ----------
    data_dir : str
        Root directory to scan for class subdirectories.

    Returns
    -------
    list of str
        List of subdirectory names (class labels) sorted alphabetically.

    Notes
    -----
    Only includes directories that contain at least one .png file.
    Hidden directories (starting with '.') are excluded.
    """
    class_dirs = []
    for entry in sorted(os.listdir(data_dir)):
        entry_path = os.path.join(data_dir, entry)
        # Skip hidden directories and non-directories
        if entry.startswith('.') or not os.path.isdir(entry_path):
            continue
        # Check if directory contains PNG files
        png_files = [f for f in os.listdir(entry_path) if f.lower().endswith('.png')]
        if png_files:
            class_dirs.append(entry)

    return class_dirs


# =============================================================================
# Plot Generation
# =============================================================================

def create_diagnostic_plot(data_dir, min_dimensions=None, plot_path=None, class_names=None):
    """
    Scans class subdirectories for PNG images, extracts dimensions, applies
    optional filtering, and generates visualisation plots.
    If class_names is not provided, automatically discovers all subdirectories 
    containing PNG files.

    Parameters
    ----------
    data_dir : str
        Root directory containing class subfolders (e.g., '1_bat', '0_other').
    min_dimensions : int or None, optional
        Minimum width and height threshold. Images with either dimension
        below this value are excluded from analysis. Default is None.
    plot_path : str or None, optional
        Directory for saving output plots. If None, plots are not generated.
        Creates directory if it doesn't exist.
    class_names : list of str or None, optional
        List of class subdirectory names to process. If None, automatically
        discovers all subdirectories containing PNG files.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - filepath: Absolute path to image file
        - label: Class label string (e.g., '1_bat')
        - width: Image width in pixels
        - height: Image height in pixels
        - label_int: Integer class label

    Raises
    ------
    ValueError
        If no valid PNG files are found in any class directories.

    Notes
    -----
    Generates separate plots for each class:
    1. Hexbin heatmap showing width vs height distribution for each class
    2. Box plot comparing bounding box areas across all classes

    Images that fail to load are skipped with warnings. Excluded images
    (due to min_dimensions) are counted and reported separately.
    """
    # Discover class directories if not provided
    if class_names is None:
        class_names = discover_class_directories(data_dir)
        print(f"Discovered {len(class_names)} class directories: {class_names}")

    if not class_names:
        raise ValueError(f"No class subdirectories with PNG files found in {data_dir}")

    # Create class-to-integer mapping
    name_to_id = {name: i for i, name in enumerate(class_names)}

    print("Initialising dataframe")
    records = []
    excluded_counts = {label: 0 for label in class_names}
    excluded_counts['total'] = 0

    # -------------------------------------------------------------------------
    # SCAN CLASS DIRECTORIES
    # -------------------------------------------------------------------------
    for label in class_names:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Missing class directory: {class_dir} (skipping)")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.png')]

        for fname in tqdm(image_files, desc=f"Processing {label}", unit="img"):
            filepath = os.path.join(class_dir, fname)

            # Extract image dimensions using PIL
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                tqdm.write(f"[WARN] Could not open {filepath}: {e} (skipping)")
                continue

            # Apply minimum dimension filter
            if min_dimensions is not None:
                if width < min_dimensions or height < min_dimensions:
                    excluded_counts[label] += 1
                    excluded_counts['total'] += 1
                    continue

            records.append({
                'filepath': filepath,
                'label': label,
                'width': width,
                'height': height
            })

    # -------------------------------------------------------------------------
    # BUILD DATAFRAME
    # -------------------------------------------------------------------------
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"No .png files found under {data_dir} in discovered class folders {class_names} "
            f"(after applying min_dimensions={min_dimensions})."
        )

    df['label_int'] = df['label'].map(name_to_id)

    # -------------------------------------------------------------------------
    # GENERATE DIAGNOSTIC PLOTS
    # -------------------------------------------------------------------------
    if plot_path and not df.empty:
        os.makedirs(plot_path, exist_ok=True)

        title_size = 16
        label_size = 12
        tick_size = 10
        annotation_size = 10
        colorbar_label_size = 10

        # Reference line for model input size
        ref_lines = (64,)

        # Heatmap generation
        for class_label in class_names:
            df_class = df[df['label'] == class_label].copy()
            if df_class.empty:
                print(f"[WARN] No data for class {class_label}; skipping heatmap.")
                continue

            # Output path with class name
            output_path = os.path.join(plot_path, f"bounding_box_heatmap_{class_label}.png")

            # Calculate axis limits for this class
            x_min = min(df_class['width'].min(), min(ref_lines))
            x_max = max(df_class['width'].max(), max(ref_lines))
            y_min = min(df_class['height'].min(), min(ref_lines))
            y_max = max(df_class['height'].max(), max(ref_lines))

            # Create single-subplot figure for this class
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # Create hexbin plot with logarithmic colour scale
            hb = ax.hexbin(
                df_class['width'],
                df_class['height'],
                gridsize=80,
                cmap='cividis',
                bins='log',      # Logarithmic binning for better visibility
                mincnt=1,
            )
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("log10(count)", fontsize=colorbar_label_size)
            cb.ax.tick_params(labelsize=tick_size)

            # Calculate statistics for annotation
            pct_w_lt64 = (df_class['width'] < 64).mean() * 100
            pct_h_lt64 = (df_class['height'] < 64).mean() * 100
            pct_both = ((df_class['width'] < 64) & (df_class['height'] < 64)).mean() * 100

            # Add text annotation with statistics
            ax.text(
                0.98, 0.98,
                f"w<64: {pct_w_lt64:.1f}%\nh<64: {pct_h_lt64:.1f}%\nboth<64: {pct_both:.1f}%",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=annotation_size,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Bounding box width", fontsize=label_size)
            ax.set_ylabel("Bounding box height", fontsize=label_size)
            ax.set_title(f"Bounding Box Dimensions: {class_label}", fontsize=title_size)
            ax.grid(False)
            ax.tick_params(labelsize=tick_size)

            fig.tight_layout()
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
            print(f"Saved heatmap for {class_label} to: {output_path}")


        # Generate box plots
        df['area'] = df['width'] * df['height']

        area_output_path = os.path.join(plot_path, "bounding_box_area_boxplot_all_classes.png")
        fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 2), 5))
        df.boxplot(column='area', by='label', ax=ax, showfliers=False)

        # Add reference line for model input size
        model_area = 64 * 64
        ax.axhline(model_area, linestyle='--', linewidth=1, color='red',
                  label='Model input resolution (64×64)')

        ax.set_xlabel("Class", fontsize=label_size)
        ax.set_ylabel("Bounding box area [px²]", fontsize=label_size)
        ax.set_title("Distribution of bounding-box area per class", fontsize=title_size)
        plt.suptitle("")
        ax.tick_params(labelsize=tick_size)
        ax.legend()
        fig.tight_layout()
        fig.savefig(area_output_path, dpi=300)
        plt.close(fig)
        print(f"Saved combined area box plot to: {area_output_path}")

    # -------------------------------------------------------------------------
    # REPORT EXCLUSION STATISTICS
    # -------------------------------------------------------------------------
    if min_dimensions is not None:
        print("\nExcluded files by class (due to min_dimensions):")
        for label in class_names:
            print(f"  {label}: {excluded_counts[label]}")
        print(f"  Total excluded: {excluded_counts['total']}")

    return df


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Main entry point for the bounding box analysis tool.

    Command-Line Usage
    ------------------
    Basic usage:
        python extract_BB_info.py \\
            --input_dir /path/to/dataset

    With dimension filtering and plots:
        python extract_BB_info.py \\
            --input_dir /path/to/dataset \\
            --min_dimensions 64 \\
            --plot_path /path/to/output

    Save results to CSV:
        python extract_BB_info.py \\
            --input_dir /path/to/dataset \\
            --output_csv dimensions.csv

    Key Arguments
    -------------
    --input_dir : str (required)
        Root directory containing class subdirectories with images.
    --min_dimensions : int, optional
        Minimum width and height threshold. Images below this are excluded.
        Default is None (no filtering).
    --plot_path : str, optional
        Directory for saving diagnostic plots. Default is None (no plots).
    --output_csv : str, optional
        Path to save dimension analysis CSV. Default is None (no CSV).

    Notes
    -----
    The script automatically discovers all subdirectories containing PNG files.
    Each subdirectory is treated as a separate class.

    Expected directory structure:
        input_dir/
            1_bat/          # Bat class images
            0_other/        # Other class images
            ...             # Any number of class directories

    Generated plots are saved to plot_path:
        - bounding_box_heatmap_<class_name>.png (one per class)
        - bounding_box_area_boxplot_all_classes.png (combined)
    """
    parser = argparse.ArgumentParser(
        description="Analyse bounding box dimensions and generate diagnostic plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python extract_BB_info.py --input_dir /data/dataset

  # With filtering and plots
  python extract_BB_info.py \\
      --input_dir /data/dataset \\
      --min_dimensions 64 \\
      --plot_path /data/plots

  # Save to CSV
  python extract_BB_info.py \\
      --input_dir /data/dataset \\
      --output_csv dimensions.csv
        """
    )

    # Required arguments
    parser.add_argument(
        '--input_dir',
        required=True,
        help="Root directory containing class subdirectories (auto-discovered)"
    )

    # Optional arguments
    parser.add_argument(
        '--min_dimensions',
        type=int,
        default=None,
        help="Minimum width and height threshold (images below this are excluded)"
    )
    parser.add_argument(
        '--plot_path',
        default=None,
        help="Directory for saving diagnostic plots (default: same as input_dir)"
    )
    parser.add_argument(
        '--output_csv',
        default=None,
        help="Path to save dimension analysis CSV (optional)"
    )

    args = parser.parse_args()

    # Validate input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Use input_dir as plot_path if not specified
    plot_path = args.plot_path if args.plot_path else args.input_dir

    # Display configuration
    print(f"{'='*60}")
    print("BOUNDING BOX DIMENSION ANALYSIS")
    print(f"{'='*60}")
    print(f"Input directory:    {args.input_dir}")
    print(f"Min dimensions:     {args.min_dimensions if args.min_dimensions else 'None (no filtering)'}")
    print(f"Plot output:        {plot_path if args.plot_path or args.input_dir else 'Disabled'}")
    print(f"CSV output:         {args.output_csv if args.output_csv else 'Disabled'}")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # RUN ANALYSIS
    # -------------------------------------------------------------------------
    try:
        df = create_diagnostic_plot(
            data_dir=args.input_dir,
            min_dimensions=args.min_dimensions,
            plot_path=plot_path
        )

        # Save CSV if requested
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"\nSaved dimension data to: {args.output_csv}")

        # Print summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total images processed: {len(df)}")
        print(f"\nDimensions by class:")

        # Get discovered class names from the dataframe
        discovered_classes = sorted(df['label'].unique())
        for label in discovered_classes:
            subset = df[df['label'] == label]
            if not subset.empty:
                print(f"\n  {label}:")
                print(f"    Count:        {len(subset)}")
                print(f"    Width range:  {subset['width'].min()}-{subset['width'].max()} px")
                print(f"    Height range: {subset['height'].min()}-{subset['height'].max()} px")
                print(f"    Mean area:    {(subset['width'] * subset['height']).mean():.0f} px²")

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
