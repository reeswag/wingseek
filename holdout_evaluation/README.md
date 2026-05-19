# Holdout Evaluation

Evaluates the bat detector with a pre-defined operational threshold, tracking workload reduction and emergence recall on independent holdout data.

## Overview

This script assesses the performance of a trained bat detection model on a held-out test set that was not used during training or threshold calibration. It provides metrics for evaluating the model's ability to reduce manual review workload whilst maintaining high recall for bat detections.

The operational threshold is calibrated externally and is NOT derived from the holdout dataset itself.

A companion plotting script, `holdout_violin_plots.py`, visualises the distribution of aggregated bat probabilities across event-level aggregation methods for bat versus non-bat events.

## Quick Start


#### With Configuration File

```bash
# Using config file
python holdout_evaluation.py \
    --config holdout_evaluation.ini
```
#### With CLI Arguments Only

```bash
# With video duration baseline
python holdout_evaluation.py \
    --csv holdout_evaluation.csv \
    --threshold 0.404 \
    --duration-baseline videos \
    --video-total "41h26m44s" \
    --out-dir results/holdout
```

#### Paper Results Reproduction
```bash
CSV="holdout_evaluation.csv"
SCRIPT="holdout_evaluation.py"
BASE_OUT="./results/"

# --------- Common args ----------
DURATION_BASELINE="videos"
VIDEO_TOTAL="41:26:44"
THRESHOLD=0.404

metrics=(average_probability median_probability min_probability max_probability top_frac_probability)


for metric in "${metrics[@]}"; do
    outdir="${BASE_OUT}/${metric}/"
    mkdir -p "$outdir"

    echo ">>> Running ${metric} => ${outdir}"
    python3 "$SCRIPT" \
      --csv "$CSV" \
      --event-col file \
      --label-col label \
      --notes-col notes \
      --threshold $THRESHOLD \
      --score-col "$metric" \
      --duration-baseline "$DURATION_BASELINE" \
      --video-total "$VIDEO_TOTAL" \
      --out-dir "$outdir"
done

python holdout_violin_plots.py \
    holdout_evaluation.csv \
```

## Input Data Format


### Required Columns

| Column | Description | Example Values |
|--------|-------------|----------------|
| Event identifier | Unique event ID | `video_001.mp4`, `frame_042` |
| Label | Ground truth class | `bat`, `bird`, `other`, `human` |
| Score | Predicted bat probability [0,1] | `0.85`, `0.23` |

### Optional Columns

| Column | Description | Default |
|--------|-------------|---------|
| `notes` | Text annotations, here used to highlight emergence events | Empty string |
| `duration_s` | Event duration in seconds | `1.0` |


## Ground Truth Label Handling

### Binary Classification
- **Positive class**: Any event labelled as "bat"
- **Negative class**: All other events (bird, human, other)

### Emergence Detection
Events are flagged as emergences if:
1. The ground truth label is "bat", AND
2. The notes field contains emergence-related text

## Configuration Options

### Configuration File Parameters

All parameters can be set via a `.ini` configuration file. CLI arguments override config file values.

#### Input Columns

| Parameter | Description | Default |
|-----------|-------------|---------|
| `event_col` | Event identifier column name | `file` |
| `label_col` | Ground truth label column name | `label` |
| `notes_col` | Free-text notes column name | `notes` |
| `score_col` | Predicted probability column name | `top_frac_probability` |

#### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_thresholds` | Max threshold points for curves | `5000` |

#### Output

| Parameter | Description | Default |
|-----------|-------------|---------|
| `out_dir` | Output directory for results | `output` |

#### Duration Baseline

| Parameter | Description | Default |
|-----------|-------------|---------|
| `baseline` | Workload denominator (events/videos) | `videos` |
| `video_total` | Total video duration (required if baseline=videos) | None |

## Command-Line Arguments

### Required Arguments

**`--config PATH`** (optional but recommended)
- Path to `.ini` configuration file
- Provides default values that can be overridden via CLI
- Example: `--config my_holdout_config.ini`

**`--csv PATH`**
- Path to input CSV file with predictions and ground truth
- Must contain columns for events, labels, scores, and optionally notes/durations

**`--threshold FLOAT`**
- Operational threshold for bat classification (0.0 to 1.0)
- Should be derived from cross-validation or separate calibration set
- Events with `score >= threshold` are classified as "bat"
- Example: `--threshold 0.40`

### Optional Arguments

**`--event-col NAME`** (default: `file`)
- Name of column containing event identifiers
- Example: `--event-col filename`

**`--label-col NAME`** (default: `label`)
- Name of column containing ground truth labels
- Example: `--label-col ground_truth`

**`--score-col NAME`** (default: `average_probability`)
- Name of column containing predicted bat probabilities
- Must contain values in range [0, 1]
- Example: `--score-col bat_score`

**`--notes-col NAME`** (default: `notes`)
- Name of column containing free-text annotations
- Used only for emergence detection
- Example: `--notes-col comments`

**`--out-dir PATH`** (default: `output`)
- Directory where all results will be saved
- Created automatically if it doesn't exist
- Example: `--out-dir results/experiment_001`

**`--max-thresholds INT`** (default: `5000`)
- Maximum number of threshold points for curve generation
- Reduces computation time for very large datasets
- Example: `--max-thresholds 1000`

### Duration Baseline Arguments

**`--duration-baseline CHOICE`** (default: `events`)
- Method for calculating duration-based workload reduction
- Choices:
  - `events`: Use sum of event durations as denominator (100% = review all detected events)
  - `videos`: Use total source video duration as denominator (100% = review all footage)
- Example: `--duration-baseline videos`

**`--video-total DURATION`** (default: `None`)
- Total source video duration (required when `--duration-baseline videos`)
- Accepts multiple formats:
  - `HH:MM:SS` — e.g., `41:26:44`
  - `MM:SS` — e.g., `90:30`
  - Time with units — e.g., `41h26m44s`, `41h 26m 44s`, `90m`, `1.5h`
  - Seconds — e.g., `149204` (plain number)
- Examples:
  - `--video-total "41:26:44"`
  - `--video-total "41h26m44s"`
  - `--video-total 149204`

## Output Files

All outputs are saved to the directory specified by `--out-dir`:

### Metrics and Data

**`threshold_metrics.csv`**
- Complete threshold sweep from 0.0 to 1.0
- Contains metrics at each threshold for downstream analysis
- Columns:
  - `threshold`: Classification threshold value
  - `tp`, `fp`, `fn`, `tn`: Confusion matrix counts
  - `precision`, `recall`, `specificity`, `f1`, `f2`: Classification metrics
  - `events_to_review`, `pct_events_to_review`: Event-based workload
  - `duration_to_review_s`, `pct_duration_to_review`: Duration-based workload
  - `emergence_recall`: Recall specifically for emergence events

**`summary_table.csv` / `.md` / `.tex`**
- Summary metrics at the operational threshold
- Available in three formats:
  - CSV: Machine-readable tabular format
  - Markdown: Human-readable table for documentation
  - LaTeX: Publication-ready table

### Visualisations

**`roc.png`**
- ROC curve (Receiver Operating Characteristic)
- Shows trade-off between true positive rate and false positive rate
- Includes AUC (Area Under Curve) score
- Threshold-free visualisation (does not depend on operational threshold)

**`pr.png`**
- Precision-Recall curve
- Shows trade-off between precision and recall
- Includes AP (Average Precision) score
- More informative than ROC for imbalanced datasets

**`workload_recall.png`**
- Dual-panel plot showing:
  - Left: Event-based workload reduction vs recall
  - Right: Duration-based workload reduction vs recall
- Shows operating point at the specified threshold

**Review Composition Plots:**

**`review_composition_vs_threshold_events.png`**
- Stacked area chart showing composition of review queue by event count
- X-axis: Threshold value (0 to 1)
- Y-axis: Absolute number of events to review
- Coloured areas: bat, bird, other_not_bat

**`review_composition_vs_threshold_duration.png`**
- Same as above but using duration (seconds) instead of event count

**`review_composition_at_operating_point_events.png`**
- Stacked bar chart at the operational threshold (event count)
- Shows absolute counts and percentages for each subtype

**`review_composition_at_operating_point_duration.png`**
- Same as above but using duration instead of event count


## Understanding Workload Metrics

### Event-Based Workload

**Definition:** Fraction of detected events that must be manually reviewed

**Calculation:**
```
Events to review = Number of events with score >= threshold
Workload % = (Events to review / Total events) × 100
```
### Duration-Based Workload

**Two modes available:**

#### Mode 1: Events Baseline (default)
```
Duration to review = Sum of durations for events with score >= threshold
Workload % = (Duration to review / Total event duration) × 100
```
- Denominator = sum of all event durations
- 100% = review all detected events

#### Mode 2: Videos Baseline
```
Duration to review = Sum of durations for events with score >= threshold
Workload % = (Duration to review / Total video duration) × 100
```
- Denominator = total source video footage
- 100% = review entire video corpus
- Requires `--video-total` argument

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn