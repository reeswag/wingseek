# Holdout Evaluation

Evaluates bat-detector performance on independent holdout data using a pre-defined operational threshold.

The script can either:

* select the event-level probability aggregation method using leave-one-site-out (LOO) validation, or
* evaluate one fixed score column directly.

The operational threshold is supplied externally and is **not derived from the holdout dataset**.

---

## LOO approach

If `score_col` is left blank, the script compares:

* `average_probability`
* `median_probability`
* `max_probability`
* `top_frac_probability`

For each site:

1. The site is held out.
2. Average Precision (AP) is calculated for each aggregation method using the remaining N-1 sites.
3. The method with the highest AP is selected.
4. That method is applied to the held-out site.
5. The process is repeated for every site.

The held-out predictions are then combined internally and used for the final ROC/PR curves, threshold analysis, workload analysis and summary metrics.

ROC-AUC is also reported during method comparison, but **AP is used for method selection**.

---

## Reproducing the reported results

The reported holdout results can be reproduced using the supplied input CSV and configuration file:

```bash
python holdout_evaluation_LOO.py \
    --config holdout_config_loo.ini \
    --csv holdout_evaluation_LOO.csv \
    --threshold 0.404
```

This runs the leave-one-site-out aggregation-method selection described above and writes all reported tables, CSV files and figures to the output directory specified in holdout_config_loo.ini.

The operational threshold of `0.404` was determined independently of the holdout dataset and is supplied here for evaluation.

### CLI only

```bash
python holdout_evaluation_LOO.py \
    --csv holdout_evaluation_LOO.csv \
    --threshold 0.404 \
    --site-col site \
    --duration-baseline videos \
    --video-total "41:26:44" \
    --out-dir output
```

To evaluate a fixed aggregation method instead of using LOO:

```bash
python holdout_evaluation_LOO.py \
    --csv holdout_evaluation_LOO.csv \
    --threshold 0.404 \
    --score-col top_frac_probability
```

---

## Example configuration

```ini
[input]
event_col = file
site_col = site
label_col = label
notes_col = notes

# Leave blank for LOO pooling selection
score_col =


[evaluation]
max_thresholds = 5000


[output]
out_dir = output


[duration]
baseline = videos
video_total = 41:26:44
```

---

## Input data

### Required columns

| Column  | Description                         |
| ------- | ----------------------------------- |
| `file`  | Unique event identifier             |
| `site`  | Independent site/group used for LOO |
| `label` | Ground-truth class                  |

### LOO probability columns

At least two of the following must be present:

| Column                 | Aggregation |
| ---------------------- | ----------- |
| `average_probability`  | Mean        |
| `median_probability`   | Median      |
| `max_probability`      | Maximum     |
| `top_frac_probability` | Top 20%     |

### Optional columns

| Column       | Description                       |
| ------------ | --------------------------------- |
| `notes`      | Used to identify emergence events |
| `duration_s` | Event duration in seconds         |

If `duration_s` is missing, a duration of 1 second is assumed.

---

## Ground-truth handling

Events are treated as:

* **bat** — positive class
* **not bat** — all other labels

Non-bat events are also divided into:

* bird
* other

An event is treated as an emergence when it is labelled as a bat and the `notes` field contains an emergence-related term such as `emergence` or `emerging`.

---

## Operational threshold

The operational threshold is supplied with:

```bash
--threshold 0.404
```

Events are classified as:

```text
score >= threshold  -> send to review
score < threshold   -> auto-discard
```

The threshold should be calibrated independently of the holdout dataset.

---

## Main configuration options

| Parameter        | Description                                | Default  |
| ---------------- | ------------------------------------------ | -------- |
| `event_col`      | Event identifier                           | `file`   |
| `site_col`       | Site/group for LOO                         | `site`   |
| `label_col`      | Ground-truth label                         | `label`  |
| `notes_col`      | Notes column                               | `notes`  |
| `score_col`      | Fixed score column; blank activates LOO    | blank    |
| `max_thresholds` | Maximum threshold points                   | `5000`   |
| `out_dir`        | Output directory                           | `output` |
| `baseline`       | Duration denominator: `events` or `videos` | `events` |
| `video_total`    | Total source-video duration                | None     |

CLI arguments override values in the configuration file.

---

# Outputs

In LOO mode the script produces:

```text
output/
├── loo_pooling_selection_by_fold.csv
├── loo_pooling_stability.csv
├── threshold_metrics.csv
├── summary_table.csv
├── summary_table.md
├── summary_table.tex
├── roc.png
├── pr.png
├── workload_recall.png
├── review_composition_vs_threshold_events.png
├── review_composition_vs_threshold_duration.png
├── review_composition_at_operating_point_events.png
└── review_composition_at_operating_point_duration.png
```

---

## LOO outputs

### `loo_pooling_selection_by_fold.csv`

Shows the performance of every candidate aggregation method for every LOO fold.

Includes:

* `fold`
* `held_out_site`
* `score_col`
* `pooling_method`
* `selection_ap`
* `selection_roc_auc`
* `selection_n`
* `selection_rank`

This is the main output showing how the aggregation method was selected independently of the held-out site.

### `loo_pooling_stability.csv`

Summarises aggregation-method selection across folds.

Includes:

* `score_col`
* `pooling_method`
* `mean_selection_ap`
* `mean_selection_rank`
* `times_selected`

The held-out event predictions are used internally but are not written to a separate CSV.

---

## Performance outputs

### `threshold_metrics.csv`

Contains threshold-specific metrics including:

* TP, FP, TN and FN
* recall
* specificity
* accuracy
* balanced accuracy
* MCC
* event workload reduction
* duration workload reduction
* review composition
* emergence recall

### `summary_table.csv`, `.md`, `.tex`

Summary of performance at the supplied operational threshold, including:

* ROC AUC
* Average Precision
* recall
* specificity
* balanced accuracy
* MCC
* workload reduction
* review composition
* Brier score
* emergence recall

---

## Figures

### `roc.png`

ROC curve and ROC AUC.

### `pr.png`

Precision–Recall curve and Average Precision.

### `workload_recall.png`

Recall versus percentage of events or duration automatically discarded.

### `review_composition_vs_threshold_events.png`

Composition of the review queue across thresholds by event count.

### `review_composition_vs_threshold_duration.png`

Composition of the review queue across thresholds weighted by event duration.

### `review_composition_at_operating_point_events.png`

Review composition at the operational threshold by event count.

### `review_composition_at_operating_point_duration.png`

Review composition at the operational threshold weighted by duration.

---

## Workload metrics

### Event-based

```text
Review fraction =
events sent to review / total events

Scrubbed fraction =
1 - review fraction
```

### Duration-based

With:

```ini
baseline = events
```

the denominator is the summed duration of all detected events.

With:

```ini
baseline = videos
```

the denominator is the total duration of the original source videos.

---

## Violin plots

A companion script, `holdout_violin_plots.py`, can be used to visualise the distribution of aggregated bat probabilities for bat versus non-bat events across the different aggregation methods.

Run with:

```bash
python holdout_violin_plots.py holdout_evaluation_LOO.csv
```
---

## Requirements

* Python 3.9+
* pandas
* numpy
* matplotlib
* scikit-learn

Install with:

```bash
pip install pandas numpy matplotlib scikit-learn
```