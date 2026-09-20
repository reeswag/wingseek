"""
Holdout Evaluation — Bat Detector Performance
================================================================

Evaluates a bat detector with pre-defined operational threshold, tracking
workload reduction and emergence recall.

- Operational threshold is REQUIRED via --threshold and is NOT derived from the holdout dataset.

Ground Truth Handling:
    bat           : Label says 'bat'
    bird          : Label says 'bird'
    other_not_bat : Everything else not matching above
    is_emergence  : Notes contain 'emerg...' (e.g., emergence, emerging)

Reports:
    * Standard classification metrics at the provided threshold
    * Workload reduction (events + duration) and review composition
    * ROC/PR curves (threshold-free visualisation only)
    * Emergence recall

Inputs (CSV must have):
    - Event identifier column (default: file)
    - Ground-truth label column (default: label) [contains 'bat', 'bird', or other]
    - Free-text notes column (default: notes) for emergences
    - duration_s: Event length in seconds (if missing, assumed 1.0)

Score modes:
    1. Single-score mode: set score_col to one probability column.
    2. LOO pooling mode: leave score_col blank/omitted. The script compares
       average_probability, median_probability, max_probability and
       top_frac_probability using AP on N-1 sites, then evaluates the selected
       pooling method on the omitted site.

Outputs (in out_dir):
    - In LOO mode: loo_pooling_selection_by_fold.csv, loo_pooling_stability.csv
    - threshold_metrics.csv (sweep over thresholds; for visualisation/ops planning)
    - roc.png, pr.png, workload_recall.png
    - review_composition_vs_threshold_events.png
    - review_composition_vs_threshold_duration.png
    - review_composition_at_operating_point_events.png
    - review_composition_at_operating_point_duration.png
    - summary_table.md / .tex / .csv
    - Console summary
"""

import argparse
import configparser
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


# ---------------------- Helpers ----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def percent(x: float) -> str:
    return f"{100.0 * x:.1f}%"

def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        .str.lower()
    )
    return df

def parse_duration_arg(s):
    """
    Parse a duration string into seconds.
    Accepts:
      - 'HH:MM:SS' or 'MM:SS'
      - '41h26m44s', '41h 26m 44s', '90m', '1.5h'
      - plain number (interpreted as seconds)
    Returns 0.0 if s is None or unparsable.
    """
    if not s:
        return 0.0
    s = str(s).strip().lower()

    # HH:MM:SS or MM:SS
    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            parts = []
        if len(parts) == 3:  # H:M:S
            h, m, sec = parts
            return float(max(0.0, 3600*h + 60*m + sec))
        if len(parts) == 2:  # M:S
            m, sec = parts
            return float(max(0.0, 60*m + sec))

    # tokens with units (h/m/s)
    toks = re.findall(r'(\d+(?:\.\d+)?)\s*([hms])', s)
    if toks:
        total = 0.0
        for val, unit in toks:
            v = float(val)
            if unit == "h": total += 3600.0 * v
            elif unit == "m": total += 60.0 * v
            elif unit == "s": total += v
        return float(max(0.0, total))

    # plain number => seconds
    try:
        v = float(s)
        return float(max(0.0, v))
    except Exception:
        return 0.0


# ---------- Label mapping ----------

EMERG_RE = r"\bemerg[a-z]*\b"  # emergence, emerging, etc.

def map_labels(df: pd.DataFrame, label_col: str, score_col: str, event_col: str, notes_col: str) -> pd.DataFrame:
    """
    Map labels and extract emergence information from notes.

    Args:
        df: Input dataframe
        label_col: Column containing labels ('bat', 'bird', etc.)
        score_col: Column containing prediction scores [0,1]
        event_col: Column containing event identifiers
        notes_col: Column containing notes (only used for emergence detection)

    Returns:
        DataFrame with mapped labels and emergence flags
    """
    d = df.copy()
    # Require key columns
    for c in (label_col, score_col, event_col):
        if c not in d.columns:
            raise ValueError(f"Missing required column: {c}")

    if notes_col not in d.columns:
        print(f"[WARN] notes column '{notes_col}' not found; treating all notes as empty.")
        d[notes_col] = ""

    d["label_chr"] = d[label_col].astype(str).str.lower().str.strip()
    d["notes_chr"] = d[notes_col].astype(str).str.lower()

    # Core class flags (from dominant label)
    is_bird = d["label_chr"].str.contains(r"\bbird\b", na=False)
    # Allow boolean/1/true style "bat" positives
    is_bat  = d["label_chr"].str.contains(r"\bbat\b",  na=False) | d["label_chr"].isin(["1","true","yes"])

    # Emergence detection from notes only
    is_emerg = d["notes_chr"].str.contains(EMERG_RE, na=False)

    # Subtyping
    subtype = np.full(len(d), "other_not_bat", dtype=object)
    subtype[np.where(is_bird)] = "bird"
    subtype[np.where(is_bat)] = "bat"

    d["subtype"] = pd.Categorical(
        subtype,
        categories=["bat","bird","other_not_bat"],
        ordered=True
    )

    # Binary truth
    truth_bin = np.where(is_bat.values, "bat", "not_bat")
    d["truth_bin"] = pd.Categorical(truth_bin, categories=["not_bat","bat"], ordered=True)

    # Score
    d["score"] = pd.to_numeric(d[score_col], errors="coerce").astype(float).clip(0,1)

    # Duration (seconds) — default to 1.0 if missing
    if "duration_s" not in d.columns:
        print("[WARN] 'duration_s' column not found; assuming each event has duration_s=1.0 second.")
        d["duration_s"] = 1.0
    d["duration_s"] = pd.to_numeric(d["duration_s"], errors="coerce").fillna(0.0).astype(float)
    d.loc[d["duration_s"] < 0, "duration_s"] = 0.0  # Just in case

    # Standardise event id name
    if event_col != "event":
        d = d.rename(columns={event_col: "event"})

    # Emergence flag (only for bat events)
    d["is_emergence"] = is_emerg & (d["subtype"] == "bat")

    return d


# ---------------------- Metrics ----------------------

@dataclass
class MetricsAtThreshold:
    threshold: float
    tp: int; fp: int; tn: int; fn: int
    recall: float; specificity: float; 
    accuracy: float; bal_acc: float; mcc: float
    # Workload by EVENTS
    review_fraction: float; scrubbed_fraction: float
    review_total: int
    review_bat: int; review_bird: int; review_other: int
    # Workload by DURATION
    total_duration_events: float
    review_fraction_dur: float; scrubbed_fraction_dur: float
    review_total_dur: float
    review_bat_dur: float; review_bird_dur: float; review_other_dur: float
    # Shares for reporting (events-based)
    share_bird_in_review: float; share_bird_in_fp: float
    # Emergence-specific
    emergence_total: int
    emergence_tp: int
    emergence_fn: int
    emergence_recall: float


def _safe_div(a, b):
    return (a / b) if b > 0 else (float("nan") if a > 0 else 0.0)

def metrics_at_threshold(d: pd.DataFrame, t: float, duration_denominator: float) -> MetricsAtThreshold:
    """
    Calculate metrics at a given threshold.

    Args:
        d: DataFrame with predictions and truth
        t: Threshold value
        duration_denominator:
            - If None or <=0: use sum of event durations for fractions
            - If float>0: use that as the denominator for duration-based fractions (e.g., total seconds of source videos)

    Returns:
        MetricsAtThreshold object with all computed metrics
    """
    pred = np.where(d["score"].values >= t, "bat", "not_bat")
    truth = d["truth_bin"].astype(str).values
    subtype = d["subtype"].astype(str).values
    dur = d["duration_s"].values.astype(float)
    is_emerg = d.get("is_emergence", pd.Series(False, index=d.index)).values.astype(bool)

    # Confusion counts (events)
    tp = int(np.sum((pred == "bat")     & (truth == "bat")))
    fp = int(np.sum((pred == "bat")     & (truth == "not_bat")))
    tn = int(np.sum((pred == "not_bat") & (truth == "not_bat")))
    fn = int(np.sum((pred == "not_bat") & (truth == "bat")))

    denom_pos = (tp + fn)
    denom_neg = (tn + fp)
    denom_p   = (tp + fp)
    denom_all = (tp + fp + tn + fn)

    recall      = _safe_div(tp, denom_pos)
    specificity = _safe_div(tn, denom_neg)
    accuracy = _safe_div((tp + tn), denom_all)
    bal_acc  = np.nanmean([recall, specificity])

    # MCC
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = _safe_div(mcc_num, mcc_den)

    # Workload: EVENTS
    review_mask = (pred == "bat")
    review_total = int(np.sum(review_mask))
    review_bat  = int(np.sum(review_mask & (subtype == "bat")))
    review_bird = int(np.sum(review_mask & (subtype == "bird")))
    review_other = int(np.sum(review_mask & (subtype == "other_not_bat")))
    review_fraction   = _safe_div(review_total, denom_all)
    scrubbed_fraction = 1.0 - (review_fraction if np.isfinite(review_fraction) else 0.0)

    # Workload: DURATION
    total_duration_events = float(dur.sum()) or 1e-12
    review_total_dur  = float(dur[review_mask].sum())
    review_bat_dur   = float(dur[review_mask & (subtype == "bat")].sum())
    review_bird_dur  = float(dur[review_mask & (subtype == "bird")].sum())
    review_other_dur = float(dur[review_mask & (subtype == "other_not_bat")].sum())

    denom_dur = total_duration_events if (duration_denominator is None or duration_denominator <= 0) else float(duration_denominator)
    review_fraction_dur   = _safe_div(review_total_dur, (denom_dur or 1e-12))
    scrubbed_fraction_dur = 1.0 - (review_fraction_dur if np.isfinite(review_fraction_dur) else 0.0)

    # Shares (events-based)
    share_bird_in_review = _safe_div(review_bird, (review_total or 1e-12))
    fp_bird  = int(np.sum((pred == "bat") & (truth == "not_bat") & (subtype == "bird")))
    fp_other = int(np.sum((pred == "bat") & (truth == "not_bat") & (subtype == "other_not_bat")))
    share_bird_in_fp = _safe_div(fp_bird, (fp_bird + fp_other))

    # Emergence
    emerg_mask = is_emerg & (truth == "bat")
    emergence_total = int(np.sum(emerg_mask))
    emergence_tp    = int(np.sum((pred == "bat")     & emerg_mask))
    emergence_fn    = int(np.sum((pred == "not_bat") & emerg_mask))
    emergence_recall = _safe_div(emergence_tp, (emergence_tp + emergence_fn))

    return MetricsAtThreshold(
        threshold=t, tp=tp, fp=fp, tn=tn, fn=fn,
        recall=recall, specificity=specificity,
        accuracy=accuracy, bal_acc=bal_acc, mcc=mcc,
        # Events
        review_fraction=review_fraction, scrubbed_fraction=scrubbed_fraction,
        review_total=review_total,
        review_bat=review_bat, review_bird=review_bird, review_other=review_other,
        # Duration
        total_duration_events=total_duration_events,
        review_fraction_dur=review_fraction_dur, scrubbed_fraction_dur=scrubbed_fraction_dur,
        review_total_dur=review_total_dur,
        review_bat_dur=review_bat_dur, review_bird_dur=review_bird_dur, review_other_dur=review_other_dur,
        # Shares (events)
        share_bird_in_review=share_bird_in_review, share_bird_in_fp=share_bird_in_fp,
        # Emergence
        emergence_total=emergence_total,
        emergence_tp=emergence_tp,
        emergence_fn=emergence_fn,
        emergence_recall=emergence_recall
    )


# ---------- Publication tables ----------

def write_publication_tables(out_dir, cm_row, op_t, n, prev,
                             event_total_hours, video_total_hours,
                             duration_baseline, auc_roc, ap_score, brier):
    """Create Markdown and LaTeX summary tables from point estimates."""
    def pct(x):   return f"{100*x:.1f}%"
    def q3(x):    return f"{x:.3f}"

    rows = [
        {"Measure":"Dataset size (events)",      "Value": f"{n}"},
        {"Measure":"Positive prevalence",        "Value": pct(prev)},
        {"Measure":"Total duration (events, h)", "Value": q3(event_total_hours)},
        {"Measure":"Total duration (videos, h)", "Value": q3(video_total_hours)},
        {"Measure":"Duration baseline",          "Value": duration_baseline},

        {"Measure":"ROC AUC",                    "Value": q3(auc_roc)},
        {"Measure":"Average Precision (AP)",     "Value": q3(ap_score)},

        {"Measure":"Operational threshold",      "Value": q3(op_t)},
        {"Measure":"TP / FP / TN / FN",          "Value": f"{cm_row.tp} / {cm_row.fp} / {cm_row.tn} / {cm_row.fn}"},

        {"Measure":"Recall (TPR)",               "Value": pct(cm_row.recall)},
        {"Measure":"Specificity (TNR)",          "Value": pct(cm_row.specificity)},
        {"Measure":"False Positive Rate (FPR)",  "Value": pct(1.0 - cm_row.specificity)},
        {"Measure":"Balanced accuracy",          "Value": pct(cm_row.bal_acc)},
        {"Measure":"Matthews corr. coeff. (MCC)","Value": q3(cm_row.mcc)},

        {"Measure":"Scrubbed fraction (events)",   "Value": pct(cm_row.scrubbed_fraction)},
        {"Measure":"Review fraction (events)",     "Value": pct(cm_row.review_fraction)},
        {"Measure":"Scrubbed fraction (duration)", "Value": pct(cm_row.scrubbed_fraction_dur)},
        {"Measure":"Review fraction (duration)",   "Value": pct(cm_row.review_fraction_dur)},

        {"Measure":"Share bird in review (events)",         "Value": pct(cm_row.share_bird_in_review)},
        {"Measure":"Share bird in false positives (events)","Value": pct(cm_row.share_bird_in_fp)},

        {"Measure":"Brier score",               "Value": q3(brier)},

        # Emergence reporting
        {"Measure":"Emergence events",          "Value": f"{cm_row.emergence_total}"},
        {"Measure":"Emergence recall (at OP)",  "Value": pct(cm_row.emergence_recall)},
    ]

    df = pd.DataFrame(rows, columns=["Measure","Value"]).fillna("")

    md_path = os.path.join(out_dir, "summary_table.md")
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"Wrote: {md_path}")

    latex = df.to_latex(index=False, escape=True, longtable=False, bold_rows=False,
                        caption="Operating-point performance, workload savings, and emergence recall.",
                        label="tab:summary", column_format="ll")
    tex_path = os.path.join(out_dir, "summary_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"Wrote: {tex_path}")

    csv_path = os.path.join(out_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")



# ---------------------- LOO pooling selection ----------------------

POOLING_SCORE_COLUMNS = [
    "average_probability",
    "median_probability",
    "max_probability",
    "top_frac_probability",
]

POOLING_LABELS = {
    "average_probability": "Mean",
    "median_probability": "Median",
    "max_probability": "Maximum",
    "top_frac_probability": "Top-k (top 20%)",
}

def _score_metrics(d: pd.DataFrame, score_col: str):
    """Threshold-free AP and ROC-AUC for one score column."""
    scores = pd.to_numeric(d[score_col], errors="coerce")
    mask = scores.notna()
    if not mask.any():
        return float("nan"), float("nan"), 0
    y = (d.loc[mask, "truth_bin"].astype(str) == "bat").astype(int).values
    s = scores.loc[mask].clip(0, 1).values
    ap = float(average_precision_score(y, s)) if np.any(y == 1) else float("nan")
    roc_auc = float("nan")
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = float(auc(fpr, tpr))
    return ap, roc_auc, int(mask.sum())


def run_loo_pooling_selection(df: pd.DataFrame, site_col: str, out_dir: str):
    """
    Select the pooling method by AP on N-1 sites, then apply that selected
    method to the omitted site.

    Only two LOO summary CSVs are written:
      - loo_pooling_selection_by_fold.csv
      - loo_pooling_stability.csv

    Out-of-fold predictions are retained in memory because they are required
    for the downstream threshold metrics, plots, and publication summaries.
    """
    if site_col not in df.columns:
        raise ValueError(
            f"LOO pooling mode requires site column '{site_col}'. "
            f"Set [input] site_col in the INI or pass --site-col."
        )

    candidates = [c for c in POOLING_SCORE_COLUMNS if c in df.columns]
    if len(candidates) < 2:
        raise ValueError(
            "LOO pooling mode requires at least two pooling score columns. "
            f"Expected any of: {', '.join(POOLING_SCORE_COLUMNS)}. "
            f"Found: {', '.join(candidates) if candidates else 'none'}."
        )

    sites = [x for x in pd.unique(df[site_col]) if pd.notna(x)]
    if len(sites) < 2:
        raise ValueError(f"LOO pooling mode requires >=2 sites; found {len(sites)}.")

    print(f"\nLOO pooling selection across {len(sites)} sites")
    print("Candidate pooling columns: " + ", ".join(candidates))
    print("Selection metric: Average Precision (AP) on N-1 sites\n")

    selection_rows = []
    selected_methods = []
    oof_parts = []

    # Deterministic tie-break follows POOLING_SCORE_COLUMNS order.
    order = {c: i for i, c in enumerate(POOLING_SCORE_COLUMNS)}

    for fold_i, held_site in enumerate(sites, start=1):
        select_df = df.loc[df[site_col] != held_site].copy()
        eval_df = df.loc[df[site_col] == held_site].copy()

        method_results = []
        for c in candidates:
            ap, roc_auc, n_valid = _score_metrics(select_df, c)
            method_results.append((c, ap, roc_auc, n_valid))
            selection_rows.append({
                "fold": fold_i,
                "held_out_site": held_site,
                "score_col": c,
                "pooling_method": POOLING_LABELS.get(c, c),
                "selection_ap": ap,
                "selection_roc_auc": roc_auc,
                "selection_n": n_valid,
            })

        valid = [r for r in method_results if np.isfinite(r[1])]
        if not valid:
            raise ValueError(f"No finite AP values available when holding out site {held_site}.")

        selected = sorted(valid, key=lambda r: (-r[1], order.get(r[0], 999)))[0]
        selected_col, selection_ap, _, _ = selected
        selected_methods.append(selected_col)

        # Apply the method selected from the N-1 sites to the omitted site.
        # These OOF scores are not written as a separate CSV; they feed the
        # standard evaluation outputs later in the script.
        eval_scores = pd.to_numeric(eval_df[selected_col], errors="coerce")
        keep = eval_scores.notna()
        dropped = int((~keep).sum())
        eval_df = eval_df.loc[keep].copy()
        eval_df["score"] = eval_scores.loc[keep].clip(0, 1).values
        eval_df["loo_fold"] = fold_i
        eval_df["held_out_site"] = held_site
        eval_df["selected_score_col"] = selected_col
        eval_df["selected_pooling_method"] = POOLING_LABELS.get(selected_col, selected_col)
        oof_parts.append(eval_df)

        if dropped:
            print(f"Fold {fold_i}: held out {held_site} | selected "
                  f"{POOLING_LABELS.get(selected_col, selected_col)} "
                  f"(selection AP={selection_ap:.3f}) | dropped {dropped} missing held-out scores")
        else:
            print(f"Fold {fold_i}: held out {held_site} | selected "
                  f"{POOLING_LABELS.get(selected_col, selected_col)} "
                  f"(selection AP={selection_ap:.3f})")

    selection_df = pd.DataFrame(selection_rows)
    oof_df = pd.concat(oof_parts, axis=0).sort_index().copy()

    # Rank methods within each fold and summarise selection stability.
    selection_df["selection_rank"] = selection_df.groupby("fold")["selection_ap"].rank(
        ascending=False, method="min"
    )

    stability = (
        selection_df.groupby(["score_col", "pooling_method"], as_index=False)
        .agg(
            mean_selection_ap=("selection_ap", "mean"),
            mean_selection_rank=("selection_rank", "mean"),
        )
    )
    counts = pd.Series(selected_methods).value_counts().rename("times_selected")
    stability["times_selected"] = stability["score_col"].map(counts).fillna(0).astype(int)
    stability = stability.sort_values(
        ["times_selected", "mean_selection_rank", "mean_selection_ap"],
        ascending=[False, True, False],
    )

    # Only the two LOO reporting CSVs required for method-selection evidence.
    selection_df.to_csv(
        os.path.join(out_dir, "loo_pooling_selection_by_fold.csv"), index=False
    )
    stability.to_csv(
        os.path.join(out_dir, "loo_pooling_stability.csv"), index=False
    )

    return oof_df, selection_df, stability

# ---------------------- Configuration Loading ----------------------

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
        'event_col': 'file',
        'label_col': 'label',
        'notes_col': 'notes',
        'score_col': None,
        'site_col': 'site',
        'out_dir': 'output',
        'max_thresholds': 5000,
        'duration_baseline': 'events',
        'video_total': None,
    }

    cfg = {}

    # Input section
    if config.has_section('input'):
        cfg['event_col'] = config.get('input', 'event_col', fallback=defaults['event_col'])
        cfg['label_col'] = config.get('input', 'label_col', fallback=defaults['label_col'])
        cfg['notes_col'] = config.get('input', 'notes_col', fallback=defaults['notes_col'])
        raw_score_col = config.get('input', 'score_col', fallback='').strip()
        cfg['score_col'] = raw_score_col if raw_score_col else None
        cfg['site_col'] = config.get('input', 'site_col', fallback=defaults['site_col']).strip() or defaults['site_col']
    else:
        cfg['event_col'] = defaults['event_col']
        cfg['label_col'] = defaults['label_col']
        cfg['notes_col'] = defaults['notes_col']
        cfg['score_col'] = defaults['score_col']
        cfg['site_col'] = defaults['site_col']

    # Evaluation section
    if config.has_section('evaluation'):
        cfg['max_thresholds'] = config.getint('evaluation', 'max_thresholds', fallback=defaults['max_thresholds'])
    else:
        cfg['max_thresholds'] = defaults['max_thresholds']

    # Output section
    if config.has_section('output'):
        cfg['out_dir'] = config.get('output', 'out_dir', fallback=defaults['out_dir'])
    else:
        cfg['out_dir'] = defaults['out_dir']

    # Duration section
    if config.has_section('duration'):
        cfg['duration_baseline'] = config.get('duration', 'baseline', fallback=defaults['duration_baseline'])
        cfg['video_total'] = config.get('duration', 'video_total', fallback=defaults['video_total'])
        if cfg['video_total'] == 'None' or cfg['video_total'] == '':
            cfg['video_total'] = None
    else:
        cfg['duration_baseline'] = defaults['duration_baseline']
        cfg['video_total'] = defaults['video_total']

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
        description="Holdout evaluation for binary bat detector with pre-defined operational threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using config file
  python holdout_evaluation.py --config holdout_config.ini --csv predictions.csv --threshold 0.404

  # Using CLI arguments only
  python holdout_evaluation.py --csv predictions.csv --threshold 0.404 --out-dir results

  # Mixed (CLI overrides config)
  python holdout_evaluation.py --config holdout_config.ini --csv predictions.csv --threshold 0.404 --recall-target 0.98
        """
    )

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to .ini configuration file')

    # Required arguments
    parser.add_argument("--csv", required=True,
                       help="Path to holdout_evaluation.csv (or similar)")
    parser.add_argument("--threshold", type=float, required=True,
                       help="Operational threshold to apply (0..1). Not learned from the evaluation data")

    # Input columns
    parser.add_argument("--event-col", type=str, default=None,
                       help="Event identifier column name")
    parser.add_argument("--label-col", type=str, default=None,
                       help="Ground truth label column name")
    parser.add_argument("--notes-col", type=str, default=None,
                       help="Free-text notes column for emergence detection")
    parser.add_argument("--score-col", type=str, default=None,
                       help="Predicted probability column name. If omitted/blank in config, run LOO pooling selection across standard pooling columns")
    parser.add_argument("--site-col", type=str, default=None,
                       help="Site/group column used for leave-one-site-out pooling selection (default: site)")

    # Evaluation parameters
    parser.add_argument("--recall-target", type=float, default=None,
                       help="Target recall for reference line in plots")
    parser.add_argument("--max-thresholds", type=int, default=None,
                       help="Maximum number of threshold points for curve generation")

    # Output
    parser.add_argument("--out-dir", type=str, default=None,
                       help="Output directory for results")

    # Duration baseline
    parser.add_argument("--duration-baseline", type=str, choices=["events","videos"], default=None,
                       help="Denominator for duration-based workload")
    parser.add_argument("--video-total", type=str, default=None,
                       help="Total source video duration (HH:MM:SS, 41h26m44s, MM:SS, or seconds)")

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
        if key not in ['config', 'csv', 'threshold'] and value is not None:
            # Convert hyphens to underscores for consistency
            key_normalized = key.replace('-', '_')
            cfg[key_normalized] = value

    return cfg


# ---------------------- Main ----------------------

def main():
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
        config_dict = load_config_from_ini(None)  # Will use all defaults

    # Merge config file with CLI arguments (CLI takes precedence)
    cfg = merge_config(config_dict, args)

    # Create namespace-like object with merged config
    class ConfigNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # Rebuild args with merged configuration
    merged_args = ConfigNamespace(
        csv=args.csv,
        threshold=args.threshold,
        event_col=cfg['event_col'],
        label_col=cfg['label_col'],
        notes_col=cfg['notes_col'],
        score_col=cfg['score_col'],
        site_col=cfg['site_col'],
        out_dir=cfg['out_dir'],
        max_thresholds=cfg['max_thresholds'],
        duration_baseline=cfg['duration_baseline'],
        video_total=cfg['video_total']
    )

    args = merged_args

    # Print configuration summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"CSV file: {args.csv}")
    print(f"Threshold: {args.threshold}")
    print(f"Score column: {args.score_col if args.score_col else 'AUTO: LOO across pooling columns'}")
    print(f"Site column: {args.site_col}")
    print(f"Duration baseline: {args.duration_baseline}")
    if args.video_total:
        print(f"Video total: {args.video_total}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 60 + "\n")

    ensure_dir(args.out_dir)

    # Load & clean
    raw = pd.read_csv(args.csv, low_memory=False)
    raw = clean_names(raw)

    # score_col set => legacy single-score evaluation.
    # score_col blank/omitted => LOO pooling selection across standard score columns.
    loo_mode = args.score_col is None or str(args.score_col).strip() == ""
    if loo_mode:
        available = [c for c in POOLING_SCORE_COLUMNS if c in raw.columns]
        if not available:
            raise ValueError(
                "score_col is blank, so LOO pooling mode was requested, but none of the standard "
                f"pooling columns were found: {', '.join(POOLING_SCORE_COLUMNS)}"
            )
        mapping_score_col = available[0]
    else:
        mapping_score_col = str(args.score_col).lower()
        if mapping_score_col not in raw.columns:
            raise ValueError(f"Missing required score column: {mapping_score_col}")

    df = map_labels(raw,
                    label_col=args.label_col.lower(),
                    score_col=mapping_score_col,
                    event_col=args.event_col.lower(),
                    notes_col=args.notes_col.lower())

    if loo_mode:
        # map_labels renames event column but otherwise retains all pooling columns.
        df, loo_selection_df, loo_stability_df = run_loo_pooling_selection(
            df=df,
            site_col=args.site_col.lower(),
            out_dir=args.out_dir,
        )
        print("\nPooling-selection stability:")
        print(loo_stability_df.to_string(index=False))
        selected_unique = int((loo_stability_df["times_selected"] > 0).sum())
        if selected_unique > 1:
            print("\n[WARN] Different pooling methods were selected across LOO folds. "
                  "The combined out-of-fold AP/ROC-AUC estimates the performance of the "
                  "pooling-selection procedure, not one fixed pooling method. Applying one "
                  "operational threshold to mixed pooling scores assumes that threshold is "
                  "appropriate across the selected score types.")
    else:
        # Legacy single-score mode: drop rows without predictions in chosen score column.
        missing_scores = df["score"].isna()
        if missing_scores.any():
            miss_ct = int(missing_scores.sum())
            df = df.loc[~missing_scores].copy()
            print(f"[WARN] Dropping {miss_ct} rows with NA '{mapping_score_col}' (no prediction). Remaining events: {len(df)}")

    if df.empty:
        raise ValueError("All rows are missing scores; nothing to evaluate.")

    n = len(df)
    prev = float((df["truth_bin"] == "bat").mean())
    event_total_seconds = float(df["duration_s"].sum())
    event_total_hours = event_total_seconds / 3600.0

    # Choose denominator for duration fractions
    if args.duration_baseline == "videos":
        video_total_seconds = parse_duration_arg(args.video_total)
        if video_total_seconds <= 0:
            print("[WARN] --duration-baseline videos but --video-total not provided or parsed as 0; "
                  "falling back to event durations as denominator.")
            duration_denominator = float(event_total_seconds)
            baseline_label = "events"
            video_total_seconds = 0.0
        else:
            duration_denominator = float(video_total_seconds)
            baseline_label = "videos"
    else:
        duration_denominator = float(event_total_seconds)
        video_total_seconds = 0.0
        baseline_label = "events"

    video_total_hours = float(video_total_seconds) / 3600.0
    print(f"Loaded {n} evaluation events. Positive prevalence = {percent(prev)}.")
    print(f"Event duration total = {event_total_hours:.3f} h | Video duration total = {video_total_hours:.3f} h "
          f"(baseline for duration fractions: {baseline_label})")

    # Threshold-free metrics
    y_true = (df["truth_bin"].astype(str) == "bat").astype(int).values
    scores = df["score"].values

    fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=False)
    auc_roc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap_score = average_precision_score(y_true, scores)

    # Threshold sweep metrics
    uniq = np.unique(np.concatenate(([0.0], scores[~np.isnan(scores)], [1.0])))
    ts = np.linspace(0, 1, args.max_thresholds) if len(uniq) > args.max_thresholds else np.sort(uniq)

    rows = []
    for t in ts:
        m = metrics_at_threshold(df, float(t), duration_denominator=duration_denominator)
        row = m.__dict__.copy()
        row["duration_baseline"] = baseline_label
        row["duration_denominator_seconds"] = float(duration_denominator)
        rows.append(row)
    thresh_df = pd.DataFrame(rows)
    thresh_csv = os.path.join(args.out_dir, "threshold_metrics.csv")
    thresh_df.to_csv(thresh_csv, index=False)
    print(f"Wrote: {thresh_csv}")

    # Operating point evaluation
    op_t = float(args.threshold)
    if not (0.0 <= op_t <= 1.0):
        raise ValueError("--threshold must be in [0,1].")
    cm_row = metrics_at_threshold(df, op_t, duration_denominator=duration_denominator)

    # Calibration + Brier scores
    if np.any(~np.isnan(scores)):
        q = np.quantile(scores[~np.isnan(scores)], q=np.linspace(0, 1, 11))
        q = np.unique(q)
    else:
        q = np.linspace(0,1,11)
    bins = pd.cut(df["score"], bins=q, include_lowest=True, duplicates="drop")
    cal = df.assign(bin=bins).groupby("bin", dropna=False).agg(
        mean_score=("score", "mean"),
        obs_rate = ("truth_bin", lambda s: np.mean(s.astype(str) == "bat")),
        n=("score", "size")
    ).reset_index()
    brier = float(np.mean((y_true - scores) ** 2))


    # ---------------------- Plots ----------------------

    # ROC
    plt.figure(figsize=(6.5,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.text(0.65, 0.1, f"AUC = {auc_roc:.3f}")
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    roc_path = os.path.join(args.out_dir, "roc.png")
    plt.savefig(roc_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote: {roc_path}")

    # PR
    plt.figure(figsize=(6.5,5))
    plt.plot(recall, precision)
    ax = plt.gca()
    ax.text(0.02, 0.05, f"AP = {ap_score:.3f}",
            transform=ax.transAxes, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    plt.title("Precision–Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    pr_path = os.path.join(args.out_dir, "pr.png")
    plt.savefig(pr_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote: {pr_path}")

    # Workload–Recall frontier (events vs duration)
    plt.figure(figsize=(7,5))
    plt.plot(thresh_df["scrubbed_fraction"],     thresh_df["recall"], label="By events")
    plt.plot(thresh_df["scrubbed_fraction_dur"], thresh_df["recall"], label=f"By video duration")

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    # vertical markers at the calibrated OP
    ax.axvline(x=cm_row.scrubbed_fraction,     ymin=0, ymax=_safe_div(cm_row.recall, ymax), linestyle=":", zorder=2)
    ax.axvline(x=cm_row.scrubbed_fraction_dur, ymin=0, ymax=_safe_div(cm_row.recall, ymax), linestyle=":", zorder=2)

    ax.scatter([cm_row.scrubbed_fraction],     [cm_row.recall], zorder=3)
    ax.scatter([cm_row.scrubbed_fraction_dur], [cm_row.recall], zorder=3)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))

    plt.title("Workload Reduction vs Recall (Operational point shown)")
    plt.xlabel("% auto-discarded")
    plt.ylabel("Recall")
    plt.legend(loc="lower left", frameon=False)

    wr_path = os.path.join(args.out_dir, "workload_recall.png")
    plt.savefig(wr_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote: {wr_path}")

    # Review composition vs threshold (EVENTS)
    comp = thresh_df[[
        "threshold","review_bat","review_bird","review_other","review_total"
    ]].copy()
    for col in ["review_bat","review_bird","review_other"]:
        comp[col] = comp[col] / (comp["review_total"].replace(0, np.nan))
    comp = comp.fillna(0.0)

    plt.figure(figsize=(7.8,5))
    x = comp["threshold"].values
    s1 = comp["review_bat"].values
    s2 = s1 + comp["review_bird"].values
    s3 = s2 + comp["review_other"].values
    plt.fill_between(x, 0, s1, alpha=0.85, label="Bat")
    plt.fill_between(x, s1, s2, alpha=0.85, label="Bird")
    plt.fill_between(x, s2, s3, alpha=0.85, label="Other")
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    plt.title("Composition of items sent to review vs threshold (EVENTS)")
    plt.xlabel("Threshold")
    plt.ylabel("Share of review by subtype")
    plt.legend(loc="upper right", frameon=False)
    rc_events = os.path.join(args.out_dir, "review_composition_vs_threshold_events.png")
    plt.savefig(rc_events, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote: {rc_events}")

    # Review composition vs threshold (DURATION)
    compd = thresh_df[[
        "threshold","review_bat_dur","review_bird_dur","review_other_dur","review_total_dur"
    ]].copy()
    for col in ["review_bat_dur","review_bird_dur","review_other_dur"]:
        compd[col] = compd[col] / (compd["review_total_dur"].replace(0, np.nan))
    compd = compd.fillna(0.0)

    plt.figure(figsize=(7.8,5))
    x = compd["threshold"].values
    s1 = compd["review_bat_dur"].values
    s2 = s1 + compd["review_bird_dur"].values
    s3 = s2 + compd["review_other_dur"].values
    plt.fill_between(x, 0, s1, alpha=0.85, label="Bat")
    plt.fill_between(x, s1, s2, alpha=0.85, label="Bird")
    plt.fill_between(x, s2, s3, alpha=0.85, label="Other")
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    plt.title("Composition of items sent to review vs threshold (DURATION-weighted)")
    plt.xlabel("Threshold")
    plt.ylabel("Share of review by subtype (duration-weighted)")
    plt.legend(loc="upper right", frameon=False)
    rc_dur = os.path.join(args.out_dir, "review_composition_vs_threshold_duration.png")
    plt.savefig(rc_dur, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote: {rc_dur}")

    # Composition at the operating point (EVENTS)
    oper_counts = {
        "bat (TP+)":          cm_row.review_bat,
        "bird (FP)":          cm_row.review_bird,
        "other_not_bat (FP)": cm_row.review_other
    }
    names = list(oper_counts.keys())
    vals  = np.array(list(oper_counts.values()), dtype=float)
    share = vals / (vals.sum() or 1.0)

    plt.figure(figsize=(7.6,5))
    plt.bar(range(len(names)), share)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    plt.xticks(range(len(names)), names, rotation=15, ha="right")
    plt.ylim(0,1)
    plt.title(f"Review composition at threshold = {op_t:.3f} (EVENTS)")
    plt.ylabel("Share of review")
    op_events = os.path.join(args.out_dir, "review_composition_at_operating_point_events.png")
    plt.savefig(op_events, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote: {op_events}")

    # Composition at the operating point (DURATION)
    oper_dur = {
        "bat (TP+)":          cm_row.review_bat_dur,
        "bird (FP)":          cm_row.review_bird_dur,
        "other_not_bat (FP)": cm_row.review_other_dur
    }
    names_d = list(oper_dur.keys())
    vals_d  = np.array(list(oper_dur.values()), dtype=float)
    share_d = vals_d / (vals_d.sum() or 1.0)

    plt.figure(figsize=(7.6,5))
    plt.bar(range(len(names_d)), share_d)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    plt.xticks(range(len(names_d)), names_d, rotation=15, ha="right")
    plt.ylim(0,1)
    plt.title(f"Review composition at threshold = {op_t:.3f} (DURATION-weighted)")
    plt.ylabel("Share of review (duration-weighted)")
    op_dur = os.path.join(args.out_dir, "review_composition_at_operating_point_duration.png")
    plt.savefig(op_dur, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote: {op_dur}")

    # ---------------------- Console summary ----------------------
    print("\n================ SUMMARY ================")
    print(f"Events: {n} | Positive prevalence: {percent(prev)}")
    print(f"Event total: {event_total_hours:.3f} h | Video total: {video_total_hours:.3f} h "
          f"(duration baseline: {baseline_label})")
    print(f"ROC AUC: {auc_roc:.3f} | PR AP: {ap_score:.3f} | Brier: {brier:.3f}")
    print(f"Operational threshold (provided): {op_t:.3f}\n")

    def p(x): return percent(x)
    print("Metrics at operating point:")
    print(f"  Recall={p(cm_row.recall)}")
    print(f"  Specificity={p(cm_row.specificity)}, FPR={p(1.0-cm_row.specificity)}, "
          f"Accuracy={p(cm_row.accuracy)}, Balanced Acc={p(cm_row.bal_acc)}, MCC={cm_row.mcc:.3f}")
    print(f"  Scrubbed (events)={p(cm_row.scrubbed_fraction)}, Review (events)={p(cm_row.review_fraction)}")
    print(f"  Scrubbed (duration)={p(cm_row.scrubbed_fraction_dur)}, Review (duration)={p(cm_row.review_fraction_dur)}")
    print(f"  Review shares (events): bird={p(cm_row.share_bird_in_review)}")

    # Emergence
    def maybe_pct(x): return p(x) if np.isfinite(x) else "NA" # safely handles cases with no emergences.
    print(f"\nEmergence:")
    print(f"  Total={cm_row.emergence_total} | Recalled={cm_row.emergence_tp} | Missed={cm_row.emergence_fn} | Recall={maybe_pct(cm_row.emergence_recall)}")

    print("\nCounts at operating point (events):")
    print(f"  review_total={cm_row.review_total} | bat={cm_row.review_bat}, "
          f"FP bird={cm_row.review_bird}, FP other={cm_row.review_other}")

    # Summary table (Markdown/LaTeX/CSV)
    write_publication_tables(
        out_dir=args.out_dir,
        cm_row=cm_row,
        op_t=op_t,
        n=n,
        prev=prev,
        event_total_hours=event_total_hours,
        video_total_hours=video_total_hours,
        duration_baseline=baseline_label,
        auc_roc=auc_roc,
        ap_score=ap_score,
        brier=brier
    )

    print(f"\nAll plots and CSVs written to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)