#!/usr/bin/env bash
set -euo pipefail

CSV="holdout_evaluation.csv"
SCRIPT="holdout_evaluation.py"
BASE_OUT="./results"

# --------- Common args ----------
DURATION_BASELINE="videos"
VIDEO_TOTAL="41:26:44"     # HH:MM:SS
THRESHOLD=0.404
# the metric columns to use as --score-col
metrics=(average_probability median_probability max_probability top_frac_probability)


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
