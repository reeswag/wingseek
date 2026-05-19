#!/usr/bin/env python3

"""
Generate holdout evaluation violin plots.

Each violin shows the distribution of bat-class probabilities for a given
aggregation method and label group.

Class colours:
  - Other: blue
  - Bat: red

Plot elements:
  - Pale class-coloured violin: full distribution
  - Stronger class-coloured shading: 5th–95th percentile interval
  - Dashed horizontal lines: 5th and 95th percentile bounds
  - Solid horizontal line: median

Outputs:
  - violinplots_bat_vs_rest.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ---------------------------------------------------
# Metrics to visualize
# ---------------------------------------------------
METRICS = [
    "average_probability",
    "median_probability",
    "max_probability",
    "top_frac_probability",
]

METRIC_LABELS = {
    "average_probability": "Mean",
    "median_probability": "Median",
    "max_probability": "Maximum",
    "top_frac_probability": "Top Fraction 20%",
}


# ---------------------------------------------------
# Class colours matched to modelling histograms
# ---------------------------------------------------
CLASS_COLORS = {
    "Other": "blue",
    "Bat": "red",
}

FULL_DISTRIBUTION_ALPHA = 0.15
PERCENTILE_90_ALPHA = 0.50

SUMMARY_LINE_COLOR = "black"


def validate_columns(df: pd.DataFrame) -> None:
    required = {"label", *METRICS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in sorted(missing))
        )


def style_violin_bodies_by_class(
    violin_parts: dict,
    label_order: list[str],
    alpha: float,
    edgecolor: str | None = None,
    linewidth: float = 0.8,
) -> None:
    """Apply class-specific colours to violin bodies."""
    for body, label in zip(violin_parts["bodies"], label_order):
        body.set_facecolor(CLASS_COLORS[label])
        body.set_alpha(alpha)

        if edgecolor is None:
            body.set_edgecolor(CLASS_COLORS[label])
        else:
            body.set_edgecolor(edgecolor)

        body.set_linewidth(linewidth)


def clip_violin_bodies_to_intervals(
    ax: plt.Axes,
    violin_parts: dict,
    positions: list[int],
    intervals: list[tuple[float, float]],
) -> None:
    """
    Clip each violin body so only the specified y-interval remains visible.
    """
    for body, x_pos, (y_min, y_max) in zip(
        violin_parts["bodies"],
        positions,
        intervals,
    ):
        clip_rect = Rectangle(
            (x_pos - 0.60, y_min),
            1.20,
            max(y_max - y_min, 0),
            transform=ax.transData,
        )
        body.set_clip_path(clip_rect)


def make_binary_violinplots(df: pd.DataFrame, output_dir: str) -> str:
    bin_map = {
        "bat": "Bat",
        "bird": "Other",
        "human": "Other",
        "other": "Other",
    }

    bin_df = df.copy()
    bin_df["binary_label"] = bin_df["label"].map(bin_map).fillna("Other")

    label_order = ["Other", "Bat"]
    positions = [1, 2]
    panel_titles = ["(a)", "(b)", "(c)", "(d)"]

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, axes_grid = plt.subplots(
        2,
        2,
        figsize=(10, 10),
        sharey=True,
    )
    axes = axes_grid.ravel()

    for i, metric in enumerate(METRICS):
        data_by_label = [
            bin_df.loc[bin_df["binary_label"] == lbl, metric].dropna()
            for lbl in label_order
        ]

        empty_groups = [
            lbl
            for lbl, values in zip(label_order, data_by_label)
            if values.empty
        ]
        if empty_groups:
            raise ValueError(
                f"No valid data found for metric '{metric}' in group(s): "
                + ", ".join(empty_groups)
            )

        quantiles = [
            np.quantile(values, [0.05, 0.50, 0.95])
            for values in data_by_label
        ]

        percentile_90_intervals = [
            (q05, q95)
            for q05, q50, q95 in quantiles
        ]

        # Full distribution: pale class colour
        full_violin = axes[i].violinplot(
            data_by_label,
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        style_violin_bodies_by_class(
            full_violin,
            label_order=label_order,
            alpha=FULL_DISTRIBUTION_ALPHA,
            edgecolor=None,
            linewidth=0.9,
        )

        # Central 90% interval: stronger class colour
        percentile_90_violin = axes[i].violinplot(
            data_by_label,
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        style_violin_bodies_by_class(
            percentile_90_violin,
            label_order=label_order,
            alpha=PERCENTILE_90_ALPHA,
            edgecolor="none",
            linewidth=0,
        )
        clip_violin_bodies_to_intervals(
            axes[i],
            percentile_90_violin,
            positions,
            percentile_90_intervals,
        )

        # Median and 5th/95th percentile boundary lines
        for x_pos, (q05, q50, q95) in zip(positions, quantiles):
            # 5th percentile dashed line
            axes[i].hlines(
                y=q05,
                xmin=x_pos - 0.20,
                xmax=x_pos + 0.20,
                colors=SUMMARY_LINE_COLOR,
                linestyles="--",
                linewidth=2.0,
                zorder=10,
            )

            # Median solid line
            axes[i].hlines(
                y=q50,
                xmin=x_pos - 0.20,
                xmax=x_pos + 0.20,
                colors=SUMMARY_LINE_COLOR,
                linestyles="-",
                linewidth=2.8,
                zorder=11,
            )

            # 95th percentile dashed line
            axes[i].hlines(
                y=q95,
                xmin=x_pos - 0.20,
                xmax=x_pos + 0.20,
                colors=SUMMARY_LINE_COLOR,
                linestyles="--",
                linewidth=2.0,
                zorder=10,
            )

        axes[i].set_title(
            f"{panel_titles[i]}",
            loc="left",
            pad=6,
        )
        axes[i].set_xlabel("Label")
        axes[i].set_xticks(positions)
        axes[i].set_xticklabels(label_order)
        axes[i].set_ylim(0, 1)

        if i % 2 == 0:
            axes[i].set_ylabel("Bat-class probability")

    # Shared legend for summary-line styles
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SUMMARY_LINE_COLOR,
            linestyle="--",
            linewidth=2.0,
            label="5th/95th percentile bounds",
        ),
        Line2D(
            [0],
            [0],
            color=SUMMARY_LINE_COLOR,
            linestyle="-",
            linewidth=2.8,
            label="Median",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.055, 1, 1])

    out_path = os.path.join(output_dir, "violinplots_bat_vs_rest.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate violin plots from holdout_eval.csv with "
            "class-coloured probability distributions, "
            "5th–95th percentile shading, dashed percentile bounds, "
            "and median markers."
        )
    )
    parser.add_argument(
        "csv_path",
        help="Path to holdout_eval.csv or holdout_evaluation.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots. Defaults to a 'plots' folder beside the CSV.",
    )

    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(os.path.dirname(csv_path), "results")
    )
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    df["label"] = (
        df["label"]
        .fillna("other")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    validate_columns(df)

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    violin_path = make_binary_violinplots(df, output_dir)

    print("Saved figures:")
    print(f"- {violin_path}")
    print(
        "Class colours: Other = blue; Bat = red. "
        "Paler fills show full distributions; stronger fills show the "
        "5th–95th percentile interval. Dashed black lines mark the "
        "5th and 95th percentile bounds; solid black lines mark the median."
    )


if __name__ == "__main__":
    main()