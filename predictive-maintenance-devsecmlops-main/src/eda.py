"""
Exploratory Data Analysis. Saves charts to reports/figures/ and a summary JSON.

Charts:
  1. class_balance.png        - how rare failures are (imbalanced data)
  2. sensor_distributions.png - each sensor, normal vs failure
  3. correlation_heatmap.png  - how sensors relate to each other and to failure
  4. failure_by_type.png      - failure rate per machine type (L/M/H)
"""
import json

import matplotlib

matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import SENSOR_COLS, TARGET_COL, get_logger, load_config, path

logger = get_logger(__name__)


def run_eda(df: pd.DataFrame) -> dict:
    cfg = load_config()
    fig_dir = path(f"{cfg['paths']['reports_dir']}/figures/x").parent

    # 1. class balance
    counts = df[TARGET_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Normal (0)", "Failure (1)"], counts.values, color=["#4C9F70", "#D9534F"])
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_balance.png", dpi=110)
    plt.close(fig)

    # 2. sensor distributions by class
    fig, axes = plt.subplots(1, len(SENSOR_COLS), figsize=(20, 4))
    for ax, col in zip(axes, SENSOR_COLS):
        ax.hist(df.loc[df[TARGET_COL] == 0, col], bins=40, alpha=0.6, density=True, label="Normal")
        ax.hist(df.loc[df[TARGET_COL] == 1, col], bins=40, alpha=0.6, density=True, label="Failure")
        ax.set_title(col)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "sensor_distributions.png", dpi=110)
    plt.close(fig)

    # 3. correlation heatmap
    corr = df[SENSOR_COLS + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=110)
    plt.close(fig)

    # 4. failure rate per machine type
    by_type = df.groupby("machine_type")[TARGET_COL].mean().reindex(["L", "M", "H"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(by_type.index, by_type.values * 100, color="#5B8DEF")
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Failure rate by machine type")
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_by_type.png", dpi=110)
    plt.close(fig)

    summary = {
        "rows": len(df),
        "failure_rate_pct": round(100 * df[TARGET_COL].mean(), 2),
        "failure_rate_by_type_pct": (by_type * 100).round(2).to_dict(),
        "correlation_with_failure": corr[TARGET_COL].drop(TARGET_COL).round(3).to_dict(),
        "sensor_stats": df[SENSOR_COLS].describe().round(2).to_dict(),
    }
    (fig_dir.parent / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("EDA complete -> %s (failure rate %.2f%%)", fig_dir, summary["failure_rate_pct"])
    return summary
