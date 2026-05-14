# src/05_visualizations.py
# Step 5: Generate all charts
#
# Charts produced:
#   01_value_distribution.png       — Market value distribution by position
#   02_training_curves_{key}.png    — Loss curves per position
#   03_actual_vs_predicted.png      — Scatter: actual vs predicted (all positions)
#   04_residuals.png                — Residual distribution per position
#   05_metrics_comparison.png       — MAE / R² bar chart across positions
#
# Usage:
#   python src/05_visualizations.py

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROC_DIR, METRICS_DIR, CHARTS_DIR, POS_KEYS,
    POSITION_COLORS, FIG_SIZE_WIDE, FIG_SIZE_SQUARE, DPI
)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Chart 1: Value Distribution by Position ───────────────────────────────────

def plot_value_distribution():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    position_names = list(POS_KEYS.keys())

    for i, (group, key) in enumerate(POS_KEYS.items()):
        path = os.path.join(PROC_DIR, f"{key}_data.csv")
        if not os.path.exists(path):
            continue

        df    = pd.read_csv(path)
        color = POSITION_COLORS[group]

        axes[i].hist(
            np.log1p(df["value_eur"]) if "value_eur" in df.columns else [],
            bins=40, color=color, alpha=0.8, edgecolor="white"
        )
        axes[i].set_title(f"{group} (n={len(df):,})", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("log(Market Value EUR)")
        axes[i].set_ylabel("Number of Players")

        # Annotate median
        med = np.log1p(df["value_eur"].median())
        axes[i].axvline(med, color="black", linestyle="--", linewidth=1.5,
                        label=f"Median: €{df['value_eur'].median():,.0f}")
        axes[i].legend(fontsize=9)

    fig.suptitle("Market Value Distribution by Position Group (log scale)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "01_value_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 2: Training Loss Curves ─────────────────────────────────────────────

def plot_training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (group, key) in enumerate(POS_KEYS.items()):
        hist_path = os.path.join(METRICS_DIR, f"{key}_training_history.csv")
        if not os.path.exists(hist_path):
            axes[i].set_title(f"{group} — no history found")
            continue

        hist  = pd.read_csv(hist_path)
        color = POSITION_COLORS[group]

        axes[i].plot(hist["epoch"], hist["train_loss"],
                     label="Train Loss", color=color, linewidth=2)
        axes[i].plot(hist["epoch"], hist["val_loss"],
                     label="Val Loss", color=color, linewidth=2,
                     linestyle="--", alpha=0.7)

        # Mark best epoch
        best_idx = hist["val_loss"].idxmin()
        axes[i].axvline(hist["epoch"][best_idx], color="gray",
                        linestyle=":", linewidth=1.5,
                        label=f"Best epoch: {hist['epoch'][best_idx]}")

        axes[i].set_title(f"{group} — Training Curves", fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("MSE Loss (log scale)")
        axes[i].legend(fontsize=9)
        axes[i].set_yscale("log")

    fig.suptitle("Training & Validation Loss Curves — All Position Models",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "02_training_curves.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 3: Actual vs Predicted ──────────────────────────────────────────────

def plot_actual_vs_predicted():
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (group, key) in enumerate(POS_KEYS.items()):
        pred_path = os.path.join(METRICS_DIR, f"{key}_predictions.csv")
        if not os.path.exists(pred_path):
            continue

        df    = pd.read_csv(pred_path)
        color = POSITION_COLORS[group]

        actual    = np.log1p(df["actual_eur"])
        predicted = np.log1p(df["predicted_eur"].clip(lower=0))

        axes[i].scatter(actual, predicted, alpha=0.4, color=color,
                        s=20, label="Players")

        # Perfect prediction line
        lim = [min(actual.min(), predicted.min()),
               max(actual.max(), predicted.max())]
        axes[i].plot(lim, lim, "k--", linewidth=1.5, label="Perfect prediction")

        # Trend line
        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        x_line = np.linspace(lim[0], lim[1], 100)
        axes[i].plot(x_line, p(x_line), color="red",
                     linewidth=1.5, alpha=0.8, label="Trend")

        # R² annotation
        r2 = np.corrcoef(actual, predicted)[0, 1] ** 2
        axes[i].text(0.05, 0.92, f"R² = {r2:.3f}",
                     transform=axes[i].transAxes,
                     fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        axes[i].set_title(f"{group} — Actual vs Predicted", fontweight="bold")
        axes[i].set_xlabel("Actual log(Value EUR)")
        axes[i].set_ylabel("Predicted log(Value EUR)")
        axes[i].legend(fontsize=9)

    fig.suptitle("Actual vs Predicted Market Value — All Positions (log scale)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "03_actual_vs_predicted.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 4: Residual Distribution ────────────────────────────────────────────

def plot_residuals():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (group, key) in enumerate(POS_KEYS.items()):
        pred_path = os.path.join(METRICS_DIR, f"{key}_predictions.csv")
        if not os.path.exists(pred_path):
            continue

        df    = pd.read_csv(pred_path)
        color = POSITION_COLORS[group]

        # Residuals in log space (less sensitive to outliers)
        residuals = np.log1p(df["predicted_eur"].clip(lower=0)) - \
                    np.log1p(df["actual_eur"])

        axes[i].hist(residuals, bins=40, color=color, alpha=0.8, edgecolor="white")
        axes[i].axvline(0, color="black", linewidth=2, linestyle="--",
                        label="Zero error")
        axes[i].axvline(residuals.mean(), color="red", linewidth=1.5,
                        label=f"Mean: {residuals.mean():.3f}")

        axes[i].set_title(f"{group} — Residuals", fontweight="bold")
        axes[i].set_xlabel("Prediction Error (log scale)")
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=9)

    fig.suptitle("Residual Distribution — Model Prediction Errors",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "04_residuals.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Chart 5: Cross-Position Metrics Comparison ────────────────────────────────

def plot_metrics_comparison():
    metrics_path = os.path.join(METRICS_DIR, "all_positions_metrics.json")
    if not os.path.exists(metrics_path):
        print("  SKIP: metrics JSON not found — run 04_evaluate_models.py first")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    df = pd.DataFrame(metrics)
    positions = df["position"].tolist()
    colors    = [POSITION_COLORS[p] for p in positions]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # MAE in thousands EUR
    mae_k = df["mae_eur"] / 1000
    axes[0].bar(positions, mae_k, color=colors, edgecolor="white")
    axes[0].set_title("MAE (thousands EUR)", fontweight="bold")
    axes[0].set_ylabel("€ thousands")
    for j, v in enumerate(mae_k):
        axes[0].text(j, v + 5, f"€{v:.0f}k", ha="center", fontsize=10)

    # MAPE %
    axes[1].bar(positions, df["mape_pct"], color=colors, edgecolor="white")
    axes[1].set_title("MAPE (%)", fontweight="bold")
    axes[1].set_ylabel("Percentage error")
    for j, v in enumerate(df["mape_pct"]):
        axes[1].text(j, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

    # R²
    axes[2].bar(positions, df["r2_score"], color=colors, edgecolor="white")
    axes[2].set_title("R² Score", fontweight="bold")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1.1)
    for j, v in enumerate(df["r2_score"]):
        axes[2].text(j, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle("Model Performance Comparison — All Position Groups",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "05_metrics_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Step 5: Visualizations")
    print("=" * 60)
    print()

    print("[1/5] Value distribution by position...")
    plot_value_distribution()

    print("[2/5] Training loss curves...")
    plot_training_curves()

    print("[3/5] Actual vs predicted scatter plots...")
    plot_actual_vs_predicted()

    print("[4/5] Residual distributions...")
    plot_residuals()

    print("[5/5] Cross-position metrics comparison...")
    plot_metrics_comparison()

    print("\n" + "=" * 60)
    print(" All charts saved to outputs/charts/")
    print(" NEXT STEP: python src/06_predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
