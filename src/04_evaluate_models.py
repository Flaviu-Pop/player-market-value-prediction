# src/04_evaluate_models.py
# Step 4: Evaluate trained models on test set
# Metrics: MAE (EUR), MAPE, R², residual analysis, undervalued players
# Input : models/{key}_model.pth + data/processed/{key}_test.csv
# Output: outputs/metrics/{key}_metrics.json + {key}_predictions.csv
#
# Usage:
#   python src/04_evaluate_models.py

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROC_DIR, MODELS_DIR, METRICS_DIR, POS_KEYS, DEVICE if False else None,
    USE_LOG_TARGET
)
from train_models import PlayerValueNet   # reuse the model class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(key: str, input_dim: int) -> nn.Module:
    """Load best saved model weights."""
    checkpoint = torch.load(
        os.path.join(MODELS_DIR, f"{key}_model.pth"),
        map_location=DEVICE
    )
    model = PlayerValueNet(input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — avoids division by zero."""
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_position(group: str, key: str) -> dict:
    """Evaluate one position model on its test set."""

    # ── Load test features ────────────────────────────────────────────────────
    test_path = os.path.join(PROC_DIR, f"{key}_test.csv")
    meta_path = os.path.join(PROC_DIR, f"{key}_test_meta.csv")

    if not os.path.exists(test_path):
        print(f"  SKIP {group}: test CSV not found")
        return {}

    test_df  = pd.read_csv(test_path)
    X_test   = torch.tensor(
        test_df.drop(columns=["target"]).values, dtype=torch.float32
    ).to(DEVICE)
    y_log    = test_df["target"].values   # log-scale targets

    # ── Predict ───────────────────────────────────────────────────────────────
    input_dim = X_test.shape[1]
    model     = load_model(key, input_dim)

    with torch.no_grad():
        log_preds = model(X_test).cpu().numpy()

    # Convert back from log scale to EUR
    y_true_eur = np.expm1(y_log)
    y_pred_eur = np.expm1(log_preds)
    y_pred_eur = np.maximum(y_pred_eur, 0)   # no negative values

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae_eur  = mean_absolute_error(y_true_eur, y_pred_eur)
    mape_pct = mape(y_true_eur, y_pred_eur)
    r2       = r2_score(y_true_eur, y_pred_eur)

    metrics = {
        "position":        group,
        "n_test_players":  len(y_true_eur),
        "mae_eur":         round(mae_eur, 0),
        "mape_pct":        round(mape_pct, 2),
        "r2_score":        round(r2, 4),
    }

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "actual_eur":    y_true_eur,
        "predicted_eur": y_pred_eur,
        "error_eur":     y_pred_eur - y_true_eur,
        "pct_error":     ((y_pred_eur - y_true_eur) / np.maximum(y_true_eur, 1)) * 100,
    })

    # Add player names if meta file exists
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path).reset_index(drop=True)
        pred_df = pd.concat([meta_df.reset_index(drop=True), pred_df], axis=1)

    pred_path = os.path.join(METRICS_DIR, f"{key}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # ── Top Undervalued Players ───────────────────────────────────────────────
    pred_df["undervalue_eur"] = pred_df["predicted_eur"] - pred_df["actual_eur"]
    top_undervalued = pred_df.nlargest(10, "undervalue_eur")

    print(f"\n  Top 5 undervalued {group}s (model predicts higher than actual):")
    name_col = "short_name" if "short_name" in pred_df.columns else None
    for _, row in top_undervalued.head(5).iterrows():
        name  = row[name_col] if name_col else "Player"
        print(f"    {name:<20} actual: €{row['actual_eur']:>10,.0f} | "
              f"predicted: €{row['predicted_eur']:>10,.0f}")

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Step 4: Model Evaluation")
    print("=" * 60)

    all_metrics = []

    for group, key in POS_KEYS.items():
        print(f"\n── Evaluating: {group} ({key}) ─────────────────────────────")
        m = evaluate_position(group, key)
        if m:
            all_metrics.append(m)
            print(f"\n  MAE   : €{m['mae_eur']:>12,.0f}")
            print(f"  MAPE  : {m['mape_pct']:>8.2f} %")
            print(f"  R²    : {m['r2_score']:>8.4f}")

    # ── Cross-Position Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Cross-Position Evaluation Summary")
    print("=" * 60)
    print(f"\n{'Position':<14} {'N Test':>8} {'MAE (EUR)':>14} {'MAPE %':>8} {'R²':>8}")
    print("-" * 58)
    for m in all_metrics:
        print(f"{m['position']:<14} {m['n_test_players']:>8} "
              f"{m['mae_eur']:>14,.0f} {m['mape_pct']:>8.2f} {m['r2_score']:>8.4f}")

    # Save summary
    summary_path = os.path.join(METRICS_DIR, "all_positions_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {summary_path}")

    print("\n" + "=" * 60)
    print(" NEXT STEP: python src/05_visualizations.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
