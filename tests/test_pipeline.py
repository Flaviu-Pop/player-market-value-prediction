# tests/test_pipeline.py
# Basic validation tests for data and model pipeline
#
# Usage:
#   python tests/test_pipeline.py

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import PROC_DIR, MODELS_DIR, POS_KEYS, POSITION_FEATURES, MIN_VALUE

def run_tests():
    print("=" * 60)
    print(" Pipeline Validation Tests")
    print("=" * 60)

    passed = failed = 0

    def check(condition, label):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {'✓' if condition else '✗'} {label}")
        passed += condition
        failed += (not condition)

    # ── Data files ────────────────────────────────────────────────────────────
    print("\n── Processed Data Files ──────────────────────────────────────")
    for group, key in POS_KEYS.items():
        path = os.path.join(PROC_DIR, f"{key}_data.csv")
        check(os.path.exists(path), f"{key}_data.csv exists")

        if os.path.exists(path):
            df = pd.read_csv(path)
            check(len(df) > 100,            f"{group}: >100 players")
            check("value_eur" in df.columns, f"{group}: value_eur column present")
            check("log_value_eur" in df.columns, f"{group}: log_value_eur present")
            check((df["value_eur"] >= MIN_VALUE).all(),
                  f"{group}: all values >= €{MIN_VALUE:,}")

            feat_cols = [c for c in POSITION_FEATURES[group] if c in df.columns]
            check(df[feat_cols].isna().sum().sum() == 0,
                  f"{group}: no NaN in feature columns")

    # ── Train/Val/Test splits ─────────────────────────────────────────────────
    print("\n── Train/Val/Test Splits ─────────────────────────────────────")
    for group, key in POS_KEYS.items():
        for split in ["train", "val", "test"]:
            path = os.path.join(PROC_DIR, f"{key}_{split}.csv")
            check(os.path.exists(path), f"{key}_{split}.csv exists")
            if os.path.exists(path):
                df = pd.read_csv(path)
                check("target" in df.columns, f"{key}_{split}: target column present")
                check(df.isna().sum().sum() == 0, f"{key}_{split}: no NaN values")

    # ── Model files ───────────────────────────────────────────────────────────
    print("\n── Saved Model Files ─────────────────────────────────────────")
    for group, key in POS_KEYS.items():
        model_path   = os.path.join(MODELS_DIR, f"{key}_model.pth")
        scaler_path  = os.path.join(MODELS_DIR, f"{key}_scaler.joblib")
        feature_path = os.path.join(MODELS_DIR, f"{key}_features.joblib")
        check(os.path.exists(model_path),   f"{key}_model.pth exists")
        check(os.path.exists(scaler_path),  f"{key}_scaler.joblib exists")
        check(os.path.exists(feature_path), f"{key}_features.joblib exists")

    # ── Result ────────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*60}")
    print(f" Results: {passed}/{total} passed | {failed} failed")
    print("=" * 60)
    if failed > 0:
        print("\n Some tests failed. Run the pipeline scripts in order (01→06).")
        sys.exit(1)
    else:
        print("\n All tests passed.")

if __name__ == "__main__":
    run_tests()
