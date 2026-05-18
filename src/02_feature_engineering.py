# src/02_feature_engineering.py

# Step 2: Scale features, split train/val/test, save ready-to-train tensors

# Input : data/processed/{gk,def,mid,fwd}_data.csv
# Output: data/processed/{key}_train/val/test.csv + scalers in models/

# Usage: python src/02_feature_engineering.py

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PROC_DIR, MODELS_DIR, POS_KEYS, POSITION_FEATURES,
    TARGET_COL, USE_LOG_TARGET,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)


def process_position(group: str, key: str) -> None:
    """Load, scale, split and save one position group."""

    input_path = os.path.join(PROC_DIR, f"{key}_data.csv")

    if not os.path.exists(input_path):
        print(f"  WARNING: {input_path} not found — run 01_data_preparation.py first")
        return

    df = pd.read_csv(input_path)

    # Identify feature columns present in this file
    all_features  = POSITION_FEATURES[group]
    feature_cols  = [c for c in all_features if c in df.columns]
    target_col    = "log_value_eur" if USE_LOG_TARGET else TARGET_COL

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # === Train / Val / Test Split =====================================================================================
    # First split off test set, then split remainder into train/val
    val_test_ratio = VAL_RATIO + TEST_RATIO

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=RANDOM_SEED
    )

    #val_fraction = VAL_RATIO / val_test_ratio
    #X_val, X_test, y_val, y_test = train_test_split(
    #    X_temp, y_temp, test_size=(1 - val_fraction), random_state=RANDOM_SEED
    #)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )

    # === Scale Features ===============================================================================================
    # Fit ONLY on training data — apply to val and test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save scaler — needed for prediction interface
    scaler_path = os.path.join(MODELS_DIR, f"{key}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save feature list — needed for prediction interface
    feature_list_path = os.path.join(MODELS_DIR, f"{key}_features.joblib")
    joblib.dump(feature_cols, feature_list_path)

    # === Save Splits as CSV ===========================================================================================
    def save_split(X_arr, y_arr, split_name):
        split_df = pd.DataFrame(X_arr, columns=feature_cols)
        split_df["target"] = y_arr

        out = os.path.join(PROC_DIR, f"{key}_{split_name}.csv")
        split_df.to_csv(out, index=False)

    save_split(X_train, y_train, "train")
    save_split(X_val,   y_val,   "val")
    save_split(X_test,  y_test,  "test")

    # Also save original (unscaled) test set with player names for evaluation
    test_indices = df.index[
        ~df.index.isin(
            df.sample(frac=TRAIN_RATIO + VAL_RATIO, random_state=RANDOM_SEED).index
        )
    ]

    id_cols = ["sofifa_id", "short_name", "club", TARGET_COL, "log_value_eur"]
    id_cols = [c for c in id_cols if c in df.columns]

    df_test_meta = df[id_cols].iloc[-len(y_test):].copy()
    df_test_meta.to_csv(os.path.join(PROC_DIR, f"{key}_test_meta.csv"), index=False)

    print(f"  {group:<12} ({key}): "
          f"train={len(X_train):>4} | val={len(X_val):>3} | test={len(X_test):>3} | "
          f"features={len(feature_cols)}")


def main():
    print("=" * 60)
    print(" Step 2: Feature Engineering & Data Splitting")
    print("=" * 60)

    print(f"\n Target: {'log(value_eur)' if USE_LOG_TARGET else 'value_eur'}")

    print(f" Split : {int(TRAIN_RATIO*100)}% train / "
          f"{int(VAL_RATIO*100)}% val / "
          f"{int(TEST_RATIO*100)}% test\n")

    for group, key in POS_KEYS.items():
        process_position(group, key)

    print("\n" + "=" * 60)
    print(" NEXT STEP: python src/03_train_models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
