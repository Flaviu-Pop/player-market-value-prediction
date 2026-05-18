# src/01_data_preparation.py

# Step 1: Load raw data, clean, assign position groups, apply log transform

# Input : data/raw/players_raw.csv
# Output: data/processed/{gk,def,mid,fwd}_data.csv


# Usage: python src/01_data_preparation.py


import os
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_CSV, PROC_DIR, TARGET_COL, MIN_VALUE,
    USE_LOG_TARGET, POSITION_GROUPS, POS_KEYS,
    POSITION_FEATURES, RANDOM_SEED
)


# === Position Assignment ==============================================================================================

def assign_position_group(player_positions: str) -> str:
    """
    Assign a player to one of four groups based on their primary position.
    Primary position = first position listed in player_positions column.
    """
    if pd.isna(player_positions):
        return None

    primary = str(player_positions).split(",")[0].strip()

    for group, positions in POSITION_GROUPS.items():
        if primary in positions:
            return group

    return None     # Unclassified — will be dropped


# === Main =============================================================================================================

def main():
    print("=" * 60)
    print(" Step 1: Data Preparation")
    print(f" Input : {RAW_CSV}")
    print("=" * 60)

    # === Load =========================================================================================================
    if not os.path.exists(RAW_CSV):
        print(f"\nERROR: {RAW_CSV} not found.")
        print("Place players_raw.csv in data/raw/ and retry.")
        sys.exit(1)

    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"\nRaw data loaded: {df.shape[0]:,} players × {df.shape[1]} columns")

    # === Drop rows with missing target ================================================================================
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    print(f"Dropped {before - len(df)} rows with missing value_eur")

    # === Filter out zero / near-zero values ===========================================================================
    before = len(df)
    df = df[df[TARGET_COL] >= MIN_VALUE]
    print(f"Dropped {before - len(df)} players with value < €{MIN_VALUE:,}")

    # === Assign position groups =======================================================================================
    df["position_group"] = df["player_positions"].apply(assign_position_group)                   #apply primary position

    before = len(df)
    df = df.dropna(subset=["position_group"])
    print(f"Dropped {before - len(df)} players with unclassified positions")

    print(f"\nPosition group counts:")
    counts = df["position_group"].value_counts()
    for group, count in counts.items():
        print(f"  {group:<12}: {count:>5,} players")

    # === Log-transform target =========================================================================================
    if USE_LOG_TARGET:
        df["log_value_eur"] = np.log1p(df[TARGET_COL])
        print(f"\nLog-transformed target created: log_value_eur")
        print(f"  Original range : €{df[TARGET_COL].min():>12,.0f} – €{df[TARGET_COL].max():>12,.0f}")
        print(f"  Log range      : {df['log_value_eur'].min():.3f} – {df['log_value_eur'].max():.3f}")

    # === Save one CSV per position group ==============================================================================
    print(f"\nSaving position-specific CSVs to: {PROC_DIR}")

    for group, key in POS_KEYS.items():
        group_df = df[df["position_group"] == group].copy()

        # Keep only features relevant to this position + target + identifiers
        feature_cols = POSITION_FEATURES[group]
        keep_cols    = (
            ["sofifa_id", "short_name", "long_name", "club",
             "player_positions", "position_group",
             TARGET_COL, "log_value_eur"]
            + [c for c in feature_cols if c in group_df.columns]
        )
        keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate, preserve order
        group_df  = group_df[keep_cols]

        # Fill remaining NaN in numeric features with column median
        feature_cols_present = [c for c in feature_cols if c in group_df.columns]
        for col in feature_cols_present:
            if group_df[col].isna().any():
                group_df[col] = group_df[col].fillna(group_df[col].median())

        #output_path = os.path.join(PROC_DIR, f"{key}_data.csv")                                              #
        output_path = PROC_DIR / f"{key}_data.csv"
        group_df.to_csv(output_path, index=False)

        print(f"  {group:<12} ({key}) → {len(group_df):>4,} players, "
              f"{len(feature_cols_present)} features → {output_path}")

    # === Summary Statistics ===========================================================================================
    print(f"\nMarket Value Summary by Position (EUR):")
    print(f"{'Position':<14} {'Median':>12} {'Mean':>12} {'Max':>14}")
    print("-" * 55)
    for group in POS_KEYS:
        g = df[df["position_group"] == group][TARGET_COL]
        print(f"{group:<14} {g.median():>12,.0f} {g.mean():>12,.0f} {g.max():>14,.0f}")

    print("\n" + "=" * 60)
    print(" NEXT STEP: python src/02_feature_engineering.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

