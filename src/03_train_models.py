# src/03_train_models.py

# Step 3: Train one PyTorch model per position group with early stopping

# Input : data/processed/{key}_train/val.csv
# Output: models/{key}_model.pth + training history CSVs

# Usage: python src/03_train_models.py


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import warnings

from torch.utils.data import DataLoader, TensorDataset
from config import (
    PROC_DIR, MODELS_DIR, METRICS_DIR, POS_KEYS,
    HIDDEN_DIMS, DROPOUT_RATES,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, RANDOM_SEED
)
from model import PlayerValueNet


warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Data Loading =====================================================================================================
def load_split(key: str, split: str) -> TensorDataset:
    """Load a train/val/test CSV and return a TensorDataset."""

    path = os.path.join(PROC_DIR, f"{key}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 02_feature_engineering.py first."
        )

    df     = pd.read_csv(path)
    X      = torch.tensor(df.drop(columns=["target"]).values, dtype=torch.float32)
    y      = torch.tensor(df["target"].values, dtype=torch.float32)

    return TensorDataset(X, y)


# === Training Loop ====================================================================================================
def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            preds      = model(X_batch)

            total_loss += criterion(preds, y_batch).item() * len(X_batch)

    return total_loss / len(loader.dataset)


def train_position_model(group: str, key: str) -> dict:
    """
    Full training pipeline for one position group.
    Returns history dict.
    """

    print(f"\n── Training: {group} ({key}) ────────────────────────────────")

    # Load data
    train_ds = load_split(key, "train")
    val_ds   = load_split(key, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_ds[0][0].shape[0]

    print(f"  Input dim   : {input_dim} features")
    print(f"  Train size  : {len(train_ds):,}")
    print(f"  Val size    : {len(val_ds):,}")
    print(f"  Device      : {DEVICE}")

    # Model, optimizer, scheduler
    model     = PlayerValueNet(input_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    criterion = nn.MSELoss()

    # Training with early stopping
    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    history        = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    print(f"\n  Epoch │ Train Loss │  Val Loss │    LR")
    print(f"  ──────┼────────────┼───────────┼────────────")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["lr"].append(current_lr)

        # Print every 20 epochs and on improvement
        if epoch % 20 == 0 or val_loss < best_val_loss:
            marker = " ◄ best" if val_loss < best_val_loss else ""
            print(f"  {epoch:>5} │ {train_loss:>10.4f} │ {val_loss:>9.4f} │ "
                  f"{current_lr:.2e}{marker}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            model_path     = os.path.join(MODELS_DIR, f"{key}_model.pth")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":   best_val_loss,
                "input_dim":  input_dim,
            }, model_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val loss: {best_val_loss:.4f} at epoch {best_epoch})")
                break

    print(f"\n  Best epoch  : {best_epoch}")
    print(f"  Best val MSE: {best_val_loss:.4f}")
    print(f"  Model saved : models/{key}_model.pth")

    # Save training history
    hist_df   = pd.DataFrame(history)
    hist_path = os.path.join(METRICS_DIR, f"{key}_training_history.csv")
    hist_df.to_csv(hist_path, index=False)

    return history


# === Main =============================================================================================================
def main():
    print("=" * 60)
    print(f" Step 3: Training PyTorch Models")
    print(f" Device : {DEVICE}")
    print(f" Epochs : up to {MAX_EPOCHS} (early stop patience={PATIENCE})")
    print(f" Arch   : {HIDDEN_DIMS} hidden dims | dropout {DROPOUT_RATES}")
    print("=" * 60)

    summaries = {}
    for group, key in POS_KEYS.items():
        try:
            history = train_position_model(group, key)
            best_idx = history["val_loss"].index(min(history["val_loss"]))

            summaries[group] = {
                "best_epoch":    history["epoch"][best_idx],
                "best_val_loss": min(history["val_loss"]),
                "total_epochs":  len(history["epoch"]),
            }
        except FileNotFoundError as e:
            print(f"  SKIP {group}: {e}")

    print("\n" + "=" * 60)
    print(" Training Summary")
    print("=" * 60)
    print(f"{'Position':<14} {'Best Epoch':>11} {'Val MSE':>10} {'Total Epochs':>13}")
    print("-" * 52)

    for group, s in summaries.items():
        print(f"{group:<14} {s['best_epoch']:>11} "
              f"{s['best_val_loss']:>10.4f} {s['total_epochs']:>13}")

    print("\n" + "=" * 60)
    print(" NEXT STEP: python src/04_evaluate_models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()