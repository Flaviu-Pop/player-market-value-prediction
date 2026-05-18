# src/06_predict.py

# Step 6: Prediction Interface: Given a player's attributes, predict their market value using the trained model.

# Two usage modes:
#   A) Interactive CLI — prompts you to enter attributes one by one
#   B) Direct API     — call predict_player() from another script

# Usage:
#   python src/06_predict.py
#   python src/06_predict.py --position Forward --interactive
#   python src/06_predict.py --demo   (runs built-in demo examples)

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR, POS_KEYS, POSITION_FEATURES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================================================================
# Add the class definition directly, again (instead of importing from 03_train_models.py):
# The problem is that Python does not allow to import a file that starts with a number

import torch.nn as nn
from config import HIDDEN_DIMS, DROPOUT_RATES

class PlayerValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout_rates=DROPOUT_RATES):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)
# ======================================================================================================================


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_artifacts(key: str):
    """Load model, scaler and feature list for a position group."""

    model_path   = os.path.join(MODELS_DIR, f"{key}_model.pth")
    scaler_path  = os.path.join(MODELS_DIR, f"{key}_scaler.joblib")
    feature_path = os.path.join(MODELS_DIR, f"{key}_features.joblib")

    for path in [model_path, scaler_path, feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run scripts 01–03 first to train the models."
            )

    features = joblib.load(feature_path)
    scaler   = joblib.load(scaler_path)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    input_dim  = checkpoint["input_dim"]

    model      = PlayerValueNet(input_dim).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, features


# ── Core Prediction Function ───────────────────────────────────────────────────

def predict_player(position_group: str, attributes: dict) -> dict:
    """
    Predict market value for a single player.

    Parameters
    ----------
    position_group : str
        One of 'Goalkeeper', 'Defender', 'Midfielder', 'Forward'
    attributes : dict
        Player attribute values keyed by feature name.
        Missing features will be filled with 0 (use median values for accuracy).

    Returns
    -------
    dict with keys:
        position_group   : str
        predicted_eur    : float  (market value in EUR)
        predicted_m_eur  : float  (market value in millions EUR)
        confidence_range : tuple  (low, high) — ±25% range
        features_used    : list
        missing_features : list   (features not provided)
    """

    if position_group not in POS_KEYS:
        raise ValueError(
            f"Invalid position group: '{position_group}'. "
            f"Must be one of: {list(POS_KEYS.keys())}"
        )

    key             = POS_KEYS[position_group]

    model, scaler, features = load_artifacts(key)

    # Build feature vector — fill missing with 0
    missing = [f for f in features if f not in attributes]

    feature_vector = np.array(
        [attributes.get(f, 0.0) for f in features], dtype=np.float32
    ).reshape(1, -1)


    # Scale
    feature_vector_scaled = scaler.transform(feature_vector)


    # Predict
    x_tensor = torch.tensor(feature_vector_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        log_pred = model(x_tensor).item()

    predicted_eur = float(np.expm1(log_pred))
    predicted_eur = max(predicted_eur, 0)

    # Confidence range: ±25% (rough interval — not a formal CI)
    low  = predicted_eur * 0.75
    high = predicted_eur * 1.25

    return {
        "position_group":   position_group,
        "predicted_eur":    round(predicted_eur),
        "predicted_m_eur":  round(predicted_eur / 1_000_000, 2),
        "confidence_range": (round(low), round(high)),
        "features_used":    features,
        "missing_features": missing,
    }


# === Interactive CLI ==================================================================================================

def interactive_predict(position_group: str) -> None:
    """Prompt user to enter attribute values one by one, then predict."""

    key = POS_KEYS[position_group]

    try:
        _, _, features = load_artifacts(key)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    print(f"\n{'='*60}")
    print(f" Market Value Prediction — {position_group}")
    print(f"{'='*60}")

    print(f" Enter attribute values below.")
    print(f" Press ENTER to skip (uses 0 — enter median for best accuracy).")

    print(f"{'='*60}\n")

    attributes = {}
    for feature in features:
        while True:
            raw = input(f"  {feature:<35} : ").strip()
            if raw == "":
                attributes[feature] = 0.0
                break
            try:
                attributes[feature] = float(raw)
                break
            except ValueError:
                print(f"  Please enter a number.")

    result = predict_player(position_group, attributes)
    _print_result(result)


def _print_result(result: dict) -> None:
    """Pretty-print prediction result."""
    print(f"\n{'='*60}")
    print(f" PREDICTION RESULT")
    print(f"{'='*60}")

    print(f" Position       : {result['position_group']}")
    print(f" Predicted Value: €{result['predicted_eur']:>15,.0f}")
    print(f"                : €{result['predicted_m_eur']:.2f}M")
    print(f" Confidence Range (±25%):")
    low, high = result['confidence_range']
    print(f"   Low  : €{low:>15,.0f}")
    print(f"   High : €{high:>15,.0f}")

    if result['missing_features']:
        print(f"\n NOTE: {len(result['missing_features'])} features set to 0 "
              f"(not provided):")
        for f in result['missing_features'][:5]:
            print(f"   - {f}")
        if len(result['missing_features']) > 5:
            print(f"   ... and {len(result['missing_features'])-5} more")

    print(f"{'='*60}\n")


# === Demo Examples ====================================================================================================

DEMO_PLAYERS = {

    "Goalkeeper": {
        "name": "Elite Young GK (like Donnarumma profile)",
        "attrs": {
            "age": 22, "height_cm": 196, "weight_kg": 90,
            "overall": 85, "potential": 91,
            "international_reputation": 3, "weak_foot": 3, "skill_moves": 1,
            "gk_diving": 86, "gk_handling": 82, "gk_kicking": 77,
            "gk_reflexes": 88, "gk_speed": 52, "gk_positioning": 84,
            "movement_reactions": 84, "mentality_composure": 73, "power_jumping": 78,
        }
    },

    "Defender": {
        "name": "Top CB Profile (like Van Dijk at peak)",
        "attrs": {
            "age": 27, "height_cm": 193, "weight_kg": 92,
            "overall": 90, "potential": 91,
            "international_reputation": 3, "weak_foot": 3, "skill_moves": 2,
            "pace": 77, "defending": 90, "physic": 86,
            "attacking_heading_accuracy": 86, "attacking_short_passing": 78,
            "skill_long_passing": 81, "movement_reactions": 88,
            "power_jumping": 90, "power_strength": 92, "power_stamina": 75,
            "mentality_aggression": 82, "mentality_interceptions": 89,
            "mentality_composure": 89, "defending_marking": 91,
            "defending_standing_tackle": 92, "defending_sliding_tackle": 85,
        }
    },

    "Midfielder": {
        "name": "Box-to-Box CM Profile (like De Bruyne)",
        "attrs": {
            "age": 28, "height_cm": 181, "weight_kg": 70,
            "overall": 91, "potential": 91,
            "international_reputation": 4, "weak_foot": 5, "skill_moves": 4,
            "pace": 76, "passing": 92, "dribbling": 86, "defending": 61, "physic": 78,
            "attacking_crossing": 93, "attacking_short_passing": 92,
            "skill_dribbling": 82, "skill_long_passing": 91, "skill_ball_control": 91,
            "movement_acceleration": 77, "movement_reactions": 91, "power_stamina": 89,
            "mentality_aggression": 76, "mentality_interceptions": 61,
            "mentality_positioning": 88, "mentality_vision": 94, "mentality_composure": 91,
        }
    },

    "Forward": {
        "name": "Elite Striker Profile (like Lewandowski)",
        "attrs": {
            "age": 30, "height_cm": 184, "weight_kg": 80,
            "overall": 89, "potential": 89,
            "international_reputation": 4, "weak_foot": 4, "skill_moves": 4,
            "pace": 77, "shooting": 87, "dribbling": 85, "physic": 82,
            "attacking_crossing": 62, "attacking_finishing": 88,
            "attacking_heading_accuracy": 85, "attacking_volleys": 88,
            "skill_dribbling": 81, "skill_curve": 77,
            "movement_acceleration": 77, "movement_sprint_speed": 77,
            "movement_agility": 78, "movement_reactions": 90,
            "power_shot_power": 87, "power_jumping": 84, "power_strength": 84,
            "mentality_positioning": 91, "mentality_composure": 86,
            "mentality_penalties": 86,
        }
    },
}


def run_demo() -> None:
    """Run predictions for four built-in demo player profiles."""

    print("\n" + "=" * 60)
    print(" DEMO MODE — Predicting 4 example player profiles")
    print("=" * 60)

    for position_group, demo in DEMO_PLAYERS.items():
        print(f"\n── {position_group}: {demo['name']} ──")
        try:
            result = predict_player(position_group, demo["attrs"])
            _print_result(result)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")


# === Entry Point ======================================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Football Player Market Value Predictor"
    )

    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo predictions for 4 built-in player profiles"
    )

    parser.add_argument(
        "--position", type=str,
        choices=["Goalkeeper", "Defender", "Midfielder", "Forward"],
        help="Position group for interactive prediction"
    )

    parser.add_argument(
        "--interactive", action="store_true",
        help="Launch interactive attribute input mode"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.interactive and args.position:
        interactive_predict(args.position)
    else:
        # Default: run demo then offer interactive mode
        run_demo()

        print("\nWould you like to predict a custom player?")
        choice = input("Enter position (Goalkeeper/Defender/Midfielder/Forward) "
                       "or press ENTER to exit: ").strip()
        if choice in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            interactive_predict(choice)
        else:
            print("Exiting. Run with --interactive --position <group> anytime.")


if __name__ == "__main__":
    main()


