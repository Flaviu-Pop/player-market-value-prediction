# src/config.py
# Central configuration for all scripts
# Edit hyperparameters and feature lists here

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV         = os.path.join(BASE_DIR, "data", "raw",       "players_20.csv")
PROC_DIR        = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
CHARTS_DIR      = os.path.join(BASE_DIR, "outputs", "charts")
METRICS_DIR     = os.path.join(BASE_DIR, "outputs", "metrics")

for d in [PROC_DIR, MODELS_DIR, CHARTS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Position Groups ────────────────────────────────────────────────────────────
POSITION_GROUPS = {
    "Goalkeeper": ["GK"],
    "Defender":   ["CB", "LB", "RB", "LWB", "RWB"],
    "Midfielder": ["CDM", "CM", "CAM", "LM", "RM"],
    "Forward":    ["LW", "RW", "ST", "CF", "SS"],
}

# Short keys used for file naming and model saving
POS_KEYS = {
    "Goalkeeper": "gk",
    "Defender":   "def",
    "Midfielder": "mid",
    "Forward":    "fwd",
}

# ── Target Variable ────────────────────────────────────────────────────────────
TARGET_COL      = "value_eur"
MIN_VALUE       = 10_000        # Filter out zero/near-zero value players
USE_LOG_TARGET  = True          # Predict log(value) — strongly recommended

# ── General Features (used by all positions) ───────────────────────────────────
GENERAL_FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "overall",
    "potential",
    "international_reputation",
    "weak_foot",
    "skill_moves",
]

# ── Position-Specific Features ─────────────────────────────────────────────────
POSITION_FEATURES = {

    "Goalkeeper": GENERAL_FEATURES + [
        "gk_diving",
        "gk_handling",
        "gk_kicking",
        "gk_reflexes",
        "gk_speed",
        "gk_positioning",
        "movement_reactions",
        "mentality_composure",
        "power_jumping",
    ],

    "Defender": GENERAL_FEATURES + [
        "pace",
        "defending",
        "physic",
        "attacking_heading_accuracy",
        "attacking_short_passing",
        "skill_long_passing",
        "movement_reactions",
        "power_jumping",
        "power_strength",
        "power_stamina",
        "mentality_aggression",
        "mentality_interceptions",
        "mentality_composure",
        "defending_marking",
        "defending_standing_tackle",
        "defending_sliding_tackle",
    ],

    "Midfielder": GENERAL_FEATURES + [
        "pace",
        "passing",
        "dribbling",
        "defending",
        "physic",
        "attacking_crossing",
        "attacking_short_passing",
        "skill_dribbling",
        "skill_long_passing",
        "skill_ball_control",
        "movement_acceleration",
        "movement_reactions",
        "power_stamina",
        "mentality_aggression",
        "mentality_interceptions",
        "mentality_positioning",
        "mentality_vision",
        "mentality_composure",
    ],

    "Forward": GENERAL_FEATURES + [
        "pace",
        "shooting",
        "dribbling",
        "physic",
        "attacking_crossing",
        "attacking_finishing",
        "attacking_heading_accuracy",
        "attacking_volleys",
        "skill_dribbling",
        "skill_curve",
        "movement_acceleration",
        "movement_sprint_speed",
        "movement_agility",
        "movement_reactions",
        "power_shot_power",
        "power_jumping",
        "power_strength",
        "mentality_positioning",
        "mentality_composure",
        "mentality_penalties",
    ],
}

# ── Train / Validation / Test Split ───────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── Neural Network Hyperparameters ────────────────────────────────────────────
HIDDEN_DIMS   = [128, 64, 32]   # Neurons per hidden layer
DROPOUT_RATES = [0.3, 0.2, 0.0] # Dropout after each hidden layer
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4             # L2 regularization
MAX_EPOCHS    = 300
PATIENCE      = 20               # Early stopping patience (epochs)

# ── Visualization ──────────────────────────────────────────────────────────────
POSITION_COLORS = {
    "Goalkeeper": "#3498DB",   # Blue
    "Defender":   "#2ECC71",   # Green
    "Midfielder": "#F39C12",   # Orange
    "Forward":    "#E74C3C",   # Red
}
FIG_SIZE_WIDE   = (14, 6)
FIG_SIZE_SQUARE = (10, 10)
DPI             = 150
