# Football Player Market Value Prediction
### Position-Specific Deep Learning Models | FIFA 20 Dataset | PyTorch

**Author:** Flaviu Pop  
**Tools:** Python · PyTorch · Pandas · NumPy · Matplotlib · Seaborn  
**Data Source:** FIFA 20 Complete Player Dataset (SoFIFA)  
**Status:** In Progress  

---

## Project Overview

This project builds **four separate PyTorch neural network models** — one per
position group (Goalkeeper, Defender, Midfielder, Forward) — to predict football
player market values from in-game performance attributes.

The core insight driving the architecture: a goalkeeper's market value depends on
completely different attributes than a striker's. Training a single model across
all positions forces it to average across incompatible feature spaces. Position-
specific models learn the right relationships for each role.

### Key Questions Answered
- Which attributes most strongly predict market value for each position?
- How accurately can a neural network estimate player market value?
- Which players are undervalued relative to their predicted value?
- Given a set of player attributes, what is the predicted market value?

---

## Project Structure

```
player_market_value_prediction/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore
│
├── src/
│   ├── config.py                    # Features per position, hyperparameters
│   ├── model.py                     # The model (architecture)
│   ├── 01_data_preparation.py       # Cleaning, position grouping, log transform
│   ├── 02_feature_engineering.py    # Position-specific feature selection & scaling
│   ├── 03_train_models.py           # Train 4 PyTorch models with early stopping
│   ├── 04_evaluate_models.py        # MAE, MAPE, R², residual analysis
│   ├── 05_visualizations.py         # All charts and comparison plots
│   └── 06_predict.py                # Prediction interface for new players
│
├── data/
│   ├── raw/players_raw.csv          # Source data (place here)
│   └── processed/                   # Generated cleaned CSVs per position
│
├── models/                          # Saved .pth model files (generated)
│   ├── gk_model.pth
│   ├── def_model.pth
│   ├── mid_model.pth
│   └── fwd_model.pth
│
├── outputs/
│   ├── charts/                      # All visualization outputs
│   └── metrics/                     # Evaluation results as CSV/JSON
│
├── notebooks/
│   └── 01_exploration.ipynb         # Exploratory data analysis
│
└── tests/
    └── test_pipeline.py             # Data and model validation tests
```

---

## Model Architecture

Each position uses the same neural network architecture with position-specific
input dimensions and feature sets:

```
Input Layer  →  128 neurons (BatchNorm + ReLU + Dropout 0.3)
             →   64 neurons (BatchNorm + ReLU + Dropout 0.2)
             →   32 neurons (BatchNorm + ReLU)
             →    1 neuron  (log market value output)
```

**Key design decisions:**
- **Log-transformed target:** Market values are exponentially distributed.
  Predicting log(value) then exponentiating gives dramatically better results.
- **BatchNormalization:** Stabilizes training across different feature scales.
- **Dropout:** Prevents overfitting on ~18K training samples.
- **Early stopping:** Stops training when validation loss stops improving.
- **Position-specific features:** Each model trains only on attributes
  relevant to its position group.

---

## Position-Specific Feature Groups

| Position | Key Features Used |
|---|---|
| **Goalkeeper** | gk_diving, gk_handling, gk_kicking, gk_reflexes, gk_speed, gk_positioning, movement_reactions, age, overall, potential |
| **Defender** | defending, physic, pace, defending_marking, defending_standing_tackle, defending_sliding_tackle, mentality_interceptions, power_jumping, power_strength, age, overall, potential |
| **Midfielder** | passing, dribbling, defending, physic, attacking_short_passing, skill_long_passing, mentality_vision, power_stamina, movement_reactions, age, overall, potential |
| **Forward** | shooting, dribbling, pace, attacking_finishing, skill_dribbling, movement_acceleration, mentality_positioning, mentality_composure, age, overall, potential |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Flaviu-Pop/football-market-value-prediction.git
cd football-market-value-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows CMD
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add data
Place `players_raw.csv` in the `data/raw/` folder.

---

## Usage — Run in Order

```bash
# Step 1: Clean and prepare data
python src/01_data_preparation.py

# Step 2: Engineer position-specific features
python src/02_feature_engineering.py

# Step 3: Train all four models
python src/03_train_models.py

# Step 4: Evaluate model performance
python src/04_evaluate_models.py

# Step 5: Generate visualizations
python src/05_visualizations.py

# Step 6: Predict market value for a new player
python src/06_predict.py
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error in EUR — most interpretable for a club |
| **MAPE** | Mean Absolute Percentage Error — comparable across positions |
| **R²** | Overall model fit quality |
| **Residual analysis** | Where does the model under/over-predict? |
| **Top undervalued players** | Players where predicted >> actual value |

---

## Planned Improvements (v2)
- Compare PyTorch models against XGBoost baseline
- SHAP values for feature importance explanation

---

## About the Author

Flaviu Pop is a Data Scientist/Machine Learning Engineer and Football Analyst with a PhD in Mathematics and over 15 years 
of academic experience at Babeș-Bolyai University. He completed the Data Analytics in Sport course from Johan Cruyff Institute 
and is currently completing a Professional Diploma in Football Tactical Analysis at the Barça Innovation Hub.

- GitHub: [github.com/Flaviu-Pop](https://github.com/Flaviu-Pop)
- LinkedIn: [linkedin.com/in/flaviu-pop-61b00369](https://linkedin.com/in/flaviu-pop-61b00369)
- Google Scholar: https://scholar.google.com/citations?user=KgQVIt4AAAAJ&hl=en

---

## License

Educational and portfolio use only.  
Data source: FIFA 20 Complete Player Dataset via SoFIFA / Kaggle.
