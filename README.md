# Soccer Goal Prediction (Expected Goals Model)

Machine learning model to predict goal probability from shot characteristics using StatsBomb event data.

## Overview
This project builds a Random Forest classifier to predict whether a soccer shot will result in a goal, generating Expected Goals (xG) values. The model handles class imbalance through weighting and produces well-calibrated probability estimates.

## Dataset
- Source: StatsBomb Open Data (preprocessed training/test sets)
- Features: Shot location (x,y coordinates), shot type, body part, play pattern
- Target: Binary classification (goal / no goal)
- Class distribution: ~11% goals, ~89% non-goals (heavily imbalanced)

## Methods
- **Preprocessing:** One-hot encoding for categorical features, standardization for numeric
- **Model:** Random Forest classifier with class weighting
- **Calibration:** Adjusted for real-world goal rates using custom weights
- **Validation:** Calibration curves, Brier score, game-level probability analysis

## Key Results
- Well-calibrated probabilities (calibration error < 5%)
- Accurate game-level predictions using Poisson-Binomial distribution
- Demonstrated critical importance of handling class imbalance

## Files
- `Goal_Prediction.ipynb` - Main analysis notebook
- `xg_train.csv` - Training data
- `xg_test.csv` - Test data

## Technologies
Python, scikit-learn, pandas, matplotlib, scipy

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib scipy seaborn
jupyter notebook Goal_Prediction.ipynb
```
