# Credit Card Default Prediction (CPCS 433 - Group 12)

Predicting credit card default using the UCI "Default of Credit Card Clients" dataset.

## Dataset
Download from https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
and place it in the project folder as `default_of_credit_card_clients.xls`.

## Requirements
pip install pandas scikit-learn imbalanced-learn xlrd openpyxl matplotlib

## How to run
1. Run the Pair A notebook/script (Logistic Regression).
2. It creates `train_idx.csv` / `test_idx.csv` (shared split) and `pair_a_results.csv`.

## Structure
- Pair A: Logistic Regression (Baseline, SMOTE, class_weight, threshold tuning)
- Pair B: XGBoost (coming soon)
- results/: metrics and confusion matrices
