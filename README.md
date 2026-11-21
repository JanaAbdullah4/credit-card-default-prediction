# Credit Card Default Prediction (CPCS 433 - Group 12)

Predicting whether a credit card client will default on payment in the following month, using the UCI "Default of Credit Card Clients" dataset (30,000 records, 23 features, 22.12% default rate).

**Team:** Saudah Khaled Almhari, Jana Alshehri, Retal Alzahrani, Nawal Hussein Alghamdi

**Supervisor:** Lolwah Mohammed Alharigy

## Repository Structure
```
├── data/        train/test split indices shared by both pairs
├── results/     metrics, figures, and analysis outputs
├── credit-card-default-prediction.ipynb            Pair A notebook (Logistic Regression)
├── credit_card_default_prediction_xgboost.ipynb    Pair B notebook (XGBoost)
├── requirements.txt
└── README.md
```

## Project Structure
| Pair | Model | Status |
|---|---|---|
| Pair A | Logistic Regression | Done (see results below) |
| Pair B | XGBoost | Done (see results below) |

Both pairs use the same train/test split (`data/train_idx.csv`, `data/test_idx.csv`) so results are directly comparable.

## Dataset
Download `default of credit card clients.xls` from
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
and place it in the project folder as `default_of_credit_card_clients.xls`.

## Requirements
```
pip install -r requirements.txt
```

## How to Run (Pair A)
1. Open `credit-card-default-prediction.ipynb` in Google Colab (or Jupyter).
2. Upload the dataset file (name as above) to the same folder.
3. Run all cells in order. The notebook will:
   - clean the data (drop `ID`, merge undocumented EDUCATION/MARRIAGE codes, one-hot encode),
   - create the stratified 80/20 split (`random_state=42`) and save `train_idx.csv` / `test_idx.csv`,
   - run four experiments, 5-fold cross-validation, and feature-coefficient analysis,
   - save the result files listed below.
4. The notebook saves its outputs in the working folder. In this repository they were moved into `data/` and `results/`.

To reuse the shared split, load it from `data/`:
```python
train_idx = pd.read_csv("data/train_idx.csv").iloc[:, 0]
test_idx = pd.read_csv("data/test_idx.csv").iloc[:, 0]
```

## Pair A: Experiments
All experiments use Logistic Regression (C = 1) inside a pipeline with standard scaling. SMOTE is applied inside the pipeline, so only on training data. Each experiment changes one thing only:

| ID | Experiment | Description |
|---|---|---|
| E1 | Baseline | No imbalance handling |
| E2 | SMOTE | Oversampling of the minority class |
| E3 | class_weight | `class_weight='balanced'` |
| E4 | Threshold tuning | Threshold chosen by 5-fold CV on the training set (max F1) |

C was tuned with Grid Search (5-fold CV, ROC-AUC): performance plateaued from C = 0.01 upward, so C = 1 was kept.

## Pair A: Results (test set, n = 6,000)
| Experiment | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| E1 Baseline | 0.50 | 0.81 | 0.69 | 0.24 | 0.36 | 0.71 |
| E2 SMOTE | 0.50 | 0.67 | 0.36 | 0.64 | 0.46 | 0.71 |
| E3 class_weight | 0.50 | 0.68 | 0.37 | 0.63 | 0.46 | 0.71 |
| E4 Threshold tuning | 0.29 | 0.79 | 0.54 | 0.47 | 0.50 | 0.71 |

5-fold cross-validation on the training set (mean ± std):

| Experiment | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| E1 Baseline | 0.25 ± 0.01 | 0.71 ± 0.02 | 0.37 ± 0.01 | 0.73 ± 0.01 |
| E2 SMOTE | 0.66 ± 0.01 | 0.37 ± 0.01 | 0.48 ± 0.01 | 0.73 ± 0.01 |
| E3 class_weight | 0.65 ± 0.02 | 0.38 ± 0.01 | 0.48 ± 0.01 | 0.73 ± 0.01 |

**Key findings**
- Accuracy is misleading: the baseline has the highest accuracy but detects only 24% of defaulters.
- SMOTE and class weighting perform equivalently (differences are within one standard deviation).
- For Logistic Regression, ROC-AUC is unchanged across experiments; these methods only move the decision point (precision vs. recall).
- - The most influential feature is `PAY_1` (most recent repayment status); demographic features contribute little.
- BILL_AMT features are highly correlated with each other, so their individual coefficients should be interpreted with caution.

## Output Files
**`data/`**
- `train_idx.csv`, `test_idx.csv`: shared train/test split indices

## Pair A: Output Files
- `pair_a_results.csv`: test-set metrics for E1-E4
- `pair_a_cv_results.csv`: cross-validation results
- `pair_a_top_coefficients.csv`: top logistic regression coefficients (E3 model)
- `pair_a_confusion_matrices.png`: confusion matrices for E1, E3, E4
- `bill_amt_correlation_heatmap.png`: correlation between BILL_AMT features

## How to Run (Pair B)

1. Open `credit_card_default_prediction_xgboost.ipynb` in Google Colab.
2. Place the dataset and the shared split (`data/train_idx.csv`, `data/test_idx.csv`) in Google Drive, or clone this repository and upload the dataset.
3. Run all cells in order. The notebook applies the same cleaning as Pair A, loads the shared split, tunes hyperparameters, runs four experiments, 5-fold cross-validation, and feature-importance analysis, and saves the result files listed below.

## Pair B: Experiments

All experiments use XGBoost (`subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`). No scaling is needed for tree-based models. SMOTE is applied inside the pipeline, so only on training data. Each experiment changes one thing only:

| ID | Experiment | Description |
|----|------------|-------------|
| E1 | Baseline | No imbalance handling |
| E2 | SMOTE | Oversampling of the minority class |
| E3 | scale_pos_weight | `scale_pos_weight = 3.52` (negatives / positives in the training set) |
| E4 | Threshold tuning | Threshold chosen by 5-fold CV on the training set (max F1) |

Hyperparameters were tuned with Grid Search (5-fold CV, ROC-AUC) on the training set over `max_depth` [3, 5], `learning_rate` [0.05, 0.1], and `n_estimators` [200, 400]. Best: `max_depth=5`, `learning_rate=0.05`, `n_estimators=200` (CV ROC-AUC 0.784 ± 0.006). The top three configurations differ by less than one standard deviation.

## Pair B: Results (test set, n = 6,000)

| Experiment | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------------|-----------|----------|-----------|--------|----|---------|
| E1 Baseline | 0.50 | 0.82 | 0.66 | 0.36 | 0.47 | 0.78 |
| E2 SMOTE | 0.50 | 0.77 | 0.48 | 0.51 | 0.49 | 0.75 |
| E3 scale_pos_weight | 0.50 | 0.77 | 0.48 | 0.62 | 0.54 | 0.78 |
| E4 Threshold tuning | 0.31 | 0.80 | 0.54 | 0.54 | 0.54 | 0.78 |

5-fold cross-validation on the training set (mean ± std):

| Experiment | Recall | Precision | F1 | ROC-AUC |
|------------|--------|-----------|----|---------|
| E1 Baseline | 0.37 ± 0.01 | 0.67 ± 0.01 | 0.48 ± 0.01 | 0.78 ± 0.01 |
| E2 SMOTE | 0.51 ± 0.01 | 0.49 ± 0.01 | 0.50 ± 0.01 | 0.75 ± 0.01 |
| E3 scale_pos_weight | 0.62 ± 0.01 | 0.48 ± 0.01 | 0.54 ± 0.01 | 0.78 ± 0.01 |

**Key findings**

- Accuracy is misleading: the baseline has the highest accuracy (0.82) but detects only 36% of defaulters.
- `scale_pos_weight` and threshold tuning reach the same F1 (0.54). E3 has higher recall (0.62), while E4 has higher precision (0.54) and accuracy (0.80), so the choice depends on the relative cost of missed defaulters versus false alarms.
- SMOTE performs worst among the imbalance methods and lowers ROC-AUC (0.75 vs 0.78), unlike with Logistic Regression where ROC-AUC was unchanged. A possible explanation is that synthetic samples interpolated across one-hot encoded features are not realistic, but this was not tested.
- The most important feature is PAY_1 (importance 0.317), followed by PAY_2 and PAY_3; repayment history dominates, consistent with Pair A.

## Comparison: Logistic Regression vs XGBoost (test set)

| Experiment | LR F1 | XGB F1 | LR Recall | XGB Recall |
|------------|-------|--------|-----------|------------|
| E1 Baseline | 0.36 | 0.47 | 0.24 | 0.36 |
| E2 SMOTE | 0.46 | 0.49 | 0.64 | 0.51 |
| E3 class weight | 0.46 | 0.54 | 0.63 | 0.62 |
| E4 Threshold tuning | 0.50 | 0.54 | 0.47 | 0.54 |

XGBoost has a clearly higher ROC-AUC (about 0.78 vs 0.71), meaning it ranks defaulters better regardless of the threshold, and a higher F1 in every experiment. Both pairs use the same split, so the results are directly comparable.

## Pair B: Output Files

results/

- `pair_b_results.csv`: test-set metrics for E1-E4
- `pair_b_cv_results.csv`: cross-validation results
- `pair_b_top_importances.csv`: top XGBoost feature importances (E3 model, gain-based)
- `pair_b_confusion_matrices.png`: confusion matrices for E1, E3, E4
- `pair_b_feature_importance.png`: top 10 feature importances

## Notes
- The dataset contains 35 duplicate rows (after removing `ID`); they are kept for now (team decision pending).
- Results were produced with `random_state=42`.


