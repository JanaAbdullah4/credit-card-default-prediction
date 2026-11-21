# Credit Card Default Prediction (CPCS 433 - Group 12)

Predicting whether a credit card client will default on payment in the following month, using the UCI "Default of Credit Card Clients" dataset (30,000 records, 23 features, 22.12% default rate).

**Team:** Saudah Khaled Almhari, Jana Alshehri, Retal Alzahrani, Nawal Hussein Alghamdi

**Supervisor:** Lolwah Mohammed Alharigy

## Repository Structure
```
├── data/        train/test split indices shared by both pairs
├── results/     metrics, figures, and analysis outputs
├── credit-card-default-prediction.ipynb   Pair A notebook
├── requirements.txt
└── README.md
```

## Project Structure
| Pair | Model | Status |
|---|---|---|
| Pair A | Logistic Regression | Done (see results below) |
| Pair B | XGBoost | In progress |

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
- ROC-AUC is unchanged across experiments; these methods only move the decision point (precision vs. recall).
- The most influential feature is `PAY_1` (most recent repayment status); demographic features contribute little.
- BILL_AMT features are highly correlated with each other, so their individual coefficients should be interpreted with caution.

## Output Files
**`data/`**
- `train_idx.csv`, `test_idx.csv`: shared train/test split indices

**`results/`**
- `pair_a_results.csv`: test-set metrics for E1-E4
- `pair_a_cv_results.csv`: cross-validation results
- `pair_a_top_coefficients.csv`: top logistic regression coefficients (E3 model)
- `pair_a_confusion_matrices.png`: confusion matrices for E1, E3, E4
- `bill_amt_correlation_heatmap.png`: correlation between BILL_AMT features

## Notes
- The dataset contains 35 duplicate rows (after removing `ID`); they are kept for now (team decision pending).
- Results were produced with `random_state=42`.
