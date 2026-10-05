# NASA Software Defect Prediction — XGBoost + K-Fold

Predicts whether software modules in NASA mission code contain bugs, using XGBoost with Stratified K-Fold cross-validation. Trained on four labeled datasets (CM1, JM1, KC1, PC1) and generates predictions on the blind KC2 test set.

## Notebook Steps

### 1. Import Required Libraries
Loads pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, Optuna (hyperparameter tuning), and imbalanced-learn (SMOTE oversampling).

### 2. Load Training Datasets
Reads four NASA CSV files (CM1, JM1, KC1, PC1). The first two rows after the header are metadata and are skipped. Column names are normalized to lowercase. The `defects` column is mapped to binary (1 = buggy, 0 = clean).

### 3. Exploratory Data Analysis
- Checks for missing values in each dataset.
- Reports class distribution (defect rate ranges from ~7% to ~16%).
- Plots bar charts of class distribution per dataset.
- Shows a feature correlation heatmap for CM1.

### 4. Data Preprocessing and Feature Engineering
Selects 21 common software metric features (Halstead metrics, McCabe complexity, LOC counts, etc.). Missing values are filled with the column median. A combined dataset from all four sources is also created.

### 5. Define XGBoost Model and K-Fold Strategy
Sets up Stratified K-Fold (k=5) to preserve class ratios across folds. Defines a base XGBoost model factory with `scale_pos_weight` to handle class imbalance.

### 6. Train XGBoost with K-Fold on Each Dataset
Runs 5-fold cross-validation on each individual dataset, tracking accuracy, F1-score, and AUC per fold. Uses the class imbalance ratio as `scale_pos_weight`.

### 7. Evaluate Cross-Validation Results
Builds a summary table comparing mean accuracy, F1, and AUC across all four datasets. Visualizes with grouped bar charts.

### 8. F1-Optimized Hyperparameter Tuning (Optuna + SMOTE)
Uses Optuna (100 trials) to search over 10 XGBoost hyperparameters, maximizing mean F1 across K-Fold. SMOTE oversampling is applied **inside each fold** to avoid data leakage. Key parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`.

### 9. Threshold Optimization + Final Model
- Collects out-of-fold probability predictions using the best Optuna params.
- Sweeps classification thresholds (0.05–0.95) to find the one that maximizes F1 (often lower than 0.5 for imbalanced data).
- Trains the final model on all SMOTE-resampled combined data.
- Plots feature importance from the final model.

### 10. Load and Preprocess Blind Test (KC2)
Loads `kc2_test_blind.csv`, aligns columns with training features (fills missing columns like `loccodeandcomment` with 0), and fills any NaN values with training medians.

### 11. Predict on Blind Test Set
Generates predictions using the tuned XGBoost model with the F1-optimal threshold. Outputs a histogram of predicted probabilities.

### 12. Save Submission
Writes `kc2_blind_predictions.csv` in the required format:
```
id,defects
0,0
1,1
2,0
...
```
- `id` — row index (0 to 521)
- `defects` — 1 (buggy) or 0 (clean)

## Project Structure

```
├── README.md
├── xgboost_kfold_defect_prediction.ipynb   # Main notebook
├── Team Notebook.ipynb                      # Challenge description
├── datasets/datasets/                       # Training CSVs (cm1, jm1, kc1, pc1)
├── test-blind/test-blind/                   # Blind test CSV (kc2)
├── kc2_blind_predictions.csv                # Generated predictions
└── pyproject.toml
```

## Requirements

- Python 3.12+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, optuna, imbalanced-learn
