# NASA Software Defect Prediction — Random Forest

Alternative approach using **Random Forest** with a train/test split (no K-Fold, no XGBoost).

## Notebook Steps

### 1. Import Libraries
Loads pandas, numpy, matplotlib, seaborn, scikit-learn (RandomForestClassifier), imbalanced-learn (SMOTE), and Optuna for hyperparameter tuning.

### 2. Load and Combine Training Data
Reads four NASA CSV files (CM1, JM1, KC1, PC1), skipping metadata rows. Normalizes column names and maps `defects` to binary (1/0). Combines all datasets into a single training set of ~16,900 rows with 21 features.

### 3. Train/Test Split + SMOTE
Splits combined data 80/20 with stratification to preserve class ratios. Applies SMOTE oversampling **only on the training set** to balance the minority (buggy) class.

### 4. Baseline Random Forest
Trains a Random Forest with 200 trees and `class_weight="balanced"`. Reports accuracy, F1, and AUC on the held-out test set along with a full classification report.

### 5. Optuna Hyperparameter Tuning (F1)
Runs 100 Optuna trials to find the best Random Forest parameters, optimizing for F1 score. Parameters tuned:
- `n_estimators` (100–1000)
- `max_depth` (3–30)
- `min_samples_split` (2–20)
- `min_samples_leaf` (1–10)
- `max_features` (sqrt, log2, None)
- `criterion` (gini, entropy)

### 6. Threshold Optimization
Sweeps classification thresholds from 0.05 to 0.95 to find the threshold that maximizes F1. Plots threshold vs F1 curve.

### 7. Confusion Matrix + Feature Importance
Displays the confusion matrix at the optimal threshold with a classification report. Plots horizontal bar chart of feature importances.

### 8. Final Model + Blind Predictions
Retrains the best model on **all** SMOTE-resampled data. Loads the KC2 blind test set, aligns features, and generates predictions using the optimal threshold.

## Output Format

`kc2_blind_predictions_rf.csv`:
```
id,defects
0,0
1,1
2,0
...
```

- `id` — row index (0 to 521)
- `defects` — 1 (buggy) or 0 (clean)

## Requirements

- Python 3.12+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, optuna, imbalanced-learn
