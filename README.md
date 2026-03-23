# 👨🏻‍💻 NASA Software Defect Prediction — Team Notebook

> **For computer science students** — a complete, step-by-step guide to the
> V2 optimal pipeline used in this project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset: The PROMISE Metrics](#3-dataset-the-promise-metrics)
4. [Quick Start](#4-quick-start)
5. [V1 Baseline Pipeline (recap)](#5-v1-baseline-pipeline-recap)
6. [V2 Optimal Pipeline — Step by Step](#6-v2-optimal-pipeline--step-by-step)
   - [V2-A: More Cross-Validation Splits](#v2-a-more-cross-validation-splits)
   - [V2-B: Feature Engineering](#v2-b-feature-engineering)
   - [V2-C: Threshold Optimisation](#v2-c-threshold-optimisation)
   - [V2-D: Model Comparison](#v2-d-model-comparison)
   - [V2-E: Soft-Voting Ensemble](#v2-e-soft-voting-ensemble)
   - [V2-F: Improved Predictions](#v2-f-improved-predictions)
7. [Results Summary](#7-results-summary)
8. [Key Concepts Explained](#8-key-concepts-explained)
9. [Further Reading](#9-further-reading)

---

## 1. Project Overview

This project tackles the **NASA Software Defect Prediction** challenge from
the [PROMISE repository](http://promise.site.uottawa.ca/SERepository/).
The goal is to **classify software modules as defective or clean** using
static code metrics collected from real NASA projects (cm1, pc1, jm1, kc1).

The trained model is evaluated on a **held-out blind test set** from the kc2
project, producing a CSV of predictions.

This is a classic **binary classification** problem with two complicating
factors that are very common in real-world ML:

| Challenge | Description |
|---|---|
| **Class imbalance** | Only ~15 % of modules are defective (5.6:1 ratio) |
| **Distribution shift** | Train and test come from different codebases |

---

## 2. Repository Structure

```
nasa-promise/
├── datasets/datasets/       # Training CSVs  (cm1, pc1, jm1, kc1)
├── test-blind/test-blind/   # Blind test set (kc2_test_blind.csv)
├── Team Notebook.ipynb      # Main Jupyter notebook (V1 + V2)
├── predictions_TEAMNAME.csv     # V1 predictions
├── predictions_TEAMNAME_v2.csv  # V2 (improved) predictions
├── pyproject.toml           # Python project & dependency spec
└── README.md                # This file
```

---

## 3. Dataset: The PROMISE Metrics

Each row represents one **software module** (function/file). The features are
static code metrics — no runtime information is needed.

### Halstead Complexity Metrics
Derived from the raw token counts (operators + operands) in source code.

| Column | Full Name | What it measures |
|---|---|---|
| `n` | Program length | Total number of tokens |
| `v` | Volume | Information content (`n × log₂(η)`) |
| `l` | Level | Abstraction level (inverse of difficulty) |
| `d` | Difficulty | Number of mental comparisons needed |
| `i` | Intelligence | Volume ÷ Difficulty |
| `e` | Effort | Mental effort (`d × v`) |
| `b` | Bugs | Estimated bugs (`v / 3000`) |
| `t` | Time | Estimated coding time in seconds |
| `uniq_op` | Unique operators (η₁) | Vocabulary size (operators) |
| `uniq_opnd` | Unique operands (η₂) | Vocabulary size (operands) |
| `total_op` | Total operators (N₁) | Token count (operators) |
| `total_opnd` | Total operands (N₂) | Token count (operands) |

### McCabe Complexity Metrics

| Column | Full Name | What it measures |
|---|---|---|
| `loc` | Lines of code | Module size |
| `v(g)` | Cyclomatic complexity | Number of independent paths through code |
| `ev(g)` | Essential complexity | Irreducible control flow |
| `iv(g)` | Design complexity | Calls to other modules |
| `branchcount` | Branch count | Number of branches |

### Line-count columns

| Column | Meaning |
|---|---|
| `lOCode` | Lines of code only |
| `lOComment` | Lines of comments only |
| `lOBlank` | Blank lines |

> **Note:** `loccodeandcomment` is present in training data but absent in the
> test set and is therefore **dropped** before modelling.

### Target

| Column | Values |
|---|---|
| `defects` | `True` / `False` → encoded as `1` / `0` |

---

## 4. Quick Start

### Prerequisites

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip

### Install dependencies

```bash
# with uv (recommended — faster)
uv sync

# or with pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Run the notebook

```bash
jupyter lab "Team Notebook.ipynb"
```

Run all cells top-to-bottom. The V2 section starts after the horizontal rule
divider titled **"V2 — Improvements & Further Experiments"**.

The final output is written to `predictions_TEAMNAME_v2.csv`.

---

## 5. V1 Baseline Pipeline (recap)

The first part of the notebook establishes a solid baseline.

```
Raw CSVs  →  Normalise column names  →  Concat all 4 datasets
         →  Drop loccodeandcomment   →  RobustScaler
         →  XGBoost (scale_pos_weight)  →  5-Fold Stratified CV
         →  Final fit on all data    →  predictions_TEAMNAME.csv
```

Key design choices in V1:

- **`RobustScaler`** instead of `StandardScaler`: robust to the extreme outliers
  present in Halstead metrics (e.g., `e` can be > 800,000).
- **`scale_pos_weight = neg / pos ≈ 5.62`**: tells XGBoost to penalise
  missing a bug 5× more than mislabelling a clean module — compensates for
  the class imbalance.
- **5-Fold Stratified K-Fold**: keeps the defect rate approximately equal in
  every fold so no fold is accidentally all-clean.

**V1 results (5-fold CV):** ROC-AUC 0.761 | F1 0.426

---

## 6. V2 Optimal Pipeline — Step by Step

The V2 section layers six improvements on top of the V1 baseline.
Each one is self-contained and can be understood independently.

---

### V2-A: More Cross-Validation Splits

**Notebook cell:** `V2-A — More CV Splits`

#### Why?

With only 5 folds, each fold uses ~80 % of data for training and ~20 % for
validation. Increasing the number of folds:

- Uses more data per validation set evaluation → lower variance in the metric
- Gives a more reliable estimate of generalisation performance
- Helps decide whether performance differences between models are real or noise

#### What we compare

| Strategy | Folds | Description |
|---|---|---|
| `StratifiedKFold(n_splits=5)` | 5 | V1 baseline |
| `StratifiedKFold(n_splits=10)` | 10 | Each fold = 10 % held out |
| `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` | 25 | 5 independent runs of 5-fold |

**Repeated K-Fold** re-shuffles the data with a different random seed each
time, producing 25 estimates instead of 5. This almost always gives the most
stable (lowest standard deviation) score.

#### Results

| Strategy | ROC-AUC | F1 |
|---|---|---|
| 5-Fold | 0.7609 ± 0.011 | 0.4259 ± 0.012 |
| 10-Fold | 0.7642 ± 0.019 | 0.4244 ± 0.024 |
| Repeated 5×5 | **0.7635 ± 0.010** | 0.4245 ± 0.012 |

**Takeaway for CS students:** The Repeated Stratified K-Fold has the smallest
standard deviation — use it when you need the most trustworthy CV estimate.
The 10-fold gives the best mean here but has higher variance.

---

### V2-B: Feature Engineering

**Notebook cell:** `V2-B — Feature Engineering`

#### Why log1p-transform?

Software metrics are **heavily right-skewed** (a few modules are enormous).
For example, `e` (Halstead effort) can range from 1 to 870,000 in the same
dataset. Tree-based models are less affected by this than linear models, but
log-transforming still helps by:

1. Compressing outliers so they don't dominate the feature scale.
2. Making splits near the origin (where most modules live) more meaningful.

We use `np.log1p(x)` = log(1 + x) rather than `log(x)` because it is defined
at x = 0.

```python
X_train_log = np.log1p(np.clip(X_train.values, 0, None))
```

The `np.clip(..., 0, None)` first clamps any negative values to 0 (defensive
programming — these metrics should never be negative, but floating-point
rounding can occasionally produce tiny negatives).

#### Why interaction features?

Two ratios capture domain knowledge about software quality that is not encoded
in the raw metrics:

| Feature | Formula | Meaning |
|---|---|---|
| `ratio_ev_vg` | `(ev(g) + 1) / (v(g) + 1)` | What fraction of cyclomatic complexity is *essential* (irreducible)? A value close to 1 means the code is hard to simplify — often a code smell. |
| `ratio_loc_n` | `(loc + 1) / (n + 1)` | Code density — how many tokens per line? Very dense or very sparse code can indicate quality issues. |

We add 1 to numerator and denominator (Laplace smoothing) to avoid division
by zero.

#### Results

| Pipeline | ROC-AUC | F1 |
|---|---|---|
| Baseline (no eng) | 0.7642 ± 0.019 | 0.4244 ± 0.024 |
| **With feature engineering** | **0.7667 ± 0.022** | **0.4308 ± 0.028** |

Both metrics improve — the domain knowledge encoded in the interactions is
genuinely useful.

---

### V2-C: Threshold Optimisation

**Notebook cell:** `V2-C — Threshold Optimisation`

#### Why not use 0.5?

A classifier outputs a **probability** (e.g., 0.37 = "37 % chance of defect").
The **decision threshold** converts this to a binary label:

```
predicted class = 1  if  probability ≥ threshold  else  0
```

The default threshold is 0.5, but this is only F1-optimal when the classes
are perfectly balanced. With our 5.6:1 imbalance, we can improve F1 by
**lowering the threshold** (predicting "defect" more often), accepting more
false positives to catch more true positives.

#### How we find the optimal threshold

We use **out-of-fold (OOF) probabilities** — predictions made only on samples
that were *not* used for training. This prevents the threshold search from
overfitting to training data.

```
For each fold in 10-fold CV:
    Train model on 9/10 of data
    Predict probabilities on the remaining 1/10  ← OOF predictions
    Store predictions

After all folds:
    Sweep threshold from 0.05 to 0.95 (181 steps)
    Pick the threshold that maximises F1 on the OOF predictions
```

#### Precision-Recall trade-off

The **Precision-Recall curve** helps visualise this trade-off:

- **Precision** = of all modules predicted defective, how many actually are?
- **Recall** = of all truly defective modules, how many did we catch?

Lowering the threshold increases recall (catch more bugs) but decreases
precision (more false alarms). F1 is the harmonic mean, balancing the two.

#### Results

| Threshold | F1 (OOF) |
|---|---|
| 0.50 (default) | 0.4308 |
| **0.54 (optimal)** | **0.4323** |
| OOF ROC-AUC | 0.7666 |

The small improvement (+0.0015 F1) is typical — threshold tuning gives
diminishing returns once the model is well-calibrated, but it's a free win.

---

### V2-D: Model Comparison

**Notebook cell:** `V2-D — Model Comparison`

We benchmark three fundamentally different algorithms, all evaluated with
10-fold stratified CV on the same engineered features:

#### XGBoost (Extreme Gradient Boosting)

An ensemble of shallow decision trees trained **sequentially**, where each
new tree corrects the errors of the previous ones (boosting).

```python
XGBClassifier(
    n_estimators=500,      # 500 sequential trees
    max_depth=5,           # each tree is at most 5 levels deep
    learning_rate=0.05,    # shrinkage — slow learning is more robust
    subsample=0.8,         # 80% of rows per tree (stochastic)
    colsample_bytree=0.8,  # 80% of columns per tree (stochastic)
    scale_pos_weight=5.62, # imbalance correction
)
```

#### Random Forest

An ensemble of **deep** decision trees trained **independently** (bagging),
each on a random subset of rows and columns, predictions averaged.

```python
RandomForestClassifier(
    n_estimators=500,    # 500 independent trees
    max_depth=10,        # deeper than XGBoost trees
    class_weight="balanced",  # sklearn's equivalent of scale_pos_weight
)
```

#### Logistic Regression

A linear model — fast, interpretable, but assumes a linear decision boundary.

```python
LogisticRegression(
    class_weight="balanced",
    solver="saga",     # efficient for large datasets
    max_iter=1000,
)
```

#### Results (10-fold CV)

| Model | ROC-AUC | F1 |
|---|---|---|
| **XGBoost (engineered)** | **0.7667 ± 0.022** | **0.4308 ± 0.028** |
| Random Forest | 0.7647 ± 0.021 | 0.4225 ± 0.025 |
| Logistic Regression | 0.7150 ± 0.015 | 0.3575 ± 0.015 |

**Why XGBoost wins:** Software metrics have complex, non-linear interactions
(e.g., a module with high `v(g)` is risky *only if* `ev(g)` is also high).
Tree-based models capture these interaction effects automatically, whereas
Logistic Regression can only model linear boundaries.

**Why Random Forest is close:** Bagging + column subsampling also captures
interactions, but boosting is more sample-efficient on small-to-medium datasets.

---

### V2-E: Soft-Voting Ensemble

**Notebook cell:** `V2-E — Soft-Voting Ensemble`

#### What is an ensemble?

The idea: combine predictions from multiple models. If Model A and Model B
both predict "defect", we're more confident than if only one of them does.

**Soft voting** averages the **probability** outputs:

```
ensemble_probability = (p_xgboost + p_randomforest) / 2
```

This is better than **hard voting** (majority vote on binary labels) because
it preserves confidence information.

#### Why XGBoost + Random Forest?

These two models are **diverse** — they make errors on *different* samples:

- XGBoost: boosting builds trees sequentially, focusing on hard examples.
- Random Forest: bagging builds trees independently with random subsets.

When two models are diverse *and* individually accurate, their ensemble is
usually better than either alone. Combining two weak models that make the
same mistakes does not help.

#### Results

| Model | ROC-AUC | F1 |
|---|---|---|
| XGBoost alone | 0.7667 ± 0.022 | 0.4308 ± 0.028 |
| Random Forest alone | 0.7647 ± 0.021 | 0.4225 ± 0.025 |
| **Soft-Voting Ensemble** | **0.7714 ± 0.022** | **0.4339 ± 0.028** |

The ensemble beats both individual models on both metrics — the canonical
result from ensemble theory.

---

### V2-F: Improved Predictions

**Notebook cell:** `V2-F — Improved Predictions`

This is the **production step** — putting all V2 improvements together to
generate the final submission file.

#### Pipeline used for final predictions

```
All training data (cm1 + pc1 + jm1 + kc1)
    │
    ▼
log1p-transform + RobustScaler + interaction features  (V2-B)
    │
    ▼
Soft-Voting Ensemble (XGBoost + RandomForest) fit on ALL training data  (V2-E)
    │
    ▼
predict_proba() on kc2 test set
    │
    ▼
Apply optimal threshold = 0.54  (V2-C)
    │
    ▼
predictions_TEAMNAME_v2.csv
```

> **Important:** When training the final model we use **all** available
> training data (no held-out fold). The CV steps were only for *estimating*
> generalisation performance — not for generating the submission.

#### Output summary

| Metric | Value |
|---|---|
| Total kc2 test samples | 522 |
| Predicted buggy (1) | 124 (23.75 %) |
| Predicted clean (0) | 398 (76.25 %) |
| Agreement with V1 | 480 / 522 (92.0 %) |

The 8 % of samples that changed label between V1 and V2 are the borderline
cases near the decision boundary — exactly the modules where ensemble + tuned
threshold makes the biggest difference.

---

## 7. Results Summary

| Pipeline | ROC-AUC (CV) | F1 (CV) | Notes |
|---|---|---|---|
| V1 Baseline | 0.761 | 0.426 | 5-fold, XGBoost, RobustScaler |
| V2-A only | 0.764 | 0.424 | 10-fold CV, same model |
| V2-A + B | 0.767 | 0.431 | Log1p + interaction features |
| V2-A + B + C | 0.767 | 0.432 | + threshold tuning |
| **V2-E (Ensemble)** | **0.771** | **0.434** | **Best — used for submission** |

Each step adds a small but consistent improvement. In practice, **stacking
multiple small improvements** is how competitive ML systems are built.

---

## 8. Key Concepts Explained

### ROC-AUC vs F1 — when to use which?

| Metric | Good for | Pitfall |
|---|---|---|
| **ROC-AUC** | Comparing model ranking ability; threshold-independent | Optimistic on imbalanced datasets |
| **F1** | Measuring actual prediction quality at a given threshold | Threshold-dependent — always specify which threshold you used |

For defect prediction we care about **both**: ROC-AUC tells us how well the
model separates the classes; F1 tells us how useful the binary predictions
are in practice (catching bugs vs. raising false alarms).

### Why Stratified K-Fold?

In a normal K-Fold split, you might accidentally get a fold with very few
(or no) positive examples. Stratified K-Fold guarantees that the defect rate
in each fold is approximately equal to the dataset-wide rate (~15 %).
This is essential when the minority class is small.

### What is scale_pos_weight?

XGBoost's `scale_pos_weight` is a weight applied to the **positive class**
during gradient computation. Setting it to `neg / pos` (≈ 5.62) is
equivalent to saying:

> "Missing a defective module is 5.62× worse than mislabelling a clean one."

This is appropriate for bug-finding, where false negatives (missed bugs)
are more costly than false positives (wasted code reviews).

### What is RobustScaler?

`RobustScaler` centres each feature using the **median** and scales it using
the **interquartile range (IQR)** rather than the mean and standard deviation:

```
x_scaled = (x - median) / IQR
```

This makes the scaler insensitive to outliers. Compare:

| Scaler | Centre | Scale | Outlier sensitive? |
|---|---|---|---|
| `StandardScaler` | mean | std dev | Yes |
| `MinMaxScaler` | min | range | Very much yes |
| **`RobustScaler`** | **median** | **IQR** | **No** |

Since Halstead metrics can have extreme outliers (e.g., `e` up to 870,000),
`RobustScaler` is the right choice here.

### What is log1p and why does it help?

`log1p(x) = ln(1 + x)`

- At x = 0: log1p(0) = 0 ✓ (safe for zero-valued metrics)
- At x = 1: log1p(1) ≈ 0.69
- At x = 999: log1p(999) ≈ 6.9 (compresses from 999 to 6.9)

The compression of large values reduces the impact of extreme outliers on
splits in tree-based models and on distance calculations in scaling.

---

## 9. Further Reading

| Topic | Resource |
|---|---|
| XGBoost paper | Chen & Guestrin, 2016 — [arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754) |
| Halstead metrics | Halstead, M. H. (1977). *Elements of Software Science*. Elsevier. |
| McCabe complexity | McCabe, T. J. (1976). "A Complexity Measure". *IEEE Transactions on Software Engineering*, 2(4). |
| PROMISE repository | [promise.site.uottawa.ca/SERepository](http://promise.site.uottawa.ca/SERepository/) |
| Imbalanced learning | He & Garcia (2009). "Learning from Imbalanced Data". *IEEE TKDE*, 21(9). |
| Ensemble methods | Dietterich, T. (2000). "Ensemble Methods in Machine Learning". *MCS 2000*. |
| sklearn CV guide | [scikit-learn.org/stable/modules/cross_validation](https://scikit-learn.org/stable/modules/cross_validation.html) |

---

> **Tip for students:** Run the notebook cell by cell and read the comments.
> Each V2 section is self-contained — you can modify the hyperparameters,
> add a new model to V2-D, or swap in a different feature transformation in
> V2-B and immediately see the effect on CV scores.
