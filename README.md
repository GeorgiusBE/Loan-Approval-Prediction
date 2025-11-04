# Loan Approval Prediction

This project builds an end-to-end **binary classification** pipeline to predict whether a loan application will be **approved** based on applicant demographics, employment, income, and credit history features.

The focus is on:

- Handling **class imbalance** (few approved/declined relative to the majority)
- Training strong **gradient boosting models** (XGBoost, LightGBM, CatBoost)
- **Hyperparameter tuning** with Optuna + pruning
- **Probability calibration** and **decision threshold tuning**
- Building a **stacking ensemble** that outperforms individual models

The core work is in the Jupyter notebook:

> `loan_approval_final.ipynb`

---

## Project Overview

**Goal:**  
Predict loan approval (`approved` vs `not approved`) and optimise for **Precision–Recall performance** and **F1-score**, rather than accuracy, due to class imbalance.

**Key techniques:**

- Tree-based gradient boosting models (XGBoost, LightGBM, CatBoost)
- Cross-validated hyperparameter tuning with **Optuna** (TPE sampler)
- **Successive Halving** and model-specific **pruning callbacks** to stop bad trials early
- **Stacking** with a regularised logistic regression meta-learner
- **Probability calibration** (Platt / isotonic) and **threshold tuning** for better decision-making

---

## Data

The dataset contains:

- **Features:** Applicant demographics, employment information, income, loan attributes, and credit history
- **Target:** Binary label indicating whether a loan was **approved** (1) or **not approved** (0)

Preprocessing steps (implemented in the notebook):

- Handling missing values and basic cleaning
- **One-hot encoding** for nominal categorical variables
- **Ordinal encoding** for ordered categories (e.g. loan/credit grades)
- **Robust scaling** of skewed numerical features
- Ensuring consistent transformations for train / validation / test splits

---

## Modeling Approach

### Base Models

The following models are trained and tuned:

- **XGBoost** (`XGBClassifier`)
- **LightGBM** (`LGBMClassifier`)
- **CatBoost** (`CatBoostClassifier`)

For each model:

- Hyperparameters are tuned with **Optuna** using the **TPE sampler**
- Evaluation metric during tuning is **Average Precision (AP / PR-AUC)**
- **Successive Halving pruner** and library-specific **pruning callbacks** (for XGBoost/LightGBM/CatBoost) are used to cut underperforming trials early

### Hyperparameter Tuning & Pruning

Each trial in Optuna:

1. Samples a hyperparameter configuration (learning rate, depth, regularisation, sampling, etc.).
2. Trains the model with cross-validation.
3. Uses **per-boosting-round pruning**:
   - The model reports validation metrics each iteration.
   - A pruning callback calls `trial.report(...)` and `trial.should_prune()`.
   - Underperforming trials are **stopped early** to save compute.

---

## Threshold Tuning & Probability Calibration

After tuning base models, the project focuses on **turning probabilities into good decisions**:

- **Out-of-fold (OOF) predictions** are generated for each model via stratified k-fold CV.
- A custom meta-estimator (`ThresholdTunedClassifier` concept) is used to:
  - Take OOF predicted probabilities and true labels
  - Optionally learn a **calibration mapping**:
    - **Platt scaling (LogisticRegression)**
    - **Isotonic regression**
  - Search for the **F1-optimal decision threshold** on calibrated (or raw) probabilities
- This separates:
  - **Model fitting** (tuned for AP / PR-AUC)
  - **Threshold selection** (tuned for F1 / business-specific trade-off)

---

## Stacking Ensemble

To combine model strengths, a **stacking ensemble** is built:

1. For each sample, collect OOF probabilities from:
   - XGBoost
   - LightGBM
   - CatBoost
2. Stack these into a feature matrix `Z` with 3 columns (one per model).
3. Train a **L2-regularised Logistic Regression** meta-learner on `Z` to predict the final probability of approval.
4. Apply **probability calibration + threshold tuning** at the meta level.

This ensemble improves over the best individual model in both **PR-AUC** and **macro-F1**.

> Example performance (from the notebook):  
> **PR-AUC:** ~0.87 · **Macro-F1:** ~0.89

---

## Evaluation

The models are evaluated with:

- **Stratified k-fold cross-validation**
- **Average Precision (PR-AUC)** – primary tuning metric  
- **ROC-AUC**
- **F1-score (per class and macro)**

Class imbalance is handled via:

- **Class weights / scale_pos_weight** in XGBoost/LightGBM/CatBoost
- **Threshold tuning** instead of naive oversampling/undersampling

---
