🧠 NervSightX — Stress Detection from Social Media using Classical Machine Learning

A complete end-to-end ML system for detecting Stress vs Non-Stress using the Dreaddit dataset.

🚀 Overview

NervSightX is a classical-ML pipeline that detects stress signals in social media text.
It combines TF-IDF, Truncated SVD, and 111 psycholinguistic features (LIWC + DAL + syntax + sentiment + social metadata) into a fused machine-learning system with:

clean preprocessing

stratified 80/20 train/test split

5-fold CV without leakage

full OOF predictions

multiple base models

stacking ensemble

statistical tests

interpretability

ablation studies

Everything is implemented with classical ML only (no deep learning).

📂 Project Structure
NervSightX/
│
├── dreaddit_StressAnalysis - Sheet1.csv
│
├── dreaddit_cv_raw_splits/
│   ├── train_raw_with_clean_text.csv
│   ├── test_frozen_raw_with_clean_text.csv
│   ├── tfidf/
│   ├── svd/
│   ├── lexical/
│   ├── fused/
│   ├── selected_features/
│   └── folds_selected/
│
├── Machine learning/
│   └── models/
│       ├── gaussiannb/
│       ├── logreg/
│       ├── svm_fast/
│       ├── dt/
│       ├── rf_baseline/
│       ├── rf_tuned/
│       ├── rf_tuned_corrected/
│       ├── lgbm/
│       └── lgbm_tuned_quick/
│
├── dreaddit_analysis_outputs/
│   └── logreg_test_preds.csv
│
└── README.md

📊 I. Dataset

Dreaddit Stress Analysis Dataset

Rows: 715

Columns: 116

Target: label

1 = Stress

0 = Non-Stress

Feature Groups
Group	Count	Description
LIWC (lex_liwc_*)	93	Psycholinguistic categories
DAL (lex_dal_*)	9	Activation, imagery, pleasantness
Syntax	2	ARI, FK grade
Sentiment	1	Polarity
Social	4	Karma, timestamp
Text	1	Raw + clean text
🧹 II. Preprocessing & Feature Engineering
✔ 1. Text Cleaning

lowercase

remove URLs, emails, markdown, mentions

keep ? and !

normalize whitespace

output → clean_text

✔ 2. Train/Test Split

Stratified 80/20

Test set frozen

Train used for 5-fold CV only

✔ 3. TF-IDF

1–2 grams

Vocabulary size: 2051

Saved TF-IDF matrices + fitted vectorizer

✔ 4. Dimensionality Reduction

TruncatedSVD (200 components)

Explained variance ≈ 65.6%

✔ 5. Lexical Features

111 LIWC + DAL + syntax + sentiment + social features

Imputed (though no missing values)

Standardized (fit only on train)

✔ 6. Feature Fusion

Combined:
200 SVD + 111 lexical = 311 features

✔ 7. L1 Feature Selection

L1 Logistic Regression

Reduced 311 → 34 final features

These 34 were used for all CV folds & all models

🔁 III. Cross-Validation Pipeline

Stratified 5-fold CV

Each fold contains:

X_train_selected.npy

X_val_selected.npy

train/val CSV with orig_index, label, clean_text

Imputer + scaler applied within each fold only

No leakage into test set

🤖 IV. Base Models (with OOF Predictions)

For each model:

✔ trained on 5 folds
✔ generated OOF predictions
✔ per-fold metrics
✔ saved pipelines and CSV outputs

Models Implemented

Gaussian Naive Bayes

Logistic Regression

Linear SVM

Decision Tree

Random Forest (baseline)

Random Forest (tuned & corrected)

LightGBM (baseline)

LightGBM (quick tuned)

Best models:

Random Forest (tuned)

Linear SVM

Logistic Regression

🧬 V. Stacking Ensemble

Using OOF predictions from all strong base models:

Constructed OOF meta-matrix

Meta-learner options:

Logistic Regression

LightGBM

Evaluated on frozen test set

Outputs saved for reproducibility

📈 VI. Evaluation & Analysis
1. Metrics

Accuracy

Macro F1

Weighted F1

ROC-AUC

PR-AUC

Precision-Recall curves

Confusion matrix (raw + normalized)

Calibration curves

Brier score

2. Confidence Intervals

Bootstrap (1000 samples) to compute 95% CI for:

F1

ROC-AUC

PR-AUC

3. Statistical Significance

McNemar test: compares paired model predictions

Wilcoxon signed-rank test: compares probability outputs

Uses:

logreg_test_preds.csv

other model test predictions

🔎 VII. Explainability
✔ Logistic Regression Coefficients

Interpret strongest positive/negative predictors.

✔ SHAP

global summary

per-feature importance

per-sample local explanations

✔ LIME

Token-level interpretability of raw text.

✔ TF-IDF-only LR Baseline

Parallel interpretable system for human inspection.

❌ VIII. Error Analysis

Extracted FP / FN from test set:

For each:

clean text

true label

predicted label

probability

SHAP explanation

Categorized common failure causes:

sarcasm

long trauma posts

very short posts

ambiguous sentiment

annotation noise in Dreaddit

🧨 IX. Ablation Studies

Ablations performed:

TF-IDF only

Lexical only

Fused features

Feature selection ON/OFF

SMOTE ON/OFF

SVD dimension = 100 / 200 / 300

Time-based train/test split simulation

🏁 X. Final Notes

Entire system is 100% classical ML

No transformers, no deep learning

Strict prevention of data leakage

Fully modular pipeline

Suitable for academic papers, hackathons, and production demos

✨ Citation

Dataset: Dreaddit Stress Analysis
Author: Ayush Singh (NervSightX)
Pipeline: Custom classical-ML architecture
