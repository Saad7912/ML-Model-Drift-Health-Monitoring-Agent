# Loan Default Prediction & Temporal Data Drift Analysis

A machine learning project for **loan default prediction and temporal data drift analysis** using the Lending Club dataset. The project focuses not only on model performance but also on understanding how model behavior and feature distributions change as financial data evolves over time.

---

## 📌 Project Overview

Loan default prediction is a highly imbalanced classification problem where correctly identifying potential defaulters is critical.

This project investigates:

* Loan default prediction
* Feature selection and engineering
* Class imbalance handling
* Time-based model validation
* Temporal data drift
* Model stability across future time periods

Instead of using a conventional random train-test split, the project uses **historical data for training and future data for evaluation**, providing a more realistic simulation of deploying a credit-risk model in the real world.

---

## 📊 Dataset

**Lending Club Loan Dataset**

* **1.3M+ loan records**
* Initial features: **151**
* Final selected features: **58**
* Target: `loan_status`
* Converted target variable: `target`

  * `0` → Fully Paid
  * `1` → Charged Off
* Class distribution:

  * ~80% Fully Paid
  * ~20% Charged Off

---

## 🔍 Feature Selection

The original dataset contained **151 features**, which were progressively reduced to **58 features**.

The feature selection pipeline included:

1. Exploratory Data Analysis (EDA)
2. Initial domain-based feature selection
3. Removal of features with **>85% missing values**
4. Numerical feature-feature correlation analysis
5. Numerical feature-target correlation
6. Chi-Square test for categorical features
7. Cramér's V analysis
8. Temporal feature engineering
9. Mutual Information
10. Random Forest feature importance

This reduced the dimensionality while retaining features with meaningful predictive information.

---

## ⚙️ Feature Engineering

Several temporal features were engineered from the available date information.

### Credit History

Created:

```text
credit_history_months
```

using:

```text
issue_d - earliest_cr_line
```

### Loan Issue Date

The `issue_d` feature was converted into a proper datetime representation and used to analyze the distribution of loans across years.

An additional feature was created:

```text
issue_month
```

to capture possible seasonal patterns.

---

## 🕒 Time-Based Model Validation

A major focus of the project was evaluating **temporal generalization**.

Instead of randomly splitting the dataset, historical data was used for training and future data was used for evaluation.

### Training Data

```text
2007 – 2014
451,060 loans
```

### Future / Drift Data

```text
2015
375,546 loans

2016
293,105 loans

2017
169,321 loans

2018
56,318 loans
```

This setup allows the project to investigate whether a model trained on historical lending behavior remains reliable as the underlying data distribution changes.

---

## ⚖️ Class Imbalance

The dataset contains approximately:

```text
80% Fully Paid
20% Charged Off
```

To address this imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

Importantly, SMOTE was applied **only to the training data**, while the test/future data remained unchanged to preserve realistic evaluation conditions.

---

## 🤖 Models

Three classification algorithms were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

Evaluation metrics included:

* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

Because identifying loan defaults is particularly important, **Recall for the default class** was given significant consideration.

---

## 📈 Temporal Data Drift Analysis

Model performance alone does not tell the complete story.

The project therefore investigates whether the input data distribution changes over time.

Two major statistical approaches were used:

### Population Stability Index (PSI)

PSI was used to quantify distribution changes between the historical reference period and subsequent years.

```text
Reference → 2007–2014
Current   → 2015–2018
```

### Kolmogorov–Smirnov (KS) Test

The KS test was used to identify statistically significant distribution shifts in numerical features across different years.

Categorical drift was additionally investigated using:

* Chi-Square
* Cramér's V

---

## 📊 Model Stability Over Time

The model was evaluated separately on each future year:

```text
2015
2016
2017
2018
```

This makes it possible to observe whether predictive performance deteriorates as the model moves further away from its training period.

The analysis includes:

* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC

This provides a practical view of **model degradation under temporal distribution shift**.

---

## 🏗️ Project Pipeline

```text
Raw Lending Club Dataset
          │
          ▼
        EDA
          │
          ▼
Initial Feature Selection
          │
          ▼
Missing Value Analysis
          │
          ▼
Feature Correlation Analysis
          │
          ▼
Feature-Target Analysis
          │
          ▼
Chi-Square / Cramér's V
          │
          ▼
Temporal Feature Engineering
          │
          ▼
Mutual Information
          │
          ▼
Random Forest Feature Selection
          │
          ▼
       58 Features
          │
          ▼
Time-Based Split
   ┌──────┴────────┐
   ▼               ▼
2007–2014       2015–2018
 Training       Future Data
   │
   ▼
     SMOTE
   │
   ▼
ML Model Training
   │
   ├── Logistic Regression
   ├── Random Forest
   └── Gradient Boosting
          │
          ▼
Performance Evaluation
          │
          ▼
Temporal Drift Analysis
          │
          ├── PSI
          ├── KS Test
          ├── Chi-Square
          └── Cramér's V
```

---

## 🔑 Key Findings

* Reduced the original feature space from **151 → 58 features**.
* Used a **time-based split** instead of conventional random validation.
* Evaluated model behavior on **future unseen years**.
* Addressed severe class imbalance using **SMOTE**.
* Gradient Boosting achieved the strongest overall ROC-AUC/PR-AUC among the evaluated models.
* Identified significant distribution shifts in several features between historical and future periods.
* Observed changes in model performance across 2015–2018, demonstrating the importance of monitoring **temporal data drift** in credit-risk models.

---

## 🛠️ Technologies Used

```text
Python
Pandas
NumPy
Scikit-learn
SciPy
Imbalanced-learn
Matplotlib
Seaborn
Jupyter Notebook
Kaggle
```

---

## 📁 Project Structure

```text
loan-default-drift-analysis/
│
├── notebook/
│   └── loan_default_prediction.ipynb
│
├── data/
│   └── README.md
│
├── results/
│   ├── feature_selection/
│   ├── model_evaluation/
│   └── drift_analysis/
│
├── README.md
└── requirements.txt
```

> The original dataset is not included in this repository due to its size.

---

## 🚀 Future Improvements

* Automated drift monitoring pipeline
* Model retraining when drift exceeds predefined thresholds
* Probability calibration for better default-risk estimation
* Hyperparameter optimization
* SHAP-based model explainability
* Real-time prediction API
* Interactive drift monitoring dashboard
* Agentic AI system for automatically explaining detected drift

---

## 👨‍💻 Project Focus

This project demonstrates an end-to-end approach to **credit-risk machine learning**, going beyond simply training a classifier by investigating:

> **"Does a model trained on historical financial data remain reliable when the underlying data changes over time?"**

The combination of **time-based validation, class-imbalance handling, feature selection, model comparison, and temporal drift analysis** makes this project focused on a realistic machine learning deployment challenge rather than only achieving a high test-set score.
