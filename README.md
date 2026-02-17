# 🎓 Student Retention Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Optimized-blueviolet.svg)](https://optuna.org/)

## 📌 Objective
The objective of this machine learning project is to classify whether a university student will successfully complete their degree or program. By analyzing a rich dataset of demographic information, academic history, and extracurricular engagement, this classification pipeline aims to identify at-risk students. The ultimate goal is to enable data-driven, proactive intervention and improve overall student retention strategies.

## 📊 Data Pipeline & Preprocessing
The dataset consists of 748 instances with a mix of numerical and categorical features (e.g., generation, school, major, class formats). To prepare the raw data for the machine learning algorithms, a robust preprocessing pipeline was implemented:

* **Categorical Encoding:** Applied `LabelEncoder` to transform 24 distinct object and boolean features into machine-readable numerical formats.
* **Handling Missing Data:** Utilized mean imputation to systematically fill `NaN` values across continuous features.
* **Feature Scaling:** Applied `StandardScaler` to normalize numerical features, ensuring that variables with larger magnitudes do not disproportionately dominate the objective functions of the models.
* **Validation Strategy:** Data was partitioned using an 80/20 train-test split to evaluate model generalization.

## 🧠 Modeling Strategy
A systematic, multi-stage approach was taken to develop the final predictive model, moving from rapid prototyping to complex ensemble learning.

### 1. Baseline Benchmarking
* Utilized **PyCaret** to rapidly train and evaluate a diverse suite of 15 standard machine learning algorithms. 
* This baseline testing identified tree-based models and Naive Bayes as strong candidates for capturing the underlying patterns in the data.

### 2. Bayesian Hyperparameter Optimization
* Selected three robust models for deeper tuning: **Gaussian Naive Bayes**, **Decision Tree Classifier**, and **XGBoost Classifier**.
* Leveraged **Optuna** to conduct 100 trials of Bayesian optimization for each model, maximizing the Macro F1-Score via 10-fold cross-validation to ensure stability against overfitting.

### 3. Ensemble Learning (Stacking)
To maximize predictive performance and reduce variance, a **Stacking Classifier** was implemented to synthesize the strengths of the individual tuned models:
* **Level-0 (Base Estimators):** Tuned `GaussianNB` and `DecisionTreeClassifier` configured to capture distinct probabilistic and non-linear boundaries.
* **Level-1 (Meta-Learner):** A highly tuned `XGBClassifier` trained on the cross-validated predictions of the base estimators to make the final classification.

<br>

> <p align="center">
>   <img width="708" alt="Stacking Architecture" src="https://github.com/user-attachments/assets/67856bda-a33f-4d32-acd2-703839519da7">
> </p>

## 📈 Results & Evaluation
The final stacked ensemble was evaluated on the hold-out validation set. The metrics reflect the model's performance in identifying the minority class (at-risk students) within an imbalanced dataset:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.610 |
| **F1-Score** | 0.277 |
| **Precision** | 0.241 |
| **Recall** | 0.325 |

## 🚀 Challenges & Future Work
* **Class Imbalance:** Initial baseline testing revealed a majority-class distribution of roughly 70%. Because the cost of missing an at-risk student is high, future iterations will implement techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or adjust class weights to penalize false negatives more heavily.
* **Feature Engineering:** Further exploration into interaction terms between academic performance and extracurricular participation to boost signal-to-noise ratio.
* **Deployment:** Containerize the finalized inference pipeline using **Docker** and expose it via a **FastAPI** endpoint for real-time predictions.
