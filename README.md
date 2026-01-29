# Stellar Classification: Galaxy, Star, and Quasar Identification

## 🌌 Project Overview
This project addresses the challenge of stellar classification using the **SDSS-DR17 (Sloan Digital Sky Survey)** dataset. The objective is to classify celestial objects into three distinct categories: **Galaxies, Stars, or Quasars (QSO)** based on their spectral and photometric characteristics.

The core of this project is a robust machine learning pipeline built on **Extreme Gradient Boosting (XGBoost)**, optimized through rigorous hyperparameter tuning and validated via stratified cross-validation.

## 📊 Dataset Insights
The dataset contains **100,000 observations** captured by the SDSS telescope. 
- **Key Features:** Photometric filters (u, g, r, i, z) and **Redshift** (the most significant predictor for classification).
- **Target Classes:** - `GALAXY`
  - `STAR`
  - `QSO` (Quasar)
  - **Source:** [Kaggle - Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-data)

## ⚙️ Methodology
- **Preprocessing:** Label Encoding for target variables and feature selection (dropping non-predictive IDs).
- **Model:** `XGBClassifier` with `logloss` evaluation metric.
- **Optimization Strategy:** `GridSearchCV` was employed to find the optimal configuration for:
  - `learning_rate`
  - `max_depth`
  - `n_estimators`
  - `gamma`
- **Validation:** **10-Fold Stratified Cross-Validation** was used to ensure the model's reliability across different data segments while maintaining class balance.

## 📈 Performance Comparison
The following table highlights the impact of **Hyperparameter Tuning** on the model's performance across different metrics:

| Metric | Untuned XGBoost (Baseline) | Tuned XGBoost (Optimized) |
| :--- | :---: | :---: |
| **Training Accuracy** | 98.67% | 98.71% |
| **Cross-Validation Mean Score** | 97.57% | 97.72% |
| **Testing Accuracy** | 97.45% | 97.53% |

### 🔍 Key Observations:
1. **Generalization:** After tuning, the gap between training and testing accuracy remained minimal, indicating a highly generalized model that doesn't suffer from overfitting.
2. **Stability:** The increase in the **Cross-Validation Mean Score** confirms that the hyperparameter optimization led to a more stable model across different subsets of the data.
3. **Precision & Recall:** The tuned model showed improved performance in identifying the `QSO` (Quasar) class, which is typically the most challenging due to its spectral similarity to certain stars.

## 🛠️ Technologies Used
- **Python** (Core Language)
- **XGBoost** (Gradient Boosting Framework)
- **Scikit-Learn** (Preprocessing, Tuning, and Validation)
- **Plotly & Matplotlib** (Data Visualization)
- **Pandas & NumPy** (Data Manipulation)

## 📜 Conclusion
By applying **XGBoost** coupled with **Stratified K-Fold** validation and automated tuning, the project achieved a high-fidelity classification system. The final model is capable of classifying celestial bodies with a testing accuracy of **97.53%**, making it a reliable tool for astronomical data analysis.
