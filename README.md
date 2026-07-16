# Customer Churn Prediction: A Machine Learning Project

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Business Objectives](#2-business-objectives)
3.  [Dataset](#3-dataset)
4.  [Methodology](#4-methodology)
    *   [Data Loading and Initial Inspection](#41-data-loading-and-initial-inspection)
    *   [Data Cleaning](#42-data-cleaning)
    *   [Exploratory Data Analysis (EDA)](#43-exploratory-data-analysis-eda)
    *   [Data Preprocessing](#44-data-preprocessing)
    *   [Model Training and Evaluation](#45-model-training-and-evaluation)
    *   [Hyperparameter Tuning](#46-hyperparameter-tuning)
    *   [Feature Importance](#47-feature-importance)
5.  [Key Findings and Insights](#5-key-findings-and-insights)
6.  [Model Selection and Performance](#6-model-selection-and-performance)
7.  [Future Work and Next Steps](#7-future-work-and-next-steps)
8.  [Technologies Used](#8-technologies-used)

## 1. Project Overview

This project aims to build and evaluate machine learning models to predict customer churn for a telecommunications company. By identifying customers at risk of churning, the company can proactively implement targeted retention strategies, thereby minimizing customer loss and optimizing business performance.

## 2. Business Objectives

*   To accurately predict which customers are likely to churn (binary classification).
*   To identify the key demographic, service, and contractual factors that contribute to customer churn.
*   To provide a robust and deployable machine learning model that can assist in data-driven decision-making for customer retention programs.

## 3. Dataset

The dataset, `Churn Dataset.csv`, contains information on a telecommunication company's customers, including various demographic details, service subscriptions, and account information, alongside a `Churn` column indicating whether the customer churned or not.

## 4. Methodology

### 4.1. Data Loading and Initial Inspection

The dataset was loaded using `pandas`. Initial inspection confirmed **7043 entries** and **21 columns**, with `TotalCharges` identified as an `object` type requiring conversion.

### 4.2. Data Cleaning

*   **`TotalCharges` Conversion:** The `TotalCharges` column was successfully converted from `object` to `float`, coercing errors to `NaN`. This revealed **11 rows** where `TotalCharges` was initially an empty string, corresponding to customers with `0` tenure.
*   **Missing Value Imputation:** The **11 `NaN` values** in `TotalCharges` were imputed with `0`, based on the logical assumption that new customers with `0` tenure would have `0` total charges.
*   **Duplicate Check:** No duplicate rows were found in the dataset.
*   **Feature Removal:** The `customerID` column was dropped as it's a unique identifier and holds no predictive value for churn.

### 4.3. Exploratory Data Analysis (EDA)

*   **Target Variable Imbalance:** The `Churn` variable showed a significant class imbalance: **73.46% 'No' (non-churn)** vs. **26.54% 'Yes' (churn)**.
*   **Numerical Features:**
    *   **Tenure:** Bimodal distribution, with peaks at low (`<10 months`) and high (`>60 months`) tenure, suggesting customers either churn early or stay long-term.
    *   **MonthlyCharges:** Bimodal distribution, with peaks around `$20` and `$70-100`, likely indicating different service tiers.
    *   **TotalCharges:** Heavily right-skewed, peaking near `0`, correlating with lower tenure customers.
*   **Categorical Features vs. Churn:**
    *   **Contract Type:** `Month-to-month` contracts showed significantly higher churn rates than `One year` or `Two year` contracts.
    *   **Internet Service:** Customers with `Fiber optic` internet had higher churn, while those with `No internet service` had very low churn.
    *   **Payment Method:** `Electronic check` was associated with the highest churn rate.
    *   **Additional Services (OnlineSecurity, TechSupport, etc.):** Absence of these services correlated with higher churn, acting as retention factors.
*   **Correlation:** `tenure` and `TotalCharges` showed a strong positive correlation (`0.83`).

### 4.4. Data Preprocessing

*   **Target Encoding:** The `Churn` variable was converted to numerical: `Yes` -> `1`, `No` -> `0`.
*   **One-Hot Encoding:** All remaining categorical features were one-hot encoded using `pd.get_dummies`, resulting in a DataFrame with **31 numerical columns**.
*   **Data Splitting:** The data was split into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` and `random_state=42` to maintain the class distribution in both sets.
*   **Feature Scaling:** `StandardScaler` was applied to both `X_train` and `X_test`, fitted only on `X_train` to prevent data leakage.

### 4.5. Model Training and Evaluation

Several classification models were trained and evaluated on the scaled data, using `class_weight='balanced'` or equivalent techniques to address imbalance. Performance was primarily assessed using ROC AUC, F1-Score, Precision, and Recall.

*   **Logistic Regression (Baseline):**
    *   ROC AUC: **0.8413**
    *   Accuracy: **0.7402**
    *   Precision: **0.5069**
    *   Recall: **0.7861**
    *   F1-Score: **0.6164**
*   **Decision Tree:** ROC AUC: **0.6500**
*   **Random Forest:** ROC AUC: **0.8217**
*   **Gradient Boosting:** ROC AUC: **0.8415**
*   **K-Nearest Neighbors:** ROC AUC: **0.7716**

### 4.6. Hyperparameter Tuning

`GridSearchCV` was used to tune the `RandomForestClassifier` with `scoring='roc_auc'` and `class_weight='balanced'`.

*   **Best Parameters:** `{'class_weight': 'balanced', 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}`
*   **Best Cross-Validation ROC AUC:** **0.8464**
*   **Tuned Random Forest Performance on Test Set:**
    *   ROC AUC: **0.8425**
    *   Accuracy: **0.7672**
    *   Precision: **0.5449**
    *   Recall: **0.7460**
    *   F1-Score: **0.6298**

### 4.7. Feature Importance

Analysis of the Tuned Random Forest model's feature importances revealed:

1.  `tenure`: **0.1706**
2.  `TotalCharges`: **0.1457**
3.  `Contract_Two year`: **0.1059**
4.  `MonthlyCharges`: **0.0991**
5.  `InternetService_Fiber optic`: **0.0721**
6.  `PaymentMethod_Electronic check`: **0.0518**

These features are the most influential predictors of customer churn.

### 4.8. XGBoost Classifier

An XGBoost Classifier was also trained, handling class imbalance with `scale_pos_weight`.

*   **XGBoost Performance on Test Set:**
    *   ROC AUC: **0.8425**
    *   Accuracy: **0.7551**
    *   Precision: **0.5267**
    *   Recall: **0.7647**
    *   F1-Score: **0.6238**

## 5. Key Findings and Insights

*   **Class Imbalance:** The target variable `Churn` is significantly imbalanced, necessitating careful handling during model training and evaluation.
*   **Customer Lifecycle:** Customers with very short or very long tenure are more common than those in the mid-range. Churners predominantly have lower tenure and lower total charges but higher monthly charges.
*   **Service & Contract Impact:**
    *   `Month-to-month` contracts are a strong indicator of churn risk.
    *   `Fiber optic` internet users have higher churn rates.
    *   Absence of value-added services (e.g., Online Security, Tech Support) increases churn likelihood.
*   **Payment Method:** `Electronic check` users are more prone to churn.
*   **Model Consistency:** Logistic Regression, Tuned Random Forest, and XGBoost all achieved very similar and strong ROC AUC scores (approx. **0.84**), indicating comparable predictive power.

## 6. Model Selection and Performance

The **Tuned Random Forest Model** was selected as the preferred model due to its robust performance (ROC AUC of **0.8425** on the test set), interpretability (through feature importances), and competitive metrics across Accuracy, Precision, Recall, and F1-Score. It offers a good balance of identifying churners (high recall) while maintaining reasonable precision.

The model has been saved as `best_churn_prediction_model.joblib` for future deployment.

## 7. Future Work and Next Steps

*   **Deployment:** Integrate the saved model into a production environment (e.g., via a REST API, batch prediction service).
*   **Monitoring:** Implement continuous monitoring of model performance, data drift, and concept drift to ensure its effectiveness over time.
*   **A/B Testing:** Evaluate the impact of retention strategies derived from the model through A/B tests.
*   **Further Optimization:** Explore advanced techniques such as stacking ensembles, deep learning models, or more sophisticated imbalance handling if higher performance is required.

## 8. Technologies Used

*   `Python`
*   `pandas` (Data manipulation)
*   `numpy` (Numerical operations)
*   `matplotlib` (Data visualization)
*   `seaborn` (Statistical data visualization)
*   `scikit-learn` (Machine learning models, preprocessing, evaluation)
*   `xgboost` (Gradient Boosting)
*   `joblib` (Model persistence)

## Author: Lawrence Ngukusuk
## Email:  ngukusuklawrence@gmail.com
## Phone:  +237081442142
## Date :  July,2026

