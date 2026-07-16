## Executive Summary: Quantifying Customer Churn Prediction

**Project Goal:** To develop a robust machine learning model to predict customer churn in a telecommunications company, enabling proactive retention strategies and minimizing customer attrition.

**Key Quantifiable Findings & Methodology:**

1.  **Data Initialisation & Cleaning:**
    *   Started with a dataset of **7043 customer records** across **21 features**.
    *   Identified and corrected a data type issue for `TotalCharges`, converting **11 non-numeric entries** to `NaN` and subsequently imputing them with `0` (for customers with 0 tenure).
    *   Confirmed **no duplicate records** and dropped the non-predictive `customerID` column.

2.  **Exploratory Data Analysis (EDA):**
    *   Uncovered a significant class imbalance in the target variable `Churn`: **73.46% non-churners** vs. **26.54% churners**.
    *   Observed strong positive correlation (`0.83`) between `tenure` and `TotalCharges`.
    *   Identified that churners typically have **lower median tenure** and `TotalCharges` but **higher median MonthlyCharges**.
    *   Categorical analysis highlighted `Month-to-month` contracts, `Fiber optic` internet service, and `Electronic check` payment methods as having notably higher churn rates.

3.  **Feature Engineering & Preprocessing:**
    *   Converted the `Churn` target variable to numerical (`1` for Yes, `0` for No).
    *   Applied one-hot encoding to **15 categorical features**, resulting in a processed dataset (`df_processed`) with **31 numerical features**.
    *   Split the data into training (80%, **5634 samples**) and testing (20%, **1409 samples**) sets, ensuring proportional class distribution via stratification.
    *   Applied `StandardScaler` to all features, standardizing their range for model training.

4.  **Model Training & Evaluation (Key Metrics on Test Set):**
    *   **Logistic Regression (Baseline):** Achieved a **ROC AUC of 0.8413**, with Accuracy: `0.7402`, Precision: `0.5069`, Recall: `0.7861`, F1-Score: `0.6164`.
    *   **Tuned Random Forest:** Optimized via `GridSearchCV` (best cross-validation ROC AUC: `0.8464`). Performance on test set: **ROC AUC of 0.8425**, Accuracy: `0.7672`, Precision: `0.5449`, Recall: `0.7460`, F1-Score: `0.6298`.
    *   **XGBoost Classifier:** Achieved a **ROC AUC of 0.8425**, with Accuracy: `0.7551`, Precision: `0.5267`, Recall: `0.7647`, F1-Score: `0.6238`.
    *   The `class_weight='balanced'` or `scale_pos_weight` parameter was crucial in all top models to mitigate the class imbalance effect.

5.  **Feature Importance:**
    *   The top **3 most influential features** from the Tuned Random Forest model were: `tenure` (importance: `0.1706`), `TotalCharges` (importance: `0.1457`), and `Contract_Two year` (importance: `0.1059`). These quantitatively underscore the primary drivers of churn.

**Conclusion:**
All top-performing models (Logistic Regression, Tuned Random Forest, and XGBoost) demonstrated comparable and robust predictive capabilities with ROC AUC scores around **0.84**. The **Tuned Random Forest Model** was selected and saved for its strong performance and interpretability, offering a reliable tool for identifying at-risk customers and informing retention strategies.
