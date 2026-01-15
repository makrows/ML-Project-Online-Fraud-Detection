# Online Payments Fraud Detection
## Machine Learning in Finance - Final Project

### 📌 Project Overview
This project focuses on building a robust Machine Learning pipeline to detect fraudulent online payment transactions. Using a large-scale dataset, we implement and compare multiple binary classification models to identify fraudulent behavior (`isFraud = 1`) while minimizing false alarms.

**Key Objectives:**
*   Analyze transaction patterns to identify fraud.
*   Handle extreme class imbalance (Fraud occurs in only ~0.13% of cases).
*   Avoid data leakage using proper Time-Series Splitting.
*   Compare performance of **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest**.

---

### 📂 Dataset
*   **Source:** Kaggle "Online Payments Fraud Detection" (`PS_20174392719_1491204439457_log.csv`)
*   **Size:** ~6.3 million transactions.
*   **Features:** `step`, `type`, `amount`, `oldbalance`, `newbalance`, etc.
*   **Target:** `isFraud` (Binary).

**Note:** The dataset contains a `step` column representing time (1 hour per step), which is critical for our splitting strategy.

---

### ⚙️ Methodology

#### 1. Data Preprocessing
*   **Filtering:** Restricted analysis to `TRANSFER` and `CASH_OUT` transaction types, as fraud only occurs in these categories.
*   **Encoding:** Applied **One-Hot Encoding** to the `type` column.
*   **Splitting:** Utilized a **Time-Series Split** (First 80% Train, Last 20% Test) to strictly prevent future data leakage.
*   **Scaling:** Standardized numerical features using `StandardScaler`.

#### 2. Handling Imbalance
*   **Training Set:** We used a massive training sample of **1,000,000** non-fraud transactions and all available fraud transactions.
*   **SMOTE:** Applied Synthetic Minority Over-sampling Technique (SMOTE) **only to the training set** to achieve a perfectly balanced 50/50 distribution for model training.
*   **Test Set:** Kept the **Full Natural Test Set (~554k samples)** with its original 0.77% fraud rate to ensure realistic evaluation.

#### 3. Models Implemented
*   **Logistic Regression:** A baseline linear model interpreted for its coefficients.
*   **K-Nearest Neighbors (KNN):** A distance-based classifier (k=5). *Note: Trained on the full 1M dataset, which is computationally intensive but provides the highest fidelity comparison.*
*   **Random Forest:** An ensemble of decision trees (`n_estimators=100`) capable of capturing non-linear patterns.

---

### 📊 Results Summary

We prioritized **Precision-Recall AUC (PR-AUC)** and **Recall** due to the class imbalance.

| Model | Recall (Fraud Capture) | Precision (Reliability) | PR-AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~86% | ~19% | 0.72 |
| **KNN** | ~63% | ~71% | 0.57 |
| **Random Forest** | **~92%** | **~17%** | **0.86** |

**Conclusion:**
**Random Forest is the recommended model.** Although its precision dropped on the massive test set (indicating more false positives to screen), it achieved the highest **Recall (92%)**, meaning it successfully catches the vast majority of fraud attempts. Its overall **PR-AUC of 0.86** demonstrates the best trade-off capability among all models tested.

---

### 🚀 How to Run
1.  Ensure you have Python 3.8+ installed.
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```
3.  Place the dataset (`PS_20174392719_1491204439457_log.csv`) in the project root.
4.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook Payment_Fraud_Detection.ipynb
    ```

---
*Author: [Your Name]*
