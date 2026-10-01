# 🚗 Used Car Price Prediction using Deep Artificial Neural Networks (ANN)

This repository contains an end-to-end machine learning and deep learning implementation to predict used car selling prices based on vehicle technical specifications and ownership history using **TensorFlow / Keras**.

---

## 📌 Project Overview
Used car market valuations depend on multiple non-linear factors such as manufacturing year, mileage, engine output (*max power*), torque, and engine speed. This project focuses on:
1. **Data Cleaning & Regex Feature Extraction**: Parsing composite torque strings into distinct numerical variables (`torque_nm` and `torque_rpm`).
2. **Preprocessing**: Handling missing values via median/mode imputation, encoding categorical attributes with `LabelEncoder`, and scaling numerical features with `StandardScaler`.
3. **Model Comparison**: Benchmarking a deep multi-layer ANN architecture (Sigmoid activation) against a regularized model (ReLU activation, Dropout, EarlyStopping, and ReduceLROnPlateau).

---

## 📊 Dataset & Features
The data is sourced from the tabular file `2B.parquet`, which includes detailed vehicle specifications:
- **Target:** `selling_price` (used car sale price)
- **Key Features:** `name`, `year`, `mileage`, `engine`, `max_power`, `seats`, `fuel`, `transmission`, `owner`, `seller_type`, `Region`, `State or Province`, `City`.
- **Engineered Features (Extracted via Regex):**
  - `torque_nm`: Engine torque output (in Nm).
  - `torque_rpm`: Rotational speed at peak torque (in rpm).

---

## 🛠️ Preprocessing Pipeline
1. **Missing Value Imputation:** Numerical features are imputed using the median, while categorical features use the mode.
2. **Torque Parsing:** Regular expressions (`r'(\d+)\s*Nm'` and `r'@\s*(\d+)\s*rpm'`) extract torque magnitude and RPM thresholds.
3. **Encoding:** Categorical variables are converted into numerical representations using `LabelEncoder`.
4. **Data Splitting:** The dataset is split into 70% Training, 15% Validation, and 15% Test subsets using a fixed random seed.
5. **Feature Scaling:** `StandardScaler` is applied independently to the input feature matrix ($X$) and target values ($y$).

---

## 🧠 Model Architectures & Experiments

### 1. Baseline Model (Sigmoid Deep ANN)
- **Input:** 14 features
- **Hidden Layers:**
  - Dense (512 units, `sigmoid` activation)
  - Dense (256 units, `sigmoid` activation)
  - Dense (128 units, `sigmoid` activation)
- **Output:** Dense (1 unit, linear activation)
- **Optimizer & Loss:** Adam ($lr = 0.01$), Loss: Mean Squared Error (MSE), Metric: Mean Absolute Error (MAE)
- **Epochs:** 30

### 2. Modified Model (ReLU + Regularization)
- **Input:** 14 features
- **Hidden Layers:**
  - Dense (256 units, `relu` activation) + Dropout (0.2)
  - Dense (128 units, `relu` activation) + Dropout (0.2)
  - Dense (64 units, `relu` activation) + Dropout (0.2)
  - Dense (32 units, `relu` activation)
- **Output:** Dense (1 unit, linear activation)
- **Callbacks:** `EarlyStopping` (patience = 5, restore best weights) and `ReduceLROnPlateau` (factor = 0.5, patience = 3, min_lr = 1e-5)
- **Optimizer & Loss:** Adam ($lr = 0.0001$), Loss: MAE, Metric: MAE
- **Epochs:** Up to 100

---

## 📈 Evaluation & Performance Comparison (Test Set)

| Metric | Baseline Model (Sigmoid) | Modified Model (ReLU + Dropout) |
| :--- | :---: | :---: |
| **Test Loss** | **0.0646** | 0.1973 |
| **Test MAE** | **0.1644** | 0.1973 |
| **$R^2$ Score** | **0.9277 (~92.8%)** | 0.8352 (~83.5%) |

> **Analysis:** The baseline multi-layer Sigmoid network achieved superior predictive accuracy ($R^2 \approx 92.77\%$), outperforming the regularized ReLU architecture on this normalized tabular dataset under the tested hyperparameters.

---

## 📁 Repository Structure
```text
├── 2B.parquet              # Raw tabular dataset
├── 2BSolution.ipynb        # Data preparation, model training, and evaluation notebook
└── README.md               # Project documentation
```
