# AI4I Predictive Maintenance System

End-to-end machine learning project to predict industrial machine failures using sensor data.
The project covers data preprocessing, model training, experiment tracking, API deployment,
containerization, and a user-facing dashboard.

---

## Problem Statement
Unexpected machine failures cause downtime and high maintenance costs in industrial environments.
The goal of this project is to **predict machine failure in advance** using sensor measurements,
allowing preventive maintenance.

---

## Dataset
**AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)**

Features include:
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine type (L, M, H)

Target:
- `Machine failure` (binary classification)

---

## Machine Learning Approach
- Data preprocessing & encoding
- Feature scaling using `StandardScaler`
- Baseline and ensemble models
- Final model: **Random Forest Classifier**
- Metrics: Precision, Recall, F1-score (recall prioritized)

---

## Experiment Tracking
Experiments are tracked using **MLflow**:
- Model parameters
- Evaluation metrics
- Saved artifacts

Run MLflow UI:
```bash
mlflow ui
