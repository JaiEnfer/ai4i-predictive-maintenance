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


---

### Download
Download the dataset from:
https://archive.ics.uci.edu/ml/datasets/ai4i+2020+predictive+maintenance+dataset

Place the CSV file in:
data/ai4i2020.csv

## 3. Machine Learning Approach
- Data cleaning and preprocessing
- One-hot encoding for categorical features
- Feature scaling using `StandardScaler`
- Model training with **Random Forest Classifier**
- Evaluation using Precision, Recall, and F1-score
- Experiment tracking using **MLflow**

---

## 4. Project Structure
ai4i-predictive-maintenance/
├── app/ # FastAPI service
├── src/ # Training and preprocessing code
├── dashboard/ # Streamlit dashboard
├── data/ # Dataset (not committed)
├── models/ # Saved model artifacts
├── requirements.txt
├── Dockerfile
└── README.md

yaml
Copy code

---

## 5. Requirements
- Python 3.10 or higher
- Git
- (Optional) Docker

---

## 6. How to Run Locally

### Step 1: Clone the repository
```bash
git clone https://github.com/<your-username>/ai4i-predictive-maintenance.git
cd ai4i-predictive-maintenance
```

### Step 2: Create and activate virtual environment

Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

### Step 3: Install dependencies
pip install -r requirements.txt

### Step 4: Train the model
python src/train.py

### Step 5: Run the FastAPI service
uvicorn app.api:app --reload


API documentation:

http://127.0.0.1:8000/docs

### Step 6: Run the Streamlit dashboard
streamlit run dashboard/streamlit_app.py

7. Run Using Docker (Recommended)

Docker allows running the API without local Python setup.

docker build -t ai4i-api .
docker run -p 8000:8000 ai4i-api


### API will be available at:

http://localhost:8000/docs

