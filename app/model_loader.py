import joblib
import numpy as np

MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_names.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

def predict_proba(payload_dict: dict) -> float:
    # payload_dict keys must match training feature names
    x = np.array([[payload_dict.get(f, 0) for f in feature_names]], dtype=float)
    x_s = scaler.transform(x)
    proba = model.predict_proba(x_s)[0, 1]
    return float(proba)
