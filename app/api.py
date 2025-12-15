from fastapi import FastAPI
from app.schemas import PredictInput
from app.model_loader import predict_proba

app = FastAPI(title="AI4I Predictive Maintenance API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: PredictInput):
    payload = inp.model_dump(by_alias=True)
    # Map aliases back to exact training column names
    mapped = {
        "Air temperature [K]": payload["Air temperature [K]"],
        "Process temperature [K]": payload["Process temperature [K]"],
        "Rotational speed [rpm]": payload["Rotational speed [rpm]"],
        "Torque [Nm]": payload["Torque [Nm]"],
        "Tool wear [min]": payload["Tool wear [min]"],
        "Type_H": payload["type_h"],
        "Type_M": payload["type_m"],
    }
    proba = predict_proba(mapped)
    return {"failure_probability": proba, "predicted_label": int(proba >= 0.5)}
