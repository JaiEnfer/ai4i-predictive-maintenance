import streamlit as st
import requests

st.title("AI4I Predictive Maintenance Dashboard")

api_url = st.text_input("API URL", "http://127.0.0.1:8000")

air = st.slider("Air temperature [K]", 295.0, 310.0, 300.0)
proc = st.slider("Process temperature [K]", 305.0, 320.0, 310.0)
rpm = st.slider("Rotational speed [rpm]", 1000.0, 3000.0, 1500.0)
torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0)
wear = st.slider("Tool wear [min]", 0.0, 300.0, 120.0)

type_choice = st.selectbox("Type", ["L", "M", "H"])
type_h = 1 if type_choice == "H" else 0
type_m = 1 if type_choice == "M" else 0

if st.button("Predict"):
    payload = {
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "type_h": type_h,
        "type_m": type_m
    }
    r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    r.raise_for_status()
    out = r.json()
    st.metric("Failure probability", f"{out['failure_probability']:.3f}")
    st.write("Predicted label:", out["predicted_label"])
