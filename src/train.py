import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_and_preprocess

MODEL_DIR = "models"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_and_preprocess("data/ai4i2020.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )

    mlflow.set_experiment("ai4i_predictive_maintenance")

    with mlflow.start_run():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 300,
            "max_depth": 10,
            "scaler": "StandardScaler",
        })
        mlflow.log_metrics({"precision": precision, "recall": recall, "f1": f1})

        # Save artifacts for deployment
        joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("=== Classification report ===")
        print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
