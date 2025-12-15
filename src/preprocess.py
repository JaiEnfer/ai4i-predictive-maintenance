import pandas as pd

def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop identifiers
    df = df.drop(columns=["UDI", "Product ID"])

    # One-hot encode Type
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    # Target
    y = df["Machine failure"].astype(int)
    X = df.drop(columns=["Machine failure"])

    return X, y
