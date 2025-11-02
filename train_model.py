import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def load_and_train_model(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include='number').dropna()

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature and one target column.")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LinearRegression()
    model.fit(X, y)

    # ✅ Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    joblib.dump((model, list(X.columns)), "models/ev_range_model.pkl")
    return model, list(X.columns)
