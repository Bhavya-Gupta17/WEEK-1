# train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings("ignore")

# EDIT: Put your dataset path here (exact path you use)
CSV_PATH = r"C:\Users\bhavya gupta\Downloads\archive (5)\Electric_Vehicle_Population_Data.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "ev_range_rf_best.pkl")

def load_numeric_data(csv_path):
    df = pd.read_csv(csv_path)
    # keep only numeric columns (drop NaNs rows)
    df_num = df.select_dtypes(include=[np.number]).dropna()
    if df_num.shape[1] < 2:
        raise RuntimeError("Need at least 1 feature + 1 target numeric column.")
    return df_num

def optimize_and_train(df_num, random_state=42):
    # features = all numeric cols except last -> this is a simple convention:
    X = df_num.iloc[:, :-1].values
    y = df_num.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    # Baseline model
    base_model = RandomForestRegressor(random_state=random_state)

    # Randomized search space (fast & effective)
    param_dist = {
        "n_estimators": randint(50, 400),
        "max_depth": randint(5, 40),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 8),
        "max_features": uniform(0.3, 0.7)
    }

    rnd_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_root_mean_squared_error",  # minimize RMSE
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    rnd_search.fit(X_train, y_train)

    best = rnd_search.best_estimator_

    # Evaluate
    y_pred = best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    return best, rnd_search.best_params_, metrics, list(df_num.columns[:-1])

def save_model(model, features, metrics, model_path=MODEL_FILE):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, features, metrics), model_path)
    print(f"Saved model to: {model_path}")

def main():
    print("Loading numeric dataset...")
    df_num = load_numeric_data(CSV_PATH)
    print(f"Numeric shape: {df_num.shape}. Running tuning...")

    model, best_params, metrics, features = optimize_and_train(df_num)
    print("Best params:", best_params)
    print("Metrics:", metrics)

    save_model(model, features, metrics)
    print("Done.")

if __name__ == "__main__":
    main()
