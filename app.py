# app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# --- IMPORTANT: Force legacy serialization to avoid r.at JS error ---
# Use Streamlit's internal config API to set legacy DataFrame serialization.
# This bypasses the Arrow/Quiver path that causes the "r.at is not a function" JS error.
try:
    # st._config exists in modern Streamlit; set legacy serialization
    st._config.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    # if any error, ignore — it's not fatal (we still try other fallbacks)
    pass

# ---- Edit CSV path if needed ----
CSV_PATH = r"C:\Users\bhavya gupta\Downloads\archive (5)\Electric_Vehicle_Population_Data.csv"
MODEL_FILE = "models/ev_range_rf_best.pkl"

st.set_page_config(page_title="EV Analysis (Optimized)", layout="wide")
st.title("⚡ EV Analysis & Range Prediction (Optimized)")

@st.cache_data
def load_raw(path):
    df = pd.read_csv(path)
    return df

def safe_show_dataframe(df, max_rows=500):
    """
    Show dataframe safely by converting to pandas values if Arrow still causes issues.
    """
    try:
        st.dataframe(df)
    except Exception:
        # fallback: show head or to_records
        st.write(df.head(max_rows).to_dict(orient="records"))

# Load data
try:
    df = load_raw(CSV_PATH)
    st.success("✅ Dataset loaded successfully.")
except Exception as e:
    st.error(f"Failed loading CSV: {e}")
    df = pd.DataFrame()

# Sidebar
st.sidebar.header("Navigation")
choice = st.sidebar.radio("Page", ["EDA", "Train/Optimize Model", "Predict", "Metrics", "About"])

# EDA
if choice == "EDA":
    st.header("Exploratory Data Analysis")
    if df.empty:
        st.warning("No data loaded.")
    else:
        st.subheader("Preview")
        safe_show_dataframe(df.head(200))

        st.subheader("Numeric Summary")
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            st.warning("No numeric columns available for EDA.")
        else:
            # show describe - safe
            try:
                st.dataframe(numeric.describe().T)
            except Exception:
                st.write(numeric.describe().T.to_dict())

            st.subheader("Correlation heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            if "Base MSRP" in numeric.columns and "Electric Range" in numeric.columns:
                st.subheader("Base MSRP vs Electric Range")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(x="Base MSRP", y="Electric Range", data=df)
                st.pyplot(fig2)

# Train / Optimize
elif choice == "Train/Optimize Model":
    st.header("Train and Optimize RandomForest (minimize RMSE)")
    st.write("This will run randomized search. It may take some minutes depending on dataset size & CPU.")
    if st.button("Start Training"):
        from train_model import load_numeric_data, optimize_and_train, save_model
        try:
            df_num = load_numeric_data(CSV_PATH)
            model, best_params, metrics, features = optimize_and_train(df_num)
            save_model(model, features, metrics, model_path="models/ev_range_rf_best.pkl")
            st.success("Training completed and model saved.")
            st.write("Best params:", best_params)
            st.write("Metrics:", metrics)
        except Exception as e:
            st.error(f"Training failed: {e}")

# Predict
elif choice == "Predict":
    st.header("Predict Electric Range (uses saved best model)")
    if os.path.exists(MODEL_FILE):
        model, features, metrics = joblib.load(MODEL_FILE)
        st.write("Model features:", features)
        user_vals = []
        for f in features:
            val = st.number_input(f"{f}", value=0.0, format="%f")
            user_vals.append(val)
        if st.button("Predict"):
            try:
                pred = model.predict([user_vals])[0]
                st.success(f"Predicted range: {pred:.2f} miles")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Model not found. Train model first (Train/Optimize Model).")

# Metrics
elif choice == "Metrics":
    st.header("Saved Model Metrics")
    if os.path.exists(MODEL_FILE):
        _, _, metrics = joblib.load(MODEL_FILE)
        st.json(metrics)
        st.metric("R²", metrics.get("r2", None))
        st.metric("RMSE", metrics.get("rmse", None))
        st.metric("MAE", metrics.get("mae", None))
        st.metric("MSE", metrics.get("mse", None))
    else:
        st.warning("Train & save a model first.")

# About
else:
    st.header("About")
    st.write("""
    - This app uses RandomizedSearchCV to optimize RandomForest hyperparameters minimizing RMSE.
    - To avoid the `r.at is not a function` frontend error, the app sets legacy DataFrame serialization.
    - If you still see a JS error, hard-refresh the browser (Ctrl+Shift+R) or open in incognito/private window.
    """)
# app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# --- IMPORTANT: Force legacy serialization to avoid r.at JS error ---
# Use Streamlit's internal config API to set legacy DataFrame serialization.
# This bypasses the Arrow/Quiver path that causes the "r.at is not a function" JS error.
try:
    # st._config exists in modern Streamlit; set legacy serialization
    st._config.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    # if any error, ignore — it's not fatal (we still try other fallbacks)
    pass

# ---- Edit CSV path if needed ----
CSV_PATH = r"C:\Users\bhavya gupta\Downloads\archive (5)\Electric_Vehicle_Population_Data.csv"
MODEL_FILE = "models/ev_range_rf_best.pkl"

st.set_page_config(page_title="EV Analysis (Optimized)", layout="wide")
st.title("⚡ EV Analysis & Range Prediction (Optimized)")

@st.cache_data
def load_raw(path):
    df = pd.read_csv(path)
    return df

def safe_show_dataframe(df, max_rows=500):
    """
    Show dataframe safely by converting to pandas values if Arrow still causes issues.
    """
    try:
        st.dataframe(df)
    except Exception:
        # fallback: show head or to_records
        st.write(df.head(max_rows).to_dict(orient="records"))

# Load data
try:
    df = load_raw(CSV_PATH)
    st.success("✅ Dataset loaded successfully.")
except Exception as e:
    st.error(f"Failed loading CSV: {e}")
    df = pd.DataFrame()

# Sidebar
st.sidebar.header("Navigation")
choice = st.sidebar.radio("Page", ["EDA", "Train/Optimize Model", "Predict", "Metrics", "About"])

# EDA
if choice == "EDA":
    st.header("Exploratory Data Analysis")
    if df.empty:
        st.warning("No data loaded.")
    else:
        st.subheader("Preview")
        safe_show_dataframe(df.head(200))

        st.subheader("Numeric Summary")
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            st.warning("No numeric columns available for EDA.")
        else:
            # show describe - safe
            try:
                st.dataframe(numeric.describe().T)
            except Exception:
                st.write(numeric.describe().T.to_dict())

            st.subheader("Correlation heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            if "Base MSRP" in numeric.columns and "Electric Range" in numeric.columns:
                st.subheader("Base MSRP vs Electric Range")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(x="Base MSRP", y="Electric Range", data=df)
                st.pyplot(fig2)

# Train / Optimize
elif choice == "Train/Optimize Model":
    st.header("Train and Optimize RandomForest (minimize RMSE)")
    st.write("This will run randomized search. It may take some minutes depending on dataset size & CPU.")
    if st.button("Start Training"):
        from train_model import load_numeric_data, optimize_and_train, save_model
        try:
            df_num = load_numeric_data(CSV_PATH)
            model, best_params, metrics, features = optimize_and_train(df_num)
            save_model(model, features, metrics, model_path="models/ev_range_rf_best.pkl")
            st.success("Training completed and model saved.")
            st.write("Best params:", best_params)
            st.write("Metrics:", metrics)
        except Exception as e:
            st.error(f"Training failed: {e}")

# Predict
elif choice == "Predict":
    st.header("Predict Electric Range (uses saved best model)")
    if os.path.exists(MODEL_FILE):
        model, features, metrics = joblib.load(MODEL_FILE)
        st.write("Model features:", features)
        user_vals = []
        for f in features:
            val = st.number_input(f"{f}", value=0.0, format="%f")
            user_vals.append(val)
        if st.button("Predict"):
            try:
                pred = model.predict([user_vals])[0]
                st.success(f"Predicted range: {pred:.2f} miles")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Model not found. Train model first (Train/Optimize Model).")

# Metrics
elif choice == "Metrics":
    st.header("Saved Model Metrics")
    if os.path.exists(MODEL_FILE):
        _, _, metrics = joblib.load(MODEL_FILE)
        st.json(metrics)
        st.metric("R²", metrics.get("r2", None))
        st.metric("RMSE", metrics.get("rmse", None))
        st.metric("MAE", metrics.get("mae", None))
        st.metric("MSE", metrics.get("mse", None))
    else:
        st.warning("Train & save a model first.")

# About
else:
    st.header("About")
    st.write("""
    - This app uses RandomizedSearchCV to optimize RandomForest hyperparameters minimizing RMSE.
    - To avoid the `r.at is not a function` frontend error, the app sets legacy DataFrame serialization.
    - If you still see a JS error, hard-refresh the browser (Ctrl+Shift+R) or open in incognito/private window.
    """)
