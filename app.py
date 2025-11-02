import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from train_model import load_and_train_model

# File path
CSV_PATH = r"C:\Users\bhavya gupta\Downloads\archive (5)\Electric_Vehicle_Population_Data.csv"

st.set_page_config(page_title="EV Data Analysis & Prediction", layout="wide")
st.title("⚡ Electric Vehicle Population Analysis & Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

df = load_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 EDA Dashboard", "🤖 Predict Vehicle Range"])

if page == "📊 EDA Dashboard":
    st.header("Exploratory Data Analysis")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Basic Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if numeric_cols:
        st.write("### Summary Statistics")
        st.write(df.describe())

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("### Distribution of a Selected Column")
        col = st.selectbox("Select column for distribution plot", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for visualization.")

elif page == "🤖 Predict Vehicle Range":
    st.header("Range Prediction")

    model, features = load_and_train_model(CSV_PATH)

    st.write("Enter values to predict electric vehicle range:")
    inputs = []
    for col in features:
        value = st.number_input(f"{col}", value=0.0)
        inputs.append(value)

    if st.button("Predict"):
        prediction = model.predict([inputs])[0]
        st.success(f"Predicted Electric Range: {prediction:.2f} miles")
