# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Fix r.at error (Streamlit Arrow bug)
try:
    st._config.set_option("global.dataFrameSerialization", "legacy")
except:
    pass

# ---------------------- FILE PATH ----------------------
CSV_PATH = r"C:\Users\bhavya gupta\Downloads\archive (5)\Electric_Vehicle_Population_Data.csv"
MODEL_PATH = "models/ev_range_rf_best.pkl"

st.set_page_config(page_title="EV Dashboard", layout="wide")
st.title("⚡ Electric Vehicle Analysis, Prediction & AI Chatbot Dashboard")


# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

try:
    df = load_data()
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    df = pd.DataFrame()


# ---------------------- SIDEBAR ----------------------
st.sidebar.title("🔍 Navigation")
choice = st.sidebar.radio(
    "Go to:",
    ["📊 EDA Dashboard", "🤖 Predict Range", "📈 Model Performance", "💬 EV Chatbot"]
)


# ---------------------- SAFE DATAFRAME ----------------------
def safe_show(df):
    try:
        st.dataframe(df)
    except:
        st.write(df.to_dict(orient="records"))


# ---------------------- EDA DASHBOARD ----------------------
if choice == "📊 EDA Dashboard":
    st.header("📊 Exploratory Data Analysis")

    if df.empty:
        st.warning("Dataset not available.")
    else:
        st.subheader("Dataset Preview")
        safe_show(df.head(200))

        st.subheader("Numeric Summary")
        numeric = df.select_dtypes(include=[np.number])

        if numeric.shape[1] > 0:
            safe_show(numeric.describe().T)

            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            if "Base MSRP" in df.columns and "Electric Range" in df.columns:
                st.subheader("Base MSRP vs Electric Range")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(x="Base MSRP", y="Electric Range", data=df)
                st.pyplot(fig2)

            if "Make" in df.columns:
                st.subheader("Top 10 EV Manufacturers")
                st.bar_chart(df["Make"].value_counts().head(10))
        else:
            st.warning("No numeric columns available in dataset.")


# ---------------------- PREDICT PAGE ----------------------
elif choice == "🤖 Predict Range":
    st.header("🔮 Predict EV Electric Range")

    if not os.path.exists(MODEL_PATH):
        st.warning("Model not trained yet. Train it from Model Performance section.")
    else:
        model, features, metrics = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")

        st.write("Enter values for prediction:")
        inputs = []

        for f in features:
            val = st.number_input(f"{f}", value=0.0)
            inputs.append(val)

        if st.button("Predict Range"):
            pred = model.predict([inputs])[0]
            st.success(f"🚗 Estimated Range: **{pred:.2f} miles**")


# ---------------------- MODEL PERFORMANCE ----------------------
elif choice == "📈 Model Performance":
    st.header("📈 Train & Evaluate Model")

    if st.button("Train Optimized Random Forest Model"):
        from train_model import load_numeric_data, optimize_and_train, save_model

        try:
            df_num = load_numeric_data(CSV_PATH)
            st.info("Training model... (1–3 minutes)")

            model, best_params, metrics, features = optimize_and_train(df_num)
            save_model(model, features, metrics)

            st.success("Training completed!")
            st.write("### Best Hyperparameters")
            st.json(best_params)

            st.write("### Model Metrics")
            st.json(metrics)

        except Exception as e:
            st.error(f"Training failed: {e}")

    if os.path.exists(MODEL_PATH):
        _, _, metrics = joblib.load(MODEL_PATH)
        st.subheader("Saved Model Performance")
        st.json(metrics)


# ---------------------- EV CHATBOT (OPENROUTER + DEEPSEEK FREE) ----------------------
elif choice == "💬 EV Chatbot":
    st.header("💬 EV Expert Chatbot (DeepSeek via OpenRouter)")

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    OR_KEY = os.getenv("OPENROUTER_API_KEY")

    if not OR_KEY:
        st.error("⚠️ Add OPENROUTER_API_KEY to your .env file.")
        st.stop()

    # OpenRouter client
    client = OpenAI(
        api_key=OR_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # Initialize memory
    if "ev_chat" not in st.session_state:
        st.session_state.ev_chat = [
            {
                "role": "system",
                "content": (
                    "You are EV-Genius, an expert in Electric Vehicles. Use real dataset insights "
                    "like EV makes, cities, MSRP, EV range patterns, charging behavior, and BEV vs PHEV differences. "
                    "Explain clearly and accurately using the dataset."
                )
            }
        ]

    # Display chat history
    for msg in st.session_state.ev_chat:
        if msg["role"] == "user":
            st.markdown(f"🧑 **You:** {msg['content']}")
        else:
            st.markdown(f"🤖 **EV-Genius:** {msg['content']}")

    user_input = st.chat_input("Ask anything about EVs✨ ...")

    if user_input:
        st.session_state.ev_chat.append({"role": "user", "content": user_input})

        # Dataset context
        try:
            stats = df.describe().to_string()
            top_makes = df["Make"].value_counts().head(5).to_string()
            top_cities = df["City"].value_counts().head(5).to_string()
        except:
            stats = top_makes = top_cities = "N/A"

        dataset_context = f"""
        Dataset Overview:
        - Rows: {df.shape[0]}, Columns: {df.shape[1]}
        - Top EV Makes:\n{top_makes}
        - Top EV Cities:\n{top_cities}
        - Summary Stats:\n{stats}
        """

        with st.spinner("Thinking... ⚡"):
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-chat-v3.1:free",
                    messages=[
                        *st.session_state.ev_chat,
                        {"role": "system", "content": dataset_context}
                    ],
                    temperature=0.5
                )

                choice = response.choices[0]

                # SAFE PARSING (final correct version)
                bot_reply = None

                # Case 1: Standard response
                if choice.message and hasattr(choice.message, "content") and choice.message.content:
                    bot_reply = choice.message.content

                # Case 2: Text field fallback
                elif hasattr(choice, "text") and choice.text:
                    bot_reply = choice.text

                # Case 3: Nothing returned
                else:
                    bot_reply = "⚠️ I received an empty response. Please try again."

            except Exception as e:
                bot_reply = f"⚠️ Chatbot Error: {str(e)}"

            st.session_state.ev_chat.append({"role": "assistant", "content": bot_reply})
            st.rerun()
