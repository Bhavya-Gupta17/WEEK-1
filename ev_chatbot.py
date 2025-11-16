# ev_chatbot.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from deepseek import DeepSeek

# Load DeepSeek API Key
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    st.error("⚠️ DeepSeek API Key not found! Add DEEPSEEK_API_KEY to your .env file.")
    st.stop()

client = DeepSeek(api_key=API_KEY)

# Load your dataset
CSV_PATH = r"C:\Users\bhavya gupta\Downloads\archive (5)\Electric_Vehicle_Population_Data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

df = load_data()

# ----------------- EV Chatbot UI -----------------
st.title("⚡ EV GEN-AI Expert Chatbot (DeepSeek Powered)")
st.caption("Ask anything about Electric Vehicles, EV Range, Pricing, Makes, Cities & Dataset Insights!")

# Initialize chat history
if "ev_chat" not in st.session_state:
    st.session_state.ev_chat = [
        {"role": "system", "content": "You are EV-Genius, an expert in electric vehicles, data analysis, and EV market trends. You answer using real dataset insights."}
    ]

# Show past chats
for msg in st.session_state.ev_chat:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **EV-Genius:** {msg['content']}")

# User input
user_input = st.chat_input("Ask about EVs, range, prices, makes, dataset insights...")

# ----------------- Chatbot Logic -----------------
if user_input:
    st.session_state.ev_chat.append({"role": "user", "content": user_input})

    # Create dataset context summary
    try:
        stats = df.describe().to_string()
        top_makes = df["Make"].value_counts().head(5).to_string()
        top_cities = df["City"].value_counts().head(5).to_string()
    except:
        stats, top_makes, top_cities = "N/A", "N/A", "N/A"

    dataset_context = f"""
    Useful dataset insights:
    - Rows: {df.shape[0]}, Columns: {df.shape[1]}

    Top 5 EV Manufacturers:
    {top_makes}

    Top 5 EV Cities:
    {top_cities}

    Summary statistics:
    {stats}
    """

    with st.spinner("⚡ EV-Genius is analyzing your question..."):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    *st.session_state.ev_chat,
                    {"role": "system", "content": dataset_context}
                ],
                temperature=0.5
            )
            bot_reply = response.choices[0].message["content"]

        except Exception as e:
            bot_reply = f"⚠️ Error from DeepSeek API: {str(e)}"

        st.session_state.ev_chat.append({"role": "assistant", "content": bot_reply})
        st.rerun()
