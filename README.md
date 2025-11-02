# WEEK-1
The goal of this project is to analyze and predict the electric range of an EV based on its specifications using supervised machine learning.

# ⚡ Electric Vehicle Data Analysis & Prediction Dashboard (Streamlit)

## 🧠 Overview
This project performs **data preprocessing, Exploratory Data Analysis (EDA), and Machine Learning** on the  
**Washington State Electric Vehicle Population Dataset (2024)**.  

It provides an **interactive Streamlit dashboard** for visualizing EV trends and predicting electric vehicle range using a trained regression model.

---

## 🎯 Objective
The goal of this project is to:
- Analyze **electric vehicle adoption and distribution** across Washington State.
- Explore relationships between **vehicle range, price, model year, and make**.
- Predict a vehicle’s **electric range** based on its specifications.

This helps in understanding **EV growth trends** and building intelligent insights using **data science**.

---

## 📁 Folder Structure
C:\<your path>\ev_analysis_streamlit
│
├── app.py # Streamlit main dashboard
├── train_model.py # ML model training script
├── Electric_Vehicle_Population_Data.csv # Dataset
└── README.md # Documentation (this file)


---

## 🧩 Dataset Information
**Dataset Name:** Electric Vehicle Population Data (Washington State, 2024)  

### Key Columns:
| Column | Description |
|---------|--------------|
| `VIN (1-10)` | Unique vehicle identifier |
| `County` | County name |
| `City` | Registered city |
| `Model Year` | Year of vehicle manufacture |
| `Make` | Vehicle manufacturer |
| `Model` | Model name |
| `Electric Vehicle Type` | BEV or PHEV |
| `Electric Range` | Range (in miles) |
| `Base MSRP` | Manufacturer’s Suggested Retail Price |
| `CAFV Eligibility` | Clean Alternative Fuel Vehicle eligibility |
| `Electric Utility` | Power provider |
| `Census Tract` | Census classification of area |

---

## ⚙️ Installation & Setup

### 1️⃣ Install Python Libraries
Open PowerShell or Command Prompt and run:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
### 2️⃣ Run the Streamlit App
streamlit run app.py

