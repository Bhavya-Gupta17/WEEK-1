🚗⚡ EV IntelliSense
Predictive Analytics & Generative AI Chatbot for Electric Vehicles

EV IntelliSense is an end-to-end EV analytics system that integrates
📊 Exploratory Data Analysis,
🤖 Machine Learning–based Range Prediction, and
💬 Generative AI Chatbot (DeepSeek via OpenRouter)
into a single Streamlit dashboard.

It uses the Full Electric Vehicle Dataset 2024 – Washington State BEV & PHEV Population to generate insights, predict electric range, and answer EV-related queries with AI.

📌 Features
🔍 1. Exploratory Data Analysis

Dataset preview

Summary statistics

Correlation heatmap

EV Range distribution

MSRP vs Range scatterplot

Top EV manufacturers & cities

🤖 2. EV Range Prediction (Machine Learning)

Random Forest Regression

Hyperparameter tuning

Feature selection & preprocessing

Performance metrics (RMSE, MAE, MSE, R²)

Saved trained model (.pkl)

💬 3. EV Expert Chatbot (DeepSeek via OpenRouter)

Conversational EV assistant

Uses dataset context for smart answers

Powered by model: deepseek/deepseek-chat-v3.1:free

Fully integrated inside Streamlit

🌐 4. Streamlit Dashboard

4 interactive pages:

EDA Dashboard

Predict EV Range

Model Performance

EV Chatbot

📁 Project Structure
ev_intellisense/
│── app.py                    # Main Streamlit UI
│── train_model.py            # Model training + optimization
│── models/
│     └── ev_range_rf_best.pkl
│── .env                      # API key (ignored in Git)
│── README.md

📊 Dataset Used
Full Electric Vehicle Dataset 2024 – Washington State BEV & PHEV Population (Kaggle)

Contains:

Model Year, Make, Model

Electric Range

Base MSRP

EV Type (BEV/PHEV)

Location: City, County, ZIP

Utilities, CAFV eligibility

VIN, Vehicle IDs

Why this dataset?

170k+ real EV registrations

Perfect for EDA + ML + chatbot context

Strong correlations between MSRP, range, and model year

⚙️ Installation & Setup
1️⃣ Clone the repo
git clone https://github.com/your-username/ev-intellisense.git
cd ev-intellisense

2️⃣ Install dependencies
pip install -r requirements.txt


Or individually:

pip install streamlit pandas numpy seaborn matplotlib scikit-learn joblib python-dotenv openai

3️⃣ Add OpenRouter API Key

Create .env:

OPENROUTER_API_KEY=your_key_here

4️⃣ Set dataset path

Update CSV_PATH in app.py:

CSV_PATH = r"C:\Your\Path\Electric_Vehicle_Population_Data.csv"

5️⃣ Run Streamlit
streamlit run app.py

🧠 How the ML Model Works

Selects numeric columns

Removes missing values

Train-test split

Random Forest Regression

Hyperparameter tuning with RandomizedSearchCV

Evaluates using:

RMSE

MAE

MSE

R²

Model is saved as:

models/ev_range_rf_best.pkl

💬 Chatbot Details

Powered by OpenRouter API

Model: deepseek/deepseek-chat-v3.1:free

Dataset-aware:

Top makes

Top cities

Descriptive stats

Provides:

EV comparisons

Range queries

General EV knowledge

Dataset-based insights

📜 License

This project is licensed under the MIT License.

⭐ Acknowledgments

Kaggle for the dataset

OpenRouter for API access

DeepSeek for the AI model

Streamlit for frontend framework
