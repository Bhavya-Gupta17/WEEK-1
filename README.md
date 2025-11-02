# ⚡ Electric Vehicle Population Analysis (Streamlit)

## 📌 Overview
A Streamlit-based frontend for exploring and predicting Electric Vehicle data using machine learning.

## 🧠 Features
- EDA Dashboard (interactive visualizations using Seaborn + Matplotlib)
- Linear Regression-based range prediction
- Auto model training from dataset

## 🗂️ Project Structure
```
ev_analysis_streamlit/
│
├── app.py              # Streamlit frontend
├── train_model.py      # Model training script
├── requirements.txt    # Dependencies list
├── README.md           # Setup instructions
└── models/             # Trained model saved here
```

## 🚀 How to Run in VS Code

1. Open **VS Code** → Terminal → Run the following:

```bash
pip install -r requirements.txt
streamlit run app.py
```

2. Make sure your dataset is present at the path you mentioned.

3. Open the local URL shown in terminal (e.g., http://localhost:8501).

✅ Done! You’ll see the full EDA dashboard and range predictor.
