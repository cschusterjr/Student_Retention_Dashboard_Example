
# Student Retention Dashboard

**Objective:** Predict student (or customer) retention risk and explain drivers to support interventions.

**Tech Stack:** Python, scikit-learn, SHAP, Streamlit

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train a toy model on synthetic data
python -m src.train

# Launch dashboard
streamlit run app/streamlit_app.py
```

## 🗂️ Project Structure
├── app/ # Streamlit dashboard UI code
│ └── streamlit_app.py
├── src/ # Model training and helper scripts
│ ├── train.py
│ └── utils.py # (coming in a later step)
├── artifacts/ # Auto-generated files: model, metrics, sample data
├── requirements.txt # Project dependencies
├── README.md # Project overview and instructions
├── .gitignore # Ignore unnecessary files and folders
└── assets/ # Screenshots, SHAP plots, etc. (optional)
