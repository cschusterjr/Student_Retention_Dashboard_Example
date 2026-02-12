# 🎓 Student Retention Risk Dashboard

A portfolio-grade end-to-end machine learning project that predicts student retention risk and explains the drivers behind that risk to support targeted interventions.

This project demonstrates:
- Realistic synthetic data generation
- Model training with reproducible configuration
- Evaluation with business-aligned metrics
- Probability-based risk scoring
- Global and local explainability using SHAP
- A production-style Streamlit dashboard
- Clean separation between training, inference, and UI

---

## 📌 Problem Statement

Educational institutions often struggle to identify students at risk of dropping out early enough to intervene effectively.

This project simulates a real-world retention analytics workflow:

1. Generate realistic student engagement and performance data  
2. Train a risk prediction model  
3. Score students with probability-based risk estimates  
4. Provide interpretable explanations for why a student is at risk  
5. Present results in an interactive dashboard  

The goal is not just prediction — it is decision support.

---

## 🧠 Modeling Approach

### Target
`label`  
- 1 = at risk / not retained  
- 0 = retained  

### Features (examples)
- Attendance rate (30 days)
- Grade percentage
- Grade trend (60 days)
- Missing assignments
- Platform logins
- Behavior incidents
- Socioeconomic proxy indicators (FRL, ELL, SPED)

These features are intentionally designed to mimic realistic retention drivers.

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- **ROC-AUC** – ranking performance across thresholds
- **PR-AUC** – performance under class imbalance
- **Precision / Recall / F1** at selected threshold
- **Confusion Matrix**
- **Precision in Top 10% Risk Bucket** (business-oriented metric)

Why top 10%?
In real interventions, resources are limited. Schools often target the highest-risk segment. This metric simulates that decision workflow.

---

## 🏗️ Architecture

```
data/
  student_retention_synthetic.csv

src/
  make_synthetic_data.py    # Realistic synthetic dataset generator
  train.py                  # Model training + evaluation
  inference.py              # Safe scoring + validation logic
  data_dictionary.py        # Human-readable column descriptions

app/
  streamlit_app.py          # Interactive dashboard UI

artifacts/
  model.pkl
  features.json
  metrics.json
  config.json
  data_schema.json
  sample.csv
```

### Design Principles

- Separation of concerns (training vs inference vs UI)
- Schema validation before scoring
- Cached SHAP explainability for performance
- Config + schema artifacts for reproducibility
- Probability-based decision support (not hard classification)

---

## 🚀 Quickstart

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# or
.venv\Scripts\activate          # Windows (PowerShell)
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Generate synthetic dataset

```bash
python -m src.make_synthetic_data
```

Creates:

```
data/student_retention_synthetic.csv
```

---

### 4️⃣ Train the model

```bash
python -m src.train
```

Creates:

```
artifacts/
  model.pkl
  features.json
  metrics.json
  config.json
  data_schema.json
  sample.csv
```

---

### 5️⃣ Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Dashboard Features

### 🔎 Overview Tab
- Top 50 highest-risk students
- Risk distribution
- Risk band counts
- Downloadable scored CSV
- Data dictionary

### 🧑 Record Explorer
- Select a specific student
- Local SHAP waterfall explanation
- Understand feature contributions to risk

### 🌍 Explainability
- Global SHAP summary plot
- Feature importance visualization

### 📉 Model Quality
- ROC-AUC and PR-AUC
- Confusion matrix
- Precision / Recall / F1
- Precision in top 10% risk bucket
- Training configuration details

---

## 📦 Reproducibility & Artifacts

Each training run saves:

- Model binary (`model.pkl`)
- Feature schema (`features.json`)
- Training config (`config.json`)
- Evaluation metrics (`metrics.json`)
- Data schema (`data_schema.json`)

This mirrors production ML workflows where artifact tracking is essential.

---

## 🛠 Tech Stack

- Python 3.10+
- scikit-learn
- pandas / numpy
- SHAP
- Streamlit
- matplotlib

---

## 🔍 Why This Is Portfolio-Ready

This project demonstrates:

- Structured ML workflow
- Thoughtful metric selection
- Data validation before inference
- Interpretability tooling
- Clean modular architecture
- End-to-end reproducibility

It reflects real-world ML product design, not just notebook experimentation.

---

## ⚠️ Limitations

- Data is synthetic (not real student data)
- Model is a baseline Random Forest (no hyperparameter tuning yet)
- No production deployment infrastructure (API, Docker, CI/CD)

Future improvements could include:
- Time-based train/test splits
- Calibration curves
- Model versioning
- API deployment
- CI pipeline with automated tests

---

## 📬 Contact

Built as a portfolio project demonstrating applied ML and product-oriented analytics.
