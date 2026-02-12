"""
src/data_dictionary.py

A simple mapping of column -> description used by the Streamlit UI.
This is an easy portfolio win: it shows you think about users, not just code.
"""

DATA_DICTIONARY = {
    "student_id": "Unique identifier for the student (not used as a model feature).",
    "grade_level": "Student grade level (e.g., 6–12).",
    "age": "Approximate student age in years.",
    "frl_eligible": "Proxy for socioeconomic status (1 = eligible for free/reduced lunch).",
    "ell": "English language learner flag (1 = yes).",
    "sped": "Special education services flag (1 = yes).",
    "attendance_rate_30d": "Attendance rate over the last 30 days (0–1).",
    "tardies_30d": "Count of tardies in the last 30 days.",
    "behavior_incidents_30d": "Count of behavior incidents in the last 30 days.",
    "logins_7d": "Number of platform logins in the last 7 days.",
    "minutes_active_7d": "Total active minutes in the last 7 days.",
    "assignments_submitted_30d": "Assignments submitted in the last 30 days.",
    "current_grade_pct": "Current grade percentage (0–100).",
    "grade_trend_60d": "Grade change trend over 60 days (positive = improving).",
    "missing_assignments": "Count of missing assignments.",
    "label": "Target variable (1 = at-risk / not retained, 0 = retained).",
    "risk_score": "Model-predicted probability of being at risk (0–1).",
    "risk_band": "Human-friendly category derived from risk_score and threshold.",
}
