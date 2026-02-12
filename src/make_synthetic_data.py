"""
src/make_synthetic_data.py

Creates a realistic-looking synthetic student retention dataset.
This is still synthetic, but it looks like a real student dataset and is useful for demos.

Outputs:
  data/student_retention_synthetic.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_student_retention_dataset(
    n_students: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # --- "Student profile" features
    grade_level = rng.integers(6, 13, size=n_students)  # middle + high school
    age = grade_level + rng.normal(6.5, 0.7, size=n_students)  # rough mapping
    frl_eligible = rng.binomial(1, 0.45, size=n_students)  # free/reduced lunch proxy
    ell = rng.binomial(1, 0.12, size=n_students)  # English language learner
    sped = rng.binomial(1, 0.10, size=n_students)  # special education

    # --- Engagement features (more predictive)
    attendance_rate_30d = np.clip(rng.normal(0.92, 0.06, size=n_students), 0.5, 1.0)
    tardies_30d = np.clip(rng.poisson(1.2, size=n_students), 0, 20)
    behavior_incidents_30d = np.clip(rng.poisson(0.4, size=n_students), 0, 10)

    logins_7d = np.clip(rng.poisson(4.5, size=n_students), 0, 30)
    minutes_active_7d = np.clip(rng.normal(120, 60, size=n_students), 0, 800)
    assignments_submitted_30d = np.clip(rng.poisson(7, size=n_students), 0, 40)

    # --- Performance features
    current_grade_pct = np.clip(rng.normal(78, 12, size=n_students), 0, 100)
    grade_trend_60d = np.clip(rng.normal(0, 6, size=n_students), -25, 25)  # + means improving
    missing_assignments = np.clip(rng.poisson(2.3, size=n_students), 0, 25)

    # --- Construct an underlying risk score with realistic relationships
    # Higher risk if: low attendance, low grades, worsening trend, more incidents, fewer logins, more missing work
    linear_risk = (
        + 3.0 * (0.90 - attendance_rate_30d)          # attendance below ~90% increases risk
        + 0.018 * (80 - current_grade_pct)            # grades below ~80 increase risk
        + 0.030 * (-grade_trend_60d)                  # declining trend increases risk
        + 0.18 * behavior_incidents_30d
        + 0.06 * tardies_30d
        + 0.10 * missing_assignments
        - 0.05 * logins_7d
        - 0.001 * minutes_active_7d
        - 0.02 * assignments_submitted_30d
        + 0.25 * frl_eligible
        + 0.15 * ell
        + 0.10 * sped
        + rng.normal(0, 0.35, size=n_students)        # noise so it's not too clean
    )

    # Convert to probability of dropout / non-retention
    dropout_prob = sigmoid(linear_risk)

    # Sample the label from the probability
    label = rng.binomial(1, dropout_prob)

    df = pd.DataFrame(
        {
            "student_id": [f"S{100000+i}" for i in range(n_students)],
            "grade_level": grade_level,
            "age": np.round(age, 1),
            "frl_eligible": frl_eligible,
            "ell": ell,
            "sped": sped,
            "attendance_rate_30d": np.round(attendance_rate_30d, 3),
            "tardies_30d": tardies_30d,
            "behavior_incidents_30d": behavior_incidents_30d,
            "logins_7d": logins_7d,
            "minutes_active_7d": np.round(minutes_active_7d, 1),
            "assignments_submitted_30d": assignments_submitted_30d,
            "current_grade_pct": np.round(current_grade_pct, 1),
            "grade_trend_60d": np.round(grade_trend_60d, 1),
            "missing_assignments": missing_assignments,
            "label": label,  # 1 = at risk / not retained
        }
    )

    return df


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_student_retention_dataset(n_students=3000, random_state=42)
    out_path = out_dir / "student_retention_synthetic.csv"
    df.to_csv(out_path, index=False)

    # Print quick quality checks
    print(f"Wrote {len(df)} rows to {out_path}")
    print("Label rate (dropout):", df["label"].mean().round(3))
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
