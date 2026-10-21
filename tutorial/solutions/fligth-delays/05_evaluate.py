"""
05_evaluate.py — Detailed evaluation of the trained model.

Loads the weather-enhanced dataset, trains the v2 model (with weather),
and prints a full classification report along with the confusion matrix
elements (TP, TN, FP, FN) for deeper insight.

Also compares v1 vs v2 performance if both datasets are available.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    """Load weather-enhanced data if available, otherwise clean data."""
    weather_path = "data/flights_with_weather.csv"
    clean_path = "data/flights_clean.csv"

    if os.path.exists(weather_path):
        print(f"Loading weather-enhanced data from {weather_path}...")
        df = pd.read_csv(weather_path)
        return df, True
    else:
        print(f"Weather-enhanced data not found; loading clean data from {clean_path}...")
        df = pd.read_csv(clean_path)
        return df, False


def train_and_evaluate(df, label):
    """Train a logistic regression model and evaluate it."""
    # Determine available features based on what columns exist in the dataframe
    base_features = ["Reporting_Airline", "Origin", "Dest", "DayOfWeek", "dep_hour", "Distance"]
    weather_features = ["temperature", "precipitation", "cloud_cover_low",
                        "relative_humidity", "wind_speed", "wind_gusts"]

    # Check which weather columns are actually present
    available_weather = [c for c in weather_features if c in df.columns]
    all_base = [c for c in base_features if c in df.columns]

    # Build feature list: base + weather if available
    features = all_base + available_weather

    X = df[features]
    y = df["ArrDel15"]

    # Drop rows with missing values
    df_clean = df.dropna(
        subset=[
            "temperature", "precipitation", "cloud_cover_low",
            "relative_humidity", "wind_speed", "wind_gusts", "ArrDel15",
        ]
        if available_weather
        else ["DepDelay", "ArrDelay", "ArrDel15"]  # fallback
    )
    # More careful: just drop NaN in features and target
    df_clean = df_clean.dropna(subset=features + ["ArrDel15"])
    X = df_clean[features]
    y = df_clean["ArrDel15"]

    # One-hot encode
    X_encoded = pd.get_dummies(
        X,
        columns=["Reporting_Airline", "Origin", "Dest", "DayOfWeek"],
        drop_first=False,
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, stratify=y, random_state=0
    )

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    baseline = (y_test == 0).mean()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n=== {label} ===")
    print(f"Test set accuracy:   {test_acc:.4f}")
    print(f"Baseline accuracy:   {baseline:.4f}")
    print(f"Confusion matrix:    TN={tn:,.0f} FP={fp:,.0f} FN={fn:,.0f} TP={tp:,.0f}")
    print(f"Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return test_acc, baseline, tp, fp, fn, tn


def main():
    # Evaluate v2 on available data (weather-enhanced or clean)
    df, has_weather = load_data()
    v2_acc, v2_baseline, v2_tp, v2_fp, v2_fn, v2_tn = train_and_evaluate(
        df, "Model v2 (with weather)"
    )

    # Evaluate v1 on clean data if it exists
    clean_path = "data/flights_clean.csv"
    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
        v1_acc, v1_baseline, v1_tp, v1_fp, v1_fn, v1_tn = train_and_evaluate(
            df_clean, "Model v1 (no weather)"
        )
        print("\n=== Comparison ===")
        print(f"Model v1:     {v1_acc:.4f} test / {v1_baseline:.4f} baseline")
        print(f"Model v2:     {v2_acc:.4f} test / {v2_baseline:.4f} baseline")
        print(f"Δ accuracy:   {v2_acc - v1_acc:.4f} percentage points")
    else:
        print("\n(clean.csv not found; v1 comparison skipped.)")


if __name__ == "__main__":
    main()