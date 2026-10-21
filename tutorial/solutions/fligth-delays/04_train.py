"""
04_train.py — Train and evaluate two logistic regression models:

  Model v1 (baseline): 6 features without weather
  Model v2 (with weather): 12 features including weather

Uses the same train/test split procedure as the reference project:
  - Stratified 75/25 split with random_state=0
  - One-hot encoding with drop_first=False
  - LogisticRegression(max_iter=1000)

Reports test set accuracy and baseline accuracy (always predict "not delayed")
for both model versions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    """Load the weather-enhanced dataset if available, else the clean dataset."""
    weather_path = "data/flights_with_weather.csv"
    clean_path = "data/flights_clean.csv"

    if os.path.exists(weather_path):
        print(f"Loading weather-enhanced data from {weather_path}...")
        df = pd.read_csv(weather_path)
        has_weather = True
    else:
        print(f"Weather-enhanced data not found; loading clean data from {clean_path}...")
        df = pd.read_csv(clean_path)
        has_weather = False

    return df, has_weather


def train_v1(df):
    """Train model v1: 6 features without weather."""
    print("\n=== Model v1: Baseline (no weather) ===")

    features = ["Reporting_Airline", "Origin", "Dest", "DayOfWeek", "dep_hour", "Distance"]
    X = df[features]
    y = df["ArrDel15"]

    # Drop rows with missing values
    df_clean = pd.concat([X, y], axis=1).dropna()
    X = df_clean[features]
    y = df_clean["ArrDel15"]

    # One-hot encode categorical features (keep all dummies, drop_first=False)
    X_encoded = pd.get_dummies(X, columns=["Reporting_Airline", "Origin", "Dest", "DayOfWeek"], drop_first=False)

    # Stratified 75/25 train/test split with random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, stratify=y, random_state=0
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Test set accuracy
    test_accuracy = accuracy_score(y_test, y_pred)

    # Baseline accuracy: always predict "not delayed" (class 0)
    baseline_accuracy = (y_test == 0).mean()

    print(f"Test set accuracy (logistic regression v1): {test_accuracy:.4f}")
    print(f"Baseline accuracy (always predict not delayed): {baseline_accuracy:.4f}")

    return test_accuracy, baseline_accuracy


def train_v2(df):
    """Train model v2: 12 features including weather."""
    print("\n=== Model v2: With weather features ===")

    features = [
        "Reporting_Airline", "Origin", "Dest", "DayOfWeek", "dep_hour", "Distance",
        "temperature", "precipitation", "cloud_cover_low",
        "relative_humidity", "wind_speed", "wind_gusts",
    ]
    X = df[features]
    y = df["ArrDel15"]

    # Drop rows with missing values in any feature or the target
    df_clean = df.dropna(
        subset=[
            "temperature", "precipitation", "cloud_cover_low",
            "relative_humidity", "wind_speed", "wind_gusts", "ArrDel15",
        ]
    )
    X = df_clean[features]
    y = df_clean["ArrDel15"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(
        X,
        columns=["Reporting_Airline", "Origin", "Dest", "DayOfWeek"],
        drop_first=False,
    )

    # Stratified 75/25 train/test split with random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, stratify=y, random_state=0
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Test set accuracy
    test_accuracy = accuracy_score(y_test, y_pred)

    # Baseline accuracy
    baseline_accuracy = (y_test == 0).mean()

    print(f"Test set accuracy (logistic regression with weather): {test_accuracy:.4f}")
    print(f"Baseline accuracy (always predict not delayed): {baseline_accuracy:.4f}")

    return test_accuracy, baseline_accuracy


def main():
    df, has_weather = load_data()

    # Train and evaluate both models
    v1_acc, v1_baseline = train_v1(df)
    v2_acc, v2_baseline = train_v2(df)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Model v1 (no accuracy):       {v1_acc:.4f} test / {v1_baseline:.4f} baseline")
    print(f"Model v2 (with weather):      {v2_acc:.4f} test / {v2_baseline:.4f} baseline")
    if not has_weather:
        print("\nNote: Model v2 was trained on the clean dataset without weather;")
        print("      weather features are NaN and those rows were dropped.")


if __name__ == "__main__":
    main()