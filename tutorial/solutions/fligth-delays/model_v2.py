import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------
# 1. Load the cleaned flight data
# ------------------------------------------------------------------
flights = pd.read_csv('data/flights_clean.csv')

# ------------------------------------------------------------------
# 2. Pull hourly historical weather from Open-Meteo for SFO, OAK, SJC
# ------------------------------------------------------------------
airports = {
    'SFO': {'lat': 37.6186, 'lon': -122.375},
    'OAK': {'lat': 37.7127, 'lon': -122.224},
    'SJC': {'lat': 37.3668, 'lon': -121.9277}
}

weather_dfs = []
for airport, info in airports.items():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "start_date": "2025-03-01",
        "end_date": "2025-03-31",
        "hourly": "temperature_2m,precipitation,cloud_cover_low,relative_humidity_2m,wind_speed_10m,wind_gusts_10m",
        "timezone": "America/Los_Angeles"
    }
    r = requests.get(url, params=params)
    data = r.json()
    df_w = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "precipitation": data["hourly"]["precipitation"],
        "cloud_cover_low": data["hourly"]["cloud_cover_low"],
        "relative_humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "wind_gusts": data["hourly"]["wind_gusts_10m"],
        "airport": airport
    })
    weather_dfs.append(df_w)

weather_all = pd.concat(weather_dfs, ignore_index=True)

# ------------------------------------------------------------------
# 3. Create a date‑hour key for joining with the flight records
# ------------------------------------------------------------------
flights = flights.copy()
# use the same format as the API (ISO 8601 with a T)
flights["date_hour"] = (
    flights["FlightDate"] + "T"
    + flights["dep_hour"].astype(str).str.zfill(2) + ":00"
)

# ------------------------------------------------------------------
# 4. Build a lookup dictionary: (airport, datetime) → weather tuple
# ------------------------------------------------------------------
weather_lookup = {}
for _, row in weather_all.iterrows():
    key = (row["airport"], row["datetime"])
    weather_lookup[key] = (
        row["temperature"],
        row["precipitation"],
        row["cloud_cover_low"],
        row["relative_humidity"],
        row["wind_speed"],
        row["wind_gusts"],
    )


def fetch_weather(row):
    """Return weather features for a given flight row, or NaN if missing."""
    key = (row["Origin"], row["date_hour"])
    if key in weather_lookup:
        return weather_lookup[key]
    else:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


weather_mat = flights.apply(fetch_weather, axis=1, result_type="expand")
weather_mat.columns = [
    "temperature",
    "precipitation",
    "cloud_cover_low",
    "relative_humidity",
    "wind_speed",
    "wind_gusts",
]

# ------------------------------------------------------------------
# 5. Attach weather columns to the flight dataframe
# ------------------------------------------------------------------
df = pd.concat([flights.reset_index(drop=True), weather_mat], axis=1)

# ------------------------------------------------------------------
# 6. Remove rows with missing values in any feature or the target
# ------------------------------------------------------------------
df = df.dropna(
    subset=[
        "temperature",
        "precipitation",
        "cloud_cover_low",
        "relative_humidity",
        "wind_speed",
        "wind_gusts",
        "ArrDel15",
    ]
)

# ------------------------------------------------------------------
# 7. Define the feature set (the six original features + six weather vars)
# ------------------------------------------------------------------
features = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DayOfWeek",
    "dep_hour",
    "Distance",
    "temperature",
    "precipitation",
    "cloud_cover_low",
    "relative_humidity",
    "wind_speed",
    "wind_gusts",
]

X = df[features]
y = df["ArrDel15"]

# ------------------------------------------------------------------
# 8. One‑hot encode the categorical columns
# ------------------------------------------------------------------
X_encoded = pd.get_dummies(
    X,
    columns=["Reporting_Airline", "Origin", "Dest", "DayOfWeek"],
    drop_first=False,
)

# ------------------------------------------------------------------
# 9. Stratified 75 % / 25 % train‑test split (random_state = 0)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, stratify=y, random_state=0
)

# ------------------------------------------------------------------
# 10. Fit logistic regression (increase max_iter for convergence)
# ------------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 11. Predict on the test set and report metrics
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
# Baseline: always predict the majority class ( “not delayed” → class 0)
baseline_accuracy = (y_test == 0).mean()

print(f"Test set accuracy (logistic regression with weather): {test_accuracy:.4f}")
print(f"Baseline accuracy (always predict not delayed): {baseline_accuracy:.4f}")