"""
03_weather.py — Download hourly historical weather from Open-Meteo and join
with the cleaned flight data.

Produces an enhanced dataset with 6 weather features attached per flight row.
Uses an exact (airport, date_hour) lookup; rows without a matching weather
record are dropped via dropna().
"""

import os
import requests
import pandas as pd
import numpy as np


# --- Airport coordinates ---
AIRPORTS = {
    "SFO": {"lat": 37.6186, "lon": -122.375},
    "OAK": {"lat": 37.7127, "lon": -122.224},
    "SJC": {"lat": 37.3668, "lon": -121.9277},
}

WEATHER_PARAMS = "temperature_2m,precipitation,cloud_cover_low,relative_humidity_2m,wind_speed_10m,wind_gusts_10m"
START_DATE = "2025-03-01"
END_DATE = "2025-03-31"
TIMEZONE = "America/Los_Angeles"


def fetch_weather_for_airport(airport, lat, lon):
    """Fetch hourly weather from Open-Meteo for a single airport."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": WEATHER_PARAMS,
        "timezone": TIMEZONE,
    }
    print(f"  Fetching weather for {airport}...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "precipitation": data["hourly"]["precipitation"],
        "cloud_cover_low": data["hourly"]["cloud_cover_low"],
        "relative_humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "wind_gusts": data["hourly"]["wind_gusts_10m"],
        "airport": airport,
    })
    return df


def build_weather_lookup(*dfs):
    """Build a dictionary keyed by (airport, datetime) → weather tuple."""
    lookup = {}
    for df in dfs:
        for _, row in df.iterrows():
            key = (row["airport"], row["datetime"])
            lookup[key] = (
                row["temperature"],
                row["precipitation"],
                row["cloud_cover_low"],
                row["relative_humidity"],
                row["wind_speed"],
                row["wind_gusts"],
            )
    return lookup


def join_weather(flights_path="data/flights_clean.csv", output_path="data/flights_with_weather.csv"):
    """Join weather data with flight records and drop rows with missing values."""
    print(f"Reading flight data from {flights_path}...")
    flights = pd.read_csv(flights_path)

    # --- Create date-hour key ---
    # Format: 2025-03-01T06:00
    flights = flights.copy()
    flights["date_hour"] = (
        flights["FlightDate"] + "T"
        + flights["dep_hour"].astype(str).str.zfill(2) + ":00"
    )

    # --- Fetch weather for all three airports ---
    weather_dfs = []
    for airport, info in AIRPORTS.items():
        df_w = fetch_weather_for_airport(airport, info["lat"], info["lon"])
        weather_dfs.append(df_w)

    weather_all = pd.concat(weather_dfs, ignore_index=True)

    # --- Build lookup dictionary ---
    weather_lookup = build_weather_lookup(weather_all)

    # --- Look up weather for each flight row ---
    def fetch_weather_row(row):
        key = (row["Origin"], row["date_hour"])
        if key in weather_lookup:
            return weather_lookup[key]
        else:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    weather_mat = flights.apply(fetch_weather_row, axis=1, result_type="expand")
    weather_mat.columns = [
        "temperature",
        "precipitation",
        "cloud_cover_low",
        "relative_humidity",
        "wind_speed",
        "wind_gusts",
    ]

    # --- Attach weather columns ---
    df = pd.concat([flights.reset_index(drop=True), weather_mat], axis=1)

    # --- Remove rows with missing values in weather or target ---
    initial = len(df)
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
    removed = initial - len(df)
    print(f"Removed {removed:,} rows with missing weather/target data")
    print(f"Remaining rows: {len(df):,}")

    # Save enhanced dataset
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nWrote {len(df):,} rows to {output_path}")
    print(f"Columns: {list(df.columns)}")

    # Print class distribution of target
    print(f"ArrDel15 distribution:\n{df['ArrDel15'].value_counts().sort_index()}")

    return df


if __name__ == "__main__":
    join_weather()