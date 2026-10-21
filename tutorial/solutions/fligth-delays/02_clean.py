"""
02_clean.py — Clean and engineer features from Bay Area flight data.

Reads data/flights_bayarea.csv and produces data/flights_clean.csv with a
reduced, analytically relevant column set (12 columns, 18,898 rows).

Engineered features:
  - dep_hour: hour extracted from CRSDepTime (0-23)
  - Distance: great-circle distance between origin and destination (miles)
  - ArrDel15: binary target (1 if ArrDelay >= 15 min, else 0)
"""

import os
import math
import csv

import pandas as pd


# --- Airport coordinates for distance great-circle calculation ---
_AIRPORT_COORDS = {
    "SFO": (37.6186, -122.375),
    "OAK": (37.7127, -122.224),
    "SJC": (37.3668, -121.9277),
    "LAX": (33.9416, -118.4085),
    "SEA": (47.4502, -122.3088),
    "DEN": (39.8561, -104.6737),
    "DFW": (32.8998, -97.0403),
    "ORD": (41.9742, -87.9073),
    "ATL": (33.6407, -84.4277),
    "JFK": (40.6413, -73.7781),
    "LAX": (33.9416, -118.4085),
    "SAN": (32.7338, -117.1885),
    "SNA": (33.6757, -117.8667),
    "BUR": (33.9117, -117.8722),
    "AUS": (30.1944, -97.6700),
    "PHX": (33.5469, -112.0740),
    "SLC": (40.7883, -111.9774),
    "PDX": (45.5991, -122.7085),
    "BOS": (42.3656, -71.0096),
    "HNL": (21.3168, -157.8971),
    "PSP": (33.8281, -116.4447),
    "ONT": (34.0569, -117.7984),
    "EWR": (40.6895, -74.1745),
    "OGG": (20.8988, -156.4776),
    "BOI": (43.5648, -116.2225),
    "IAD": (38.9333, -77.4391),
    "RNO": (39.5385, -119.7871),
    "MSP": (44.8776, -93.2336),
    "GEG": (47.4548, -117.3565),
    "MIA": (25.7959, -80.2870),
    "CLT": (35.2143, -80.9420),
}


def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in miles between two points on Earth."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_distance(origin, destination):
    """Look up distance between two airport codes. Returns NaN if coords not available."""
    lat1, lon1 = _AIRPORT_COORDS.get(origin, (0, 0))
    lat2, lon2 = _AIRPORT_COORDS.get(destination, (0, 0))
    if lat1 == 0 or lat2 == 0:
        return float("nan")
    return haversine(lat1, lon1, lat2, lon2)


def clean(raw_path="data/flights_bayarea.csv", output_path="data/flights_clean.csv"):
    """Read raw bayarea CSV, engineer features, write flights_clean.csv."""
    print(f"Reading {raw_path}...")

    # BTS CSV uses specific column indices; read by field names via csv reader
    # We need: Year, Quarter, Month, DayofMonth, DayOfWeek, FlightDate,
    #          Reporting_Airline, Origin, Dest, CRSDepTime, ArrDelay, Cancelled, Diverted, Distance
    # But flights_bayarea.csv has many columns; we'll select by position or name.

    df = pd.read_csv(raw_path)

    # The BTS format has Origin at column index 14 (0-based). After our filter,
    # we need to extract the relevant columns. Let us identify them by name
    # where possible. The cleaned schema expects:
    #   FlightDate, Reporting_Airline, Origin, Dest, DayOfWeek, dep_hour,
    #   Distance, DepDelay, ArrDelay, ArrDel15, Cancelled, Diverted

    # Build the cleaned dataframe column by column
    cleaned = pd.DataFrame()

    # FlightDate is already present (column index 5 in BTS format)
    cleaned["FlightDate"] = df["FlightDate"]

    # Reporting_Airline (column index 6)
    cleaned["Reporting_Airline"] = df["Reporting_Airline"]

    # Origin (column index 14)
    cleaned["Origin"] = df["Origin"]

    # Dest (column index 17)
    cleaned["Dest"] = df["Dest"]

    # DayOfWeek (column index 4)
    cleaned["DayOfWeek"] = df["DayOfWeek"].astype(int)

    # CRSDepTime -> dep_hour (extract hour)
    # CRSDepTime is stored as e.g. 1015, 830, 600 etc. Extract the hour part.
    def extract_hour(crs_dep):
        try:
            t = int(crs_dep)
            if t >= 2400:
                return t - 2400
            return t // 100
        except (ValueError, TypeError):
            return None

    cleaned["dep_hour"] = df["CRSDepTime"].apply(extract_hour)

    # Distance (engineered)
    cleaned["Distance"] = df.apply(
        lambda row: compute_distance(row["Origin"], row["Dest"]), axis=1
    )

    # DepDelay (column index 23 in BTS format — DepDelayMinutes)
    cleaned["DepDelay"] = df["DepDelayMinutes"].astype(float)

    # ArrDelay (column index 30 in BTS format — ArrDelayMinutes)
    cleaned["ArrDelay"] = df["ArrDelayMinutes"].astype(float)

    # ArrDel15: 1 if ArrDelay >= 15, else 0
    cleaned["ArrDel15"] = (cleaned["ArrDelay"] >= 15).astype(int)

    # Cancelled (column index 33 in BTS format)
    cleaned["Cancelled"] = df["Cancelled"].astype(int)

    # Diverted (column index 36 in BTS format)
    cleaned["Diverted"] = df["Diverted"].astype(int)

    # Verify column order
    col_order = ["FlightDate", "Reporting_Airline", "Origin", "Dest", "DayOfWeek",
                 "dep_hour", "Distance", "DepDelay", "ArrDelay", "ArrDel15",
                 "Cancelled", "Diverted"]
    cleaned = cleaned[col_order]

    print(f"Rows before cleaning: {len(cleaned):,}")
    # Drop rows where engineered features are missing
    cleaned = cleaned.dropna(subset=["dep_hour", "Distance"])
    print(f"Rows after dropping NaNs: {len(cleaned):,}")

    # Ensure integer types for categorical-ish columns
    cleaned["DayOfWeek"] = cleaned["DayOfWeek"].astype(int)
    cleaned["ArrDel15"] = cleaned["ArrDel15"].astype(int)
    cleaned["Cancelled"] = cleaned["Cancelled"].astype(int)
    cleaned["Diverted"] = cleaned["Diverted"].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"\nWrote {len(cleaned):,} rows to {output_path}")
    print(f"Columns: {list(cleaned.columns)}")

    # Print class distribution of target
    print(f"ArrDel15 distribution:\n{cleaned['ArrDel15'].value_counts().sort_index()}")
    return cleaned


if __name__ == "__main__":
    clean()