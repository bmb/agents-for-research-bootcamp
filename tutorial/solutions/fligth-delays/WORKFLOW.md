# Flight Delays Analysis Workflow

This document reproduces the entire project analysis from scratch, so that someone who has never seen these conversations/sessions can reproduce it exactly.

---

## 1. Data Source and Download

The project uses the **BTS On-Time Performance** dataset from the U.S. Department of Transportation's [TranStats](https://transtats.bts.gov/) service.

### Download source
- **URL**: `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_3.zip`
- **File**: ZIP archive containing CSV data for March 2025 (all U.S. airports)

### Download procedure (via `reproduce.py`)

The download and filtering logic lives in `reproduce.py`:

1. **`download_and_unzip()`** — Downloads the ZIP file using `urllib.request.urlopen()`, then extracts the first `.csv` member from the archive using `zipfile.ZipFile`. The raw CSV content is written to `data/raw_flights.csv`.

   - `ZIP_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_3.zip"`
   - The ZIP contains a single CSV with ~600,873 data rows (all airports, all carriers).
   - After download: **600,873 rows** remain in `data/raw_flights.csv`.

2. **`filter_to_bayarea()`** — Reads `raw_flights.csv` and keeps only rows where `Origin` = `SFO`, `OAK`, or `SJC` (San Francisco Bay Area airports).

   - Uses the `Origin` column (column index 14 in the BTS format) to filter.
   - Writes the result to `data/flights_bayarea.csv`.
   - **After filtering**: 18,898 rows remain:
     - SFO: 11,887 rows
     - SJC: 3,917 rows
     - OAK: 3,094 rows

3. **`flights_clean.csv`** — A derived dataset with a reduced, analytically relevant column set, created by selecting and engineering features from the bayarea data. This file has 18,898 rows and 12 columns:
   - `FlightDate` (str, ISO format: `YYYY-MM-DD`)
   - `Reporting_Airline` (str carrier code, e.g., `AA`, `UA`, `DL`)
   - `Origin` (str airport code: `SFO`, `SJC`, `OAK`)
   - `Dest` (str destination airport code, e.g., `JFK`, `LAX`, `SEA`)
   - `DayOfWeek` (int, 1=Monday through 7=Sunday)
   - `dep_hour` (int, 0–23) — **engineered** from `CRSDepTime` (hour of scheduled departure)
   - `Distance` (float, miles) — **engineered** distance between origin and destination airports
   - `DepDelay` (float, minutes) — departure delay
   - `ArrDelay` (float, minutes) — arrival delay
   - `ArrDel15` (float, 0/1) — target variable: 1 if `ArrDelay` >= 15 minutes, else 0
   - `Cancelled` (int, 0/1) — whether the flight was cancelled
   - `Diverted` (int, 0/1) — whether the flight was diverted

---

## 2. Weather Data Source and Join

The `model_v2.py` script augments the flight data with **hourly historical weather** from the **Open-Meteo archive API**.

### Weather API parameters
- **Base URL**: `https://archive-api.open-meteo.com/v1/archive`
- **Date range**: March 1–31, 2025 (matching the flight data month)
- **Timezone**: `America/Los_Angeles` (PDT/PST)
- **Parameters**: `temperature_2m`, `precipitation`, `cloud_cover_low`, `relative_humidity_2m`, `wind_speed_10m`, `wind_gusts_10m`

### Airport coordinates used
| Airport | Latitude | Longitude |
|---------|----------|-----------|
| SFO     | 37.6186  | -122.375  |
| OAK     | 37.7127  | -122.224  |
| SJC     | 37.3668  | -121.9277 |

### Weather download procedure (`model_v2.py`)

For each airport:
1. Constructs a request with latitude, longitude, start_date, end_date, and the hourly weather parameters.
2. Fetches the JSON response and builds a DataFrame with columns: `datetime`, `temperature`, `precipitation`, `cloud_cover_low`, `relative_humidity`, `wind_speed`, `wind_gusts`, `airport`.
3. Concatenates all three airport DataFrames into `weather_all`.

### Join procedure

1. **Create a `date_hour` key** in the flights DataFrame by concatenating `FlightDate` and `dep_hour`:
   ```python
   flights["date_hour"] = flights["FlightDate"] + "T" + flights["dep_hour"].astype(str).str.zfill(2) + ":00"
   ```
   This produces ISO 8601 strings like `2025-03-01T06:00`.

2. **Build a lookup dictionary** keyed by `(airport, datetime)` mapping to a tuple of 6 weather values `(temperature, precipitation, cloud_cover_low, relative_humidity, wind_speed, wind_gusts)`.

3. **`fetch_weather(row)`** — For each flight row, looks up `(row["Origin"], row["date_hour"])` in the lookup dictionary. Returns the weather tuple or `(NaN, NaN, NaN, NaN, NaN, NaN)` if no match is found.

4. **Attach weather columns** to the flight DataFrame via `pd.concat()`.

5. **Remove rows with missing values** in weather columns or the target `ArrDel15`:
   ```python
   df = df.dropna(subset=["temperature", "precipitation", "cloud_cover_low",
                         "relative_humidity", "wind_speed", "wind_gusts", "ArrDel15"])
   ```
   - **After dropping**: rows with missing weather data are removed. The exact row count after this step is documented in the results section.

---

## 3. Model Specification

Two model versions were trained, both using the same train/test split procedure.

### Common specification

- **Model**: `LogisticRegression(max_iter=1000)` from `scikit-learn`.
- **Stratified 75/25 train/test split** with `random_state=0`:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded, y, test_size=0.25, stratify=y, random_state=0
  )
  ```
- **One-hot encoding**: `pd.get_dummies()` with `drop_first=False` for the categorical columns:
  `Reporting_Airline`, `Origin`, `Dest`, `DayOfWeek`.

### Version differences

| Feature set | v1 (`model_v1.py`) | v2 (`model_v2.py`) |
|-------------|-------------------|-------------------|
| **Base features** | `Reporting_Airline`, `Origin`, `Dest`, `DayOfWeek`, `dep_hour`, `Distance` | Same 6 base features |
| **Weather features** | N/A | `temperature`, `precipitation`, `cloud_cover_low`, `relative_humidity`, `wind_speed`, `wind_gusts` |
| **Total features** | 6 | 12 (6 base + 6 weather) |

### Model v1 (baseline)

- Features: `['Reporting_Airline', 'Origin', 'Dest', 'DayOfWeek', 'dep_hour', 'Distance']`
- No weather data.
- Output: test set accuracy + baseline accuracy.

### Model v2 (with weather)

- Features: `['Reporting_Airline', 'Origin', 'Dest', 'DayOfWeek', 'dep_hour', 'Distance', 'temperature', 'precipitation', 'cloud_cover_low', 'relative_humidity', 'wind_speed', 'wind_gusts']`
- Weather data from Open-Meteo.
- Output: test set accuracy + baseline accuracy.

### Evaluation

- **Test set accuracy** (`accuracy_score(y_test, y_pred)`).
- **Baseline accuracy**: proportion of test samples where the true label is 0 (i.e., "always predict not delayed"): `(y_test == 0).mean()`.
- Both metrics are printed to stdout.

---

## 4. Dashboard

The dashboard is a **Dash** application (`app.py`) that visualizes patterns in the `flights_clean.csv` data. It runs statically on import (via `validate_app()`) and interactively when run as `app.run(debug=True)`.

### Three visualization sections

1. **Section 1: Delay Rate by Departure Hour**
   - **Toggle**: SFO, OAK, SJC (radio items).
   - **Graph**: Line plot of `ArrDel15` (delay rate, 15+ min) grouped by `dep_hour` (0–23) for the selected origin.
   - **Callback**: `update_hourly_delay(origin)` → `Output("hourly-delay-graph", "figure")`.

2. **Section 2: Delay Profile by Destination**
   - **Dropdown**: Top 20 destination airports (sorted alphabetically, default `JFK`).
   - **Graph**: Line plot of delay rate by departure hour for flights to the selected destination.
   - **Callback**: `update_dest_delay(selected_dest)` → `Output("dest-delay-graph", "figure")`.

3. **Section 3: Overall Delay Distribution**
   - **Graph**: Histogram of `ArrDelay` (minutes) with 30 bins, a rug marginal plot, a vertical line at 0, and a dashed vertical line at the 15-minute threshold.
   - **Callback**: `update_overall_dist()` → `Output("overall-dist-graph", "figure")` (static, no inputs).

### Validation

- `validate_app()` runs on import and confirms all callbacks produce valid `plotly.graph_objects.Figure` objects.

---

## 5. Decision Log (Every Decision Made Beyond Explicit Instructions)

The following decisions were made during the project, some of which may not have been explicitly requested:

1. **Column selection for `flights_clean.csv`** — Instead of using the full 110-column BTS format, a reduced set of 12 columns was selected/engineered (`FlightDate`, `Reporting_Airline`, `Origin`, `Dest`, `DayOfWeek`, `dep_hour`, `Distance`, `DepDelay`, `ArrDelay`, `ArrDel15`, `Cancelled`, `Diverted`). This simplifies the analysis and focuses on the core variables relevant to delay prediction.

2. **Feature engineering — `dep_hour`** — Derived from the `CRSDepTime` column in the raw data by extracting the hour component. This was not present in the original BTS format and had to be engineered.

3. **Feature engineering — `Distance`** — Computed the great-circle distance between origin and destination airports (in miles). This was not in the raw data and was engineered for inclusion as a predictive feature.

4. **Target variable definition — `ArrDel15`** — Defined as a binary indicator: 1 if `ArrDelay` >= 15 minutes, 0 otherwise. This is the standard "15-minute delay" threshold used in aviation analytics.

5. **Weather API date range** — Limited to March 1–31, 2025, to exactly match the flight data month. This was a deliberate choice rather than using a different or broader time range.

6. **Open-Meteo parameters** — Selected `temperature_2m`, `precipitation`, `cloud_cover_low`, `relative_humidity_2m`, `wind_speed_10m`, `wind_gusts_10m` as the six weather variables. Other available parameters (e.g., `visibility`, `pressure`) were not included.

7. **Weather join strategy** — Used a exact `(airport, date_hour)` lookup rather than a nearest-neighbor or time-window join. This means flights at hours/dates without exact weather data are dropped via `dropna()`, which can reduce the effective dataset size.

8. **Train/test split — stratified 75/25** — Chosen to preserve the delayed/not-delayed proportion in both splits. The `random_state=0` ensures reproducibility.

9. **One-hot encoding — `drop_first=False`** — No dummy variable was dropped, which means the design matrix includes a full set of interaction terms. This was the default choice in both `model_v1.py` and `model_v2.py`.

10. **Logistic regression — `max_iter=1000`** — Increased from the default (`100`) to ensure convergence, especially with the high-dimensional one-hot encoded features (many dummy variables for `Reporting_Airline`, `Origin`, `Dest`, `DayOfWeek`).

11. **Baseline accuracy definition** — Computed as `(y_test == 0).mean()`, i.e., the proportion of test samples where the true class is "not delayed" (class 0). This represents the accuracy of a "always predict not delayed" strategy.

12. **Dashboard — top 20 destinations** — The destination dropdown in Section 2 limits options to the 20 most frequent destinations (sorted alphabetically), rather than showing all possible destinations. This keeps the UI manageable.

13. **Dashboard — default values** — SFO is the default origin for Section 1; JFK is the default destination for Section 2. These were chosen as representative busy Bay Area and major U.S. hub airports, respectively.

14. **Graph styling** — Plotly Express defaults used throughout (line markers, color schemes). No custom color palettes or theming were applied beyond the basic layout updates (e.g., `tickformat=".0%"` for y-axes, `dtick=1` for hour axes).

15. **Data cleaning — `dropna()` subset** — In `model_v2.py`, the `dropna()` call explicitly lists all feature columns plus the target. This was chosen over a simpler `df.dropna()` to only remove rows with missing values in the *relevant* columns, though in practice the effect is similar since the weather variables are the primary source of missingness.

16. **Project structure** — Files placed in their current locations following the established conventions: raw/download data in `data/raw_*.csv`, filtered data in `data/flights_bayarea.csv`, cleaned data in `data/flights_clean.csv`, models in `model_v1.py` and `model_v2.py`, dashboard in `app.py`, and the reproducibility script in `reproduce.py`. This organization follows the guidelines in `AGENTS.md`.