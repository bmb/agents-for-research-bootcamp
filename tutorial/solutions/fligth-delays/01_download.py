"""
01_download.py — Download and filter BTS On-Time Performance data.

Produces:
  data/raw_flights.csv    - raw unzipped CSV (~600k rows)
  data/flights_bayarea.csv - filtered to SFO, OAK, SJC departures (~19k rows)
"""

import csv
import io
import os
import urllib.request
import zipfile

# --- Configuration ---
ZIP_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_3.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_flights.csv")
FILTERED_CSV = os.path.join(DATA_DIR, "flights_bayarea.csv")


def download_and_unzip():
    """Download the ZIP file and extract the CSV to data/raw_flights.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading from {ZIP_URL}...")

    with urllib.request.urlopen(ZIP_URL) as response:
        zip_data = response.read()

    # Extract the CSV from the ZIP, counting rows and writing to disk
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = [m for m in zf.namelist() if m.endswith(".csv")][0]
        # Read and decode the CSV content
        raw_content = zf.read(csv_name).decode("utf-8")
        lines = raw_content.split("\n")
        total_lines = sum(1 for line in lines if line.strip())

        # Write the clean CSV to disk
        with open(RAW_CSV, "w", encoding="utf-8") as f:
            f.write(raw_content)

        print(f"  Raw file has {total_lines:,} data rows")
        print(f"  Saved raw data to {RAW_CSV}")
        return RAW_CSV


def filter_to_bayarea(raw_path):
    """Filter flights to Origin = SFO, OAK, or SJC and save flights_bayarea.csv."""
    print(f"\nFiltering to Bay Area airports (SFO, OAK, SJC)...")

    with open(raw_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        headers = next(reader)

        # Find the Origin column index (column 14 based on BTS format)
        origin_idx = headers.index("Origin")

        # Prepare output
        os.makedirs(DATA_DIR, exist_ok=True)
        filtered_rows = 0
        counts = {"SFO": 0, "OAK": 0, "SJC": 0}

        with open(FILTERED_CSV, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(headers)  # write header

            for row in reader:
                if not row:
                    continue  # skip empty rows
                origin = row[origin_idx].strip()
                if origin in ("SFO", "OAK", "SJC"):
                    writer.writerow(row)
                    counts[origin] += 1
                    filtered_rows += 1

    # Print summary
    total = sum(counts.values())
    print(f"\nSelected {total:,} rows from SFO+OAK+SJC airports")
    print(f"After filtering: {filtered_rows:,} rows remain")
    print(f"  SFO: {counts['SFO']:,}")
    print(f"  OAK: {counts['OAK']:,}")
    print(f"  SJC: {counts['SJC']:,}")
    return filtered_rows


def main():
    raw_path = download_and_unzip()
    filter_to_bayarea(raw_path)


if __name__ == "__main__":
    main()