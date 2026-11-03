"""
dataset_loader.py
Provides loading and CSV generation utilities for movie data.
"""

import os
import json
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_PATH = os.path.join(DATA_DIR, "movies.json")
CSV_PATH = os.path.join(DATA_DIR, "movies.csv")


def load_movies_from_json():
    """Loads movies from the JSON dataset file."""
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Missing {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def export_to_csv():
    """Exports movies from JSON to CSV format."""
    movies = load_movies_from_json()
    fieldnames = [
        "id", "title", "genres", "overview", "director",
        "cast", "release_year", "rating", "vote_count", "runtime", "poster_path"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in movies:
            row = dict(m)
            if isinstance(row.get("genres"), list):
                row["genres"] = ", ".join(row["genres"])
            if isinstance(row.get("cast"), list):
                row["cast"] = ", ".join(row["cast"])
            writer.writerow(row)
    return CSV_PATH


if __name__ == "__main__":
    csv_file = export_to_csv()
    print(f"Exported dataset to {csv_file}")
