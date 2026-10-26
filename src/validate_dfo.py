"""OPTIONAL external check of the labels against REAL reported floods.
1. Download the Excel archive from https://floodobservatory.colorado.edu/Archives/index.html
   and save it as data/external/FloodArchive.xlsx
2. python -m src.validate_dfo
For DFO events (any country) that began 2019+ within ~1 degree of one of our points, reports how often
our discharge-based level reached >= Moderate during the reported dates (+/- 2 days).
Column names are auto-detected; the script prints them if detection fails."""
import json
import numpy as np
import pandas as pd
from config import RAW, MODELS, REPORTS


def col(df, *keys):
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    raise SystemExit(f"Could not find a column like {keys}. Columns: {list(df.columns)}")


def main():
    ev = pd.read_excel("data/external/FloodArchive.xlsx")
    b, e, lat, lon = col(ev, "began"), col(ev, "ended"), col(ev, "lat", "centroid y"), col(ev, "long", "centroid x")
    ev[b], ev[e] = pd.to_datetime(ev[b], errors="coerce"), pd.to_datetime(ev[e], errors="coerce")
    ev = ev[(ev[b] >= "2019-01-01")].dropna(subset=[b, e, lat, lon])
    th = json.loads((MODELS / "thresholds.json").read_text())
    hits = tot = 0
    for _, r in ev.iterrows():
        for slug, t in th.items():
            if abs(t["lat"] - r[lat]) < 1 and abs(t["lon"] - r[lon]) < 1:
                d = pd.read_csv(RAW / f"{slug}.csv", parse_dates=["date"])
                w = d[(d.date >= r[b] - pd.Timedelta(days=2)) & (d.date <= r[e] + pd.Timedelta(days=2))]
                if len(w):
                    tot += 1
                    hits += int((w["discharge"] >= t["q2"]).any())
    print(f"{hits}/{tot} reported flood events near our points reached >= Moderate (2-year level) in GloFAS discharge")
    if tot < 10:
        print("Few matches - add more LOCATIONS near flood-affected areas for a stronger check.")


if __name__ == "__main__":
    main()
