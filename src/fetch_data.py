"""Download REAL data (no synthetic data anywhere):
  rain      : Open-Meteo Historical Weather API (ERA5 reanalysis, ECMWF/Copernicus)
  discharge : Open-Meteo Global Flood API (GloFAS v4, Copernicus Emergency Mgmt Service)
  elevation : Open-Meteo Elevation API
Resumable: re-run and finished locations are skipped.   python -m src.fetch_data"""
import json
import time
import requests
import pandas as pd
from config import RAW, LOCATIONS, START_DATE, END_DATE, TZ

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FLOOD = "https://flood-api.open-meteo.com/v1/flood"
ELEV = "https://api.open-meteo.com/v1/elevation"


def get(url, params):
    for i in range(6):
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(20 * (i + 1))
            continue
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    raise RuntimeError("too many retries")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    meta_path = RAW / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for slug, (name, lat, lon) in LOCATIONS.items():
        out = RAW / f"{slug}.csv"
        if out.exists():
            print("skip (already downloaded):", slug)
            continue
        try:
            w = get(ARCHIVE, dict(latitude=lat, longitude=lon, start_date=START_DATE,
                                  end_date=END_DATE, daily="precipitation_sum", timezone=TZ))
            f = get(FLOOD, dict(latitude=lat, longitude=lon, start_date=START_DATE,
                                end_date=END_DATE, daily="river_discharge"))
            e = get(ELEV, dict(latitude=lat, longitude=lon))
        except Exception as ex:
            print("FAILED", slug, ex)
            continue
        rain = pd.DataFrame({"date": w["daily"]["time"], "precip": w["daily"]["precipitation_sum"]})
        q = pd.DataFrame({"date": f["daily"]["time"], "discharge": f["daily"]["river_discharge"]})
        df = rain.merge(q, on="date", how="inner")
        if df["discharge"].notna().sum() < 3650:
            print("no usable river coverage, dropped:", slug)
            continue
        df.to_csv(out, index=False)
        meta[slug] = {"name": name, "lat": lat, "lon": lon, "elevation": e["elevation"][0]}
        meta_path.write_text(json.dumps(meta, indent=1))
        print("saved", slug, len(df), "days")
        time.sleep(1)


if __name__ == "__main__":
    main()
