"""Real-time inputs for one location (Open-Meteo Forecast API + GloFAS forecast)."""
import pandas as pd
import requests
from config import TZ
from src.features import add_features, FEATURES

FORECAST = "https://api.open-meteo.com/v1/forecast"
FLOOD = "https://flood-api.open-meteo.com/v1/flood"


def _get(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def live_features(info: dict) -> dict:
    base = dict(latitude=info["lat"], longitude=info["lon"], past_days=30, forecast_days=4)
    w = _get(FORECAST, {**base, "daily": "precipitation_sum", "timezone": TZ})
    f = _get(FLOOD, {**base, "daily": "river_discharge"})
    rain = pd.DataFrame({"date": pd.to_datetime(w["daily"]["time"]), "precip": w["daily"]["precipitation_sum"]})
    q = pd.DataFrame({"date": pd.to_datetime(f["daily"]["time"]), "discharge": f["daily"]["river_discharge"]})
    df = rain.merge(q, on="date", how="left")
    feats = add_features(df, info["q2"], info["elevation"])
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    row = feats[feats["date"] == today]
    if row.empty:
        raise RuntimeError("today's row not available from the API")
    i = row.index[0]
    series = [{"date": str(feats.date[j].date()), "rain": _num(df.precip[j]), "q_ratio": _num(feats.q_over_q2[j])}
              for j in range(max(0, i - 6), min(len(feats), i + 4))]
    return {"features": {k: _num(row.iloc[0][k]) for k in FEATURES}, "series": series, "today": str(today.date())}
