"""Feature engineering shared by training AND the live app (no train/serve skew)."""
import numpy as np
import pandas as pd

FEATURES = ["rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "days_since_rain",
            "rain_next_1d", "rain_next_3d", "elevation",
            "q_over_q2", "q_max3_over_q2", "q_trend3"]
# +1: risk may only go UP as the value rises, -1: only DOWN, 0: free.
MONOTONIC = {"rain_1d": 1, "rain_3d": 1, "rain_7d": 1, "rain_14d": 1, "rain_30d": 1,
             "days_since_rain": -1, "rain_next_1d": 1, "rain_next_3d": 1, "elevation": 0,
             "q_over_q2": 1, "q_max3_over_q2": 1, "q_trend3": 1}


def add_features(df: pd.DataFrame, q2: float, elevation: float) -> pd.DataFrame:
    """df: daily rows with columns date, precip (mm/day), discharge (m3/s)."""
    d = df.sort_values("date").reset_index(drop=True)
    p, q = d["precip"], d["discharge"]
    out = pd.DataFrame({"date": d["date"]})
    out["rain_1d"] = p
    for n in (3, 7, 14, 30):
        out[f"rain_{n}d"] = p.rolling(n, min_periods=n).sum()
    idx = np.arange(len(d))
    last_wet = pd.Series(np.where(p.fillna(0) >= 1.0, idx, np.nan)).ffill()
    out["days_since_rain"] = (idx - last_wet).clip(upper=30).fillna(30)
    out["rain_next_1d"] = p.shift(-1)
    out["rain_next_3d"] = p.shift(-1) + p.shift(-2) + p.shift(-3)
    out["elevation"] = elevation
    out["q_over_q2"] = q / q2
    out["q_max3_over_q2"] = q.rolling(3, min_periods=3).max() / q2
    out["q_trend3"] = (q - q.shift(3)) / q2
    return out
