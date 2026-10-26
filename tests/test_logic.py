"""Run:  python -m tests.test_logic   (tiny hand-made inputs that test the code, not data to train on)"""
import numpy as np
import pandas as pd
from src.features import add_features
from src.predict import is_dry


def test_features():
    d = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=40), "precip": 0.0, "discharge": 10.0})
    d.loc[35, "precip"] = 50.0
    f = add_features(d, q2=100.0, elevation=5)
    assert f.loc[35, "rain_1d"] == 50 and f.loc[36, "rain_3d"] == 50
    assert f.loc[34, "rain_next_1d"] == 50 and f.loc[32, "rain_next_3d"] == 50
    assert f.loc[37, "days_since_rain"] == 2 and abs(f.loc[20, "q_over_q2"] - 0.1) < 1e-9


def test_gate():
    base = dict(rain_3d=0, rain_7d=0, rain_next_3d=0, q_over_q2=0.4)
    assert is_dry(base)                                    # no rain + normal river => zero risk
    assert not is_dry({**base, "rain_next_3d": 20})        # heavy rain forecast
    assert not is_dry({**base, "q_over_q2": 1.3})          # river already above flood level


if __name__ == "__main__":
    test_features(); test_gate(); print("all tests passed")
