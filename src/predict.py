"""Risk assessment = physical gate + ML model."""
import numpy as np
import pandas as pd
from src.features import FEATURES

LEVELS = ["Low", "Moderate", "High", "Critical"]
ADVICE = {
    "None": "No rain recently or forecast, and the river is below flood level. No flood risk.",
    "Low": "Routine monitoring. No action needed.",
    "Moderate": "Stay alert. Track weather and river-level updates.",
    "High": "Prepare to evacuate. Secure property and move valuables up.",
    "Critical": "Immediate evacuation. Alert local disaster-management authorities.",
}


def is_dry(f) -> bool:
    """Rain is the trigger. No meaningful rain in the last 3/7 days, none forecast for the
    next 3 days, and the river below its 2-year flood level => risk is exactly zero.
    (If the river is ALREADY above flood level, e.g. water arriving from upstream rain, the
    gate stays open and the model decides - that flood is real even with a dry sky.)"""
    def g(k):
        v = f.get(k)
        return 0.0 if v is None or v != v else float(v)
    return g("rain_3d") < 1.0 and g("rain_7d") < 5.0 and g("rain_next_3d") < 1.0 and g("q_over_q2") < 1.0


def assess(bundle, f: dict) -> dict:
    if is_dry(f):
        return {"level": "None", "score": 0.0, "probs": {n: 0.0 for n in LEVELS},
                "gated": True, "advice": ADVICE["None"]}
    m = bundle["model"]
    P = m.proba_ge(pd.DataFrame([f])[FEATURES])[0]
    lvl = int((P >= np.array(m.thr)).sum())
    probs = [1 - P[0], P[0] - P[1], P[1] - P[2], P[2]]
    return {"level": LEVELS[lvl], "score": float(P[0]), "gated": False, "advice": ADVICE[LEVELS[lvl]],
            "probs": {n: float(x) for n, x in zip(LEVELS, probs)}}
