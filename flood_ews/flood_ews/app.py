"""Flood early-warning website.   python app.py  ->  http://127.0.0.1:5000"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
import joblib
from flask import Flask, jsonify, render_template, request
from config import MODELS
from src.live import live_features
from src.predict import assess

app = Flask(__name__)
BUNDLE = joblib.load(MODELS / "flood_model.joblib")
LOCS = json.loads((MODELS / "thresholds.json").read_text())
CACHE, TTL = {}, 600


def live(slug):
    hit = CACHE.get(slug)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    info = LOCS[slug]
    d = live_features(info)
    r = {"slug": slug, "name": info["name"], "lat": info["lat"], "lon": info["lon"], "today": d["today"],
         "features": d["features"], "series": d["series"], **assess(BUNDLE, d["features"])}
    CACHE[slug] = (time.time(), r)
    return r


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/overview")
def overview():
    def one(s):
        try:
            r = live(s)
            return {k: r[k] for k in ("slug", "name", "lat", "lon", "level", "score")}
        except Exception as e:
            return {"slug": s, "name": LOCS[s]["name"], "lat": LOCS[s]["lat"], "lon": LOCS[s]["lon"], "level": "Unavailable", "score": None, "error": str(e)}
    with ThreadPoolExecutor(6) as ex:
        return jsonify(list(ex.map(one, LOCS)))


@app.get("/api/live/<slug>")
def api_live(slug):
    if slug not in LOCS:
        return jsonify(error="unknown location"), 404
    try:
        return jsonify(live(slug))
    except Exception as e:
        return jsonify(error=str(e)), 502


@app.post("/api/whatif")
def whatif():
    b = request.get_json(force=True)
    num = lambda k, d=0.0: max(0.0, float(b.get(k, d) or 0.0))
    r1 = num("rain_1d")
    r3 = max(num("rain_3d"), r1)
    r7 = max(num("rain_7d"), r3)
    r14 = max(num("rain_14d"), r7)
    r30 = max(num("rain_30d"), r14)
    q = num("river_pct", 30) / 100.0
    f = {"rain_1d": r1, "rain_3d": r3, "rain_7d": r7, "rain_14d": r14, "rain_30d": r30,
         "days_since_rain": 0.0 if r1 >= 1 else min(30.0, num("days_since_rain", 30)),
         "rain_next_1d": num("rain_next_1d"), "rain_next_3d": max(num("rain_next_3d"), num("rain_next_1d")),
         "elevation": num("elevation", 50), "q_over_q2": q, "q_max3_over_q2": q, "q_trend3": 0.0}
    return jsonify({"features": f, **assess(BUNDLE, f)})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
