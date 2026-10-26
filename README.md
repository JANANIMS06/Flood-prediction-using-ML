# AI-Based Flood Risk Prediction & Early Warning System (real-data version)

No synthetic or sample data is used. Every number comes from the sources below.

## Data sources (all free, no API key)
| What | Source | Website |
|---|---|---|
| Daily rainfall 1990-present (ERA5 reanalysis) | Open-Meteo Historical Weather API | https://open-meteo.com/en/docs/historical-weather-api |
| Daily river discharge 1984-present + forecast (GloFAS v4, Copernicus/ECMWF) | Open-Meteo Global Flood API | https://open-meteo.com/en/docs/flood-api |
| Live rain forecast | Open-Meteo Forecast API | https://open-meteo.com/en/docs |
| Elevation | Open-Meteo Elevation API | https://open-meteo.com/en/docs/elevation-api |
| Reported real floods (optional validation) | Dartmouth Flood Observatory, Global Active Archive of Large Flood Events | https://floodobservatory.colorado.edu/Archives/index.html |

Open-Meteo is free for non-commercial use with attribution; cite Open-Meteo, Copernicus (ERA5, GloFAS) and DFO in your report.

## Run (needs internet)
```
pip install -r requirements.txt
python -m src.fetch_data       # downloads real data for every point in config.py (resumable)
python -m src.build_dataset    # builds data/processed/flood_dataset.csv + models/thresholds.json
python -m src.train            # trains, compares, writes reports/ and models/flood_model.joblib
python app.py                  # website at http://127.0.0.1:5000
python -m tests.test_logic     # quick logic tests
python -m src.validate_dfo     # optional: compare labels with DFO reported floods
```

## How it fixes "rain = 0 but Moderate risk"
1. **Rain-based features** (today, 3/7/14/30-day totals, days since rain, forecast next 1/3 days) plus river level relative to the location's 2-year flood level.
2. **Monotonic model**: more rain can never lower risk, less rain can never raise it (`src/model.py`).
3. **Dry-conditions rule** (`src/predict.py`): <1 mm in 3 days, <5 mm in 7 days, <1 mm forecast for 3 days AND river below its 2-year flood level => risk is exactly 0 ("No risk").
   If a river is already above flood level (water from upstream rain) the rule does not apply, because that flood is real even under a dry sky.
4. `src/train.py` measures the false-alarm rate on dry days in the test years and writes it to `reports/metrics.json`.

## Labels and honest limitations
- Label = next-day river discharge class by return period (Low <2-yr, Moderate >=2-yr, High >=5-yr, Critical >=20-yr), thresholds fitted per location on training years only. This is hydrological flood *level*, not damage or inundation extent. Use `validate_dfo.py` to check against reported events.
- Train 1990-2015, validation 2016-2018 (alert thresholds tuned here), test 2019+. Never a random split.
- Persistence (tomorrow = today) is reported as a baseline; river level is a strong predictor, so judge the model by how much it beats persistence on High/Critical recall.
- Training uses ERA5 "forecast" rain of the next days; live use takes real forecasts, which are less accurate. ERA5 tends to underestimate extreme rain.
- GloFAS is a 5 km model; coordinates must sit near a large river. Small urban drains (e.g. Mumbai's Mithi) are not resolved. Edit `LOCATIONS` in `config.py`.
- The score shown is not a calibrated probability.
