"""Raw daily data -> training table.
Label = flood level on the NEXT day from GloFAS river discharge, using each location's own
return-period thresholds (2-, 5-, 20-year, the same levels GloFAS alerts use), fitted by a
Gumbel distribution on TRAIN-years annual maxima only (no look-ahead).
0 Low (<2-yr) | 1 Moderate (>=2-yr) | 2 High (>=5-yr) | 3 Critical (>=20-yr)
python -m src.build_dataset"""
import json
import numpy as np
import pandas as pd
from config import RAW, PROC, MODELS, TRAIN_END, VAL_END
from src.features import add_features, FEATURES


def gumbel_levels(am):
    beta = am.std(ddof=1) * np.sqrt(6) / np.pi
    mu = am.mean() - 0.5772157 * beta
    return {T: float(mu - beta * np.log(-np.log(1 - 1 / T))) for T in (2, 5, 20)}


def to_level(q, th):
    q = np.asarray(q, dtype=float)
    return np.select([q >= th[20], q >= th[5], q >= th[2]], [3, 2, 1], 0)


def main():
    meta = json.loads((RAW / "meta.json").read_text())
    PROC.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    parts, thresholds = [], {}
    for slug, m in meta.items():
        df = pd.read_csv(RAW / f"{slug}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        tr = df[df["date"] <= TRAIN_END]
        am = tr.groupby(tr["date"].dt.year)["discharge"].max().dropna()
        if len(am) < 15 or am.std() == 0 or am.mean() <= 0:
            print("skipped (not enough river signal):", slug)
            continue
        th = gumbel_levels(am)
        if th[2] <= 0:
            print("skipped (bad thresholds):", slug)
            continue
        d = add_features(df, th[2], m["elevation"])
        nxt = df["discharge"].shift(-1)
        d["level"] = to_level(nxt, th)
        d["level_now"] = to_level(df["discharge"], th)          # persistence baseline
        d["valid"] = nxt.notna() & df["discharge"].notna()
        d["location"] = slug
        parts.append(d[d["valid"]].drop(columns="valid"))
        thresholds[slug] = {**m, "q2": th[2], "q5": th[5], "q20": th[20]}
    data = pd.concat(parts).dropna(subset=FEATURES)
    data.to_csv(PROC / "flood_dataset.csv", index=False)
    (MODELS / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    print(f"{len(data):,} rows, {len(thresholds)} locations")
    split = pd.cut(data["date"], [pd.Timestamp("1900"), pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END), pd.Timestamp("2100")],
                   labels=["train", "val", "test"])
    print(data.groupby(split, observed=True)["level"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
