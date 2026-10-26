"""Train + honestly evaluate (time-based split, never random).   python -m src.train"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, fbeta_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import PROC, MODELS, REPORTS, TRAIN_END, VAL_END
from src.features import FEATURES
from src.model import OrdinalFloodModel
from src.predict import assess, LEVELS

NAMES = ["Low", "Moderate", "High", "Critical"]


def tune_thresholds(model, val):
    P = model.proba_ge(val[FEATURES])
    thr = []
    for k in range(3):
        y = (val["level"] >= k + 1).astype(int)
        if y.sum() == 0:
            thr.append(0.5)
            continue
        grid = np.arange(0.05, 0.96, 0.05)
        scores = [fbeta_score(y, P[:, k] >= t, beta=2, zero_division=0) for t in grid]
        thr.append(float(grid[int(np.argmax(scores))]))
    return thr


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(PROC / "flood_dataset.csv", parse_dates=["date"])
    tr = df[df.date <= TRAIN_END]
    va = df[(df.date > TRAIN_END) & (df.date <= VAL_END)]
    te = df[df.date > VAL_END]
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    ordm = OrdinalFloodModel().fit(tr[FEATURES], tr["level"])
    ordm.thr = tune_thresholds(ordm, va)
    print("alert thresholds tuned on validation years:", ordm.thr)

    others = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    preds = {"Persistence (tomorrow = today)": te["level_now"].values,
             "Ordinal monotonic GBM (deployed)": ordm.predict(te[FEATURES])}
    for n, m in others.items():
        preds[n] = m.fit(tr[FEATURES], tr["level"]).predict(te[FEATURES])

    rows = []
    for n, p in preds.items():
        rep = classification_report(te["level"], p, labels=range(4), target_names=NAMES, output_dict=True, zero_division=0)
        rows.append({"Model": n, "Accuracy": accuracy_score(te["level"], p),
                     "Macro F1": f1_score(te["level"], p, average="macro", zero_division=0),
                     "Recall Moderate": rep["Moderate"]["recall"], "Recall High": rep["High"]["recall"],
                     "Recall Critical": rep["Critical"]["recall"], "Precision High": rep["High"]["precision"]})
    table = pd.DataFrame(rows).set_index("Model").round(3)
    print(table.to_string())
    table.to_csv(REPORTS / "model_comparison.csv")
    p = preds["Ordinal monotonic GBM (deployed)"]
    print(classification_report(te["level"], p, labels=range(4), target_names=NAMES, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(te["level"], p, labels=range(4), display_labels=NAMES, cmap="Blues")
    plt.title("Test years (2019+) - deployed model")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close()

    s = te.sample(min(20000, len(te)), random_state=1)
    imp = permutation_importance(ordm, s[FEATURES], s["level"], n_repeats=3, random_state=1,
                                 scoring=lambda e, X, y: f1_score(y, e.predict(X), average="macro", zero_division=0))
    pd.Series(imp.importances_mean, index=FEATURES).sort_values().plot.barh(color="#3B6FD4")
    plt.title("Permutation importance (drop in macro-F1)")
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close()

    # --- the prototype bug: no rain must mean no risk ---
    dry = te[(te.rain_3d < 1) & (te.rain_7d < 5) & (te.rain_next_3d < 1) & (te.q_over_q2 < 1)]
    raw_alarm = float((ordm.predict(dry[FEATURES]) >= 1).mean()) if len(dry) else None
    gated = [assess({"model": ordm}, r)["level"] for r in dry.head(2000).to_dict("records")]
    sanity = {"dry_days_in_test": int(len(dry)), "raw_model_alarm_rate_on_dry_days": raw_alarm,
              "final_system_nonzero_risk_on_dry_days": float(np.mean([g != "None" for g in gated])) if gated else None,
              "alert_thresholds": ordm.thr}
    print("dry-day sanity:", sanity)
    (REPORTS / "metrics.json").write_text(json.dumps({"comparison": table.reset_index().to_dict("records"), "sanity": sanity}, indent=1))
    joblib.dump({"model": ordm, "features": FEATURES, "levels": LEVELS}, MODELS / "flood_model.joblib")
    print("saved models/flood_model.joblib and reports/")


if __name__ == "__main__":
    main()
