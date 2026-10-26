"""Ordinal flood-risk model: three monotonic gradient-boosting classifiers answering
P(level>=Moderate), P(level>=High), P(level>=Critical) (Frank & Hall ordinal scheme).
Monotonic constraints mean MORE rain can never lower risk and LESS rain can never raise it."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import FEATURES, MONOTONIC


class OrdinalFloodModel:
    def __init__(self):
        self.features = FEATURES
        self.mono = [MONOTONIC[f] for f in FEATURES]
        self.thr = [0.5, 0.5, 0.5]
        self.models = []

    def fit(self, X, y):
        y = np.asarray(y)
        self.models = []
        for k in (1, 2, 3):
            pos = int((y >= k).sum())
            if pos < 8:
                raise RuntimeError(f"Only {pos} training rows with level>={k}; add more LOCATIONS in config.py.")
            if pos < 40:
                print(f"WARNING: only {pos} training rows with level>={k}; High/Critical results will be noisy.")
            m = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.07, max_depth=6, min_samples_leaf=40,
                l2_regularization=1.0, monotonic_cst=self.mono, random_state=42)
            self.models.append(m.fit(X[self.features], (y >= k).astype(int)))
        return self

    def proba_ge(self, X):
        P = np.column_stack([m.predict_proba(X[self.features])[:, 1] for m in self.models])
        return np.minimum.accumulate(P, axis=1)       # P(>=2) <= P(>=1) etc.

    def predict(self, X):
        return (self.proba_ge(X) >= np.array(self.thr)).sum(axis=1)
