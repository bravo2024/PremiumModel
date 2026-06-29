from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = ["age", "policy_holder_tenure", "coverage_amount", "num_dependents", "smoker", "bmi", "health_score", "premium_history_ontime_pct", "previous_claims", "region", "occupation_risk"]
CATEGORICAL_FEATURES = ["smoker", "region", "occupation_risk"]
NUMERICAL_FEATURES = ["age", "policy_holder_tenure", "coverage_amount", "num_dependents", "bmi", "health_score", "premium_history_ontime_pct", "previous_claims"]

def make_synthetic(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(45, 15, size=n).clip(18, 80).astype(int),
        "policy_holder_tenure": rng.gamma(shape=3, scale=5, size=n).clip(0, 40).round(1),
        "coverage_amount": rng.lognormal(mean=11.5, sigma=0.8, size=n).clip(25000, 2000000).astype(int),
        "num_dependents": rng.poisson(lam=1.5, size=n).clip(0, 6),
        "smoker": rng.choice(["yes", "no"], size=n, p=[0.15, 0.85]),
        "bmi": rng.normal(27, 5, size=n).clip(15, 50).round(1),
        "health_score": rng.beta(7, 3, size=n).clip(0.1, 1.0).round(3),
        "premium_history_ontime_pct": rng.beta(8, 2, size=n).clip(0.5, 1.0).round(3),
        "previous_claims": rng.poisson(lam=0.3, size=n).clip(0, 4),
        "region": rng.choice(["urban", "suburban", "rural"], size=n, p=[0.45, 0.35, 0.20]),
        "occupation_risk": rng.choice(["low", "medium", "high"], size=n, p=[0.40, 0.40, 0.20]),
    })
    smoker = (df["smoker"] == "yes").astype(int); health = df["health_score"]
    bmi = np.clip((df["bmi"] - 25) / 15, 0, 1); claims = np.clip(df["previous_claims"], 0, 3)
    tenure = df["policy_holder_tenure"] / 40; coverage = np.log(df["coverage_amount"] / 10000) / 5
    log_odds = -2.8 + 0.8 * smoker + 0.4 * bmi - 0.5 * health + 0.6 * claims + 0.2 * coverage - 0.15 * tenure + rng.normal(0, 0.4, size=n)
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > np.percentile(prob, 80)).astype(np.float64)
    return {"X": df, "y": y, "features": FEATURE_NAMES, "df": df.assign(claim=y), "categorical_features": CATEGORICAL_FEATURES, "numerical_features": NUMERICAL_FEATURES, "n_samples": n, "n_features": len(FEATURE_NAMES), "positive_rate": y.mean()}
