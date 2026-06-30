from __future__ import annotations
import numpy as np
import pandas as pd


def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n).astype(float)
    territory = rng.choice(["A", "B", "C"], n)
    coverage = rng.choice(["basic", "standard", "premium"], n)
    exposure = rng.uniform(0.5, 1.0, n)
    freq_rate = 0.08 + 0.20 * (age / 70) + 0.12 * (territory == "B") + 0.25 * (territory == "C") + 0.18 * (coverage == "premium")
    claim_count = rng.poisson(freq_rate * exposure)
    mean_severity = 250 + 120 * (age / 70) + 60 * (territory == "B") + 140 * (territory == "C") + 100 * (coverage == "premium")
    severity = np.where(claim_count > 0, rng.gamma(2, mean_severity / 2, size=n), 0.0)
    claims = (claim_count * severity).round(2)
    zero_claim = (claims == 0).astype(int)
    df = pd.DataFrame({
        "age": age,
        "territory": territory,
        "coverage": coverage,
        "exposure": exposure,
        "claim_count": claim_count,
        "severity": severity,
        "claims": claims,
    })
    return {
        "df": df,
        "categorical_features": ["territory", "coverage"],
        "numerical_features": ["age"],
        "exposure": "exposure",
        "target": "claims",
        "n_samples": n,
        "positive_rate": (claims > 0).mean(),
    }
