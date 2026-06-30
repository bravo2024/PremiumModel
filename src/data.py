# Insurance premium data: age, territory, coverage tier, claims
import numpy as np
import pandas as pd

def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n).astype(float)
    territory = rng.choice(["A", "B", "C"], n)
    coverage = rng.choice(["basic", "standard", "premium"], n)
    exposure = rng.uniform(0.5, 1.0, n)
    true_rate = 100 + 30 * (age / 70) + 20 * (territory == "B") + 50 * (territory == "C") + 40 * (coverage == "premium")
    claims = rng.gamma(3, true_rate * (exposure / 100), n).round(2)
    df = pd.DataFrame({
        "age": age, "territory": territory, "coverage": coverage,
        "exposure": exposure, "claims": claims,
    })
    return {"df": df, "categorical_features": ["territory", "coverage"], "numerical_features": ["age"], "exposure": "exposure", "target": "claims", "n_samples": n}