# Premium model: Tweedie GLM for pure premium (compound Poisson-Gamma)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.core import tweedie_deviance, premium_adequacy, loss_ratio

def fit_and_evaluate(data, seed=42):
    df = data["df"]
    cat_cols = data["categorical_features"]
    X = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
    X["age"] = df["age"].values
    X = sm.add_constant(X)
    y = df[data["target"]].values
    offset = np.log(df[data["exposure"]].values)
    model = sm.GLM(
        y, X,
        family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
        offset=offset,
    ).fit()
    pred = model.predict(X)
    tv = tweedie_deviance(y, pred)
    return (
        {"model": model, "features": list(X.columns)},
        {
            "tweedie_deviance": tv,
            "adequacy": premium_adequacy(pred, y),
            "loss_ratio": loss_ratio(y, pred),
        },
    )