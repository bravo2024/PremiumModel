from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from src.core import tweedie_deviance, premium_adequacy, loss_ratio, compute_regression_metrics, lift_chart


TWEEDIE_P = 1.5


def _prepare_features(data):
    df = data["df"]
    cat_cols = data["categorical_features"]
    num_cols = data["numerical_features"]
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
    X_num = df[num_cols].values.astype(float)
    X = np.column_stack([X_num, X_cat.values])
    feature_names = num_cols + list(X_cat.columns)
    return X, feature_names


def fit_glm(data, X_train, y_train, offset_train, feature_names):
    X_sm = sm.add_constant(pd.DataFrame(X_train, columns=feature_names))
    model = sm.GLM(
        y_train, X_sm,
        family=sm.families.Tweedie(var_power=TWEEDIE_P, link=sm.families.links.Log()),
        offset=np.log(offset_train),
    ).fit()
    return {"model": model, "features": list(X_sm.columns), "feature_names_raw": feature_names}


def predict_glm(glm_dict, X):
    X_sm = sm.add_constant(X, has_constant="add")
    return glm_dict["model"].predict(X_sm)


def fit_xgboost(data, X_train, y_train, offset_train, seed=42):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        objective="reg:tweedie",
        eval_metric="tweedie-nloglik@1.5",
        max_depth=4,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
        verbosity=0,
    )
    if offset_train is not None:
        y_rate = y_train / offset_train
        sample_weight = offset_train
    else:
        y_rate = y_train
        sample_weight = None
    model.fit(X_train, y_rate, sample_weight=sample_weight)
    return {"model": model, "features": []}


def predict_xgboost(xgb_dict, X, offset=None):
    pred_rate = xgb_dict["model"].predict(X)
    if offset is not None:
        return pred_rate * offset
    return pred_rate


def train_all_models(data, seed=42, test_size=0.2):
    df = data["df"]
    y = df[data["target"]].values.astype(float)
    offset = df[data["exposure"]].values.astype(float)
    X, feature_names = _prepare_features(data)
    X_train, X_test, y_train, y_test, offset_train, offset_test = train_test_split(
        X, y, offset, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = {}
    models = {}
    glm_dict = fit_glm(data, X_train_scaled, y_train, offset_train, feature_names)
    y_pred_glm = predict_glm(glm_dict, X_test_scaled)
    glm_metrics = compute_regression_metrics(y_test, y_pred_glm)
    glm_metrics["adequacy"] = premium_adequacy(y_pred_glm, y_test)
    glm_metrics["loss_ratio"] = loss_ratio(y_test, y_pred_glm)
    results["Tweedie GLM"] = {"metrics": glm_metrics, "y_pred": y_pred_glm.tolist(), "y_true": y_test.tolist(), "lift": lift_chart(y_test, y_pred_glm)}
    models["Tweedie GLM"] = glm_dict
    xgb_dict = fit_xgboost(data, X_train_scaled, y_train, offset_train, seed=seed)
    y_pred_xgb = predict_xgboost(xgb_dict, X_test_scaled, offset_test)
    xgb_metrics = compute_regression_metrics(y_test, y_pred_xgb)
    xgb_metrics["adequacy"] = premium_adequacy(y_pred_xgb, y_test)
    xgb_metrics["loss_ratio"] = loss_ratio(y_test, y_pred_xgb)
    results["XGBoost"] = {"metrics": xgb_metrics, "y_pred": y_pred_xgb.tolist(), "y_true": y_test.tolist(), "lift": lift_chart(y_test, y_pred_xgb)}
    models["XGBoost"] = xgb_dict
    best_name = min(results, key=lambda n: results[n]["metrics"]["tweedie_deviance"])
    return {
        "results": results,
        "models": models,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "features": feature_names,
        "scaler": scaler,
        "glm_features": glm_dict["features"],
        "best_model": best_name,
    }


def cross_validate(data, seed=42, n_folds=5):
    df = data["df"]
    y = df[data["target"]].values.astype(float)
    offset = df[data["exposure"]].values.astype(float)
    X, feature_names = _prepare_features(data)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    glm_scores = []
    xgb_scores = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        off_tr, off_te = offset[train_idx], offset[test_idx]
        scaler_cv = StandardScaler()
        X_tr_s = scaler_cv.fit_transform(X_tr)
        X_te_s = scaler_cv.transform(X_te)
        glm_dict = fit_glm(data, X_tr_s, y_tr, off_tr, feature_names)
        y_p_glm = predict_glm(glm_dict, X_te_s)
        glm_scores.append(tweedie_deviance(y_te, y_p_glm))
        xgb_dict = fit_xgboost(data, X_tr_s, y_tr, off_tr, seed=seed)
        y_p_xgb = predict_xgboost(xgb_dict, X_te_s, off_te)
        xgb_scores.append(tweedie_deviance(y_te, y_p_xgb))
    return {
        "Tweedie GLM": {"tweedie_deviance": {"mean": float(np.mean(glm_scores)), "std": float(np.std(glm_scores))}},
        "XGBoost": {"tweedie_deviance": {"mean": float(np.mean(xgb_scores)), "std": float(np.std(xgb_scores))}},
    }
