from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from src.core import tweedie_deviance, premium_adequacy, loss_ratio, compute_regression_metrics, lift_chart, decile_balance

TWEEDIE_P = 1.5
RANDOM_SEED = 42


def _prepare_features(data):
    df = data["df"]
    cat_cols = data["categorical_features"]
    num_cols = data["numerical_features"]
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
    X_num = df[num_cols].values.astype(float)
    X = np.column_stack([X_num, X_cat.values])
    feature_names = num_cols + list(X_cat.columns)
    return X, feature_names


def fit_glm(X_train, y_train, offset_train, feature_names):
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


def fit_xgboost(X_train, y_train, offset_train, seed=42):
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


def fit_freq_severity(X_train, y_freq_train, y_sev_train, offset_train, has_claim_train, feature_names):
    freq_mask = np.asarray(has_claim_train, bool)
    sev_mask = np.asarray(has_claim_train, bool)
    if sev_mask.sum() < 2:
        return None
    X_sm_freq = sm.add_constant(pd.DataFrame(X_train, columns=feature_names))
    freq_model = sm.GLM(
        y_freq_train, X_sm_freq,
        family=sm.families.Poisson(),
        offset=np.log(offset_train),
    ).fit()
    X_sev = X_train[sev_mask]
    y_sev = y_sev_train[sev_mask]
    if len(y_sev) < 2:
        return None
    X_sm_sev = sm.add_constant(pd.DataFrame(X_sev, columns=feature_names))
    sev_model = sm.GLM(
        y_sev, X_sm_sev,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    return {
        "freq_model": freq_model,
        "sev_model": sev_model,
        "features": list(X_sm_freq.columns),
        "feature_names_raw": feature_names,
    }


def predict_freq_severity(fs_dict, X, offset):
    X_sm = sm.add_constant(X, has_constant="add")
    pred_freq = fs_dict["freq_model"].predict(X_sm)
    pred_sev = fs_dict["sev_model"].predict(X_sm)
    return pred_freq * pred_sev


def fit_autocalibrated(X_train, y_train, offset_train, X_cal, y_cal, offset_cal, feature_names, seed=42):
    base = fit_glm(X_train, y_train, offset_train, feature_names)
    y_pred_cal = predict_glm(base, X_cal)
    eps = 1e-10
    log_pred = np.log(np.clip(y_pred_cal / offset_cal, eps, None))
    log_pred_sm = sm.add_constant(log_pred)
    cal_model = sm.GLM(
        y_cal, log_pred_sm,
        family=sm.families.Tweedie(var_power=TWEEDIE_P, link=sm.families.links.Log()),
        offset=np.log(offset_cal),
    ).fit()
    return {"base": base, "cal_model": cal_model, "feature_names_raw": feature_names}


def predict_autocalibrated(ac_dict, X, offset):
    base_pred = predict_glm(ac_dict["base"], X)
    eps = 1e-10
    log_pred = np.log(np.clip(base_pred, eps, None))
    log_pred_sm = sm.add_constant(log_pred)
    raw = ac_dict["cal_model"].predict(log_pred_sm)
    return base_pred * (raw / base_pred.mean())


def train_all_models(data, seed=42, test_size=0.2):
    df = data["df"]
    y = df[data["target"]].values.astype(float)
    offset = df[data["exposure"]].values.astype(float)
    claim_count = df["claim_count"].values.astype(float)
    severity = df["severity"].values.astype(float)
    has_claim = (claim_count > 0).astype(float)
    X, feature_names = _prepare_features(data)
    X_rest, X_test, y_rest, y_test, off_rest, off_test, cc_rest, cc_test, sv_rest, sv_test, hc_rest, hc_test = train_test_split(
        X, y, offset, claim_count, severity, has_claim, test_size=test_size, random_state=seed
    )
    X_train, X_cal, y_train, y_cal, off_train, off_cal, cc_train, cc_cal, sev_train, sev_cal, hc_train, hc_cal = train_test_split(
        X_rest, y_rest, off_rest, cc_rest, sv_rest, hc_rest, test_size=0.25, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)
    results = {}
    models = {}
    glm_d = fit_glm(X_train_s, y_train, off_train, feature_names)
    y_p_glm = predict_glm(glm_d, X_test_s)
    gm = compute_regression_metrics(y_test, y_p_glm)
    gm["adequacy"] = premium_adequacy(y_p_glm, y_test)
    gm["loss_ratio"] = loss_ratio(y_test, y_p_glm)
    bal = decile_balance(y_test, y_p_glm)
    results["Tweedie GLM"] = {"metrics": gm, "y_pred": y_p_glm.tolist(), "y_true": y_test.tolist(), "lift": lift_chart(y_test, y_p_glm), "balance": bal}
    models["Tweedie GLM"] = glm_d
    fs_d = fit_freq_severity(X_train_s, cc_train, sev_train * hc_train, off_train, hc_train, feature_names)
    if fs_d is not None:
        y_p_fs = predict_freq_severity(fs_d, X_test_s, off_test)
        fm = compute_regression_metrics(y_test, y_p_fs)
        fm["adequacy"] = premium_adequacy(y_p_fs, y_test)
        fm["loss_ratio"] = loss_ratio(y_test, y_p_fs)
        bal_fs = decile_balance(y_test, y_p_fs)
        results["Freq-Severity"] = {"metrics": fm, "y_pred": y_p_fs.tolist(), "y_true": y_test.tolist(), "lift": lift_chart(y_test, y_p_fs), "balance": bal_fs}
        models["Freq-Severity"] = fs_d
    xgb_d = fit_xgboost(X_train_s, y_train, off_train, seed=seed)
    y_p_xgb = predict_xgboost(xgb_d, X_test_s, off_test)
    xm = compute_regression_metrics(y_test, y_p_xgb)
    xm["adequacy"] = premium_adequacy(y_p_xgb, y_test)
    xm["loss_ratio"] = loss_ratio(y_test, y_p_xgb)
    bal_x = decile_balance(y_test, y_p_xgb)
    results["XGBoost"] = {"metrics": xm, "y_pred": y_p_xgb.tolist(), "y_true": y_test.tolist(), "lift": lift_chart(y_test, y_p_xgb), "balance": bal_x}
    models["XGBoost"] = xgb_d
    ac_d = fit_autocalibrated(X_train_s, y_train, off_train, X_cal_s, y_cal, off_cal, feature_names, seed=seed)
    y_p_ac = predict_autocalibrated(ac_d, X_test_s, off_test)
    am = compute_regression_metrics(y_test, y_p_ac)
    am["adequacy"] = premium_adequacy(y_p_ac, y_test)
    am["loss_ratio"] = loss_ratio(y_test, y_p_ac)
    bal_ac = decile_balance(y_test, y_p_ac)
    results["Autocalibrated"] = {"metrics": am, "y_pred": y_p_ac.tolist(), "y_true": y_test.tolist(), "lift": lift_chart(y_test, y_p_ac), "balance": bal_ac}
    models["Autocalibrated"] = ac_d
    best_name = min(results, key=lambda n: results[n]["metrics"]["tweedie_deviance"])
    return {
        "results": results,
        "models": models,
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "features": feature_names,
        "scaler": scaler,
        "glm_features": glm_d["features"],
        "best_model": best_name,
    }


def cross_validate(data, seed=42, n_folds=5):
    df = data["df"]
    y = df[data["target"]].values.astype(float)
    offset = df[data["exposure"]].values.astype(float)
    X, feature_names = _prepare_features(data)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = {k: [] for k in ["Tweedie GLM", "Freq-Severity", "XGBoost", "Autocalibrated"]}
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        off_tr, off_te = offset[train_idx], offset[test_idx]
        cc_tr = df["claim_count"].values[train_idx]
        sev_tr = df["severity"].values[train_idx]
        hc_tr = (cc_tr > 0).astype(float)
        scaler_cv = StandardScaler()
        X_tr_s = scaler_cv.fit_transform(X_tr)
        X_te_s = scaler_cv.transform(X_te)
        glm_d = fit_glm(X_tr_s, y_tr, off_tr, feature_names)
        scores["Tweedie GLM"].append(tweedie_deviance(y_te, predict_glm(glm_d, X_te_s)))
        fs_d = fit_freq_severity(X_tr_s, cc_tr, sev_tr * hc_tr, off_tr, hc_tr, feature_names)
        if fs_d is not None:
            scores["Freq-Severity"].append(tweedie_deviance(y_te, predict_freq_severity(fs_d, X_te_s, off_te)))
        xgb_d = fit_xgboost(X_tr_s, y_tr, off_tr, seed=seed)
        scores["XGBoost"].append(tweedie_deviance(y_te, predict_xgboost(xgb_d, X_te_s, off_te)))
        X_cal_cv, X_val_cv, y_cal_cv, y_val_cv, off_cal_cv, off_val_cv = train_test_split(
            X_tr_s, y_tr, off_tr, test_size=0.25, random_state=seed
        )
        ac_d = fit_autocalibrated(X_cal_cv, y_cal_cv, off_cal_cv, X_val_cv, y_val_cv, off_val_cv, feature_names, seed=seed)
        scores["Autocalibrated"].append(tweedie_deviance(y_te, predict_autocalibrated(ac_d, X_te_s, off_te)))
    return {k: {"tweedie_deviance": {"mean": float(np.mean(v)), "std": float(np.std(v))}} for k, v in scores.items() if v}
