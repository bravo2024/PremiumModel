from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def tweedie_deviance(y, yhat, p=1.5):
    y = np.asarray(y, float)
    yhat = np.clip(np.asarray(yhat, float), 1e-10, None)
    t1 = np.power(np.clip(y, 0, None), 2 - p) / ((1 - p) * (2 - p))
    t2 = y * np.power(yhat, 1 - p) / (1 - p)
    t3 = np.power(yhat, 2 - p) / (2 - p)
    return float(2 * np.sum(t1 - t2 + t3))


def premium_adequacy(predicted, actual):
    return float(np.sum(np.asarray(actual)) / np.sum(np.asarray(predicted)))


def loss_ratio(claims, premiums):
    return float(np.sum(claims) / np.sum(premiums)) if np.sum(premiums) > 0 else 0.0


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "tweedie_deviance": tweedie_deviance(y_true, y_pred),
        "adequacy": premium_adequacy(y_pred, y_true),
    }


def lift_chart(y_true, y_pred, n_deciles=10):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    n = len(y_true)
    decile_size = n // n_deciles
    results = []
    for i in range(n_deciles):
        start = i * decile_size
        end = start + decile_size if i < n_deciles - 1 else n
        actual = y_true_sorted[start:end].sum()
        predicted = y_pred_sorted[start:end].sum()
        results.append({
            "decile": i + 1,
            "actual_sum": float(actual),
            "predicted_sum": float(predicted),
            "count": end - start,
        })
    return results
