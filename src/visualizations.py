from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

THEME = {
    "bg": "#0e1117",
    "fg": "#ffffff",
    "grid": "#1a1f2e",
    "cyan": "#22d3ee",
    "violet": "#a78bfa",
    "orange": "#f97316",
    "rose": "#f43f5e",
    "amber": "#fbbf24",
    "green": "#22c55e",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["bg"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["fg"],
        "text.color": THEME["fg"],
        "xtick.color": THEME["fg"],
        "ytick.color": THEME["fg"],
        "grid.color": THEME["grid"],
        "grid.alpha": 0.3,
        "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": THEME["grid"],
        "legend.labelcolor": THEME["fg"],
    })


def plot_actual_vs_predicted(y_true, y_pred_dict):
    _style()
    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"]]
    for ax, (name, y_pred), c in zip(axes, y_pred_dict.items(), colors):
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=c, edgecolors="none")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color=THEME["fg"], lw=1.5, alpha=0.5, label="Ideal")
        ax.set_xlabel("Actual Claims ($)")
        ax.set_ylabel("Predicted Premium ($)")
        ax.set_title(name, color=THEME["fg"])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    plt.tight_layout()
    return fig


def plot_lift_chart(lift_dict):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"]]
    x = np.arange(1, 11)
    for (name, lift), c in zip(lift_dict.items(), colors):
        actual = np.array([l["actual_sum"] for l in lift])
        predicted = np.array([l["predicted_sum"] for l in lift])
        actual_cum = np.cumsum(actual)
        total_actual = actual_cum[-1] if actual_cum[-1] > 0 else 1
        lift_val = actual_cum / (x / 10 * total_actual)
        ax.plot(x, lift_val, "o-", color=c, lw=2, ms=5, label=name)
    ax.axhline(1.0, color=THEME["fg"], ls=":", lw=1.5, alpha=0.5, label="Random")
    ax.set_xlabel("Decile (highest predicted risk \u2192)")
    ax.set_ylabel("Lift (cumulative actual / expected)")
    ax.set_title("Decile Lift Curve", color=THEME["fg"])
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def plot_residuals(y_true, y_pred_dict):
    _style()
    n_models = len(y_pred_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    colors = [THEME["cyan"], THEME["violet"], THEME["orange"]]
    for ax, (name, y_pred), c in zip(axes, y_pred_dict.items(), colors):
        resid = np.asarray(y_true) - np.asarray(y_pred)
        ax.scatter(y_pred, resid, alpha=0.3, s=10, color=c, edgecolors="none")
        ax.axhline(0, color=THEME["fg"], ls="--", lw=1.5, alpha=0.5)
        ax.set_xlabel("Predicted Premium ($)")
        ax.set_ylabel("Residual (Actual \u2212 Predicted)")
        ax.set_title(name, color=THEME["fg"])
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_glm_coefficients(glm_model, feature_names):
    _style()
    coef = glm_model.params.values
    names = list(glm_model.params.index)
    idx = np.argsort(np.abs(coef)) if len(coef) > 1 else np.arange(len(coef))
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    colors = [THEME["green"] if c >= 0 else THEME["rose"] for c in coef[idx]]
    bars = ax.barh(range(len(names)), coef[idx], color=colors, alpha=0.8, edgecolor=THEME["grid"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.set_xlabel("GLM Coefficient (log-premium scale)")
    ax.set_title("Tweedie GLM Coefficients", color=THEME["fg"])
    ax.axvline(0, color=THEME["fg"], lw=1, alpha=0.4)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    return fig


def plot_premium_distribution(df, pred_dict):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [THEME["cyan"], THEME["violet"]]
    ax.hist(df["claims"].values, bins=50, alpha=0.4, color=THEME["fg"], label=f"Actual (n={len(df)})", density=True)
    for (name, y_pred), c in zip(pred_dict.items(), colors):
        ax.hist(y_pred, bins=50, alpha=0.5, color=c, label=f"{name}", density=True)
    ax.set_xlabel("Premium ($)")
    ax.set_ylabel("Density")
    ax.set_title("Premium Distribution: Actual vs Predicted", color=THEME["fg"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def plot_feature_importance(importances, features, title="Feature Importance", color=None):
    _style()
    imp = np.array([v["mean"] if isinstance(v, dict) else v for v in importances])
    idx = np.argsort(np.abs(imp)) if len(imp) > 1 else np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.3)))
    bars = ax.barh(range(len(features)), imp[idx], color=color or THEME["cyan"], alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title, color=THEME["fg"])
    ax.grid(True, alpha=0.2, axis="x")
    return fig
