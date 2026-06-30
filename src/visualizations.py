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
    "colors": ["#22d3ee", "#a78bfa", "#f97316", "#f43f5e"],
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


COLORS_POOL = ["#22d3ee", "#a78bfa", "#f97316", "#f43f5e"]


def plot_actual_vs_predicted(y_true, y_pred_dict):
    _style()
    models_list = list(y_pred_dict.items())
    n = len(models_list)
    fig, axes = plt.subplots(1, max(1, n), figsize=(max(6, 5 * n), 5))
    if n == 1:
        axes = [axes]
    for ax, (name, y_pred), c in zip(axes, models_list, COLORS_POOL):
        ax.scatter(y_true, y_pred, alpha=0.25, s=8, color=c, edgecolors="none")
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
    x = np.arange(1, 11)
    for (name, lift), c in zip(lift_dict.items(), COLORS_POOL):
        actual = np.array([l["actual_sum"] for l in lift])
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
    models_list = list(y_pred_dict.items())
    n = len(models_list)
    fig, axes = plt.subplots(1, max(1, n), figsize=(max(6, 5 * n), 5))
    if n == 1:
        axes = [axes]
    for ax, (name, y_pred), c in zip(axes, models_list, COLORS_POOL):
        resid = np.asarray(y_true) - np.asarray(y_pred)
        ax.scatter(y_pred, resid, alpha=0.25, s=8, color=c, edgecolors="none")
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
    colors_bar = [THEME["green"] if c >= 0 else THEME["rose"] for c in coef[idx]]
    ax.barh(range(len(names)), coef[idx], color=colors_bar, alpha=0.8, edgecolor=THEME["grid"])
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
    ax.hist(df["claims"].values, bins=50, alpha=0.3, color=THEME["fg"], label=f"Actual (n={len(df)})", density=True)
    for (name, y_pred), c in zip(pred_dict.items(), COLORS_POOL):
        ax.hist(np.asarray(y_pred), bins=50, alpha=0.5, color=c, label=name, density=True)
    ax.set_xlabel("Premium ($)")
    ax.set_ylabel("Density")
    ax.set_title("Premium Distribution: Actual vs Predicted", color=THEME["fg"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def plot_correlation_heatmap(df, num_cols, cat_cols):
    _style()
    numeric = df[num_cols].copy()
    for c in cat_cols:
        numeric[f"{c}_code"] = df[c].astype("category").cat.codes
    cols = num_cols + [f"{c}_code" for c in cat_cols] + ["claim_count", "severity", "claims"]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Not enough columns", ha="center", va="center", color=THEME["fg"])
        return fig
    corr = df[cols].corr().values
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = LinearSegmentedColormap.from_list("corr", [THEME["rose"], THEME["bg"], THEME["green"]], N=64)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color=THEME["fg"] if abs(corr[i, j]) < 0.5 else THEME["bg"])
    ax.set_title("Feature Correlation Matrix", color=THEME["fg"])
    plt.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()
    return fig


def plot_box_outliers(df, features):
    _style()
    n = len(features)
    fig, axes = plt.subplots(1, max(1, n), figsize=(max(4, 3 * n), 4))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        bp = ax.boxplot(df[feat].values, vert=False, patch_artist=True,
                         boxprops=dict(facecolor=THEME["cyan"], alpha=0.5),
                         whiskerprops=dict(color=THEME["fg"]),
                         capprops=dict(color=THEME["fg"]),
                         medianprops=dict(color=THEME["amber"]))
        ax.set_title(feat, color=THEME["fg"])
        ax.set_xlabel("Value")
        ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    return fig


def plot_partial_dependence(pd_results, feature_name):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (x_vals, y_vals), c in zip(pd_results.keys(), pd_results.values(), COLORS_POOL):
        ax.plot(x_vals, y_vals, "-", color=c, lw=2, label=label)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted Premium ($)")
    ax.set_title(f"Partial Dependence: {feature_name}", color=THEME["fg"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return fig


def plot_balance_decile(balance_dict):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, 11)
    for (name, bal), c in zip(balance_dict.items(), COLORS_POOL):
        ratios = np.array([b["ratio"] for b in bal])
        ax.plot(x, ratios, "o-", color=c, lw=2, ms=5, label=name)
    ax.axhline(1.0, color=THEME["fg"], ls=":", lw=1.5, alpha=0.5, label="Perfect (1.0)")
    ax.fill_between(x, 0.95, 1.05, alpha=0.1, color=THEME["green"], label="\u00b15% band")
    ax.set_xlabel("Decile (highest predicted risk \u2192)")
    ax.set_ylabel("Actual / Predicted Ratio")
    ax.set_title("Decile Balance (Autocalibration Diagnostic)", color=THEME["fg"])
    ax.set_xticks(x)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    return fig


def plot_bootstrap_ci(bootstrap_results):
    _style()
    n = len(bootstrap_results)
    fig, ax = plt.subplots(figsize=(max(5, 2 * n), 4))
    names = list(bootstrap_results.keys())
    means = [r["mean"] for r in bootstrap_results.values()]
    lowers = [r["lower"] for r in bootstrap_results.values()]
    uppers = [r["upper"] for r in bootstrap_results.values()]
    x_pos = np.arange(n)
    colors_bar = COLORS_POOL[:n]
    ax.bar(x_pos, means, yerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)],
           color=colors_bar, alpha=0.8, capsize=4, edgecolor=THEME["grid"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("R\u00b2 Score")
    ax.set_title("Bootstrap 95% CI for R\u00b2", color=THEME["fg"])
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    return fig


def plot_kneighbors(query_df, neighbors_df, features):
    _style()
    n_feat = len(features)
    fig, axes = plt.subplots(1, max(1, n_feat), figsize=(max(4, 3 * n_feat), 4))
    if n_feat == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        q = query_df[feat].values[0]
        n_vals = neighbors_df[feat].values
        ax.scatter([0] * len(n_vals), n_vals, alpha=0.6, s=30, color=THEME["cyan"], label="Neighbors")
        ax.scatter(0, q, s=80, color=THEME["rose"], marker="*", label="Query", zorder=5)
        ax.set_xticks([])
        ax.set_ylabel(feat)
        ax.set_title(feat, color=THEME["fg"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    return fig
