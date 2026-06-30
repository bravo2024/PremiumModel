from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.core import compute_regression_metrics
from src.visualizations import plot_actual_vs_predicted, plot_lift_chart, plot_residuals, plot_glm_coefficients, plot_premium_distribution, _style

st.set_page_config(page_title="PremiumModel | Pure Premium Pricing", layout="wide", page_icon="\U0001f3e6")

with st.sidebar:
    st.header("\u2699 Config")
    n = st.slider("Policies", 2000, 20000, 10000, 1000)
    st.caption("WTW | NAIC AI Bulletin | State Rate Filing")

data = make_synthetic(n=n)
b = train_all_models(data)

y_test = b["y_test"]
y_preds = {n: np.array(r["y_pred"]) for n, r in b["results"].items()}
lifts = {n: r["lift"] for n, r in b["results"].items()}
best = b["best_model"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Policies", f"{n:,}")
c2.metric("Claim Rate", f"{data['positive_rate']:.1%}")
c3.metric("Avg Claim ($)", f"${data['df']['claims'].mean():.0f}")
c4.metric("Best Model", best)

t1, t2, t3, t4 = st.tabs([
    "\U0001f4ca Explorer",
    "\U0001f52c Model Lab",
    "\U0001f3af GLM Pricing",
    "\U0001f4b0 Portfolio",
])

with t1:
    st.dataframe(data["df"].head(50), use_container_width=True, height=200)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3))
        _style()
        ax.hist(data["df"]["claims"].values, bins=50, color="#22d3ee", alpha=0.7, edgecolor="none")
        ax.set_xlabel("Claim Amount ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Pure Premium Distribution", color="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        _style()
        for i, (title, col) in enumerate(zip(
            ["Avg Premium by Territory", "Avg Premium by Coverage"],
            ["territory", "coverage"],
        )):
            means = data["df"].groupby(col)["claims"].mean()
            colors_plot = ["#22d3ee", "#a78bfa", "#f97316"]
            axes[i].bar(means.index, means.values, color=colors_plot[:len(means)], alpha=0.7)
            axes[i].set_title(title, color="white", fontsize=9)
            axes[i].tick_params(axis="x", labelsize=7)
            axes[i].grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        st.pyplot(fig)

with t2:
    rows = []
    for n, r in b["results"].items():
        m = r["metrics"]
        rows.append({
            "Model": n,
            "MAE": f"{m['mae']:.2f}",
            "RMSE": f"{m['rmse']:.2f}",
            "R\u00b2": f"{m['r2']:.4f}",
            "Tweedie Dev.": f"{m['tweedie_deviance']:.2f}",
            "Adequacy": f"{m['adequacy']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_actual_vs_predicted(y_test, y_preds))
    with col_b:
        st.pyplot(plot_lift_chart(lifts))
    col_c, col_d = st.columns(2)
    with col_c:
        st.pyplot(plot_residuals(y_test, y_preds))
    with col_d:
        st.pyplot(plot_premium_distribution(data["df"], y_preds))
    cv = cross_validate(data)
    cv_rows = [{
        "Model": n,
        "Tweedie Dev.": f"{s['tweedie_deviance']['mean']:.2f}",
        "\u00b1": f"\u00b1{s['tweedie_deviance']['std']:.4f}",
    } for n, s in cv.items()]
    st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)

with t3:
    st.subheader("GLM Rate Relativities")
    st.latex(r"\log(\mu_i) = \beta_0 + \sum \beta_j x_{ij} + \log(\text{exposure}_i)")
    st.markdown(
        "Tweedie GLM (var power=1.5, log-link) for pure premium modeling \u2014 the CAS Monograph No. 5 standard. "
        "Coefficients represent log-premium impact; exponentiate for multiplicative relativity."
    )
    glm_mdl = b["models"]["Tweedie GLM"]["model"]
    glm_features = b["glm_features"]
    coefs = pd.DataFrame({
        "Feature": glm_features,
        "Coefficient": glm_mdl.params.values.round(4),
        "Relativity": np.exp(glm_mdl.params.values).round(4),
    })
    st.dataframe(coefs, use_container_width=True, hide_index=True)
    st.markdown("Relativity > 1.0 means higher premium factor; < 1.0 means lower factor.")
    st.pyplot(plot_glm_coefficients(glm_mdl, glm_features))
    st.subheader("Key Diagnostics")
    st.markdown(
        f"- **Adequacy**: {b['results']['Tweedie GLM']['metrics']['adequacy']:.4f} "
        "(1.0 = perfect balance, >1 = premiums exceed claims)\n"
        f"- **Loss Ratio**: {b['results']['Tweedie GLM']['metrics']['loss_ratio']:.2%}\n"
        f"- **Tweedie Deviance**: {b['results']['Tweedie GLM']['metrics']['tweedie_deviance']:.2f}"
    )

with t4:
    st.subheader("Portfolio Premium Simulation")
    base_prem = st.number_input("Base Premium ($)", 200, 2000, 800, 100)
    df_p = data["df"].iloc[:len(y_test)].copy()
    df_p["Predicted_GLM"] = y_preds["Tweedie GLM"]
    df_p["Predicted_XGB"] = y_preds["XGBoost"]
    df_p["Premium_GLM"] = base_prem * (df_p["Predicted_GLM"] / df_p["Predicted_GLM"].mean())
    df_p["Premium_XGB"] = base_prem * (df_p["Predicted_XGB"] / df_p["Predicted_XGB"].mean())
    st.dataframe(
        df_p[["age", "territory", "coverage", "exposure", "claims", "Premium_GLM", "Premium_XGB"]].head(10),
        use_container_width=True, hide_index=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4))
        _style()
        for label, series, c in [
            ("Tweedie GLM", df_p["Premium_GLM"], "#22d3ee"),
            ("XGBoost", df_p["Premium_XGB"], "#a78bfa"),
        ]:
            ax.hist(series, bins=30, alpha=0.5, color=c, label=label, density=True)
        ax.set_xlabel("Annual Premium ($)")
        ax.set_ylabel("Density")
        ax.set_title("Premium Distribution by Model", color="white")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(8, 4))
        _style()
        for label, series, c in [
            ("Tweedie GLM", df_p["Premium_GLM"], "#22d3ee"),
            ("XGBoost", df_p["Premium_XGB"], "#a78bfa"),
        ]:
            ax.scatter(df_p["claims"].values, series, alpha=0.3, s=8, color=c, label=label)
        ax.set_xlabel("Actual Claims ($)")
        ax.set_ylabel("Premium Charged ($)")
        ax.set_title("Premium vs Actual Claims", color="white")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    total_prem_glm = df_p["Premium_GLM"].sum()
    total_prem_xgb = df_p["Premium_XGB"].sum()
    total_claims = y_test.sum()
    lr_glm = total_claims / total_prem_glm if total_prem_glm > 0 else 0
    lr_xgb = total_claims / total_prem_xgb if total_prem_xgb > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Written Premium (GLM)", f"${total_prem_glm:,.0f}")
    c2.metric("Loss Ratio (GLM)", f"{lr_glm:.1%}")
    c3.metric("Total Written Premium (XGB)", f"${total_prem_xgb:,.0f}")
    c4.metric("Loss Ratio (XGB)", f"{lr_xgb:.1%}")
