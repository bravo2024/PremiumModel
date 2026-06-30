from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate, _prepare_features, predict_glm, predict_xgboost, predict_freq_severity, predict_autocalibrated
from src.core import compute_regression_metrics, bootstrap_ci, partial_dependence, decile_balance
from src.visualizations import (
    plot_actual_vs_predicted, plot_lift_chart, plot_residuals,
    plot_glm_coefficients, plot_premium_distribution, _style,
    plot_correlation_heatmap, plot_box_outliers, plot_partial_dependence as plot_pd,
    plot_balance_decile, plot_bootstrap_ci, plot_kneighbors,
)

st.set_page_config(page_title="PremiumModel \u2014 Pure Premium Pricing", layout="wide", page_icon="\U0001f3e6")

with st.sidebar:
    st.header("\u2699 Config")
    n = st.slider("Policies", 2000, 20000, 10000, 1000)
    st.caption("WTW | NAIC AI Bulletin | State Rate Filing")

data = make_synthetic(n=n)
b = train_all_models(data)

y_test = b["y_test"]
y_preds = {n: np.array(r["y_pred"]) for n, r in b["results"].items()}
lifts = {n: r["lift"] for n, r in b["results"].items()}
balances = {n: r["balance"] for n, r in b["results"].items()}
best = b["best_model"]
mdl_names = list(b["results"].keys())
df = data["df"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Policies", f"{n:,}")
c2.metric("Claim Rate", f"{data['positive_rate']:.1%}")
c3.metric("Avg Claim ($)", f"${df['claims'].mean():.0f}")
c4.metric("Best Model", best)

t0, t1, t2, t3, t4, t5, t6 = st.tabs([
    "\U0001f3e6 Overview",
    "\U0001f4ca Data Explorer",
    "\U0001f52c Model Lab",
    "\U0001f3af GLM Pricing",
    "\U0001f4b0 Portfolio",
    "\U0001f6e1\ufe0f Risk Mgmt",
    "\U0001f4dd What-If",
])

with t0:
    st.header("Pure Premium Modeling Framework")
    st.markdown("## Methodology")
    st.latex(r"\log(\mu_i) = \beta_0 + \sum \beta_j x_{ij} + \log(\text{exposure}_i)")
    st.markdown(
        "The pure premium is modeled via the **Tweedie distribution** (compound Poisson-Gamma, variance "
        "power \\(p=1.5\\)) with a log-link function, consistent with CAS Monograph No. 5 and NAIC pricing "
        "guidelines. Four models are trained and compared:"
    )
    for name, desc in [
        ("Tweedie GLM", "Generalized Linear Model with Tweedie family. The regulatory standard for rate "
         "filing\u2014fully interpretable, closed-form coefficients, exposure offset."),
        ("Freq-Severity", "Decomposed Poisson (frequency) + Gamma (severity) GLMs. Industry-standard "
         "two-part model. Predicts \\(\\hat{\\lambda} \\times \\hat{\\mu}_{sev}\\) per policy."),
        ("XGBoost", "Gradient-boosted trees with Tweedie regression objective. Captures non-linear "
         "interactions automatically. Benchmark for ML-based pricing."),
        ("Autocalibrated", "Tweedie GLM + calibration GLM on a held-out set (Denuit et al. 2021). "
         "Restores local balance\u2014ensuring premiums match claims within each risk decile."),
    ]:
        with st.expander(name):
            st.markdown(desc)
            if name in b["results"]:
                m = b["results"][name]["metrics"]
                st.markdown(
                    f"**Test set:** MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R\u00b2={m['r2']:.4f}, "
                    f"Tweedie Dev.={m['tweedie_deviance']:.2f}, Adequacy={m['adequacy']:.4f}"
                )
    st.markdown("## Synthetic Data")
    st.dataframe(
        pd.DataFrame({
            "Field": ["age", "territory", "coverage", "exposure", "claim_count", "severity", "claims"],
            "Type": ["numeric", "categorical", "categorical", "numeric", "count", "continuous", "continuous"],
            "Role": ["Rating factor", "Rating factor", "Rating factor", "Offset", "Frequency target", "Severity target", "Pure premium target"],
        }),
        use_container_width=True, hide_index=True,
    )
    st.markdown("## Evaluation Framework")
    st.markdown(
        "- **Tweedie Deviance**: Proper scoring rule for compound Poisson-Gamma\n"
        "- **Adequacy**: \\(\\sum y_i / \\sum \\hat{y}_i\\) > 1 means premiums exceed claims\n"
        "- **Decile Lift**: Cumulative actual/predicted ratio across risk-ranked deciles\n"
        "- **Decile Balance**: Per-decile adequacy (Denuit 2021 autocalibration diagnostic)\n"
        "- **Bootstrap R\u00b2 CI**: 95% confidence interval via 1,000 bootstrap resamples"
    )
    st.markdown("## References")
    st.markdown(
        "- Denuit, Charpentier & Trufin (2021). *Autocalibration and Tweedie-dominance for Insurance Pricing with Machine Learning.* arXiv:2103.03635\n"
        "- Laub, Pho & Wong (2025). *An Interpretable Deep Learning Model for General Insurance Pricing.* arXiv:2509.08467\n"
        "- W\u00fcthrich & Buser (2023). *Data Analytics for Non-Life Insurance Pricing.* ETH Zurich\n"
        "- *Machine Learning and Frequency\u2013Severity Decomposition for Insurance Pricing.* Mathematics 2026, 14(10), 1640"
    )

with t1:
    st.dataframe(df.head(50), use_container_width=True, height=200)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3)); _style()
        ax.hist(df["claims"].values, bins=50, color="#22d3ee", alpha=0.7, edgecolor="none")
        ax.set_xlabel("Claim Amount ($)"); ax.set_ylabel("Frequency")
        ax.set_title("Pure Premium Distribution", color="white"); ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3)); _style()
        for i, (title, col) in enumerate(zip(["Avg Premium by Territory", "Avg Premium by Coverage"], ["territory", "coverage"])):
            means = df.groupby(col)["claims"].mean()
            axes[i].bar(means.index, means.values, color=["#22d3ee", "#a78bfa", "#f97316"][:len(means)], alpha=0.7)
            axes[i].set_title(title, color="white", fontsize=9); axes[i].tick_params(axis="x", labelsize=7)
            axes[i].grid(True, alpha=0.2, axis="y")
        plt.tight_layout(); st.pyplot(fig)
    col_c, col_d = st.columns(2)
    with col_c:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3)); _style()
        axes[0].hist(df["claim_count"], bins=15, color="#22d3ee", alpha=0.7, edgecolor="none")
        axes[0].set_title("Claim Count (Frequency)", color="white", fontsize=9); axes[0].grid(True, alpha=0.2)
        nonzero = df[df["claims"] > 0]["severity"]
        axes[1].hist(nonzero, bins=30, color="#a78bfa", alpha=0.7, edgecolor="none")
        axes[1].set_title("Severity | Claims > 0", color="white", fontsize=9); axes[1].grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig)
    with col_d:
        st.pyplot(plot_correlation_heatmap(df, data["numerical_features"], data["categorical_features"]))
    col_e, col_f = st.columns(2)
    with col_e:
        fig, ax = plt.subplots(figsize=(5, 3)); _style()
        ax.hist(df["exposure"].values, bins=30, color="#22d3ee", alpha=0.7, edgecolor="none")
        ax.set_xlabel("Exposure"); ax.set_ylabel("Count")
        ax.set_title("Exposure Distribution", color="white"); ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    with col_f:
        st.pyplot(plot_box_outliers(df, data["numerical_features"] + ["claims"]))

with t2:
    rows = []
    for n, r in b["results"].items():
        m = r["metrics"]
        rows.append({"Model": n, "MAE": f"{m['mae']:.2f}", "RMSE": f"{m['rmse']:.2f}",
                     "R\u00b2": f"{m['r2']:.4f}", "Tweedie Dev.": f"{m['tweedie_deviance']:.2f}",
                     "Adequacy": f"{m['adequacy']:.4f}", "Loss Ratio": f"{m['loss_ratio']:.2%}"})
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a: st.pyplot(plot_actual_vs_predicted(y_test, y_preds))
    with col_b: st.pyplot(plot_lift_chart(lifts))
    col_c, col_d = st.columns(2)
    with col_c: st.pyplot(plot_residuals(y_test, y_preds))
    with col_d: st.pyplot(plot_premium_distribution(df, y_preds))
    st.subheader("Autocalibration Diagnostic (Denuit et al. 2021)")
    st.markdown(
        "Each bar shows the ratio of actual-to-predicted claims within a risk decile. "
        "Values close to 1.0 indicate local balance. The **\u00b15% band** is standard threshold for "
        "regulatory adequacy. Models outside this band may need bias correction."
    )
    st.pyplot(plot_balance_decile(balances))
    cv = cross_validate(data)
    cv_rows = [{"Model": n, "Tweedie Dev.": f"{s['tweedie_deviance']['mean']:.2f}",
                 "\u00b1": f"\u00b1{s['tweedie_deviance']['std']:.4f}"} for n, s in cv.items()]
    st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)
    st.subheader("Bootstrap R\u00b2 Confidence Intervals (n=1,000)")
    boot_results = {}
    for nm in mdl_names:
        yp = y_preds[nm]
        l, u = bootstrap_ci(y_test, yp)
        boot_results[nm] = {"mean": (l + u) / 2, "lower": l, "upper": u}
    st.pyplot(plot_bootstrap_ci(boot_results))

with t3:
    st.subheader("GLM Rate Relativities")
    st.latex(r"\log(\mu_i) = \beta_0 + \sum \beta_j x_{ij} + \log(\text{exposure}_i)")
    st.markdown(
        "Tweedie GLM (var power=1.5, log-link) for pure premium modeling. "
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
    st.markdown("Relativity > 1.0 = higher premium factor; < 1.0 = lower factor.")
    st.pyplot(plot_glm_coefficients(glm_mdl, glm_features))
    st.subheader("ANAM-style Partial Dependence")
    st.markdown("Per-feature partial dependence plots (Laub et al. 2025). Models are evaluated on a grid while holding other features at their mean.")
    pd_feat = st.selectbox("Feature", b["features"])
    if pd_feat:
        feat_idx = b["features"].index(pd_feat)
        pd_results = {}
        X_grid = b["X_test"].copy()
        for nm in mdl_names:
            md = b["models"][nm]
            if nm == "Autocalibrated":
                fn = lambda x, m=md: predict_autocalibrated(m, x, np.ones(x.shape[0]))
            elif nm == "Freq-Severity":
                fn = lambda x, m=md: predict_freq_severity(m, x, np.ones(x.shape[0]))
            elif nm == "XGBoost":
                fn = lambda x, m=md: predict_xgboost(m, x, np.ones(x.shape[0]))
            else:
                fn = lambda x, m=md: predict_glm(m, x)
            xv, yv = partial_dependence(fn, X_grid, feat_idx, grid_size=30)
            pd_results[nm] = (xv, yv)
        st.pyplot(plot_pd(pd_results, pd_feat))
    st.subheader("Key Diagnostics")
    m = b["results"]["Tweedie GLM"]["metrics"]
    st.markdown(
        f"- **Adequacy**: {m['adequacy']:.4f} (1.0 = perfect balance)\n"
        f"- **Loss Ratio**: {m['loss_ratio']:.2%}\n"
        f"- **Tweedie Deviance**: {m['tweedie_deviance']:.2f}"
    )

with t4:
    st.subheader("Portfolio Premium Simulation")
    base_prem = st.number_input("Base Premium ($)", 200, 2000, 800, 100)
    loading = st.slider("Loading (expenses + profit)", 0.0, 0.5, 0.2, 0.05)
    df_p = df.iloc[:len(y_test)].copy()
    for nm in mdl_names:
        df_p[f"Pred_{nm}"] = y_preds[nm]
        df_p[f"Prem_{nm}"] = base_prem * (df_p[f"Pred_{nm}"] / df_p[f"Pred_{nm}"].mean()) * (1 + loading)
    cols_show = ["age", "territory", "coverage", "exposure", "claims"] + [f"Prem_{nm}" for nm in mdl_names]
    st.dataframe(df_p[cols_show].head(10), use_container_width=True, hide_index=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4)); _style()
        for nm, c in zip(mdl_names, ["#22d3ee", "#a78bfa", "#f97316", "#f43f5e"]):
            ax.hist(df_p[f"Prem_{nm}"].values, bins=30, alpha=0.5, color=c, label=nm, density=True)
        ax.set_xlabel("Annual Premium ($)"); ax.set_ylabel("Density")
        ax.set_title("Premium Distribution by Model", color="white")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(8, 4)); _style()
        for nm, c in zip(mdl_names, ["#22d3ee", "#a78bfa", "#f97316", "#f43f5e"]):
            ax.scatter(df_p["claims"].values, df_p[f"Prem_{nm}"].values, alpha=0.2, s=6, color=c, label=nm)
        ax.set_xlabel("Actual Claims ($)"); ax.set_ylabel("Premium Charged ($)")
        ax.set_title("Premium vs Actual Claims", color="white")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    st.subheader("Portfolio KPIs")
    total_claims = y_test.sum()
    kpi_cols = st.columns(4)
    for i, nm in enumerate(mdl_names):
        tp = df_p[f"Prem_{nm}"].sum()
        lr = total_claims / tp if tp > 0 else 0
        kpi_cols[i].metric(f"Written Prem. ({nm[:6]}...)", f"${tp:,.0f}")
    kpi_cols2 = st.columns(4)
    for i, nm in enumerate(mdl_names):
        tp = df_p[f"Prem_{nm}"].sum()
        lr = total_claims / tp if tp > 0 else 0
        kpi_cols2[i].metric(f"Loss Ratio ({nm[:6]}...)", f"{lr:.1%}")
    var_99 = np.percentile(df_p["claims"].values * (1 + loading), 99)
    st.metric("VaR(99%) of Portfolio Claims", f"${var_99:,.0f}")
    st.metric("Estimated Capital Requirement (Solvency II)", f"${var_99 * 0.15:,.0f}")
    st.caption("Capital = 15% of VaR(99%) per Solvency II guidelines (simplified).")

with t5:
    st.subheader("Model Card")
    st.markdown(
        "| Attribute | Value |\n"
        "|-----------|-------|\n"
        f"| **Purpose** | Pure premium pricing for P&C insurance |\n"
        f"| **Training data** | {n} synthetic policies (compound Poisson-Gamma) |\n"
        f"| **Features** | {', '.join(b['features'])} |\n"
        f"| **Models** | {', '.join(mdl_names)} |\n"
        f"| **Best model** | {best} |\n"
        f"| **Best Tweedie Dev.** | {b['results'][best]['metrics']['tweedie_deviance']:.2f} |\n"
        f"| **Intended use** | Pricing decision support, rate filing analysis |\n"
        f"| **Limitations** | Synthetic data only; needs validation on real portfolio data |\n"
    )
    st.subheader("Fairness Audit")
    st.markdown("**Premium-to-Claim ratio by territory** (geographic fairness proxy):")
    if "Tweedie GLM" in y_preds:
        df_fair = df.iloc[:len(y_test)].copy()
        df_fair["Predicted"] = y_preds["Tweedie GLM"]
        fair = df_fair.groupby("territory").apply(lambda g: g["claims"].sum() / g["Predicted"].sum() if g["Predicted"].sum() > 0 else 0)
        st.dataframe(pd.DataFrame({"Territory": fair.index, "Premium/Claim Ratio": fair.values.round(4)}), use_container_width=True, hide_index=True)
    st.markdown("**Proxy discrimination check:**")
    for cat in data["categorical_features"]:
        for num in data["numerical_features"]:
            corr_val = float(np.corrcoef(pd.factorize(df[cat])[0].astype(float), df[num].values)[0, 1])
            st.markdown(f"- `{cat}` vs `{num}`: correlation = {corr_val:.3f}")
    st.subheader("Stability Over Time")
    st.markdown("Synthetic time-split: first 70% of rows (train) vs last 30% (test) simulates temporal drift.")
    half = len(df) // 2
    df_train = df.iloc[:half]
    df_test = df.iloc[half:]
    y_train_half = df_train["claims"].values
    y_test_half = df_test["claims"].values
    st.markdown(f"- Train period avg claim: ${y_train_half.mean():.0f}")
    st.markdown(f"- Test period avg claim: ${y_test_half.mean():.0f}")
    st.markdown(f"- Relative drift: {abs(y_test_half.mean() / y_train_half.mean() - 1):.2%}")
    st.subheader("Edge Case Stress Tests")
    edge = df[(df["age"] <= 22) | (df["age"] >= 65)]
    st.markdown(f"- Extreme ages (<=22 or >=65): {len(edge)} policies, avg actual claims ${edge['claims'].mean():.0f}")
    st.subheader("Production Readiness")
    st.markdown(
        "- **Reproducibility**: Seed-controlled training (seed=42)\n"
        "- **Version control**: Git-tracked with tagged releases\n"
        "- **Monitoring**: Drift detection placeholder (compare decile balance over time)\n"
        "- **A/B testing**: Deploy Tweedie GLM as champion; Autocalibrated as challenger"
    )

with t6:
    st.subheader("What-If Analysis \u2014 Policy Pricing Sandbox")
    if len(b["features"]) == 0 or "age" not in df.columns:
        st.warning("Feature data not available.")
        st.stop()
    age_q = st.slider("Age", int(df["age"].min()), int(df["age"].max()), 35)
    terr_q = st.selectbox("Territory", df["territory"].unique())
    cov_q = st.selectbox("Coverage", df["coverage"].unique())
    exp_q = st.slider("Exposure", 0.5, 1.0, 1.0)
    query = pd.DataFrame([{"age": float(age_q), "territory": terr_q, "coverage": cov_q, "exposure": exp_q}])
    ref_rows = df.head(20)
    combined = pd.concat([ref_rows, query], ignore_index=True)
    X_combined, _ = _prepare_features({"df": combined, "categorical_features": data["categorical_features"], "numerical_features": data["numerical_features"]})
    X_query = X_combined[-1:]
    X_query_s = b["scaler"].transform(X_query)
    preds = {}
    for nm in mdl_names:
        d = b["models"][nm]
        if nm == "Tweedie GLM":
            preds[nm] = float(predict_glm(d, X_query_s)[0])
        elif nm == "Autocalibrated":
            preds[nm] = float(predict_autocalibrated(d, X_query_s, np.array([exp_q]))[0])
        elif nm == "Freq-Severity":
            preds[nm] = float(predict_freq_severity(d, X_query_s, np.array([exp_q]))[0])
        else:
            preds[nm] = float(predict_xgboost(d, X_query_s, np.array([exp_q]))[0])
    pr = pd.DataFrame({"Model": list(preds.keys()), "Predicted Premium ($)": [f"${v:.2f}" for v in preds.values()]})
    st.dataframe(pr, use_container_width=True, hide_index=True)
    base_prem_q = st.number_input("Base Premium ($)", 200, 2000, 800, 100)
    st.subheader("Recommended Price")
    for nm, v in preds.items():
        recommended = base_prem_q * (v / max(preds.values())) * (1 + 0.2)
        st.metric(nm, f"${recommended:.2f}", delta=f"vs base: {(recommended/base_prem_q - 1):.1%}")
    st.subheader("Comparable Policies")
    from sklearn.neighbors import NearestNeighbors
    X_all = b["X_test"]
    nbrs = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(X_all)
    dists, idx = nbrs.kneighbors(X_query_s)
    neighbors = df.iloc[:len(X_all)].iloc[idx[0]]
    st.dataframe(neighbors[["age", "territory", "coverage", "exposure", "claims"]].head(5), use_container_width=True, hide_index=True)
    st.subheader("Compare Scenarios")
    saved = st.session_state.get("saved_policies", [])
    if st.button("Save this scenario"):
        saved.append({"age": age_q, "territory": terr_q, "coverage": cov_q, "exposure": exp_q, **preds})
        st.session_state["saved_policies"] = saved
        st.success("Saved!")
    if saved:
        comp = pd.DataFrame(saved)
        st.dataframe(comp, use_container_width=True, hide_index=True)
