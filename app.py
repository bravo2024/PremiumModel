from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.core import compute_metrics
from src.visualizations import *
st.set_page_config(page_title="PremiumModel | WTW Insurance Pricing", layout="wide", page_icon="\U0001f3e6")
with st.sidebar:
    st.header("\u2699 Config"); n = st.slider("Samples",2000,20000,10000,1000)
    tau = st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("WTW | NAIC AI Bulletin | State Rate Filing")
data = make_synthetic(n=n); b = train_all_models(data)
y_test = b["y_test"]; y_probas = {n: b["results"][n]["y_proba"] for n in b["results"]}
best = max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4 = st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Claim Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4 = st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f3af GLM Pricing","\U0001f4b0 Portfolio"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax = plt.subplots(figsize=(5,3)); _style()
    ax.bar(["No Claim","Claim"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Claim Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows = [{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b = st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test, y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test, y_probas))
    cv = cross_validate(data)
    cvr = [{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("GLM Price Relativity")
    st.latex(r"\log(\mu_i) = \beta_0 + \sum \beta_j x_{ij}")
    st.markdown("Using log-link Gamma GLM (Tweedie) for pure premium modeling — the CAS Monograph No. 5 standard.")
    from sklearn.linear_model import PoissonRegressor as GLM
    X = b["X_train"]
    num_cols = data["numerical_features"]
    glm = GLM(alpha=0.1, max_iter=1000).fit(X[num_cols], b["y_train"] + 0.01)
    coefs = pd.DataFrame({"Feature": num_cols, "Coefficient": glm.coef_.round(4), "Relativity": np.exp(glm.coef_).round(4)})
    st.dataframe(coefs, use_container_width=True, hide_index=True)
    st.markdown("Relativity > 1.0 means higher risk (higher premium factor). Relativity < 1.0 means lower risk.")
    fig,ax = plt.subplots(figsize=(7,4)); _style()
    idx = np.argsort(np.abs(glm.coef_))
    ax.barh(range(len(num_cols)), glm.coef_[idx], color="#22d3ee", alpha=0.7)
    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels([num_cols[i] for i in idx], fontsize=8)
    ax.set_xlabel("GLM Coefficient"); ax.set_title("GLM Coefficient Relativities",color="white")
    ax.grid(True,alpha=.2,axis="x")
    st.pyplot(fig)
with t4:
    st.subheader("Portfolio Pricing Simulation")
    base_prem = st.number_input("Base Premium ($)",200,2000,800,100)
    xgb_y = b["results"]["XGBoost"]["y_proba"]
    loaded_prem = base_prem * (1 + 2.0 * xgb_y)
    df_p = data["df"].copy()
    df_p["PD"] = xgb_y; df_p["Premium"] = loaded_prem
    st.dataframe(df_p[["age","smoker","bmi","health_score","PD","Premium"]].head(10),use_container_width=True,hide_index=True)
    fig,ax = plt.subplots(figsize=(8,4)); _style()
    for label,mask,c in [("Non-Smoker",df_p["smoker"]=="no","#22c55e"),("Smoker",df_p["smoker"]=="yes","#f43f5e")]:
        ax.hist(df_p.loc[mask,"Premium"],bins=30,alpha=0.5,color=c,label=f"{label} (n={mask.sum()})",density=True)
    ax.set_xlabel("Annual Premium ($)"); ax.set_ylabel("Density")
    ax.set_title("Premium Distribution: Smoker vs Non-Smoker",color="white")
    ax.legend(fontsize=8); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    total_prem = loaded_prem.sum()
    total_claims = (y_test * loaded_prem * 0.6).sum()
    st.metric("Total Written Premium",f"${total_prem:,.0f}")
    st.metric("Expected Claim Cost (60% LR)",f"${total_claims:,.0f}")
    st.metric("Loss Ratio",f"{total_claims/total_prem:.1%}")
