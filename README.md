# PremiumModel

> Insurance premium pricing with GLM relativities and portfolio simulation.

Trains four classifiers on synthetic policyholder data to predict claim probability. Dashboard provides data exploration, multi-model comparison, log-link Gamma GLM (Tweedie) price relativity factors per CAS Monograph No. 5 standard, and portfolio-level premium simulation with XGBoost-based risk loading.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.906 |
| Gini | 0.813 |
| KS Statistic | 0.692 |
| F1 Score | 0.655 |
| Accuracy | 0.811 |

5-fold CV AUC: 0.900 ± 0.025. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Policyholder dataset overview, claim rate distribution |
| **Model Lab** | Multi-model comparison, ROC/calibration curves, CV results |
| **GLM Pricing** | Log-link Poisson GLM coefficients, price relativities (exp(β)), factor importance |
| **Portfolio** | Base premium configuration, risk-loaded pricing, portfolio-level metrics |

## Repo Structure

```
PremiumModel/
  src/         data, model, core, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic policyholder dataset: age, vehicle age, credit score, driving experience, annual mileage, prior claims, coverage type, region, and claim status.

## License

MIT
