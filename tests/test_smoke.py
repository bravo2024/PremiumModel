import sys;from pathlib import Path;sys.path.insert(0,str(Path(__file__).parent.parent))
from src.data import make_synthetic;from src.model import train_all_models, cross_validate;from src.core import tweedie_deviance, premium_adequacy, compute_regression_metrics
def test_data():d=make_synthetic(300);assert d["n_samples"]==300;assert 0<d["positive_rate"]<=1
def test_deviance():assert tweedie_deviance([100,200],[90,210])>0
def test_regression_metrics():m=compute_regression_metrics([100,200],[110,190]);assert all(k in m for k in("mae","rmse","r2","tweedie_deviance"))
def test_fit():d=make_synthetic(300);b=train_all_models(d);assert"Tweedie GLM"in b["results"];assert"XGBoost"in b["results"];assert len(b["y_test"])>0
def test_cv():d=make_synthetic(300);c=cross_validate(d);assert"Tweedie GLM"in c;assert"XGBoost"in c
if __name__=="__main__":test_data();test_deviance();test_regression_metrics();test_fit();test_cv();print("OK")
