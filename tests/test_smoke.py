import sys;from pathlib import Path;sys.path.insert(0,str(Path(__file__).parent.parent))
from src.data import make_synthetic;from src.model import fit_and_evaluate;from src.core import tweedie_deviance,premium_adequacy
def test_data():d=make_synthetic(300);assert d["n_samples"]==300
def test_deviance():assert tweedie_deviance([100,200],[90,210])>0
def test_fit():d=make_synthetic(300);m,met=fit_and_evaluate(d);assert"adequacy"in met
if __name__=="__main__":test_data();test_deviance();test_fit();print("OK")
