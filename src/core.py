from __future__ import annotations
import numpy as np
def tweedie_deviance(y,yhat,p=1.5):
    y=np.asarray(y,float);yhat=np.clip(np.asarray(yhat,float),1e-10,None)
    t1=np.power(np.clip(y,0,None),2-p)/((1-p)*(2-p));t2=y*np.power(yhat,1-p)/(1-p);t3=np.power(yhat,2-p)/(2-p)
    return float(2*np.sum(t1-t2+t3))
def premium_adequacy(predicted,actual):
    return float(np.sum(np.asarray(actual))/np.sum(np.asarray(predicted)))
def loss_ratio(claims,premiums):return float(np.sum(claims)/np.sum(premiums)) if np.sum(premiums)>0 else 0.0
