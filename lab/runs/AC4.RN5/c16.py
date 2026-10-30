import numpy as np, json, os
from scipy.optimize import brentq
# ADAM multiplicative score proxy zero-drift: iETS_I: p log x + (1-p) log(1+x)=0 ; iETS_O: p log(2-x) + (1-p) log(1-x)=0 (from log(1+s), s=x-o for I, o-x for O)
fI=lambda p,x:p*np.log(x)+(1-p)*np.log(1+x)
fO=lambda p,x:p*np.log(2-x)+(1-p)*np.log(1-x)
def dev(p):
    xi=brentq(lambda x:fI(p,x),1e-300 if False else 1e-12,1-1e-12); xo=brentq(lambda x:fO(p,x),1e-12,1-1e-12)
    return xi/p-1, xo/p-1
ps=np.linspace(0.02,0.98,9601); d=np.array([dev(p) for p in ps])
res=[dict(quantity="max over p in [0.02,0.98] (grid step 1e-4) of relative deviation of the multiplicative-score ADAM zero-drift point from p",value=float(np.max(np.abs(d[:,0]))),conditions={"model":"iETS_I"}),
     dict(quantity="max over p in [0.02,0.98] (grid step 1e-4) of relative deviation of the multiplicative-score ADAM zero-drift point from p",value=float(np.max(np.abs(d[:,1]))),conditions={"model":"iETS_O"}),
     dict(quantity="argmax p of the iETS_I deviation",value=float(ps[np.argmax(np.abs(d[:,0]))]),conditions={}),
     dict(quantity="argmax p of the iETS_O deviation",value=float(ps[np.argmax(np.abs(d[:,1]))]),conditions={})]
for p in (0.001,0.005,0.01,0.02,0.03,0.05):
    a,b=dev(p); res.append(dict(quantity="relative deviation of the zero-drift point from p",value=float(a),conditions={"model":"iETS_I","p":p})); print(p,a,b)
# p at which iETS_I deviation equals 1.69
pp=brentq(lambda p:dev(p)[0]-1.69,0.001,0.5); res.append(dict(quantity="p at which iETS_I deviation equals 1.69",value=pp,conditions={}))
print(res)
if os.environ.get("CLAIMSET_RESULTS"): json.dump(res,open(os.environ["CLAIMSET_RESULTS"],"w"))
