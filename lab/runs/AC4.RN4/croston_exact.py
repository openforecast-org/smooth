# Exact stationary E[1/qhat] for Croston's SES of Geometric(p) intervals, by the Laplace-transform identity
# 1/q = int_0^inf exp(-s q) ds,  q_hat = sum_k alpha(1-alpha)^k q_{t-k},  q_i iid Geometric(p) on {1,2,...}
import numpy as np, json, os
from numpy.polynomial.legendre import leggauss
def Einv(p,a,npan=60,ng=24):
    K=int(min(max(1,np.ceil(32/a)),4_000_000)) if a<1 else 1
    w=a*(1-a)**np.arange(K); w=w/w.sum()   # renormalise the truncated weights (weights beyond K are < 1e-13)
    x,wg=leggauss(ng)
    edges=np.concatenate([[0],np.geomspace(1e-3,80,npan)]); tot=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=(hi-lo)/2*x+(hi+lo)/2; ws=(hi-lo)/2*wg
        lam=np.outer(s,w)                              # (ng,K)
        lp=np.log(p)-lam-np.log1p(-(1-p)*np.exp(-lam))
        tot+=np.sum(ws*np.exp(lp.sum(axis=1)))
    return tot
rows=[]; res=[]
# check at alpha=1: closed form -p ln p/(1-p)
for p in [0.02,0.3,0.7,0.98]:
    e=Einv(p,1.0); c=-p*np.log(p)/(1-p); print("alpha=1 p=%g numeric %.10f closed %.10f"%(p,e,c)); rows.append(("check",p,1.0,e,c))
mind=1e9; maxrel_small=0
for p in [0.001,0.01,0.02,0.1,0.3,0.5,0.7,0.9,0.98,0.999]:
    for a in [1.0,0.9,0.7,0.5,0.3,0.1,0.03,0.01,0.004,0.002,0.001]:
        e=Einv(p,a); rb=(e-p)/p; first=(1-p)*a/(2-a)   # first-order (delta-method) relative bias
        rows.append(("bias",p,a,e,rb,first)); print("p=%g alpha=%g E[1/qhat]=%.8f relbias=%.3e  first-order %.3e ratio %.3f"%(p,a,e,rb,first,rb/first if first>0 else float('nan')))
        mind=min(mind,rb)
        if a<=0.002: maxrel_small=max(maxrel_small,rb)
res=[dict(quantity="minimum over the (p,alpha) grid of the exact relative bias (E[1/q_hat]-p)/p of Croston's method",value=mind,conditions={"p":"0.001..0.999 (10 values)","alpha":"0.001..1 (11 values)"}),
     dict(quantity="maximum exact relative bias of Croston's method over p grid at alpha<=0.002",value=maxrel_small,conditions={"alpha":"0.001,0.002"}),
     dict(quantity="exact relative bias of Croston's method at alpha=0.002, p=0.5",value=(Einv(0.5,0.002)-0.5)/0.5,conditions={"alpha":0.002,"p":0.5})]
json.dump(rows,open(os.path.join(os.environ.get("CLAIMSET_RUN_DIR","."),"croston_exact.json"),"w"))
if os.environ.get("CLAIMSET_RESULTS"): json.dump(res,open(os.environ["CLAIMSET_RESULTS"],"w"))
print(res)
