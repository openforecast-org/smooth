import sympy as sp, json, os, numpy as np
p,x=sp.symbols('p x',positive=True)
def rat(o,isO):
    v=o-x; rO=(1+v)/(1-v); return rO if isO else 1/rO
res=[]
# multiplicative old: E[e|x]=p(r1-1)+(1-p)(r0-1)
for isO,pred in ((False,1-sp.sqrt(1-p)),(True,sp.sqrt(p))):
    h=p*(rat(1,isO)-1)+(1-p)*(rat(0,isO)-1)
    num=sp.numer(sp.together(h)); print("iO" if isO else "iI","numerator of h:",sp.factor(num))
    chk=sp.simplify(h.subs(x,pred)); print(" h at predicted point:",chk)
    res.append(dict(quantity="symbolic residual of zero-drift equation at the claimed point (0 = exact)",value=float(sp.N(chk.subs(p,sp.Rational(3,10)))),conditions={"model":"iETS_O" if isO else "iETS_I","error":"M"}))
# stability: slope of h at the point (negative for the state direction => stable). For O: phat increases with mu, drift of log mu = h; stable if dh/dx<0 ; for I phat decreases with mu: stable if dh/dx>0
for isO,pred in ((False,1-sp.sqrt(1-p)),(True,sp.sqrt(p))):
    h=p*(rat(1,isO)-1)+(1-p)*(rat(0,isO)-1)
    d=sp.simplify(sp.diff(h,x).subs(x,pred)); print(" dh/dx at point, iO" if isO else " dh/dx at point, iI", d, [float(d.subs(p,q)) for q in (0.1,0.5,0.9)])
# additive: h=p log r1+(1-p) log r0; claim: same root for I and O, and equation p log(x/(2-x))+(1-p)log((1+x)/(1-x))=0
hI=p*sp.log(rat(1,False))+(1-p)*sp.log(rat(0,False)); hO=p*sp.log(rat(1,True))+(1-p)*sp.log(rat(0,True))
claim=p*sp.log(x/(2-x))+(1-p)*sp.log((1+x)/(1-x))
print("hI - claim:",sp.simplify(sp.expand_log(hI-claim,force=True)),"; hO + claim:",sp.simplify(sp.expand_log(hO+claim,force=True)))
# monotonic in x? g(x)=claim; derivative
g=sp.diff(claim,x); gs=sp.simplify(g); print("d claim/dx:",gs)
# between p and 1/2: sign of claim at x=p and x=1/2
from scipy.optimize import brentq
f=sp.lambdify((p,x),claim,'numpy')
bad=0;n=0
for pv in np.concatenate([np.linspace(0.001,0.499,200),np.linspace(0.501,0.999,200)]):
    r=brentq(lambda t:f(pv,t),1e-9,1-1e-9); lo,hi=sorted([pv,0.5]); n+=1; bad+= (not (lo<r<hi))
print("root strictly between p and 0.5 for",n-bad,"of",n,"p values; at p=.5 root:",brentq(lambda t:f(0.5,t)+1e-300*t,0.2,0.8) if False else "see below")
res.append(dict(quantity="number of p values (of 400 in (0,1)\\{0.5}) at which the additive root is not strictly between p and 1/2",value=bad,conditions={}))
print("claim at p=.5, x=.5:",f(0.5,0.5))
res.append(dict(quantity="additive zero-drift equation evaluated at p=0.5, x=0.5 (root at 1/2, not strictly between p and 1/2)",value=float(f(0.5,0.5)),conditions={}))
if os.environ.get("CLAIMSET_RESULTS"): json.dump(res,open(os.environ["CLAIMSET_RESULTS"],"w"))
