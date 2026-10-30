import subprocess, json, os, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import brentq
D=os.path.dirname(os.path.abspath(__file__))
P=[0.02,0.07,0.17,0.31,0.5,0.66,0.83,0.93,0.98]; AL=[0.008,0.004,0.002,0.001]; N=40_000_000
names={0:("iETS_I","M","old"),1:("iETS_O","M","old"),2:("iETS_I","A","old"),3:("iETS_O","A","old"),
 4:("iETS_I","M","score"),5:("iETS_O","M","score"),6:("iETS_I","A","score"),7:("iETS_O","A","score"),
 8:("iETS_I","ADAM","expscore"),9:("iETS_O","ADAM","expscore"),10:("iETS_I","ADAM","multscore"),11:("iETS_O","ADAM","multscore"),12:("Croston","","")}
# zero-drift predictions, computed from the definitions (expected update at phat=x), not from the claims' formulas
def drift(type,p,x):
    isO = type in (1,3,5,7,9,11)
    out=[]
    for o,pr in ((1,p),(0,1-p)):
        v=o-x; rO=(1+v)/(1-v); rI=1/rO; s=(o-x) if isO else (x-o)
        if type in (0,1): e=(rO if isO else rI)-1
        elif type in (2,3): e=np.log(rO if isO else rI)
        elif type in (4,5,6,7,8,9): e=s
        else: e=np.log(1+s)
        out.append(pr*e)
    return sum(out)
def pred(type,p):
    if type==12: return p
    xs=np.linspace(1e-6,1-1e-6,20000); h=np.array([drift(type,p,x) for x in xs])
    sg=np.where(np.sign(h[:-1])!=np.sign(h[1:]))[0]
    assert len(sg)==1,(type,p,len(sg))   # unique zero
    i=sg[0]; return brentq(lambda x:drift(type,p,x),xs[i],xs[i+1],xtol=1e-14)
cells=[(t,p,a) for t in range(13) for p in P for a in AL]
def run(c):
    t,p,a=c; seed=1000+hash((t,p,a))%100000
    r=subprocess.run([os.path.join(D,'filt'),str(t),str(p),str(a),str(N),str(abs(hash((t,p,a)))%10**6+7)],capture_output=True,text=True).stdout.split()
    return (t,p,a,float(r[3]),float(r[4]))
with ThreadPoolExecutor(2) as ex: out=list(ex.map(run,cells))
json.dump(out,open(os.path.join(os.environ.get('CLAIMSET_RUN_DIR',D),'raw_cells.json'),'w'))
res=[];rows=[]
for t in range(13):
    nm,er,px=names[t]; devs=[];devs_min=[]
    for p in P:
        pts=sorted([(o[2],o[3],o[4]) for o in out if o[0]==t and o[1]==p])
        a=np.array([q[0] for q in pts]); m=np.array([q[1] for q in pts]); se=np.array([q[2] for q in pts])
        # least squares line m = b0 + b1*alpha; intercept = limit; SE by propagation
        X=np.c_[np.ones_like(a),a]; W=1/se**2; cov=np.linalg.inv(X.T@(X*W[:,None])); b=cov@(X.T@(W*m))
        pr=pred(t,p); dev=abs(b[0]-pr)/pr; devs.append(dev); dmin=abs(m[0]-pr)/pr; devs_min.append(dmin)
        rows.append(dict(type=t,model=nm,error=er,proxy=px,p=p,pred=pr,limit=b[0],limit_se=float(np.sqrt(cov[0,0])),slope=b[1],mean_alpha001=m[0],se001=se[0],reldev_limit=dev,reldev_alpha001=dmin,
                         means=m.tolist(),ses=se.tolist(),vs_p=(b[0]-p)/p, pred_vs_p=(pr-p)/p))
    key=dict(model=nm,error=er,proxy=px)
    res.append(dict(quantity="max relative deviation of the intercept-at-alpha=0 limit from the zero-drift prediction over p grid",value=max(devs),conditions=key))
    res.append(dict(quantity="max relative deviation of the smallest-alpha (0.001) mean from the prediction over p grid",value=max(devs_min),conditions=key))
    res.append(dict(quantity="max relative deviation of the predicted limit from p over p grid",value=max(abs(r['pred_vs_p']) for r in rows if r['type']==t),conditions=key))
    res.append(dict(quantity="max relative deviation of the simulated limit from p over p grid",value=max(abs(r['vs_p']) for r in rows if r['type']==t),conditions=key))
rd=os.environ.get("CLAIMSET_RUN_DIR",D)
json.dump(rows,open(os.path.join(rd,"limit_rows.json"),"w"),indent=1)
if os.environ.get("CLAIMSET_RESULTS"): json.dump(res,open(os.environ["CLAIMSET_RESULTS"],"w"))
for r in rows: print(r['model'],r['error'],r['proxy'],r['p'],"pred=%.5f limit=%.5f(se %.5f) a001=%.5f reldev=%.4f reldev001=%.4f"%(r['pred'],r['limit'],r['limit_se'],r['mean_alpha001'],r['reldev_limit'],r['reldev_alpha001']))
for q in res: print(q)
