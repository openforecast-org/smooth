import os, json, numpy as np, pandas as pd
m=pd.read_csv('/home/claude/smooth/claimset/iets-occurrence-bias/lab/runs/AC6.RN1/my_series_measures.csv')
meths=sorted(c for c in m.method.unique() if c.startswith('iETS'))
taus=[0.8,0.9,0.95,0.99]
W={}
for t in taus:
    W[('raw',t)]=m.pivot(index='row',columns='method',values=f'cov{t}')
    W[('rnd',t)]=m.pivot(index='row',columns='method',values=f'covr{t}')
n=len(W[('raw',0.8)]); B=10000; lvl=1-0.01/72
lo_q=(1-lvl)/2*100; hi_q=100-lo_q
pairs=[(me,t) for me in meths for t in taus]
pt={}
for me,t in pairs:
    a=W[('rnd',t)][me].values; r=W[('raw',t)][me].values
    pt[(me,t)]=abs(a.mean()-t)-abs(r.mean()-t)
def run(seed,shared=True):
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,n,size=(B,n)) if shared else None
    out={}
    for me,t in pairs:
        if not shared: idx2=rng.integers(0,n,size=(B,n))
        ii=idx if shared else idx2
        a=W[('rnd',t)][me].values; r=W[('raw',t)][me].values
        D=np.abs(a[ii].mean(1)-t)-np.abs(r[ii].mean(1)-t)
        lo,hi=np.percentile(D,[lo_q,hi_q]); out[(me,t)]=(lo,hi)
    return out
def count(out):
    up=sum(1 for v in out.values() if v[0]>0); dn=sum(1 for v in out.values() if v[1]<0)
    return up,dn
res=[]
allout={}
for seed in [20261030,1,2,3,4,5]:
    o=run(seed,True); allout[seed]=o; u,d=count(o)
    print('seed',seed,'shared draws: grows',u,'shrinks',d,flush=True)
    res.append({'quantity':'pairs where coverage deviation grows (interval above 0), shared bootstrap draws','value':u,'conditions':{'seed':seed}})
    res.append({'quantity':'pairs where coverage deviation shrinks (interval below 0), shared bootstrap draws','value':d,'conditions':{'seed':seed}})
for seed in [20261030,11]:
    o=run(seed,False); u,d=count(o); print('seed',seed,'independent draws per pair: grows',u,'shrinks',d,flush=True)
    res.append({'quantity':'pairs where coverage deviation grows (interval above 0), independent draws per pair','value':u,'conditions':{'seed':seed}})
    res.append({'quantity':'pairs where coverage deviation shrinks (interval below 0), independent draws per pair','value':d,'conditions':{'seed':seed}})
o=allout[20261030]
unst=[]
for k in pairs:
    cls=set(('up' if allout[s][k][0]>0 else 'down' if allout[s][k][1]<0 else 'ns') for s in allout)
    if len(cls)>1: unst.append((k,sorted(cls)))
print('pairs whose class varies across the 6 seeds:',unst)
res.append({'quantity':'pairs whose classification varies across 6 seeds','value':len(unst)})
for k in pairs:
    lo,hi=o[k]; print(k,'D=%.5f'%pt[k],'CI=(%.5f,%.5f)'%(lo,hi),'up' if lo>0 else 'down' if hi<0 else 'ns')
json.dump(res,open(os.environ['CLAIMSET_RESULTS'],'w'))
