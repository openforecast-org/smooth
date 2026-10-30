import os, json, numpy as np, pandas as pd
from greybox import rmcb
m=pd.read_csv('/home/claude/smooth/claimset/iets-occurrence-bias/lab/runs/AC6.RN1/my_series_measures.csv')
res=[]
def one(col,benchs,label):
    w=m.pivot(index='row',columns='method',values=col).dropna(axis=1)
    r=rmcb(w,level=0.99,outplot='none')
    mean=r.mean; iv=r.interval
    print(label,'n methods',w.shape[1],'series',w.shape[0])
    out={}
    for pool,name in [([c for c in w.columns if 'score' in c],'score-proxy 8'),([c for c in w.columns if c.startswith('iETS') and ('score' in c or c.endswith(('F_conv','D_conv')))],'score+F+D')]:
        best=min(pool,key=lambda c:mean[c]); out[name]=best
        for b in benchs:
            lo_b,hi_b=iv.loc[b].values; lo,hi=iv.loc[best].values
            st='better' if hi<lo_b else ('worse' if lo>hi_b else 'n.s.')
            print(f'  [{name}] best={best} rank={mean[best]:.3f} vs {b} rank={mean[b]:.3f}: {st}  (best CI {lo:.3f},{hi:.3f}; bench CI {lo_b:.3f},{hi_b:.3f})')
            if name=='score-proxy 8':
                res.append({'quantity':f'MCB result {label} best score-proxy iETS vs {b}: 1 better, -1 worse, 0 n.s.','value':{'better':1,'worse':-1,'n.s.':0}[st]})
        res.append({'quantity':f'MCB {label} mean rank of best ({name}) {best}','value':float(mean[best])})
    return out
one('rmsse',['Croston','SBA','TSB','NegBin','HSP'],'RMSSE')
for t in [0.8,0.9,0.95,0.99]:
    one(f'pin{t}',['NegBin','HSP'],f'pinball{t}')
json.dump(res,open(os.environ['CLAIMSET_RESULTS'],'w'))
