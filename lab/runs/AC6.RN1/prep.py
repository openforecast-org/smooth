# Builds per-series measures from the recorded forecast files (own code, no evaluate.py).
import os, json, numpy as np, pandas as pd
H=14
OUT=os.environ.get('CLAIMSET_RUN_DIR','.')
sub=pd.read_csv('/home/claude/smooth/claimset/code/iets-occurrence-bias/m5/subset.csv')
rows=sub['row'].tolist()
tr=pd.read_csv('/home/claude/data/m5/sales_train_evaluation.csv'); te=pd.read_csv('/home/claude/data/m5/sales_test_evaluation.csv')
assert (tr.item_id==te.item_id).all() and (tr.store_id==te.store_id).all()
dc=[c for c in tr.columns if c.startswith('d_')]
print(len(dc), te.shape, 'subset', len(rows))
Y={}; info={}
for r,(it,st) in zip(rows, zip(sub.item_id, sub.store_id)):
    assert tr.item_id[r]==it and tr.store_id[r]==st
    y=np.concatenate([tr.loc[r,dc].values.astype(float), te.loc[r,[c for c in te.columns if c.startswith('d_')]].values.astype(float)])
    nz=np.flatnonzero(y!=0); y=y[nz[0]:] if len(nz) else y[:0]
    Y[r]=y
ins={r:Y[r][:-H] for r in rows}; ho={r:Y[r][-H:] for r in rows}
excl={'few':0,'zero_scale':0}
a=pd.read_csv('/home/claude/smooth/claimset/iets-occurrence-bias/lab/runs/RN32/iets_forecasts.csv')
b=pd.read_csv('/home/claude/smooth/claimset/iets-occurrence-bias/lab/runs/RN33/benchmark_forecasts.csv')
f=pd.concat([a,b],ignore_index=True)
assert set(f.row)==set(rows)
keep=[]
for r in rows:
    x=ins[r]
    if (x!=0).sum()<5: excl['few']+=1; continue
    d=np.diff(x)
    if np.mean(d**2)==0: excl['zero_scale']+=1; continue
    keep.append(r)
print('excluded',excl,'kept',len(keep))
rec=[]
taus=[0.8,0.9,0.95,0.99]
f=f.sort_values(['method','row','horizon'])
for m,g in f.groupby('method'):
    for r,gg in g.groupby('row'):
        if r not in keep: continue
        assert len(gg)==H and (gg.horizon.values==np.arange(1,H+1)).all()
        y=ho[r]; x=ins[r]; fm=gg['mean'].values
        s2=np.mean(np.diff(x)**2); sa=np.mean(np.abs(np.diff(x)))
        d={'method':m,'row':r,'sumf':fm.sum(),'rmsse':np.sqrt(np.mean((y-fm)**2)/s2)}
        for t in taus:
            q=gg[f'q{t}'].values
            if np.isnan(q).any(): continue
            def pb(qq): return np.mean(np.where(y>=qq, t*(y-qq), (1-t)*(qq-y)))/sa
            qr=np.ceil(q)
            d[f'pin{t}']=pb(q); d[f'pinr{t}']=pb(qr)
            d[f'cov{t}']=np.mean(y<=q); d[f'covr{t}']=np.mean(y<=qr)
        rec.append(d)
m=pd.DataFrame(rec); m.to_csv(f'{OUT}/my_series_measures.csv',index=False)
print(m.groupby('method').size().to_dict())
print(m.isna().sum().to_dict())
res=[{'quantity':'series kept','value':len(keep)},{'quantity':'methods','value':m.method.nunique()},
 {'quantity':'iETS rows with missing or non-ok forecasts','value':int((a.status!='ok').sum()+a.isna().sum().sum())},
 {'quantity':'benchmark rows with non-ok status or missing mean','value':int((b.status!='ok').sum()+b['mean'].isna().sum())},
 {'quantity':'benchmark missing quantile rows for HSP and NegBin','value':int(b[b.method.isin(['HSP','NegBin'])][['q0.8','q0.9','q0.95','q0.99']].isna().sum().sum())}]
json.dump(res,open(os.environ['CLAIMSET_RESULTS'],'w'))
