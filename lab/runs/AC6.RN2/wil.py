import os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
M='/home/claude/smooth/claimset/iets-occurrence-bias/lab/runs/AC6.RN1/my_series_measures.csv'
m=pd.read_csv(M)
def wide(col): return m.pivot(index='row',columns='method',values=col)
def tests(pairs, alpha=0.01):
    out=[]
    for name,x,y,pred in pairs:
        d=(x-y).values; nzd=d[d!=0]
        p=wilcoxon(d,zero_method='wilcox',alternative='two-sided').pvalue
        out.append(dict(name=name,p=p,med=np.median(nzd),zero=np.mean(d==0),pred=pred,mean=d.mean()))
    k=len(out); order=np.argsort([o['p'] for o in out]); run=0
    for rank,i in enumerate(order):
        run=max(run,min(1,(k-rank)*out[i]['p'])); out[i]['padj']=run
    for o in out:
        sig=o['padj']<alpha; o['sig']=sig
        o['dir']= 0 if not sig else (1 if o['med']>0 else -1)
        o['good']=bool(sig and o['dir']==o['pred']); o['bad']=bool(sig and o['dir']==-o['pred'])
    return out
res=[]
def rep(tag,out):
    for o in out: print(tag,o['name'],'padj=%.3g med=%.4g meandiff=%.4g zero=%.3f sig=%s dir=%s'%(o['padj'],o['med'],o['mean'],o['zero'],o['sig'],o['dir']))
    g=sum(o['good'] for o in out); b=sum(o['bad'] for o in out)
    print(tag,'good',g,'bad',b,'nsig',sum(o['sig'] for o in out),'of',len(out))
    res.append({'quantity':f'{tag}: comparisons significant in the predicted direction','value':g})
    res.append({'quantity':f'{tag}: comparisons significant against the predicted direction','value':b})
    res.append({'quantity':f'{tag}: comparisons significant (any direction)','value':sum(o['sig'] for o in out)})
    for o in out:
        res.append({'quantity':f'{tag}: Holm-adjusted p, {o["name"]}','value':o['padj']})
        res.append({'quantity':f'{tag}: median non-zero difference, {o["name"]}','value':float(o['med'])})
# C20
sf=wide('sumf')
p=[]
for X,pred in [('I',1),('O',-1)]:
    for e in ['conv','adam']:
        if X=='I' and e=='adam': pred=-1
        p.append((f'sumf {X} {e}',sf[f'iETS_{X}_score_{e}'],sf[f'iETS_{X}_ratio_{e}'],pred))
rep('C20',tests(p))
# C21
r=wide('rmsse'); p=[]
for X in 'OIGA':
    for e in ['conv','adam']: p.append((f'rmsse {X} {e}',r[f'iETS_{X}_score_{e}'],r[f'iETS_{X}_ratio_{e}'],-1))
rep('C21',tests(p))
# C22
p=[]
for X in 'OIGA':
    for e in ['conv','adam']:
        for t in [0.8,0.9,0.95,0.99]:
            w=wide(f'pin{t}'); p.append((f'pin{t} {X} {e}',w[f'iETS_{X}_score_{e}'],w[f'iETS_{X}_ratio_{e}'],-1))
o=tests(p); rep('C22',o)
print([x['name'] for x in o if x['good']]); print([x['name'] for x in o if x['bad']])
# C25
meths=[c for c in m.method.unique() if c.startswith('iETS')]; p=[]
for me in meths:
    for t in [0.8,0.9,0.95,0.99]:
        p.append((f'{me} {t}',wide(f'pinr{t}')[me],wide(f'pin{t}')[me],-1))
o=tests(p); rep('C25',o)
print('n sig up',sum(x['bad'] for x in o),'down',sum(x['good'] for x in o), 'ns', [x['name'] for x in o if not x['sig']])
json.dump(res,open(os.environ['CLAIMSET_RESULTS'],'w'))
