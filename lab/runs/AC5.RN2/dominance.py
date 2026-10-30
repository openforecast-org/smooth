import sys, random, math, json, csv, os
sys.path.insert(0, "/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC5")
from lib import *
from collections import defaultdict
T=int(sys.argv[1]) if len(sys.argv)>1 else 10_000_000
burn=T//10
rng=random.Random(5002)
# name, n, p range, alpha lo, hi, log?, errors
designs=[
 ("logA_all",  200, (0.02,0.98), 0.001,1.0, True, ["M","A"]),
 ("high_M",    150, (0.02,0.98), 0.5,  1.0, False,["M"]),
 ("unif_A",    200, (0.02,0.98), 0.001,1.0, False,["A"]),
 ("low_M",     200, (0.02,0.98), 0.001,0.5, False,["M"]),
 ("tail_A",    100, None,        0.001,1.0, False,["A"]),   # p in [0.002,0.02] u [0.98,0.998]
]
cells=[];meta=[];sd=0
for name,n,pr,lo,hi,lg,errs in designs:
    for _ in range(n):
        if pr is None:
            p=rng.uniform(0.002,0.02); p=p if rng.random()<.5 else 1-p
        else: p=rng.uniform(*pr)
        a=math.exp(rng.uniform(math.log(lo),math.log(hi))) if lg else rng.uniform(lo,hi)
        for m in "io":
            for e in errs:
                for proxy in ("old","score"):
                    sd+=1; cells.append((m,e,proxy,p,a,7_000_000+sd)); meta.append((name,m,e,proxy,p,a))
res=run_cells(cells,T,burn)
rd=os.environ.get("CLAIMSET_RUN_DIR",".")
w=csv.writer(open(rd+"/dominance_cells.csv","w")); w.writerow("design model err proxy p alpha mean se bad".split())
for mt,r in zip(meta,res): w.writerow(list(mt)+list(r))
worse=defaultdict(int); tot=defaultdict(int); wl=[]; nonfin=defaultdict(int)
for i in range(0,len(meta),2):
    (name,m,e,_,p,a),(mo,so,bo)=meta[i],res[i]; (_,_,_,_,_,_),(ms,ss,bs)=meta[i+1],res[i+1]
    k=(name,m,e); tot[k]+=1
    if bs: worse[k]+=1; nonfin[k]+=1; wl.append((name,m,e,round(p,4),round(a,4),"nonfinite")); continue
    d=abs(ms-p)-abs(mo-p); se=math.hypot(so,ss)
    if d>4*se: worse[k]+=1; wl.append((name,m,e,round(p,4),round(a,4),round(d/se,1)))
out=[{"quantity":"cells where the score proxy is more biased than the old proxy (|bias| larger by more than 4 combined SE)","value":worse[k],
      "conditions":{"design":k[0],"model":"iETS_"+k[1].upper(),"error":k[2],"comparisons":tot[k],"T":T}} for k in sorted(tot)]
emit(out)
print(dict(worse)); print(len(wl)); 
for x in wl: print(x)
