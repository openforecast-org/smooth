import sys, random, math, json
sys.path.insert(0, "/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC5")
from lib import *
T=int(sys.argv[1]) if len(sys.argv)>1 else 10_000_000
burn=T//10
rng=random.Random(5001)
designs=[ # name, n, alpha lo, hi, log?, errors
 ("logA_all",   300, 0.001, 1.0, True,  ["M","A"]),
 ("high_M",     400, 0.80,  1.0, False, ["M"]),
 ("mid_M",      300, 0.001, 0.8, False, ["M"]),
 ("unif_A",     300, 0.001, 1.0, False, ["A"]),
]
cells=[];meta=[]
sd=0
for name,n,lo,hi,lg,errs in designs:
    for _ in range(n):
        p=rng.uniform(0.02,0.98); a=math.exp(rng.uniform(math.log(lo),math.log(hi))) if lg else rng.uniform(lo,hi)
        for m in "io":
            for e in errs:
                sd+=1; cells.append((m,e,"old",p,a,5_000_000+sd)); meta.append((name,m,e,p,a))
res=run_cells(cells,T,burn)
import csv, os
rd=os.environ.get("CLAIMSET_RUN_DIR",".")
w=csv.writer(open(rd+"/direction_cells.csv","w")); w.writerow("design model err p alpha mean se bad bias wrong".split())
from collections import defaultdict
cnt=defaultdict(int); tot=defaultdict(int); wl=[]
for (name,m,e,p,a),(mm,se,bad) in zip(meta,res):
    wr=wrong_dir(m,e,p,mm,se,bad); w.writerow([name,m,e,p,a,mm,se,bad,mm-p,int(wr)])
    cnt[(name,m,e)]+=wr; tot[(name,m,e)]+=1
    if wr: wl.append((name,m,e,round(p,4),round(a,4),round((mm-p)/se,1) if se==se else None))
out=[]
for k in sorted(cnt):
    out.append({"quantity":"cells where the old-proxy bias has the wrong direction (more than 4 SE)","value":cnt[k],
      "conditions":{"design":k[0],"model":"iETS_"+k[1].upper(),"error":k[2],"cells":tot[k],"T":T}})
emit(out)
print(dict(cnt)); print(wl)
