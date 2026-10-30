import sys, json, math, os
sys.path.insert(0, "/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC5")
from lib import *
d=json.load(open("/home/claude/smooth/claimset/iets-occurrence-bias/iets-occurrence-bias.claimset.json"))
ev={e["id"]:e for e in d["evidence"]}
M={"iETS_I":"i","iETS_O":"o"}
# study's flagged cells (computed by me from the recorded means/intervals, not from its notes)
def rows(eid): return ev[eid]["results"]
def se(r): return (r["interval"]["upper"]-r["interval"]["lower"])/2/1.96
# E9: wrong-direction cells, multiplicative old
cand9=[]
for r in rows("E9"):
    c=r["conditions"]
    if c["error"]=="M" and c["proxy"]=="old":
        if wrong_dir(M[c["model"]],"M",c["p"],r["value"],se(r),0): cand9.append((c,r["value"],se(r)))
# E17: score worse than old
byk={}
for r in rows("E17"):
    c=r["conditions"]; byk[(c["model"],c["error"],c["proxy"],c["p"],c["alpha"])]=r
cand17=[]
for k,r in byk.items():
    if k[2]!="old": continue
    s=[v for kk,v in byk.items() if kk[0]==k[0] and kk[3]==k[3] and kk[4]==k[4] and kk[2]!="old"]
    if not s: continue
    s=s[0]; p=k[3]
    dd=abs(s["value"]-p)-abs(r["value"]-p)
    if dd>4*math.hypot(se(r),se(s)): cand17.append((k,r,s))
print(len(cand9),len(cand17))
cells=[];info=[]
for c,v,s in cand9:
    cells.append((M[c["model"]],"M","old",c["p"],c["alpha"],991)); info.append(("E9",c["model"],c["p"],c["alpha"],v,s))
for k,r,s in cand17:
    for proxy in ("old","score"):
        cells.append((M[k[0]],"M",proxy,k[3],k[4],992)); info.append(("E17",k[0],k[3],k[4],proxy))
res=run_cells(cells,20_000_000,2_000_000)
out=[];
for inf,(mm,sd,bad) in zip(info,res):
    print(inf,mm,sd,bad)
    if inf[0]=="E9":
        p=inf[2]; out.append({"quantity":"my re-run bias (mean-p) in a cell the study flags as wrong-direction","value":mm-p,"conditions":{"model":inf[1],"p":p,"alpha":inf[3],"study_bias":inf[4]-p,"my_se":sd,"study_se":inf[5],"wrong_by_my_run":bool(wrong_dir(M[inf[1]],"M",p,mm,sd,bad))}})
    else:
        p=inf[2]; out.append({"quantity":"my re-run bias (mean-p), study cell where score proxy flagged worse","value":mm-p,"conditions":{"model":inf[1],"p":p,"alpha":inf[3],"proxy":inf[4],"my_se":sd}})
emit(out)
