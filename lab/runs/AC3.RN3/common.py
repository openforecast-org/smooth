import json, os
import sympy as sp, mpmath as mp
mp.mp.dps = 50
RES = []
def add(q, v, **cond):
    RES.append({"quantity": q, "value": v, "conditions": cond})
def save():
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        json.dump(RES, open(path, "w"))
    for r in RES:
        print(r)
class Tally:
    def __init__(s): s.unreduced=0; s.counter=0; s.checks=0; s.log=[]
    def ident(s, name, expr):
        s.checks+=1
        e = sp.simplify(expr)
        if e != 0:
            s.unreduced+=1; s.log.append(("UNREDUCED",name,str(e)))
        print("identity", name, "->", e)
    def ineq(s, name, ok, detail=""):
        s.checks+=1
        if not ok:
            s.counter+=1; s.log.append(("COUNTEREXAMPLE",name,detail))
        print("check", name, ok, detail)
# proxies from definition D3 (v=o-x, u=(1+v)/2)
def prox(o, x, model):
    v = o - x; u = (1 + v) / 2
    return u / (1 - u) if model == 'O' else (1 - u) / u
pgrid = [mp.mpf(i) / 2000 for i in range(1, 2000)]  # 1999 interior points
