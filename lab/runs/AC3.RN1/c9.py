import sys; sys.path.insert(0,'/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC3')
from common import *
import shutil
T=Tally()
p=sp.symbols('p',positive=True)
# expectation at x=p straight from D3 by enumeration over o
Ea=p*prox(1,p,'O')+(1-p)*prox(0,p,'O')
Eb=p*prox(1,p,'I')+(1-p)*prox(0,p,'I')
T.ident("E[1+e_a] closed form", Ea-(2-p+(1-p)**2/(1+p)))
T.ident("E[1+e_b] closed form", Eb-(1+p+p**2/(2-p)))
T.ident("E[1+e_a]-1 = 2(1-p)/(1+p)", Ea-1-2*(1-p)/(1+p))
T.ident("E[1+e_b]-1 = 2p/(2-p)", Eb-1-2*p/(2-p))
# both strictly positive for p in (0,1): numerators/denominators positive, checked on the grid and at extremes
bad=0
for q in pgrid+[mp.mpf('1e-30'),1-mp.mpf('1e-30')]:
    a=2*(1-q)/(1+q); b=2*q/(2-q)
    if not (a>0 and b>0): bad+=1
T.ineq("E[1+e]>1 on grid", bad==0, f"bad={bad}")
# additive: g
ga=p*sp.log(prox(1,p,'O'))+(1-p)*sp.log(prox(0,p,'O'))
gb=p*sp.log(prox(1,p,'I'))+(1-p)*sp.log(prox(0,p,'I'))
g=p*sp.log((2-p)/p)+(1-p)*sp.log((1-p)/(1+p))
def lg(e): return sp.expand_log(sp.logcombine(sp.expand_log(e,force=True),force=True),force=True)
T.ident("E log(1+e_a)=g", lg(ga-g))
T.ident("E log(1+e_b)=-g", lg(gb+g))
g1=g.subs(p,1-p)
T.ident("g(1-p)=-g(p)", lg(g1+g))
T.ident("g(1/2)=0", sp.simplify(g.subs(p,sp.Rational(1,2))))
# atanh form: g = 2p atanh(1-p) - 2(1-p) atanh(p)
gat=2*p*sp.atanh(1-p)-2*(1-p)*sp.atanh(p)
T.ident("g = atanh form", lg(sp.expand_log((gat-g).rewrite(sp.log),force=True)))
# numeric: atanh form equals g, and the sign of g, 50 digits, grid + extremes + fine grid
def gnum(q): return q*mp.log((2-q)/q)+(1-q)*mp.log((1-q)/(1+q))
def phi(t): return mp.atanh(t)/t
bad_id=0; bad_sign=0; bad_phi=0
pts=pgrid+[mp.mpf(i)/20000 for i in range(1,20000)]+[mp.mpf(10)**(-k) for k in range(2,41,3)]+[1-mp.mpf(10)**(-k) for k in range(2,41,3)]
for q in pts:
    gv=gnum(q)
    if abs(gv-(2*q*mp.atanh(1-q)-2*(1-q)*mp.atanh(q)))>mp.mpf(10)**-30: bad_id+=1
    if abs(gv+gnum(1-q))>mp.mpf(10)**-30: bad_id+=1
    if q<mp.mpf(1)/2 and not gv>0: bad_sign+=1
    if q>mp.mpf(1)/2 and not gv<0: bad_sign+=1
    if q==mp.mpf(1)/2 and abs(gv)>mp.mpf(10)**-40: bad_sign+=1
    # g = 2p(1-p)[phi(1-p)-phi(p)]
    if abs(gv-2*q*(1-q)*(phi(1-q)-phi(q)))>mp.mpf(10)**-30: bad_phi+=1
T.ineq("g identities (atanh form, antisymmetry) on %d points"%len(pts), bad_id==0, f"bad={bad_id}")
T.ineq("sign of g: >0 below 1/2, 0 at 1/2, <0 above", bad_sign==0, f"bad={bad_sign}")
T.ineq("g=2p(1-p)[phi(1-p)-phi(p)], phi=atanh(t)/t", bad_phi==0, f"bad={bad_phi}")
# phi strictly increasing: series sum t^(2k)/(2k+1), all coefficients positive (analytic); numeric monotonicity
ts=[mp.mpf(i)/5000 for i in range(1,5000)]
mono=all(phi(ts[i+1])>phi(ts[i]) for i in range(len(ts)-1))
T.ineq("phi increasing on grid", mono)
# direct Monte Carlo-free sanity: exact enumeration at x=p reproduces closed forms at 50 digits (from definition, no algebra)
bad=0
for q in pgrid:
    ea=q*prox(1,q,'O')+(1-q)*prox(0,q,'O'); eb=q*prox(1,q,'I')+(1-q)*prox(0,q,'I')
    if abs(ea-(2-q+(1-q)**2/(1+q)))>mp.mpf(10)**-40 or abs(eb-(1+q+q**2/(2-q)))>mp.mpf(10)**-40: bad+=1
    ga_=q*mp.log(prox(1,q,'O'))+(1-q)*mp.log(prox(0,q,'O'))
    gb_=q*mp.log(prox(1,q,'I'))+(1-q)*mp.log(prox(0,q,'I'))
    if abs(ga_-gnum(q))>mp.mpf(10)**-40 or abs(gb_+gnum(q))>mp.mpf(10)**-40: bad+=1
T.ineq("enumeration from D3 equals closed forms (1999 p)", bad==0, f"bad={bad}")
print(T.log)
add("identities sympy could not reduce to zero", T.unreduced, claim="C9")
add("numerical counterexamples", T.counter, claim="C9")
add("checks performed", T.checks, claim="C9")
add("smallest E[1+e_a]-1 on grid", float(min(2*(1-q)/(1+q) for q in pgrid)), claim="C9")
add("smallest E[1+e_b]-1 on grid", float(min(2*q/(2-q) for q in pgrid)), claim="C9")
add("g(0.25)", float(gnum(mp.mpf('0.25'))), claim="C9")
save()
