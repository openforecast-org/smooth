import sys; sys.path.insert(0,'/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC3')
from common import *
mp.mp.dps=40
T=Tally()
p,x,q,s=sp.symbols('p x q s',positive=True)
# multiplicative drift from definition: h(x)=E[e|x]=E[proxy]-1
hb=p*prox(1,x,'I')+(1-p)*prox(0,x,'I')-1
ha=p*prox(1,x,'O')+(1-p)*prox(0,x,'O')-1
T.ident("h_b closed form", hb+2*(x**2-2*x+p)/((2-x)*(1-x)))
T.ident("h_a closed form", ha-2*(p-x**2)/(x*(1+x)))
# substitute q=sqrt(1-p): p=1-q^2, x*=1-q
T.ident("h_b(1-sqrt(1-p))=0", sp.simplify(hb.subs(p,1-q**2).subs(x,1-q)))
T.ident("h_b factorised: h_b(2-x)(1-x)=2(x-x*)(1+q-x)", (hb*(2-x)*(1-x)).subs(p,1-q**2)-2*(x-(1-q))*(1+q-x))
T.ident("p-x* = q(1-q)", (1-q**2)-(1-q)-q*(1-q))
T.ident("h_a(sqrt p)=0", sp.simplify(ha.subs(p,s**2).subs(x,s)))
T.ident("h_a factorised: h_a x(1+x)=2(s-x)(s+x)", (ha*x*(1+x)).subs(p,s**2)-2*(s-x)*(s+x))
T.ident("sqrt p - p = s(1-s)", s-s**2-s*(1-s))
# roots of the quadratic: only 1-q is in (0,1)
print(sp.solve(x**2-2*x+p,x))
# numeric: signs of h_b, h_a on x-grid for sampled p (stability and uniqueness), straight from the definitions
import random
random.seed(1)
bad_sign=0; bad_ord=0
for P in pgrid[::10]+[mp.mpf('1e-9'),1-mp.mpf('1e-9')]:
    xs_=1-mp.sqrt(1-P); xo=mp.sqrt(P)
    if not (xs_<P and xo>P): bad_ord+=1
    nb=0; na=0
    for i in range(1,1000):
        X=mp.mpf(i)/1000
        hbv=P*prox(1,X,'I')+(1-P)*prox(0,X,'I')-1
        hav=P*prox(1,X,'O')+(1-P)*prox(0,X,'O')-1
        if X<xs_ and not hbv<0: bad_sign+=1
        if X>xs_ and not hbv>0: bad_sign+=1
        if X<xo and not hav>0: bad_sign+=1
        if X>xo and not hav<0: bad_sign+=1
T.ineq("multiplicative: order x*<p, sqrt p>p", bad_ord==0, f"bad={bad_ord}")
T.ineq("multiplicative: sign pattern of h (=> unique, stable) on x-grid", bad_sign==0, f"bad={bad_sign}")
# additive
A=lambda X: mp.log(X/(2-X)); B=lambda X: mp.log((1+X)/(1-X))
def hadd_I(X,P): return P*mp.log(prox(1,X,'I'))+(1-P)*mp.log(prox(0,X,'I'))
def hadd_O(X,P): return P*mp.log(prox(1,X,'O'))+(1-P)*mp.log(prox(0,X,'O'))
def R(X): return mp.atanh(1-X)/mp.atanh(X)
Pp,X=sp.symbols('P X',positive=True)
# h_b = 2 atanh(x) [ (1-p) - p R(x) ] (symbolic, via logs)
hsym=Pp*sp.log(X/(2-X))+(1-Pp)*sp.log((1+X)/(1-X))
alt=2*sp.atanh(X)*((1-Pp)-Pp*sp.atanh(1-X)/sp.atanh(X))
T.ident("h_b = 2atanh(x)[(1-p)-pR(x)]", sp.expand_log(sp.expand_log((hsym-alt).rewrite(sp.log),force=True),force=True))
def bis(f,lo,hi):
    flo=f(lo)
    for _ in range(300):
        m=(lo+hi)/2; fm=f(m)
        if (fm>0)==(flo>0): lo=m; flo=fm
        else: hi=m
    return (lo+hi)/2
bad_root=0; bad_between=0; bad_stab=0; bad_same=0; bad_R=0; bad_uniq=0
maxres=0
for P in pgrid+[mp.mpf('1e-9'),1-mp.mpf('1e-9')]:
    xr=bis(lambda X: hadd_I(X,P),mp.mpf('1e-30'),1-mp.mpf('1e-30'))
    res=abs(hadd_I(xr,P)); maxres=max(maxres,res)
    if res>mp.mpf(10)**-25 or not (0<xr<1): bad_root+=1
    # same root for iETS_O: h_a = -h_b
    if abs(hadd_O(xr,P))>mp.mpf(10)**-25 or abs(hadd_O(xr,P)+hadd_I(xr,P))>mp.mpf(10)**-30: bad_same+=1
    # R(x*) = (1-p)/p
    if abs(R(xr)-(1-P)/P)>mp.mpf(10)**-20*max(1,(1-P)/P): bad_R+=1
    half=mp.mpf(1)/2
    if P<half: ok = P<xr<half
    elif P>half: ok = half<xr<P
    else: ok = abs(xr-half)<mp.mpf(10)**-25
    if not ok: bad_between+=1
    # stability: h_I increasing through root (h_I<0 left, >0 right), h_O the reverse
    d=mp.mpf(10)**-12*min(xr,1-xr)
    if not (hadd_I(xr-d,P)<0<hadd_I(xr+d,P) and hadd_O(xr-d,P)>0>hadd_O(xr+d,P)): bad_stab+=1
T.ineq("additive: root found, residual<1e-25 (max res %s)"%mp.nstr(maxres,3), bad_root==0, f"bad={bad_root}")
T.ineq("additive: iETS_O and iETS_I share the root, h_a=-h_b", bad_same==0, f"bad={bad_same}")
T.ineq("additive: R(x*)=(1-p)/p", bad_R==0, f"bad={bad_R}")
T.ineq("additive: x* strictly between p and 1/2 for p!=1/2; =1/2 at 1/2", bad_between==0, f"bad={bad_between}")
T.ineq("additive: stability (sign change) at the root", bad_stab==0, f"bad={bad_stab}")
# uniqueness: sign changes of h_I over a fine x-grid for sampled p
for P in pgrid[::25]:
    sg=[hadd_I(mp.mpf(i)/3000,P)>0 for i in range(1,3000)]
    ch=sum(1 for i in range(len(sg)-1) if sg[i]!=sg[i+1])
    if ch!=1: bad_uniq+=1
T.ineq("additive: exactly one sign change of h on 2999 x-points, 80 p values", bad_uniq==0, f"bad={bad_uniq}")
# R strictly decreasing, limits
mono=all(R(mp.mpf(i+1)/4000)<R(mp.mpf(i)/4000) for i in range(1,3999))
T.ineq("R decreasing on grid", mono)
print("limits R at 0+,1-:", sp.limit(sp.atanh(1-X)/sp.atanh(X),X,0,'+'), sp.limit(sp.atanh(1-X)/sp.atanh(X),X,1,'-'))
# p=1/2 edge case
half_root=bis(lambda X: hadd_I(X,mp.mpf(1)/2),mp.mpf('1e-30'),1-mp.mpf('1e-30'))
print("root at p=1/2:",half_root)
print(T.log)
add("identities sympy could not reduce to zero", T.unreduced, claim="C10")
add("numerical counterexamples", T.counter, claim="C10")
add("checks performed", T.checks, claim="C10")
add("additive root at p=1/2", float(half_root), claim="C10", note="equals p, so not strictly between p and 1/2")
add("additive root at p=0.1", float(bis(lambda X: hadd_I(X,mp.mpf('0.1')),mp.mpf('1e-30'),1-mp.mpf('1e-30'))), claim="C10")
add("additive root at p=0.9", float(bis(lambda X: hadd_I(X,mp.mpf('0.9')),mp.mpf('1e-30'),1-mp.mpf('1e-30'))), claim="C10")
save()
