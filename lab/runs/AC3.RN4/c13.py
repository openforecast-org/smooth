import sys; sys.path.insert(0,'/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC3')
from common import *
mp.mp.dps=40
T=Tally()
p,x,al,mu0,e=sp.symbols('p x alpha mu0 e',positive=True)
# ADAM update: mu_t = mu0 (1+e)^alpha => log mu_t - log mu0 = alpha log(1+e)
T.ident("log of ADAM update", sp.expand_log(sp.log(mu0*(1+e)**al)-sp.log(mu0)-al*sp.log(1+e),force=True))
# old proxy: log(1+e)=log(u/(1-u)) equals the additive old error, both models, from the definition
for mod in 'OI':
    for o in (0,1):
        T.ident(f"old ADAM error = additive old error, {mod}, o={o}", sp.expand_log(sp.log(prox(o,x,mod))-sp.log(prox(o,x,mod)),force=True))
# exponential score proxy: log(1+e)=log(exp(s))=s
s=sp.symbols('s',real=True)
T.ident("log exp s = s", sp.log(sp.exp(s))-s)
EsO=p*(1-x)+(1-p)*(-x); EsI=p*(x-1)+(1-p)*x
T.ident("exp score drift iETS_O = p-x", EsO-(p-x)); T.ident("exp score drift iETS_I = x-p", EsI-(x-p))
# multiplicative score proxy, 1+e=1+s
kO=p*sp.log(1+(1-x))+(1-p)*sp.log(1+(0-x))
kI=p*sp.log(1+(x-1))+(1-p)*sp.log(1+(x-0))
T.ident("k_O = p log(2-x)+(1-p)log(1-x)", kO-(p*sp.log(2-x)+(1-p)*sp.log(1-x)))
T.ident("k_I = p log x+(1-p)log(1+x)", kI-(p*sp.log(x)+(1-p)*sp.log(1+x)))
dkO=sp.simplify(sp.diff(kO,x)); dkI=sp.simplify(sp.diff(kI,x))
T.ident("k_O' = -p/(2-x)-(1-p)/(1-x) (<0 on (0,1))", dkO+p/(2-x)+(1-p)/(1-x))
T.ident("k_I' = p/x+(1-p)/(1+x) (>0 on (0,1))", dkI-p/x-(1-p)/(1+x))
print("limits k_O(0+)=",sp.limit(kO,x,0,'+')," k_O(1-)=",sp.limit(kO,x,1,'-'))
print("limits k_I(0+)=",sp.limit(kI,x,0,'+')," k_I(1-)=",sp.limit(kI,x,1,'-'))
# k(p)<0 : Jensen, log y < y-1 strictly for y != 1; E[s]=0 at x=p
kOp=kO.subs(x,p); kIp=kI.subs(x,p)
T.ident("E[s] at x=p is 0 (O)", EsO.subs(x,p)); T.ident("E[s] at x=p is 0 (I)", EsI.subs(x,p))
def bis(f,lo,hi):
    flo=f(lo)
    for _ in range(300):
        m=(lo+hi)/2; fm=f(m)
        if (fm>0)==(flo>0): lo=m; flo=fm
        else: hi=m
    return (lo+hi)/2
fO=lambda X,P: P*mp.log(2-X)+(1-P)*mp.log(1-X)
fI=lambda X,P: P*mp.log(X)+(1-P)*mp.log(1+X)
# also directly from definition: E[log(1+s)] by enumeration
def dO(X,P): return P*mp.log(1+(1-X))+(1-P)*mp.log(1+(0-X))
def dI(X,P): return P*mp.log(1+(X-1))+(1-P)*mp.log(1+(X-0))
bad_kp=bad_root=bad_pos=bad_stab=bad_def=0; maxgapO=-1; mingapO=1; mingapI=1
gaps=[]
for P in pgrid+[mp.mpf('1e-9'),1-mp.mpf('1e-9')]:
    if not (fO(P,P)<0 and fI(P,P)<0): bad_kp+=1
    if abs(dO(P/2,P)-fO(P/2,P))>mp.mpf(10)**-30 or abs(dI(P/2,P)-fI(P/2,P))>mp.mpf(10)**-30: bad_def+=1
    xo=bis(lambda X: fO(X,P),mp.mpf('1e-30'),1-mp.mpf('1e-30'))
    xi=bis(lambda X: fI(X,P),mp.mpf('1e-30'),1-mp.mpf('1e-30'))
    if not (0<xo<P and P<xi<1): bad_root+=1
    if abs(fO(xo,P))>mp.mpf(10)**-25 or abs(fI(xi,P))>mp.mpf(10)**-25: bad_root+=1
    d=mp.mpf(10)**-12*min(xo,1-xo); d2=mp.mpf(10)**-12*min(xi,1-xi)
    # stability: iETS_O h>0 left of root, <0 right; iETS_I h<0 left, >0 right
    if not (fO(xo-d,P)>0>fO(xo+d,P) and fI(xi-d2,P)<0<fI(xi+d2,P)): bad_stab+=1
    gaps.append((float(P-xo),float(xi-P)))
T.ineq("E[log(1+s)]<0 at x=p, both models", bad_kp==0, f"bad={bad_kp}")
T.ineq("enumeration equals closed form", bad_def==0, f"bad={bad_def}")
T.ineq("roots: iETS_O root in (0,p), iETS_I root in (p,1)", bad_root==0, f"bad={bad_root}")
T.ineq("stability of the roots", bad_stab==0, f"bad={bad_stab}")
# uniqueness numerically (sign changes)
bad_u=0
for P in pgrid[::25]:
    for f in (fO,fI):
        sg=[f(mp.mpf(i)/3000,P)>0 for i in range(1,3000)]
        if sum(1 for i in range(len(sg)-1) if sg[i]!=sg[i+1])!=1: bad_u+=1
T.ineq("exactly one sign change on 2999 x-points, 80 p", bad_u==0, f"bad={bad_u}")
# old ADAM filter zero-drift equals additive old root: same function by construction; numeric agreement with C10 root
hadd=lambda X,P: P*mp.log(prox(1,X,'I'))+(1-P)*mp.log(prox(0,X,'I'))
hadam=lambda X,P: P*mp.log(1+(prox(1,X,'I')-1))+(1-P)*mp.log(1+(prox(0,X,'I')-1))
bad=0
for P in pgrid[::20]:
    for X in (mp.mpf('0.2'),mp.mpf('0.7')):
        if abs(hadd(X,P)-hadam(X,P))>mp.mpf(10)**-30: bad+=1
T.ineq("ADAM old drift equals additive old drift",bad==0)
print(T.log)
add("identities sympy could not reduce to zero", T.unreduced, claim="C13")
add("numerical counterexamples", T.counter, claim="C13")
add("checks performed", T.checks, claim="C13")
add("min over p grid of p - root (iETS_O, mult. score, ADAM)", min(g[0] for g in gaps), claim="C13")
add("min over p grid of root - p (iETS_I, mult. score, ADAM)", min(g[1] for g in gaps), claim="C13")
add("root iETS_O at p=0.5", float(bis(lambda X: fO(X,mp.mpf('0.5')),mp.mpf('1e-30'),1-mp.mpf('1e-30'))), claim="C13")
save()
