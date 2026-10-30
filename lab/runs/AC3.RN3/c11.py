import sys; sys.path.insert(0,'/tmp/claude-0/-home-claude/b1bc4455-e770-57c6-ada9-91dcfd104e34/scratchpad/AC3')
from common import *
T=Tally()
m,o,p,x,a=sp.symbols('m o p x alpha',real=True)
mu=sp.exp(m)
pO=mu/(1+mu); pI=1/(1+mu)
def ell(pp): return o*sp.log(pp)+(1-o)*sp.log(1-pp)
T.ident("dl/dlog(mu_a) = o-p", sp.diff(ell(pO),m)-(o-pO))
T.ident("dl/dlog(mu_b) = p-o", sp.diff(ell(pI),m)-(pI-o))
pp=sp.symbols('pp',positive=True)
T.ident("dp/dlog mu_a = p(1-p)", sp.simplify(sp.diff(pO,m)-pO*(1-pO)))
T.ident("dp/dlog mu_b = -p(1-p)", sp.simplify(sp.diff(pI,m)+pI*(1-pI)))
# drift: true prob p, estimate x
EO=p*(1-x)+(1-p)*(0-x); EI=p*(x-1)+(1-p)*(x-0)
T.ident("E[o-x]=p-x", EO-(p-x)); T.ident("E[x-o]=x-p", EI-(x-p))
# stability signs: iETS_O: x<p => drift>0 => mu_a up => p_hat up; iETS_I: x<p => drift<0 => mu_b down => p_hat up
bad=0
for P in pgrid[::7]:
    for i in range(1,400):
        X=mp.mpf(i)/400
        dO=P-X; dI=X-P
        if X<P and not (dO>0 and dI<0): bad+=1
        if X>P and not (dO<0 and dI>0): bad+=1
T.ineq("sign of drift moves p_hat towards p (both models)", bad==0, f"bad={bad}")
# positivity: 1+alpha*e in the four cases, alpha in (0,1], x in (0,1)
cases={"O,o=0":1+a*(0-x),"O,o=1":1+a*(1-x),"I,o=0":1+a*(x-0),"I,o=1":1+a*(x-1)}
T.ident("1-alpha x = (1-alpha)+alpha(1-x)", cases["O,o=0"]-((1-a)+a*(1-x)))
T.ident("1+alpha(x-1) = (1-alpha)+alpha x", cases["I,o=1"]-((1-a)+a*x))
bad=0
for A in [mp.mpf(i)/50 for i in range(1,51)]:
    for i in range(1,1000):
        X=mp.mpf(i)/1000
        for c in cases.values():
            if not c.subs({a:A,x:X})>0: bad+=1
T.ineq("1+alpha e >0 for alpha in (0,1] on 50x999 grid, all 4 cases", bad==0, f"bad={bad}")
# necessity: alpha>1 fails for x close to 1 (O, o=0)
T.ineq("alpha=1.1, x=0.95, O, o=0: 1+alpha e <0 (positivity needs alpha<=1)", (1-1.1*0.95)<0)
# floating-point corner: in doubles p_hat can round to exactly 1
import numpy as np
mu_big=1e17; ph=mu_big/(1+mu_big)
print("double p_hat at mu=1e17:",ph, "state after o=0, alpha=1:", mu_big*(1+1.0*(0-ph)))
add("double precision: state after o=0, alpha=1, mu=1e17 (iETS_O)", float(mu_big*(1+1.0*(0-ph))), claim="C11", note="p_hat rounds to 1.0 in double precision, so the positive state underflows to zero only through rounding")
print(T.log)
add("identities sympy could not reduce to zero", T.unreduced, claim="C11")
add("numerical counterexamples", T.counter, claim="C11")
add("checks performed", T.checks, claim="C11")
save()
