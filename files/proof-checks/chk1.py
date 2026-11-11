import numpy as np, math
rng=np.random.default_rng(1)
def kalman_ann(z,obs,alpha,l0):
    a=l0;P=0.0;ll=0.0;rows=[];S=0.0
    for t in range(len(z)):
        if obs[t]:
            v=z[t]-a;f=P+1;k=(P+alpha)/f
            ll+=-0.5*(math.log(2*math.pi*f)+v*v/f)
            rows.append((v,f,k))
            S+=v*v/f
            Pn=P+alpha**2-k*k*f
            a=a+k*v;P=Pn
        else:
            P=P+alpha**2
    return ll,rows,S
def zero_err(z,obs,alpha,l0):
    l=l0;S=0.0;n=0
    for t in range(len(z)):
        if obs[t]:
            e=z[t]-l;S+=e*e;n+=1;l+=alpha*e
    return S,n
# --- C1/C4: brute-force integral over unobserved z vs Kalman
alpha=0.4;l0=1.0;T=6
z=rng.normal(size=T)*1.0+1
obs=np.array([1,1,0,1,0,1])
def complete_density(zs,alpha,l0):
    l=l0;ll=0.0
    for t in range(len(zs)):
        e=zs[t]-l; ll+=-0.5*(math.log(2*math.pi)+e*e); l+=alpha*e
    return ll
g=np.linspace(-12,14,1201);dz=g[1]-g[0]
tot=0.0
for u1 in g:
    for u2 in g:
        zz=z.copy();zz[2]=u1;zz[4]=u2
        tot+=math.exp(complete_density(zz,alpha,l0))
tot*=dz*dz
print("brute log-lik",math.log(tot),"kalman",kalman_ann(z,obs,alpha,l0)[0])
# --- C1 occurrence separation: enumerate with discrete simulation of p constant: trivial algebra; skip
# --- C5: estimators
def sim(T,alpha,pobs,seed):
    r=np.random.default_rng(seed);l=0.0;z=np.zeros(T);
    for t in range(T):
        e=r.normal();z[t]=l+e;l+=alpha*e
    obs=(r.random(T)<pobs).astype(int);return z,obs
grid=np.linspace(0.02,0.98,193)
for pobs in (0.5,0.3):
  res_ex=[];res_ze=[]
  for seed in range(6):
    z,obs=sim(20000,0.3,pobs,seed)
    # l0 treated as true 0
    ex=[];ze=[]
    for a in grid:
        ll,rows,S=kalman_ann(z,obs,a,0.0);n=obs.sum()
        lf=sum(math.log(f) for v,f,k in rows)
        ex.append(-0.5*(lf+n*math.log(S/n)))
        S2,n2=zero_err(z,obs,a,0.0);ze.append(-0.5*n2*math.log(S2/n2))
    res_ex.append(grid[int(np.argmax(ex))]);res_ze.append(grid[int(np.argmax(ze))])
  print("pobs",pobs,"true 0.3 exact MLE mean",np.mean(res_ex),res_ex,"zero-error mean",np.mean(res_ze),res_ze)
# --- C16
for al in (0.1,0.3,0.7):
  for j in range(1,8):
    z=np.zeros(j);obs=np.zeros(j,int);obs[-1]=1
    ll,rows,S=kalman_ann(z,obs,al,0.0)
    k=rows[0][2];form=((j-1)*al**2+al)/((j-1)*al**2+1)
    assert abs(k-form)<1e-14,(al,j,k,form)
print("C16 gain formula matches Kalman for alpha in .1,.3,.7, j=1..7; j=200 alpha .3:",((199)*.09+.3)/(199*.09+1))
# direct regression check of gain: cov(l_t, z)/var(z) by simulation
al=0.3;j=4;r=np.random.default_rng(5);N=400000
e=r.normal(size=(N,j));l=np.zeros(N)
# P=0 start: level known = 0. j-1 unobserved errors then observed z
lt=al*e[:,:j-1].sum(1);zj=lt+e[:,j-1];lnext=lt+al*e[:,j-1]
print("MC gain",np.cov(lnext-lt,zj-lt)[0,1]/np.var(zj-lt)," (l_{t+1}-a)/(z-a) regression; formula",((j-1)*al**2+al)/((j-1)*al**2+1))
