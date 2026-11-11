import numpy as np, math
rng=np.random.default_rng(11)
def run(z,obs,al,l0):
    # exact filter v,f and zero-error e
    a=l0;P=0.0;l=l0;Sv=0;Se=0;rows=[];pv=None
    for t in range(len(z)):
        if obs[t]:
            v=z[t]-a;f=P+1;k=(P+al)/f;e=z[t]-l
            rows.append((t,v,f,e));Sv+=v*v/f;Se+=e*e
            P=P+al**2-k*k*f;a+=k*v;l+=al*e
        else: P+=al**2
    return Sv,Se,rows
al=0.4;z=rng.normal(size=12).cumsum();obs=np.array([1,1,0,0,1,1,0,1,1,1,1,1])
Sv,Se,rows=run(z,obs,al,0.0)
print("Sigma v^2/f =",Sv,"Sigma e^2 =",Se)
for r in rows[:4]: print(r)
# Q>0 when gap followed by last observation
obs2=np.array([1,1,0,0,1]);z2=rng.normal(size=5)
Sv,Se,rows=run(z2,obs2,al,0.0);print("last-obs-after-gap: Se-Sv =",Se-Sv,"expected e^2(1-1/f) =",rows[-1][3]**2*(1-1/rows[-1][2]))
# random sign of Se-Sv across draws
d=[]
for i in range(2000):
    zz=rng.normal(size=12).cumsum();Sv,Se,_=run(zz,obs,al,0.0);d.append(Se-Sv)
print("Se-Sv: min,max,frac nonzero",min(d),max(d),np.mean(np.abs(d)>1e-12))
