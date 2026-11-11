import numpy as np, math
rng=np.random.default_rng(7)
# ===== C6: ETS(M,N,N) ADAM lognormal with gaps; filter on log z with mean shift
def filt(logz,obs,alpha,x0,s):
    a=x0;P=0.0;ll=0.0;rows=[]
    for t in range(len(logz)):
        if obs[t]:
            v=logz[t]-(a-s/2);f=P+1;k=(P+alpha)/f
            ll+=-0.5*(math.log(2*math.pi*s*f)+v*v/(s*f))-logz[t]
            rows.append((v,f));Pn=P+alpha**2-k*k*f
            a=a-alpha*s/2+k*v;P=Pn
        else:
            a=a-alpha*s/2;P=P+alpha**2
    return ll,rows
alpha=0.35;s0=0.4;T=12;x0=2.0
x=x0;logz=np.zeros(T)
for t in range(T):
    u=rng.normal(-s0/2,math.sqrt(s0));logz[t]=x+u;x+=alpha*u
obs=np.array([1,1,0,0,1,1,0,1,1,1,0,1])
# brute force: integrate over unobserved u at periods 2,3,6,10 -> use u-space grid for 4 dims is heavy; use 2 gaps only
obs2=np.array([1,1,0,1,1,1,0,1,1,1,1,1])
s=0.4
def complete_ll(logzfull,alpha,x0,s):
    x=x0;ll=0.0
    for t in range(len(logzfull)):
        u=logzfull[t]-x;ll+=-0.5*(math.log(2*math.pi*s)+(u+s/2)**2/s)-logzfull[t];x+=alpha*u
    return ll
g=np.linspace(x0-8,x0+8,801);dz=g[1]-g[0];tot=0.0
for a_ in g:
  for b_ in g:
    lz=logz.copy();lz[2]=a_;lz[6]=b_;tot+=math.exp(complete_ll(lz,alpha,x0,s))
# note the unobserved z density in log z: complete_ll includes -logz Jacobian for log-z variable? we integrate over log z, so remove Jacobian for the unobserved ones
# complete_ll includes -logz for all; integrating over log z coordinates needs +logz for unobserved (density in log z space)
tot=0.0
for a_ in g:
  for b_ in g:
    lz=logz.copy();lz[2]=a_;lz[6]=b_;tot+=math.exp(complete_ll(lz,alpha,x0,s)+a_+b_)
tot*=dz*dz
print("C6 brute",math.log(tot),"filter",filt(logz,obs2,alpha,x0,s)[0])
# scale equation: b_t from v(s)=v0+s b
def vs(s,obs): return np.array([v for v,f in filt(logz,obs,alpha,x0,s)[1]])
for ob in (obs,obs2):
    v0=vs(1e-12,ob);b=(vs(1.0,ob)-v0)  # linear in s so exact
    # check linearity
    assert np.allclose(vs(0.7,ob),v0+0.7*b)
    f=np.array([ff for v,ff in filt(logz,ob,alpha,x0,0.3)[1]]);T1=ob.sum()
    A=(b**2/f).sum();C=-(v0**2/f).sum()
    root=(-T1+math.sqrt(T1**2-4*A*C))/(2*A)
    ss=np.linspace(0.01,3,60000);L=[filt(logz,ob,alpha,x0,q)[0] for q in ss[::20]]
    sm=ss[::20][int(np.argmax(L))]
    # refine
    ss2=np.linspace(root-0.01,root+0.01,201);L2=[filt(logz,ob,alpha,x0,q)[0] for q in ss2]
    print("C6 root",root,"grid argmax",sm,"fine argmax",ss2[int(np.argmax(L2))],"b range",b.min(),b.max())
# ===== C8
m=0.5+0.25*0.5**2*0+0.5+0  # placeholder
s=0.5;m=s+s*s/4
print("C8 m",m,"ADAM est",2*(1-math.sqrt(1-m)),"MLE",2*(math.sqrt(1+m)-1),"ratio",2*(1-math.sqrt(1-m))/s)
# direct MLE check: lognormal u~N(-s/2,s), loglik in s for given u sample
u=rng.normal(-0.25,math.sqrt(0.5),size=200000);mm=np.mean(u**2)
ss=np.linspace(0.3,0.8,50001);L=-0.5*len(u)*np.log(ss)-((u[:,None]**2).sum(0) if False else 0)
Su2=np.sum(u**2);Su=np.sum(u)
L=-0.5*len(u)*np.log(ss)-(Su2+ss*Su+len(u)*ss**2/4)/(2*ss)
print("C8 sample m",mm,"numerical MLE",ss[np.argmax(L)],"formula",2*(math.sqrt(1+mm)-1),"ADAM-type",2*(1-math.sqrt(1-mm)))
ms=np.linspace(0.001,0.999,999);print("C8 ADAM-type > MLE for all m in (0,1):",bool(np.all(2*(1-np.sqrt(1-ms))>2*(np.sqrt(1+ms)-1))))
# ===== C10
p=0.4;c=0.9;mu=3.0;sg=0.5
# sizes lognormal location mu
q=(c-(1-p))/p
from statistics import NormalDist
nd=NormalDist();u_=mu*math.exp(nd.inv_cdf(q)*sg-sg*sg/2*0)  # lognormal median mu, simple
N=2000000;o=rng.random(N)<p;zz=mu*np.exp(sg*rng.normal(size=N));y=o*zz
print("C10 q",q,"coverage P(0<=y<=u)",np.mean(y<=u_),"target",c,"; size cdf at u",np.mean(zz<=u_))
# ===== C12
al,be,ga,ph,m_=0.3,0.1,0.2,0.9,4
E=np.exp(rng.normal(-0.05,0.3,size=40))-1+1e-9  # 1+eps
l,b=5.0,1.02;sv=[1.1,0.9,1.05,0.95]
ll,bb=math.log(l),math.log(b);ss_=[math.log(x) for x in sv]
lz=[];zs=[]
for t in range(40):
    zt=l*b**ph*sv[t%m_]*(1+E[t]);zs.append(zt)
    l,b,sv[t%m_]=l*b**ph*(1+E[t])**al,b**ph*(1+E[t])**be,sv[t%m_]*(1+E[t])**ga
    u=math.log(1+E[t]);lz.append(math.log(zt)-(ll+ph*bb+ss_[t%m_]))
    ll,bb,ss_[t%m_]=ll+ph*bb+al*u,ph*bb+be*u,ss_[t%m_]+ga*u
print("C12 ADAM log-linear mismatch",max(abs(math.log(l)-ll),abs(math.log(b)-bb),max(abs(math.log(x)-y) for x,y in zip(sv,ss_)),max(abs(a-math.log(1+e)) for a,e in zip(lz,E))))
e=np.array([-.5,-.2,.3,1.0]);print("C12 conventional log(1+a e)/(a log(1+e)) for a=.3:",np.log(1+.3*e)/(.3*np.log(1+e)))
# ===== C14
# gamma shape k scale th entropy closed form vs numeric int f log f
for k_,th in ((2.0,0.5),(5.0,0.2)):
    x=np.linspace(1e-9,60,2000001);dx=x[1]-x[0]
    lf=(k_-1)*np.log(x)-x/th-math.lgamma(k_)-k_*math.log(th);f=np.exp(lf)
    num=np.sum(f*lf)*dx
    H=k_+math.log(th)+math.lgamma(k_)+(1-k_)*(math.log(k_)*0+ (lambda z: (math.log(z)-1/(2*z)-1/(12*z*z)+1/(120*z**4)) if z>6 else None)(k_) if False else 0)
    # digamma by series
    def dg(z):
        r=0.0
        while z<8: r-=1/z;z+=1
        return r+math.log(z)-1/(2*z)-1/(12*z*z)+1/(120*z**4)-1/(252*z**6)
    H=k_+math.log(th)+math.lgamma(k_)+(1-k_)*dg(k_)
    print("C14 gamma",k_,th,"int f log f =",num,"-H =",-H)
sg=0.7;print("C14 normal -H",-0.5*math.log(2*math.pi*math.e*sg*sg))
xg=np.linspace(-8,8,400001);fn=np.exp(-xg**2/(2*sg*sg))/math.sqrt(2*math.pi*sg*sg);print("   numeric",np.sum(fn*np.log(fn))*(xg[1]-xg[0]))
# ===== C19
def Eg(al,s): return math.exp(math.lgamma(1/s+al)+al*math.log(s)-math.lgamma(1/s))
s=0.32;al=0.21;print("C19 formula E(1+e)^alpha",Eg(al,s))
x=rng.gamma(1/s,s,size=4_000_000);print("   MC",np.mean(x**al),"mean x",x.mean(),"var",x.var())
for h in (1,2,5,12):
    N=1_000_000;lvl=np.full(N,10.0)
    for i in range(h-1): lvl=lvl*rng.gamma(1/s,s,size=N)**al
    zT=lvl*rng.gamma(1/s,s,size=N)
    print("   h",h,"MC E z / mu_hat",zT.mean()/10,"formula",Eg(al,s)**(h-1))
