import numpy as np, math
rng=np.random.default_rng(3)
T1,T0=60,40;e=rng.normal(scale=1.7,size=T1);S=np.sum(e**2)
def lH(s2): return -0.5*T1*math.log(2*math.pi*s2)-S/(2*s2)-T0*0.5*math.log(2*math.pi*math.e*s2)
def lobs(s2): return -0.5*T1*math.log(2*math.pi*s2)-S/(2*s2)
g=np.linspace(0.5,8,200001)
aH=g[np.argmax([lH(x) for x in g[::10]])*10];ao=g[np.argmax([lobs(x) for x in g[::10]])*10]
print("argmax l_H",aH,"S/T",S/(T1+T0),"; argmax observed",ao,"S/T1",S/T1)
s2=2.0
for _ in range(500): s2=(S+T0*s2)/(T1+T0)
print("EM iteration limit",s2)
# gradient of cross-entropy at theta=theta_k (normal var): d/ds2 int N(0,sk) log N(0,s2) = -1/(2 s2)+sk/(2 s2^2)
sk=1.3;h=1e-6
ce=lambda s2:-0.5*math.log(2*math.pi*s2)-sk/(2*s2)
print("cross-entropy grad at s2=sk",(ce(sk+h)-ce(sk-h))/(2*h),"; -dH/ds2 at sk",(-0.5*math.log(2*math.pi*math.e*(sk+h))+0.5*math.log(2*math.pi*math.e*(sk-h)))/(2*h))
# KL decomposition check
s_k,s_t=1.3,2.1;x=np.linspace(-30,30,2000001);dx=x[1]-x[0]
fk=np.exp(-x*x/(2*s_k))/math.sqrt(2*math.pi*s_k);lft=-x*x/(2*s_t)-0.5*math.log(2*math.pi*s_t)
lhs=np.sum(fk*lft)*dx;H=0.5*math.log(2*math.pi*math.e*s_k);KL=0.5*(math.log(s_t/s_k)+s_k/s_t-1)
print("cross-entropy",lhs,"-H-KL",-H-KL)
