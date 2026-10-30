// Own implementation of iETS occurrence filters on Bernoulli(p) data (AC5).
// stdin lines: model(o|i) err(A|M) proxy(old|score) p alpha seed ; args: T burn batches
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <string>
#include <iostream>
static uint64_t s[4];
static inline uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static inline uint64_t nxt(){uint64_t r=rotl(s[1]*5,7)*9,t=s[1]<<17;s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return r;}
static void seed(uint64_t z){for(int i=0;i<4;i++){z+=0x9e3779b97f4a7c15ULL;uint64_t x=z;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;s[i]=x^(x>>31);}}
int main(int argc,char**argv){
  long T=atol(argv[1]),B=atol(argv[2]);int nb=atoi(argv[3]);
  std::string m,e,pr;double p,a;unsigned long long sd;
  while(std::cin>>m>>e>>pr>>p>>a>>sd){
    seed(sd);
    bool O=(m=="o"),Add=(e=="A"),score=(pr=="score");
    // state: x = log mu (additive) or mu (multiplicative); start at the true value
    // O: p=mu/(1+mu) => mu=p/(1-p); I: p=1/(1+mu) => mu=(1-p)/p
    double mu=O? p/(1-p):(1-p)/p;
    double x=Add?log(mu):mu;
    long len=(T-B)/nb; double bs=0,tot=0; double bm[1000]; int ib=0; long cnt=0; int bad=0;
    double ob=0;
    for(long t=0;t<T;t++){
      double ph,q; // fitted probability and its complement
      if(Add){ // mu=exp(x)
        double lm=x; if(O){ if(lm>=0){double z=exp(-lm);ph=1/(1+z);q=z/(1+z);}else{double z=exp(lm);ph=z/(1+z);q=1/(1+z);} }
        else { if(lm>=0){double z=exp(-lm);q=1/(1+z);ph=z/(1+z);}else{double z=exp(lm);q=z/(1+z);ph=1/(1+z);} }
      } else { if(O){ph=x/(1+x);q=1/(1+x);} else {ph=1/(1+x);q=x/(1+x);} }
      if(t>=B){bs+=ph;cnt++; if(cnt==len){bm[ib++]=bs/len;bs=0;cnt=0;}}
      int o=((nxt()>>11)*(1.0/9007199254740992.0))<p;
      double ee; // error: (1+e) ratio for M ; log ratio / score for A
      if(score){ double sc=O? (o-ph):(ph-o);
        if(Add){x+=a*sc;} else {x*=1+a*sc;} }
      else {
        // r = u/(1-u), u=(1+o-ph)/2 ; for o=1: (2-ph)/ph ; o=0: q/(2-q)... (1-ph)/(1+ph)
        double r= o? (1+q)/ph : q/(1+ph); // 2-ph=1+q ; 1-ph=q ; 1+ph
        if(!O) r=1/r;
        if(Add){x+=a*log(r);} else {x*=1+a*(r-1);}
      }
      if(!std::isfinite(x)||(!Add&&x<=0)){bad=1;break;}
    }
    if(bad||ib<nb){printf("nan nan 1\n");continue;}
    double mm=0;for(int i=0;i<nb;i++)mm+=bm[i];mm/=nb;double v=0;for(int i=0;i<nb;i++)v+=(bm[i]-mm)*(bm[i]-mm);
    printf("%.12f %.12g 0\n",mm,sqrt(v/(nb-1)/nb));
  }
}
