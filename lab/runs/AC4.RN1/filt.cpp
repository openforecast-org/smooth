// Own implementation of occurrence filters from the definitions D3, D4, D5, D8, D16-D18 (AC4).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
// types: 0 old I M, 1 old O M, 2 old I A, 3 old O A, 4 score I M, 5 score O M, 6 score I A, 7 score O A,
//        8 expscore I, 9 expscore O, 10 multscore-ADAM I, 11 multscore-ADAM O, 12 Croston
int main(int argc, char** argv){
  int type=atoi(argv[1]); double p=atof(argv[2]), a=atof(argv[3]); long N=atol(argv[4]); unsigned long seed=atol(argv[5]);
  std::mt19937_64 g(seed); std::uniform_real_distribution<double> U(0,1);
  long burn=std::max(500000L,(long)(2000.0/a)); const int NB=100; long bl=N/NB; 
  bool isO=(type==1||type==3||type==5||type==7||type==9||type==11);
  double z; // log mu
  double qhat=1.0/p; long c=0;
  // start at the true p: O: p=mu/(1+mu) -> mu=p/(1-p); I: p=1/(1+mu) -> mu=(1-p)/p
  z = isO? log(p/(1-p)) : log((1-p)/p);
  std::vector<double> bm(NB,0.0); double tot=0;
  for(long t=-burn;t<N;t++){
    double x;
    if(type==12) x=1.0/qhat; else x = isO? 1.0/(1.0+exp(-z)) : 1.0/(1.0+exp(z));
    if(t>=0){ bm[t/bl>=NB?NB-1:t/bl]+=x; }
    int o = U(g)<p;
    if(type==12){ c++; if(o){ qhat+=a*(c-qhat); c=0;} continue; }
    double s = isO? (o-x) : (x-o);  // score
    double v=o-x, rO=(1+v)/(1-v), rI=(1-v)/(1+v);
    switch(type){
      case 0: z+=log(1+a*(rI-1)); break;
      case 1: z+=log(1+a*(rO-1)); break;
      case 2: z+=a*log(rI); break;
      case 3: z+=a*log(rO); break;
      case 4: case 5: z+=log(1+a*s); break;
      case 6: case 7: z+=a*s; break;
      case 8: case 9: z+=a*s; break;           // 1+e=exp(s), mu*(1+e)^alpha
      case 10: case 11: z+=a*log(1+s); break;  // 1+e=1+s, mu*(1+e)^alpha
    }
  }
  double m=0; for(int i=0;i<NB;i++){bm[i]/=bl; m+=bm[i];} m/=NB; double ss=0; for(int i=0;i<NB;i++) ss+=(bm[i]-m)*(bm[i]-m);
  printf("%d %.10g %.10g %.10g %.3g\n",type,p,a,m,sqrt(ss/(NB-1)/NB));
}
