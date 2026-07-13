#pragma once

// Analytic derivatives of the pure-ETS measurement / transition / update maps
// (adamWvalue / adamFvalue / adamGvalue from adamGeneral.h), used by
// adamCore::gradientSolve to propagate initial-state sensitivities in one
// forward pass instead of finite-difference probe passes.
//
// Scope: pure ETS only (nArima = 0, nXreg = 0, constant = false) — exactly the
// scope of initial="gradient". Both the ADAM-ETS maths (adamETS = true) and the
// conventional-ETS maths (adamETS = false) are covered, mirroring the branch
// structure of the original functions one-to-one. The original functions are
// NOT modified; these are stand-alone companions.
//
// Derivatives use real arithmetic: the |exp(g*log(complex(1+x)))| guards of the
// originals equal |1+x|^g on the real line, whose derivative is g*phi/(1+x)
// with phi = |1+x|^g — valid on both sides of -1. Non-finite values (states at
// zero, 1+x = 0) propagate to the caller's finiteness check, which falls back
// to finite differences.

// d(phi)/dx for phi = |1+x|^g
inline double adamPowJacHelper(double const &phi, double const &x, double const &g){
    return g * phi / (1 + x);
}

// |1+x|^g without complex arithmetic
inline double adamPowHelper(double const &x, double const &g){
    return std::pow(std::abs(1 + x), g);
}

// Derivative of the measurement: d(yhat)/d(v), a 1 x nETS row vector.
// Mirrors the pure-ETS branches of adamWvalue.
inline arma::rowvec adamWvalueJac(arma::vec const &vecVt, arma::rowvec const &rowvecW,
                                  char const &E, char const &T, char const &S,
                                  unsigned int const &nETS, unsigned int const &nSeasonal){
    arma::rowvec jacW(nETS, arma::fill::zeros);

    switch(S){
    case 'N':
        switch(T){
        case 'N':
            jacW(0) = 1;
            break;
        case 'A':
            jacW = rowvecW.cols(0,1);
            break;
        case 'M':{
            // yhat = v0^w0 * v1^w1
            double product = std::pow(vecVt(0), rowvecW(0)) * std::pow(vecVt(1), rowvecW(1));
            jacW(0) = rowvecW(0) * product / vecVt(0);
            jacW(1) = rowvecW(1) * product / vecVt(1);
            break;
        }
        }
        break;
    case 'A':
        switch(T){
        case 'N':
        case 'A':
            jacW = rowvecW.cols(0,nETS-1);
            break;
        case 'M':{
            // yhat = v0^w0 * v1^w1 + w_s * v_s
            double product = std::pow(vecVt(0), rowvecW(0)) * std::pow(vecVt(1), rowvecW(1));
            jacW(0) = rowvecW(0) * product / vecVt(0);
            jacW(1) = rowvecW(1) * product / vecVt(1);
            jacW.cols(2,nETS-1) = rowvecW.cols(2,nETS-1);
            break;
        }
        }
        break;
    case 'M':
        switch(T){
        case 'N':
        case 'M':{
            // yhat = prod(v_i^w_i) over all ETS rows
            double product = 1;
            for(unsigned int i=0; i<nETS; i=i+1){
                product *= std::pow(vecVt(i), rowvecW(i));
            }
            for(unsigned int i=0; i<nETS; i=i+1){
                jacW(i) = rowvecW(i) * product / vecVt(i);
            }
            break;
        }
        case 'A':{
            // yhat = (w0 l + w1 b) * prod(s_j^{w_j})
            double linear = rowvecW(0)*vecVt(0) + rowvecW(1)*vecVt(1);
            double productS = 1;
            for(unsigned int j=0; j<nSeasonal; j=j+1){
                productS *= std::pow(vecVt(2+j), rowvecW(2+j));
            }
            jacW(0) = rowvecW(0) * productS;
            jacW(1) = rowvecW(1) * productS;
            for(unsigned int j=0; j<nSeasonal; j=j+1){
                jacW(2+j) = rowvecW(2+j) * linear * productS / vecVt(2+j);
            }
            break;
        }
        }
        break;
    }
    return jacW;
}

// Derivative of the transition: d(F(v))/d(v), an nETS x nETS matrix.
// Mirrors the pure-ETS branches of adamFvalue: linear for T in {N, A}
// (the matrix itself), geometric for T == 'M' (v'_i = prod_j v_j^{F_ij}).
inline arma::mat adamFvalueJac(arma::vec const &vecVt, arma::mat const &matrixF,
                               char const &T, unsigned int const &nETS){
    if(T != 'M'){
        return matrixF;
    }
    arma::mat jacF(nETS, nETS, arma::fill::zeros);
    // v'_i = prod_j v_j^{F_ij}  =>  d v'_i / d v_j = F_ij * v'_i / v_j
    arma::vec vNew = arma::exp(matrixF * arma::log(arma::abs(vecVt)));
    for(unsigned int i=0; i<nETS; i=i+1){
        for(unsigned int j=0; j<nETS; j=j+1){
            if(matrixF(i,j) != 0){
                jacF(i,j) = matrixF(i,j) * vNew(i) / vecVt(j);
            }
        }
    }
    return jacF;
}

// Derivatives of the update g(v, e, yhat): fills
//   jacGv (nETS x nETS) = dg/dv (holding e, yhat),
//   jacGe (nETS)        = dg/de,
//   jacGy (nETS)        = dg/d(yhat).
// Mirrors the pure-ETS branches of adamGvalue for both adamETS variants.
inline void adamGvalueJac(arma::vec const &vecVt, arma::mat const &matrixF,
                          arma::mat const &rowvecW,
                          char const &E, char const &T, char const &S,
                          unsigned int const &nETS, unsigned int const &nSeasonal,
                          arma::vec const &vectorG, double const &error,
                          double const &fitted, bool const &adamETS,
                          arma::mat &jacGv, arma::vec &jacGe, arma::vec &jacGy){
    jacGv.zeros(nETS, nETS);
    jacGe.zeros(nETS);
    jacGy.zeros(nETS);

    if(adamETS){
        // Base: g = G * e
        jacGe = vectorG.rows(0,nETS-1);

        switch(E){
        case 'A':{
            // u enters as (1 + e/yhat)^G in the multiplicative-seasonal updates
            double u = error / fitted;
            switch(T){
            case 'N':
                if(S=='M'){
                    // g0 = G0 e / (w_s v_s); g_s = v_s (|1+u|^{G_s} - 1)
                    double sigma = arma::as_scalar(rowvecW.cols(1,nSeasonal) * vecVt.rows(1,nSeasonal));
                    jacGe(0) = vectorG(0) / sigma;
                    for(unsigned int j=1; j<=nSeasonal; j=j+1){
                        jacGv(0,j) = -vectorG(0) * error * rowvecW(j) / (sigma*sigma);
                        double phi = adamPowHelper(u, vectorG(j));
                        jacGv(j,j) = phi - 1;
                        double dPhi = vecVt(j) * adamPowJacHelper(phi, u, vectorG(j));
                        jacGe(j) = dPhi / fitted;
                        jacGy(j) = -dPhi * error / (fitted*fitted);
                    }
                }
                break;
            case 'A':
                if(S=='M'){
                    // g_{0,1} = G_{0,1} e / |prod s^{w_s}|; g_s = v_s (|1+u|^{G_s} - 1)
                    double productS = 1;
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        productS *= std::pow(std::abs(vecVt(2+j)), rowvecW(2+j));
                    }
                    for(unsigned int i=0; i<2; i=i+1){
                        jacGe(i) = vectorG(i) / productS;
                        for(unsigned int j=0; j<nSeasonal; j=j+1){
                            jacGv(i,2+j) = -vectorG(i) * error * rowvecW(2+j) / (productS * vecVt(2+j));
                        }
                    }
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        double phi = adamPowHelper(u, vectorG(2+j));
                        jacGv(2+j,2+j) = phi - 1;
                        double dPhi = vecVt(2+j) * adamPowJacHelper(phi, u, vectorG(2+j));
                        jacGe(2+j) = dPhi / fitted;
                        jacGy(2+j) = -dPhi * error / (fitted*fitted);
                    }
                }
                break;
            case 'M':
                switch(S){
                case 'N':
                case 'A':
                    // g1 = G1 e / v0
                    jacGe(1) = vectorG(1) / vecVt(0);
                    jacGv(1,0) = -vectorG(1) * error / (vecVt(0)*vecVt(0));
                    break;
                case 'M':{
                    // g0 = G0 e / (w_s v_s); g1 = b^{F11} (|1+u|^{G1} - 1);
                    // g_s = v_s (|1+u|^{G_s} - 1)
                    double sigma = arma::as_scalar(rowvecW.cols(2,2+nSeasonal-1) * vecVt.rows(2,2+nSeasonal-1));
                    jacGe(0) = vectorG(0) / sigma;
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        jacGv(0,2+j) = -vectorG(0) * error * rowvecW(2+j) / (sigma*sigma);
                    }
                    double bPow = std::pow(vecVt(1), matrixF(1,1));
                    double phi1 = adamPowHelper(u, vectorG(1));
                    jacGv(1,1) = matrixF(1,1) * bPow * (phi1 - 1) / vecVt(1);
                    double dPhi1 = bPow * adamPowJacHelper(phi1, u, vectorG(1));
                    jacGe(1) = dPhi1 / fitted;
                    jacGy(1) = -dPhi1 * error / (fitted*fitted);
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        double phi = adamPowHelper(u, vectorG(2+j));
                        jacGv(2+j,2+j) = phi - 1;
                        double dPhi = vecVt(2+j) * adamPowJacHelper(phi, u, vectorG(2+j));
                        jacGe(2+j) = dPhi / fitted;
                        jacGy(2+j) = -dPhi * error / (fitted*fitted);
                    }
                    break;
                }
                }
                break;
            }
            break;
        }
        case 'M':{
            switch(T){
            case 'N':{
                // g0 = v0 (|1+e|^{G0} - 1)
                double phi0 = adamPowHelper(error, vectorG(0));
                jacGv(0,0) = phi0 - 1;
                jacGe(0) = vecVt(0) * adamPowJacHelper(phi0, error, vectorG(0));
                if(S=='A'){
                    // g_s = G_s yhat e
                    for(unsigned int j=1; j<=nSeasonal; j=j+1){
                        jacGe(j) = vectorG(j) * fitted;
                        jacGy(j) = vectorG(j) * error;
                    }
                }
                else if(S=='M'){
                    // g_s = v_s (|1+e|^{G_s} - 1)
                    for(unsigned int j=1; j<=nSeasonal; j=j+1){
                        double phi = adamPowHelper(error, vectorG(j));
                        jacGv(j,j) = phi - 1;
                        jacGe(j) = vecVt(j) * adamPowJacHelper(phi, error, vectorG(j));
                    }
                }
                break;
            }
            case 'A':{
                // g0 = (F00 l + F01 b)(|1+e|^{G0} - 1)
                double linearF = matrixF(0,0)*vecVt(0) + matrixF(0,1)*vecVt(1);
                double phi0 = adamPowHelper(error, vectorG(0));
                jacGv(0,0) = matrixF(0,0) * (phi0 - 1);
                jacGv(0,1) = matrixF(0,1) * (phi0 - 1);
                jacGe(0) = linearF * adamPowJacHelper(phi0, error, vectorG(0));
                switch(S){
                case 'N':
                case 'A':
                    // g1 = G1 yhat e; (S='A': g_s = G_s e, the base)
                    jacGe(1) = vectorG(1) * fitted;
                    jacGy(1) = vectorG(1) * error;
                    break;
                case 'M':{
                    // g1 = (F00 l + F01 b) G1 e; g_s = v_s (|1+e|^{G_s} - 1)
                    jacGe(1) = linearF * vectorG(1);
                    jacGv(1,0) = matrixF(0,0) * vectorG(1) * error;
                    jacGv(1,1) = matrixF(0,1) * vectorG(1) * error;
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        double phi = adamPowHelper(error, vectorG(2+j));
                        jacGv(2+j,2+j) = phi - 1;
                        jacGe(2+j) = vecVt(2+j) * adamPowJacHelper(phi, error, vectorG(2+j));
                    }
                    break;
                }
                }
                break;
            }
            case 'M':{
                // g0 = |l^{F00} b^{F01}| (|1+e|^{G0} - 1);
                // g1 = b^{F11} (|1+e|^{G1} - 1)
                double product0 = std::pow(std::abs(vecVt(0)), matrixF(0,0)) *
                                  std::pow(std::abs(vecVt(1)), matrixF(0,1));
                double phi0 = adamPowHelper(error, vectorG(0));
                jacGv(0,0) = matrixF(0,0) * product0 * (phi0 - 1) / vecVt(0);
                jacGv(0,1) = matrixF(0,1) * product0 * (phi0 - 1) / vecVt(1);
                jacGe(0) = product0 * adamPowJacHelper(phi0, error, vectorG(0));
                double bPow = std::pow(vecVt(1), matrixF(1,1));
                double phi1 = adamPowHelper(error, vectorG(1));
                jacGv(1,1) = matrixF(1,1) * bPow * (phi1 - 1) / vecVt(1);
                jacGe(1) = bPow * adamPowJacHelper(phi1, error, vectorG(1));
                if(S=='A'){
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        jacGe(2+j) = vectorG(2+j) * fitted;
                        jacGy(2+j) = vectorG(2+j) * error;
                    }
                }
                else if(S=='M'){
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        double phi = adamPowHelper(error, vectorG(2+j));
                        jacGv(2+j,2+j) = phi - 1;
                        jacGe(2+j) = vecVt(2+j) * adamPowJacHelper(phi, error, vectorG(2+j));
                    }
                }
                break;
            }
            }
            break;
        }
        }
    }
    else{
        // Conventional ETS: g = c(v) % G * e, with c built per branch (ones by
        // default). So dg/de = c % G, dg/dv row i = G_i * e * dc_i/dv, dg/dyhat = 0.
        arma::vec c(nETS, arma::fill::ones);
        arma::mat jacCv(nETS, nETS, arma::fill::zeros);

        switch(E){
        case 'A':
            switch(T){
            case 'N':
                if(S=='M'){
                    // c0 = 1/(w_s v_s); c_s = 1/v0
                    double sigma = arma::as_scalar(rowvecW.cols(1,nSeasonal) * vecVt.rows(1,nSeasonal));
                    c(0) = 1 / sigma;
                    for(unsigned int j=1; j<=nSeasonal; j=j+1){
                        jacCv(0,j) = -rowvecW(j) / (sigma*sigma);
                        c(j) = 1 / vecVt(0);
                        jacCv(j,0) = -1 / (vecVt(0)*vecVt(0));
                    }
                }
                break;
            case 'A':
                if(S=='M'){
                    // c_{0,1} = 1/prod(s^{w_s}); c_s = 1/(w01 v01)
                    double productS = 1;
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        productS *= std::pow(vecVt(2+j), rowvecW(2+j));
                    }
                    double linear = arma::as_scalar(rowvecW.cols(0,1) * vecVt.rows(0,1));
                    for(unsigned int i=0; i<2; i=i+1){
                        c(i) = 1 / productS;
                        for(unsigned int j=0; j<nSeasonal; j=j+1){
                            jacCv(i,2+j) = -rowvecW(2+j) / (productS * vecVt(2+j));
                        }
                    }
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        c(2+j) = 1 / linear;
                        jacCv(2+j,0) = -rowvecW(0) / (linear*linear);
                        jacCv(2+j,1) = -rowvecW(1) / (linear*linear);
                    }
                }
                break;
            case 'M':
                switch(S){
                case 'N':
                case 'A':
                    // c1 = 1/v0
                    c(1) = 1 / vecVt(0);
                    jacCv(1,0) = -1 / (vecVt(0)*vecVt(0));
                    break;
                case 'M':{
                    // c0 = 1/(w_s v_s); c1 = 1/(v0 prod(s^{w_s})); c_s = 1/(v0^{w0} v1^{w1})
                    double sigma = arma::as_scalar(rowvecW.cols(2,2+nSeasonal-1) * vecVt.rows(2,2+nSeasonal-1));
                    double productS = 1;
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        productS *= std::pow(vecVt(2+j), rowvecW(2+j));
                    }
                    double product01 = std::pow(vecVt(0), rowvecW(0)) * std::pow(vecVt(1), rowvecW(1));
                    c(0) = 1 / sigma;
                    c(1) = 1 / (vecVt(0) * productS);
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        jacCv(0,2+j) = -rowvecW(2+j) / (sigma*sigma);
                        jacCv(1,2+j) = -rowvecW(2+j) / (vecVt(0) * productS * vecVt(2+j));
                        c(2+j) = 1 / product01;
                        jacCv(2+j,0) = -rowvecW(0) / (product01 * vecVt(0));
                        jacCv(2+j,1) = -rowvecW(1) / (product01 * vecVt(1));
                    }
                    jacCv(1,0) = -1 / (vecVt(0)*vecVt(0) * productS);
                    break;
                }
                }
                break;
            }
            break;
        case 'M':
            switch(T){
            case 'N':
                if(S=='N' || S=='M'){
                    // c = v (elementwise)
                    c = vecVt.rows(0,nETS-1);
                    jacCv.eye(nETS, nETS);
                }
                else{
                    // S='A': c_i = v_i for i = 0..nSeasonal
                    for(unsigned int i=0; i<=nSeasonal; i=i+1){
                        c(i) = vecVt(i);
                        jacCv(i,i) = 1;
                    }
                }
                break;
            case 'A':
                switch(S){
                case 'N':{
                    // c_{0,1} = w01 v01
                    double linear = arma::as_scalar(rowvecW.cols(0,1) * vecVt.rows(0,1));
                    c(0) = linear; c(1) = linear;
                    jacCv(0,0) = rowvecW(0); jacCv(0,1) = rowvecW(1);
                    jacCv(1,0) = rowvecW(0); jacCv(1,1) = rowvecW(1);
                    break;
                }
                case 'A':{
                    // c_i = w v (all rows)
                    double linear = arma::as_scalar(rowvecW.cols(0,nETS-1) * vecVt.rows(0,nETS-1));
                    c.fill(linear);
                    for(unsigned int i=0; i<nETS; i=i+1){
                        jacCv.row(i) = rowvecW.cols(0,nETS-1);
                    }
                    break;
                }
                case 'M':{
                    // c_{0,1} = w01 v01; c_s = v_s
                    double linear = arma::as_scalar(rowvecW.cols(0,1) * vecVt.rows(0,1));
                    c(0) = linear; c(1) = linear;
                    jacCv(0,0) = rowvecW(0); jacCv(0,1) = rowvecW(1);
                    jacCv(1,0) = rowvecW(0); jacCv(1,1) = rowvecW(1);
                    for(unsigned int j=0; j<nSeasonal; j=j+1){
                        c(2+j) = vecVt(2+j);
                        jacCv(2+j,2+j) = 1;
                    }
                    break;
                }
                }
                break;
            case 'M':
                switch(S){
                case 'N':
                case 'M':{
                    // c_{0,1} = exp(F_{0:1,0:1} log v01): c_i = l^{F_i0} b^{F_i1}
                    for(unsigned int i=0; i<2; i=i+1){
                        double productF = std::pow(vecVt(0), matrixF(i,0)) *
                                          std::pow(vecVt(1), matrixF(i,1));
                        c(i) = productF;
                        jacCv(i,0) = matrixF(i,0) * productF / vecVt(0);
                        jacCv(i,1) = matrixF(i,1) * productF / vecVt(1);
                    }
                    if(S=='M'){
                        for(unsigned int j=0; j<nSeasonal; j=j+1){
                            c(2+j) = vecVt(2+j);
                            jacCv(2+j,2+j) = 1;
                        }
                    }
                    break;
                }
                case 'A':{
                    // Q = v0^{w0} v1^{w1} + w_s v_s; c_i = Q for i != 1; c1 = Q/v0
                    double product01 = std::pow(vecVt(0), rowvecW(0)) * std::pow(vecVt(1), rowvecW(1));
                    double q = product01 + arma::as_scalar(rowvecW.cols(2,nETS-1) * vecVt.rows(2,nETS-1));
                    arma::rowvec dq(nETS, arma::fill::zeros);
                    dq(0) = rowvecW(0) * product01 / vecVt(0);
                    dq(1) = rowvecW(1) * product01 / vecVt(1);
                    dq.cols(2,nETS-1) = rowvecW.cols(2,nETS-1);
                    for(unsigned int i=0; i<nETS; i=i+1){
                        c(i) = q;
                        jacCv.row(i) = dq;
                    }
                    c(1) = q / vecVt(0);
                    jacCv.row(1) = dq / vecVt(0);
                    jacCv(1,0) -= q / (vecVt(0)*vecVt(0));
                    break;
                }
                }
                break;
            }
            break;
        }

        jacGe = c % vectorG.rows(0,nETS-1);
        for(unsigned int i=0; i<nETS; i=i+1){
            jacGv.row(i) = jacCv.row(i) * vectorG(i) * error;
        }
        // Conventional branches never use yhat: jacGy stays zero.
    }
}
