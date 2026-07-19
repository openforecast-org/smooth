#pragma once

#include "olsCore.h"

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

// ---------- Occurrence-model helpers (initial="gradient" for om) ----------
//
// The om cost is a function of the fitted probability p = link(yhat)
// (omLinkFunction in R/om.R) and the binary occurrence o. Both the link and
// the occurrenceError() update channel (ssOccurrence.h) need derivatives with
// respect to yhat so the state sensitivities propagate through the occurrence
// recursion. Occurrence types: 'd' direct, 'o' odds-ratio, 'i' inverse-odds.

// p and dp/dyhat for the occurrence link. The clamp of the direct model has
// derivative zero on its flat spots (same convention as the mixed-model clamp).
inline void occurrenceLinkJac(double const &yFit, char const &E, char const &O,
                              double &p, double &dpdy){
    double const a = (E=='A') ? std::exp(yFit) : yFit;
    switch(O){
    case 'o':
        p = a / (1 + a);
        dpdy = (E=='A') ? p * (1 - p) : 1 / ((1 + a) * (1 + a));
        break;
    case 'i':
        p = 1 / (1 + a);
        dpdy = (E=='A') ? -p * (1 - p) : -1 / ((1 + a) * (1 + a));
        break;
    default:
        // 'd': p = clamp(yhat, 0, 1)
        p = std::max(std::min(yFit, 1.0), 0.0);
        dpdy = (yFit > 0 && yFit < 1) ? 1.0 : 0.0;
    }
}

// d(e)/d(yhat) for the occurrence error (mirrors occurrenceError() branch by
// branch; e is the update-channel error the state recursion consumes).
inline double occurrenceErrorJac(double const &yAct, double const &yFit,
                                 char const &E, char const &O){
    double p, dpdy;
    occurrenceLinkJac(yFit, E, O, p, dpdy);
    if(O=='d'){
        if(E=='M'){
            // e = (c - p)/p with c = o(1-2k)+k
            double const kappa = 1E-10;
            double const c = yAct * (1 - 2 * kappa) + kappa;
            return -c / (p * p) * dpdy;
        }
        // E='A': e = o - p
        return -dpdy;
    }
    // 'o' / 'i': e is a transform of u = (1 + o - p)/2, du/dyhat = -dpdy/2
    double const u = (1 + yAct - p) / 2;
    double dedu;
    if(O=='o'){
        // q = u/(1-u); e = q - 1 (E='M') or log(q) (E='A')
        dedu = (E=='M') ? 1 / ((1 - u) * (1 - u)) : 1 / (u * (1 - u));
    }
    else{
        // 'i': q = (1-u)/u; e = q - 1 (E='M') or log(q) (E='A')
        dedu = (E=='M') ? -1 / (u * u) : -1 / (u * (1 - u));
    }
    return dedu * (-dpdy / 2);
}

// ---------- Loss helpers for the arbitrary-loss gradient solve ----------
//
// At fixed persistence (and concentrated scale) every supported estimation
// loss reduces to sum(rho(e_t)) over the errorf residuals, so the initial
// states can be profiled under the actual loss instead of the one-step SSE
// surrogate. rho and its derivatives below mirror the R-side loss formulas
// (CF and adam_scaler) up to positive factors and additive constants, which
// drop out of the argmin. Loss codes (kept mirror-identical in the R and
// Python wrapper mapping tables):
//   'S' SSE (default; MSE / likelihood+dnorm)      rho = e^2
//   'A' MAE (MAE / likelihood+dlaplace)            rho = |e|
//   'H' HAM (HAM / likelihood+ds)                  rho = sqrt(|e|)
//   'G' generalised normal (likelihood+dgnorm)     rho = |e|^beta
//   'l' log-normal, E='M' (likelihood+dlnorm)      rho = (log(1+e)+sigma^2/2)^2
//   'g' gamma, E='M' (likelihood+dgamma)           rho = (1+e) - log(1+e)
//   'i' inv. gaussian, E='M' (likelihood+dinvgauss) rho = e^2/(1+e) + s^2 log(1+e)
//   'B' Bernoulli (om likelihood)                   rho = -log(1-|r|)
//       (r = o - p is the probability residual; with o in {0,1} the Bernoulli
//        log-likelihood -[o log(p) + (1-o) log(1-p)] is exactly -log(1-|r|),
//        so it is separable in r like every other rho here)
//   'h' MSEh, 'T' TMSE, 't' GTMSE, 'C' MSCE, 'P' GPL: additive multistep losses
// The scale argument carries sigma for 'l' and the dispersion s^2 for 'i',
// refreshed from the current residuals via gradientLossScale(); 'g' is
// scale-free in the argmin (the dispersion only multiplies its rho).

inline bool gradientLossMultistep(char const &L){
    return L=='h' || L=='T' || L=='t' || L=='C' || L=='P';
}

inline double gradientLossRho(double const &e, char const &L,
                              double const &beta, double const &scale){
    switch(L){
    case 'A':
        return std::abs(e);
    case 'H':
        return std::sqrt(std::abs(e));
    case 'G':
        return std::pow(std::abs(e), beta);
    case 'l':{
        double const u = std::log(std::abs(1+e)) + scale*scale/2;
        return u*u;
    }
    case 'g':
        return (1+e) - std::log(std::abs(1+e));
    case 'i':
        return e*e/(1+e) + scale*std::log(std::abs(1+e));
    case 'B':
        // Non-finite for |e| >= 1 (saturated probability) by design: the
        // acceptance checks reject such candidates and the caller falls back.
        return -std::log(1 - std::abs(e));
    default:
        return e*e;
    }
}

// Sum of rho over the residuals. The 'S' case keeps arma::dot so the default
// SSE path stays bit-identical to the pre-loss-aware implementation.
inline double gradientLossSum(arma::vec const &errors, char const &L,
                              double const &beta, double const &scale){
    if(L=='S'){
        return arma::dot(errors, errors);
    }
    double total = 0;
    for(arma::uword i=0; i<errors.n_elem; i=i+1){
        total += gradientLossRho(errors(i), L, beta, scale);
    }
    return total;
}

// Concentrated scale, mirroring adam_scaler(): sigma for 'l', s^2 for 'i'.
// Residuals at skipped (ot==0) points are exactly zero and contribute nothing,
// matching the R-side sum over the observed points divided by obsInSample.
inline double gradientLossScale(arma::vec const &errors, char const &L){
    double const obs = errors.n_elem;
    double total = 0;
    switch(L){
    case 'l':
        for(arma::uword i=0; i<errors.n_elem; i=i+1){
            double const u = std::log(std::abs(1+errors(i)));
            total += u*u;
        }
        return std::sqrt(2*std::abs(1-std::sqrt(std::abs(1-total/obs))));
    case 'i':
        for(arma::uword i=0; i<errors.n_elem; i=i+1){
            total += errors(i)*errors(i)/(1+errors(i));
        }
        return total/obs;
    default:
        return 0;
    }
}

// Row scaling (a) and right-hand side (b) of the loss-aware step system: the
// least-squares solve of (a % J) d = b yields the IRLS / weighted Gauss-Newton
// step for the power losses (weight |e|^{p-2}; exact-zero residuals sit at
// their subgradient optimum and are dropped -- no epsilon floor) and the
// Gauss-Newton-Newton step for the likelihood losses (rows where rho'' is not
// positive are dropped). 'S' keeps a=1, b=e: the plain Gauss-Newton system.
inline void gradientLossStepRow(double const &e, char const &L,
                                double const &beta, double const &scale,
                                double &a, double &b){
    switch(L){
    case 'A':
    case 'H':
    case 'G':{
        double const p = (L=='A') ? 1.0 : ((L=='H') ? 0.5 : beta);
        if(e==0){
            a = 0;
            b = 0;
        }
        else{
            a = std::sqrt(std::pow(std::abs(e), p-2));
            b = a*e;
            if(!std::isfinite(a) || !std::isfinite(b)){
                a = 0;
                b = 0;
            }
        }
        break;
    }
    case 'l':
    case 'g':
    case 'i':
    case 'B':{
        double psi, psiPrime;
        if(L=='l'){
            double const u = std::log(std::abs(1+e)) + scale*scale/2;
            psi = 2*u/(1+e);
            psiPrime = 2*(1-u)/((1+e)*(1+e));
        }
        else if(L=='g'){
            psi = e/(1+e);
            psiPrime = 1/((1+e)*(1+e));
        }
        else if(L=='B'){
            // rho = -log(1-|e|): psi = sign(e)/(1-|e|), psi' = 1/(1-|e|)^2
            double const oneAbs = 1 - std::abs(e);
            psi = ((e>0) - (e<0)) / oneAbs;
            psiPrime = 1/(oneAbs*oneAbs);
        }
        else{
            psi = e*(2+e)/((1+e)*(1+e)) + scale/(1+e);
            psiPrime = 2/((1+e)*(1+e)*(1+e)) - scale/((1+e)*(1+e));
        }
        if(psiPrime<=0 || !std::isfinite(psi) || !std::isfinite(psiPrime)){
            a = 0;
            b = 0;
        }
        else{
            a = std::sqrt(psiPrime);
            b = psi/a;
        }
        break;
    }
    default:
        a = 1;
        b = e;
    }
}

// IRLS sweeps for the separable losses on the additive affine surrogate:
// e(theta) = e0 - D*theta with a constant design D, so every sweep is one
// weighted QR -- no further model passes. Sweeps are accepted only while the
// loss decreases (the majorisation holds for p <= 2; the guard also covers
// dgnorm with beta > 2) and stop on diminishing returns.
inline arma::vec gradientLossIrls(arma::mat const &design, arma::vec const &residuals,
                                  arma::vec theta, char const &L, double const &beta){
    double rhoCurrent = gradientLossSum(residuals - design*theta, L, beta, 0);
    arma::vec rowScale(residuals.n_elem);
    for(int sweep=0; sweep<100; sweep=sweep+1){
        arma::vec const e = residuals - design*theta;
        double a, b;
        for(arma::uword i=0; i<e.n_elem; i=i+1){
            gradientLossStepRow(e(i), L, beta, 0, a, b);
            rowScale(i) = a;
        }
        arma::vec thetaNew = olsCore(design.each_col() % rowScale,
                                     residuals % rowScale, 1e-7);
        thetaNew.elem(arma::find_nonfinite(thetaNew)).zeros();
        double const rhoNew = gradientLossSum(residuals - design*thetaNew, L, beta, 0);
        if(!(rhoNew < rhoCurrent)){
            break;
        }
        theta = thetaNew;
        if(rhoNew > rhoCurrent - 1e-8*std::abs(rhoCurrent)){
            break;
        }
        rhoCurrent = rhoNew;
    }
    return theta;
}

// Solvers for the additive multistep losses on the affine multistep surrogate.
// msDesign / msResiduals are stacked origin-major: row idx*h + k holds the
// (k+1)-step-ahead residual from origin idx and its design d(yhat)/d(theta),
// exactly replicating the ferrors() recursion the cost function evaluates.
// Everything below is pure linear algebra on this stack -- no model passes.
inline arma::vec gradientLossMultistepSolve(arma::mat const &msDesign,
                                            arma::vec const &msResiduals,
                                            char const &L, int const &hor,
                                            int const &nOrigins){
    unsigned int const nFree = msDesign.n_cols;

    // MSEh: quadratic in theta on the h-step rows only -- one QR.
    if(L=='h'){
        arma::uvec const rowsH =
            arma::regspace<arma::uvec>(hor-1, hor, nOrigins*hor-1);
        return olsCore(msDesign.rows(rowsH), msResiduals(rowsH), 1e-7);
    }

    // MSCE: the cumulative error is the within-origin row sum -- one QR.
    if(L=='C'){
        arma::mat designC(nOrigins, nFree);
        arma::vec residualsC(nOrigins);
        for(int idx=0; idx<nOrigins; idx=idx+1){
            designC.row(idx) = arma::sum(msDesign.rows(idx*hor, idx*hor+hor-1), 0);
            residualsC(idx) = arma::accu(msResiduals.rows(idx*hor, idx*hor+hor-1));
        }
        return olsCore(designC, residualsC, 1e-7);
    }

    // TMSE (also the starting point for GTMSE and GPL): the full stack -- one QR.
    arma::vec theta = olsCore(msDesign, msResiduals, 1e-7);
    theta.elem(arma::find_nonfinite(theta)).zeros();
    if(L=='T'){
        return theta;
    }

    // GTMSE: sum(log(SSE_k)). The log is majorised by its tangent, so each MM
    // sweep is a per-horizon weighted QR with weights 1/SSE_k -- monotone.
    if(L=='t'){
        arma::vec rowScale(nOrigins*hor);
        arma::vec sse(hor);
        double objective = arma::datum::inf;
        for(int sweep=0; sweep<30; sweep=sweep+1){
            arma::vec const e = msResiduals - msDesign*theta;
            sse.zeros();
            for(int idx=0; idx<nOrigins; idx=idx+1){
                for(int k=0; k<hor; k=k+1){
                    sse(k) += e(idx*hor+k)*e(idx*hor+k);
                }
            }
            if(sse.min() <= 0){
                break;
            }
            double const objectiveNew = arma::accu(arma::log(sse));
            if(!(objectiveNew < objective)){
                break;
            }
            objective = objectiveNew;
            for(int idx=0; idx<nOrigins; idx=idx+1){
                for(int k=0; k<hor; k=k+1){
                    rowScale(idx*hor+k) = 1/std::sqrt(sse(k));
                }
            }
            arma::vec thetaNew = olsCore(msDesign.each_col() % rowScale,
                                         msResiduals % rowScale, 1e-7);
            thetaNew.elem(arma::find_nonfinite(thetaNew)).zeros();
            theta = thetaNew;
        }
        return theta;
    }

    // GPL: log(det(E'E)). log det is concave in the covariance, so the tangent
    // majorisation turns each MM sweep into a GLS whitened by chol(Omega)^{-T},
    // starting from the TMSE solution (the Omega = I sweep). Monotone.
    if(L=='P'){
        double objective = arma::datum::inf;
        arma::mat designW(nOrigins*hor, nFree);
        arma::vec residualsW(nOrigins*hor);
        for(int sweep=0; sweep<30; sweep=sweep+1){
            arma::vec const e = msResiduals - msDesign*theta;
            arma::mat const errorsMat =
                arma::reshape(e, hor, nOrigins);
            arma::mat const omega = errorsMat * errorsMat.t();
            arma::mat cholUpper;
            if(!arma::chol(cholUpper, omega)){
                break;
            }
            // Numerically singular covariance: the profiled GPL is at the edge
            // of its domain and further whitened sweeps are meaningless.
            if(cholUpper.diag().min() < 1e-8 * cholUpper.diag().max()){
                break;
            }
            double const objectiveNew = 2*arma::accu(arma::log(cholUpper.diag()));
            if(!(objectiveNew < objective)){
                break;
            }
            objective = objectiveNew;
            for(int idx=0; idx<nOrigins; idx=idx+1){
                designW.rows(idx*hor, idx*hor+hor-1) =
                    arma::solve(arma::trimatl(cholUpper.t()),
                                msDesign.rows(idx*hor, idx*hor+hor-1));
                residualsW.rows(idx*hor, idx*hor+hor-1) =
                    arma::solve(arma::trimatl(cholUpper.t()),
                                msResiduals.rows(idx*hor, idx*hor+hor-1));
            }
            arma::vec thetaNew = olsCore(designW, residualsW, 1e-7);
            thetaNew.elem(arma::find_nonfinite(thetaNew)).zeros();
            theta = thetaNew;
        }
        return theta;
    }

    return theta;
}
