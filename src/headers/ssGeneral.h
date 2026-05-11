#pragma once

#include <limits>
#include "ssOccurrence.h"

// Define R_PosInf for Python builds
#ifndef R_PosInf
#define R_PosInf std::numeric_limits<double>::infinity()
#endif

/* # Function is needed to estimate the correct error for ETS when multisteps model selection with r(matvt) is sorted out. */
inline arma::mat matrixPower(arma::mat const &A, int const &power){
    arma::mat B(A.n_rows, A.n_rows, arma::fill::eye);

    if(power!=0){
        for(int i=0; i<power; ++i){
            B = B * A;
        }
    }
    return B;
}

/* # Function returns multiplicative or additive error for scalar.
 * Optional ot and O parameters extend it to occurrence models:
 * - O='n' (default): demand model; ot==0 returns 0 immediately.
 * - O='d'/'o'/'i': dispatches to occurrenceError() (yact IS the occurrence indicator). */
inline double errorf(double const &yact, double &yfit, char const &E,
                     double const &ot = 1.0, char const &O = 'n') {
    if(O == 'n') {
        if(ot == 0) { return 0; }
        if(E=='A') { return yact - yfit; }
        if((yact==0) & (yfit==0)) { return 0; }
        if((yact!=0) & (yfit==0)) {
#ifdef PYTHON_BUILD
            return std::numeric_limits<double>::infinity();
#else
            return R_PosInf;
#endif
        }
        return (yact - yfit) / yfit;
    }
    // Occurrence types d/o/i: yact IS the occurrence indicator (ot)
    return occurrenceError(yact, yfit, 1.0, E, 'M', O)[0];
}

/* # Function is needed to estimate the correct error for ETS when multisteps model selection with r(matvt) is sorted out. */
inline arma::mat errorvf(arma::mat yact, arma::mat yfit, char const &E){
    if(E=='A'){
        return yact - yfit;
    }
    else{
        yfit.elem(find(yfit==0)).fill(1e-100);
        return (yact - yfit) / yfit;
    }
}

/* # Function allows to multiply polinomails */
inline arma::vec polyMult(arma::vec const &poly1, arma::vec const &poly2){

    int poly1Nonzero = arma::as_scalar(find(poly1,1,"last"));
    int poly2Nonzero = arma::as_scalar(find(poly2,1,"last"));

    arma::vec poly3(poly1Nonzero + poly2Nonzero + 1, arma::fill::zeros);

    for(int i = 0; i <= poly1Nonzero; ++i){
        for(int j = 0; j <= poly2Nonzero; ++j){
            poly3(i+j) += poly1(i) * poly2(j);
        }
    }

    return poly3;
}

/* # Function returns value of w() -- y-fitted -- used in the measurement equation */
inline double wvalue(arma::vec const &vecVt, arma::rowvec const &rowvecW, char const &E, char const &T, char const &S,
              arma::rowvec const &rowvecXt, arma::vec const &vecAt){
// vecVt is a vector here!
    double yfit = 0;
    arma::mat vecYfit;

    switch(S){
// ZZN
    case 'N':
        switch(T){
        case 'N':
        case 'A':
            vecYfit = rowvecW * vecVt;
        break;
        case 'M':
            vecYfit = exp(rowvecW * log(vecVt));
        break;
        }
    break;
// ZZA
    case 'A':
        switch(T){
        case 'N':
        case 'A':
            vecYfit = rowvecW * vecVt;
        break;
        case 'M':
            vecYfit = exp(rowvecW.cols(0,1) * log(vecVt.rows(0,1))) + vecVt(2);
        break;
        }
    break;
// ZZM
    case 'M':
        switch(T){
        case 'N':
        case 'M':
            vecYfit = exp(rowvecW * log(vecVt));
        break;
        case 'A':
            vecYfit = rowvecW.cols(0,1) * vecVt.rows(0,1) * vecVt(2);
        break;
        }
    break;
    }

    switch(E){
        case 'A':
        case 'D':
            yfit = as_scalar(vecYfit + rowvecXt * vecAt);
        break;
        case 'M':
        case 'L':
            yfit = as_scalar(vecYfit * exp(rowvecXt * vecAt));
        break;
    }

    return yfit;
}

/* # Function returns value of r() -- additive or multiplicative error -- used in the error term of measurement equation.
     This is mainly needed by sim.ets */
inline double rvalue(arma::vec const &vecVt, arma::rowvec const &rowvecW, char const &E, char const &T, char const &S,
              arma::rowvec const &rowvecXt, arma::vec const &vecAt){

    switch(E){
// MZZ
    case 'M':
        return wvalue(vecVt, rowvecW, E, T, S, rowvecXt, vecAt);
    break;
// AZZ
    case 'A':
    default:
        return 1.0;
    }
}

/* # Function returns value of f() -- new states without the update -- used in the transition equation */
inline arma::vec fvalue(arma::vec const &matrixVt, arma::mat const &matrixF, char const T, char const S){
    arma::vec matrixVtnew = matrixVt;

    switch(S){
// ZZN
    case 'N':
        switch(T){
        case 'N':
        case 'A':
            matrixVtnew = matrixF * matrixVt;
        break;
        case 'M':
            matrixVtnew = exp(matrixF * log(matrixVt));
        break;
        }
    break;
// ZZA
    case 'A':
        switch(T){
        case 'N':
        case 'A':
            matrixVtnew = matrixF * matrixVt;
        break;
        case 'M':
            matrixVtnew.rows(0,1) = exp(matrixF.submat(0,0,1,1) * log(matrixVt.rows(0,1)));
            matrixVtnew(2) = matrixVt(2);
        break;
        }
    break;
// ZZM
    case 'M':
        switch(T){
        case 'N':
        case 'M':
            matrixVtnew = exp(matrixF * log(matrixVt));
        break;
        case 'A':
            matrixVtnew = matrixF * matrixVt;
        break;
        }
    break;
    }

    return matrixVtnew;
}

/* # Function returns value of g() -- the update of states -- used in components estimation for the persistence */
inline arma::vec gvalue(arma::vec const &matrixVt, arma::mat const &matrixF, arma::mat const &rowvecW, char const &E, char const &T, char const &S){
    arma::vec g(matrixVt.n_rows, arma::fill::ones);

// AZZ
    switch(E){
    case 'A':
    case 'D':
// ANZ
        switch(T){
        case 'N':
            switch(S){
            case 'M':
                g(0) = 1 / matrixVt(1);
                g(1) = 1 / matrixVt(0);
            break;
            }
        break;
// AAZ
        case 'A':
            switch(S){
            case 'M':
                g.rows(0,1) = g.rows(0,1) / matrixVt(2);
                g(2) = 1 / as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1));
            break;
            }
        break;
// AMZ
        case 'M':
            switch(S){
            case 'N':
            case 'A':
                g(1) = g(1) / matrixVt(0);
            break;
            case 'M':
                g(0) = g(0) / matrixVt(2);
                g(1) = g(1) / (matrixVt(0) * matrixVt(2));
                g(2) = g(2) / as_scalar(exp(rowvecW.cols(0,1) * log(matrixVt.rows(0,1))));
            break;
            }
        break;
        }
    break;
// MZZ
    case 'M':
    case 'L':
// MNZ
        switch(T){
        case 'N':
            switch(S){
            case 'N':
                g(0) = matrixVt(0);
            break;
            case 'A':
                g.rows(0,1).fill(matrixVt(0) + matrixVt(1));
            break;
            case 'M':
                g = matrixVt;
            break;
            }
        break;
// MAZ
        case 'A':
            switch(S){
            case 'N':
            case 'A':
                g.fill(as_scalar(rowvecW * matrixVt));
            break;
            case 'M':
                g.rows(0,1).fill(as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1)));
                g(2) = matrixVt(2);
            break;
            }
        break;
// MMZ
        case 'M':
            switch(S){
            case 'N':
                g(0) = as_scalar(exp(rowvecW * log(matrixVt)));
                g(1) = pow(matrixVt(1),rowvecW(1));
            break;
            case 'A':
                g.rows(0,2).fill(as_scalar(exp(rowvecW.cols(0,1) * log(matrixVt.rows(0,1))) + matrixVt(2)));
                g(1) = g(0) / matrixVt(0);
            break;
            case 'M':
                g = exp(matrixF * log(matrixVt));
            break;
            }
        break;
        }
    break;
    }

    return g;
}

/* # Function is needed for the proper xreg update in additive / multiplicative models.*/
// Make sure that the provided vecXt is already in logs or whatever!
inline arma::vec gXvalue(arma::vec const &vecXt, arma::vec const &vecGX, arma::vec const &error, char const &E){
    arma::vec bufferforat(vecGX.n_rows);

    switch(E){
    case 'A':
    case 'D':
        bufferforat = vecGX / vecXt * error;
    break;
    case 'M':
    case 'L':
        bufferforat = vecGX / vecXt * log(1 + error);
    break;
    }
    bufferforat.elem(find_nonfinite(bufferforat)).fill(0);

    return bufferforat;
}

/* # Function is needed for the renormalisation of seasonal components. It should be done seasonal-wise.*/
inline arma::mat normaliser(arma::mat Vt, int &obsall, unsigned int &maxlag, char const &S, char const &T){

    unsigned int nComponents = Vt.n_rows;
    double meanseason = 0;

    switch(S){
    case 'A':
        meanseason = mean(Vt.row(nComponents-1));
        Vt.row(nComponents-1) = Vt.row(nComponents-1) - meanseason;
        switch(T){
        case 'N':
        case 'A':
            Vt.row(0) = Vt.row(0) + meanseason;
        break;
        case 'M':
            Vt.row(0) = Vt.row(0) + meanseason / Vt.row(1);
        break;
        }
    break;
    case 'M':
        meanseason = exp(mean(log(Vt.row(nComponents-1))));
        Vt.row(nComponents-1) = Vt.row(nComponents-1) / meanseason;
        switch(T){
        case 'N':
        case 'M':
            Vt.row(0) = Vt.row(0) / meanseason;
        break;
        case 'A':
            Vt.row(0) = Vt.row(0) * meanseason;
            Vt.row(1) = Vt.row(1) * meanseason;
        break;
        }
    break;
    }

    return(Vt);
}



/* # The function checks for the bounds conditions and returns either zeo (everything is fine) or a huge number (not) */
inline double boundsTester(char const &boundtype, char const &T, char const &S,
                           arma::vec const &vecG, arma::rowvec const &rowvecW, arma::mat const &matrixF){

    if(boundtype=='u'){
// alpha in (0,1)
        if((vecG(0)>1) || (vecG(0)<0)){
            return 1E+300;
        }
        if(T!='N'){
// beta in (0,alpha)
            if((vecG(1)>vecG(0)) || (vecG(1)<0)){
                return 1E+300;
            }
            if(S!='N'){
// gamma in (0,1-alpha)
                if((vecG(2)>(1-vecG(0))) || (vecG(2)<0)){
                    return 1E+300;
                }
            }
        }
        if(S!='N'){
// gamma in (0,1-alpha)
            if((vecG(1)>(1-vecG(0))) || (vecG(1)<0)){
                return 1E+300;
            }
        }
    }
    else if((boundtype=='a') | (boundtype=='r')){
        // Values needed for eigenvalues calculation
        arma::cx_vec eigval;
        if(arma::eig_gen(eigval, matrixF - vecG * rowvecW)){
            if(max(abs(eigval))> (1 + 1E-50)){
                return max(abs(eigval))*1E+100;
            }
        }
        else{
            return 1E+300;
        }
    }

    return 0;
}
