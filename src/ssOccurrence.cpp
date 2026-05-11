#include <RcppArmadillo.h>
#include <iostream>
#include "headers/ssGeneral.h"
#include "headers/ssOccurrence.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// # Fitter for the occurrence part of the univariate models. This does not include iETS_G model
List occurenceFitter(arma::mat &matrixVt, arma::mat const &matrixF, arma::rowvec const &rowvecW, arma::vec const &vecG,
                     arma::vec const &vecOt, arma::uvec &lags, char const &E, char const &T, char const &S, char const &O,
                     arma::mat const &matrixXt, arma::mat &matrixAt, arma::mat const &matrixFX, arma::vec const &vecGX){
    /* # matrixVt should have a length of obs + maxlag.
    * # rowvecW should have 1 row.
    * # vecG should be a vector
    * # lags is a vector of lags
    * # matrixXt is the matrix with the exogenous variables
    * # matrixAt is the matrix with the parameters for the exogenous
    */

    arma::mat matrixXtTrans = matrixXt.t();

    int obs = vecOt.n_rows;
    int obsall = matrixVt.n_cols;
    unsigned int nComponents = matrixVt.n_rows;
    arma::uvec lagsInternal = lags;
    unsigned int maxlag = max(lagsInternal);
    int lagslength = lagsInternal.n_rows;

    lagsInternal = lagsInternal * nComponents;

    for(int i=0; i<lagslength; i=i+1){
        lagsInternal(i) = lagsInternal(i) + (lagslength - i - 1);
    }

    arma::uvec lagrows(lagslength, arma::fill::zeros);

    // The vector for the fitted values
    arma::vec vecYfit(obs, arma::fill::zeros);
    // The vector for the fitted probabilities
    arma::vec vecPfit(obs, arma::fill::zeros);
    // The vector of errors
    arma::vec vecErrors(obs, arma::fill::zeros);
    // The buffer for the calculation of transition for explanatory part
    arma::vec bufferforat(vecGX.n_rows);
    // The warning level:
    // true - the states became negative or the value was rediculous, so we substituted it with the previous one;
    bool warning = false;

    for(unsigned int i=maxlag; i<obs+maxlag; i=i+1) {
        lagrows = i * nComponents - lagsInternal + nComponents - 1;

/* # Measurement equation and the error term */
        vecYfit(i-maxlag) = wvalue(matrixVt(lagrows), rowvecW, E, T, S,
                                   matrixXt.row(i-maxlag), matrixAt.col(i-1));

        // This is a failsafe for cases of ridiculously high and ridiculously low values
        if((vecYfit(i-maxlag) > 1e+100) || (vecYfit(i-maxlag) < -1e+100)){
            warning = true;
            vecYfit(i-maxlag) = vecYfit(i-maxlag-1);
        }

        // For O==c("o","i","d") the bFit is set to 1 and aFit is the variable under consideration
        vecErrors(i-maxlag) = occurrenceError(vecOt(i-maxlag), vecYfit(i-maxlag), 1.0, E, 'M', O)[0];

/* # Transition equation */
        matrixVt.col(i) = fvalue(matrixVt(lagrows), matrixF, T, S) +
                          gvalue(matrixVt(lagrows), matrixF, rowvecW, E, T, S) % vecG * vecErrors(i-maxlag);

/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVt.col(i).is_finite()){
            warning = true;
            matrixVt.col(i) = matrixVt(lagrows);
        }
        if((S=='M') && (matrixVt(nComponents-1,i) <= 0)){
            warning = true;
            matrixVt(nComponents-1,i) = arma::as_scalar(matrixVt(lagrows.row(nComponents-1)));
        }
        if(T=='M' && ((matrixVt(0,i) <= 0) || (matrixVt(1,i) <= 0))){
                warning = true;
                matrixVt(0,i) = arma::as_scalar(matrixVt(lagrows.row(0)));
                matrixVt(1,i) = arma::as_scalar(matrixVt(lagrows.row(1)));
        }
        if(any(matrixVt.col(i)>1e+100)){
            warning = true;
            matrixVt.col(i) = matrixVt(lagrows);
        }

/* Renormalise components if the seasonal model is chosen */
        if(S!='N'){
            if(double(i+1) / double(maxlag) == double((i+1) / maxlag)){
                matrixVt.cols(i-maxlag+1,i) = normaliser(matrixVt.cols(i-maxlag+1,i), obsall, maxlag, S, T);
            }
        }

/* # Transition equation for xreg */
        bufferforat = gXvalue(matrixXtTrans.col(i-maxlag), vecGX, vecErrors.row(i-maxlag), E);
        matrixAt.col(i) = matrixFX * matrixAt.col(i-1) + bufferforat;
    }

    for (int i=obs+maxlag; i<obsall; i=i+1) {
        lagrows = i * nComponents - lagsInternal + nComponents - 1;
        matrixVt.col(i) = fvalue(matrixVt(lagrows), matrixF, T, S);
        matrixAt.col(i) = matrixFX * matrixAt.col(i-1);

/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVt.col(i).is_finite()){
            warning = true;
            matrixVt.col(i) = matrixVt(lagrows);
        }
        if((S=='M') && (matrixVt(matrixVt.n_rows-1,i) <= 0)){
            warning = true;
            matrixVt(matrixVt.n_rows-1,i) = arma::as_scalar(matrixVt(lagrows.row(matrixVt.n_rows-1)));
        }
        if(T=='M' && ((matrixVt(0,i) <= 0) || (matrixVt(1,i) <= 0))){
                warning = true;
                matrixVt(0,i) = arma::as_scalar(matrixVt(lagrows.row(0)));
                matrixVt(1,i) = arma::as_scalar(matrixVt(lagrows.row(1)));
        }
    }

    vecPfit = vecYfit;
    if(E=='M' && any(vecPfit<0)){
        warning = true;
        vecPfit.elem(find(vecPfit<0)).fill(1E-10);
    }

    switch(O){
        case 'o':
            switch(E){
                case 'A':
                    vecPfit = exp(vecPfit) / (1+exp(vecPfit));
                    // This is needed for cases when huge numbers were generated
                    // vecPfit.elem(find_nonfinite(vecPfit)).replace(1.0-1E-10);
                    vecPfit.replace(NA_REAL, 1.0-1E-10);
                break;
                case 'M':
                    vecPfit = vecPfit / (1+vecPfit);
                break;
            }
        break;
        case 'i':
            switch(E){
                case 'A':
                    vecPfit = 1 / (1+exp(vecPfit));
                    // This is needed for cases when huge numbers were generated
                    // vecPfit.elem(find_nonfinite(vecPfit)).replace(1.0-1E-10);
                    vecPfit.replace(NA_REAL, 1.0-1E-10);
                break;
                case 'M':
                    vecPfit = 1 / (1+vecPfit);
                break;
            }
        break;
        case 'd':
            vecPfit = vecYfit;
            // This is not correct statistically. See the (50) - (52) in order to see how this needs to be done properly.
            // But this works and I don't have time to do that 100% correctly.
            vecPfit.elem(find(vecPfit>1)).fill(1.0-1E-10);
            vecPfit.elem(find(vecPfit<0)).fill(1E-10);
        break;
    }

    return List::create(Named("matvt") = matrixVt, Named("yfit") = vecYfit, Named("pfit") = vecPfit,
                        Named("errors") = vecErrors, Named("matat") = matrixAt, Named("warning") = warning);
}

/* # Wrapper for fitter */
// [[Rcpp::export]]
RcppExport SEXP occurenceFitterWrap(SEXP matvt, SEXP matF, SEXP matw, SEXP vecg, SEXP ot,
                                    SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP Otype,
                                    SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX) {

    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    NumericMatrix ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.nrow(), false);

    NumericMatrix vecg_n(vecg);
    arma::vec vecG(vecg_n.begin(), vecg_n.nrow(), false);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    char O = as<char>(Otype);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow(), false);

    return wrap(occurenceFitter(matrixVt, matrixF, rowvecW, vecG, vecOt, lags, E, T, S, O,
                                matrixXt, matrixAt, matrixFX, vecGX));
}

/* # Function returns the chosen Cost Function based on the chosen model and produced errors */
double occurrenceOptimizer(arma::mat &matrixVt, arma::mat const &matrixF, arma::rowvec const &rowvecW,
                           arma::vec const &vecG, arma::vec const &vecOt,
                           arma::uvec &lags, char const &E, char const &T, char const &S, char const &O,
                           arma::mat const &matrixXt, arma::mat &matrixAt, arma::mat const &matrixFX, arma::vec const &vecGX){

    List fitting = occurenceFitter(matrixVt, matrixF, rowvecW, vecG, vecOt, lags, E, T, S, O,
                                   matrixXt, matrixAt, matrixFX, vecGX);

    NumericMatrix pfitfromfit = as<NumericMatrix>(fitting["pfit"]);
    arma::vec vecPfit = as<arma::vec>(pfitfromfit);

    // 0.5 is needed for cases, when the variable is continuous in (0, 1)
    double CFres = -sum(log(vecPfit.elem(find(vecOt>=0.5)))) - sum(log(1-vecPfit.elem(find(vecOt<0.5))));

    return CFres;
}

/* # Function is used for the occurrence model of the types "I", "O" and "D"
# If bounds are violated, it returns variance of ot. */
// [[Rcpp::export]]
RcppExport SEXP occurrenceOptimizerWrap(SEXP matvt, SEXP matF, SEXP matw, SEXP vecg, SEXP ot,
                                        SEXP modellags, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP Otype,
                                        SEXP matxt, SEXP matat, SEXP matFX, SEXP vecgX,
                                        SEXP bounds) {
/* Function is needed to implement admissible constrains on smoothing parameters */
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol());

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::rowvec rowvecW(matw_n.begin(), matw_n.ncol(), false);

    NumericMatrix ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.nrow(), false);

    NumericMatrix vecg_n(vecg);
    arma::vec vecG(vecg_n.begin(), vecg_n.nrow(), false);

    IntegerVector modellags_n(modellags);
    arma::uvec lags = as<arma::uvec>(modellags_n);

    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    char O = as<char>(Otype);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    NumericMatrix vecgX_n(vecgX);
    arma::vec vecGX(vecgX_n.begin(), vecgX_n.nrow(), false);

    char boundtype = as<char>(bounds);

    // Test the bounds for the ETS elements
    double boundsTestResult = boundsTester(boundtype, T, S, vecG, rowvecW, matrixF);
    if(boundsTestResult!=0){
        return wrap(boundsTestResult);
    }

    if(matrixAt(0,0)!=0){
        // Test the bounds for the explanatory part
        arma::rowvec rowvecWX(matFX_n.nrow(), arma::fill::ones);
        boundsTestResult = boundsTester('a', T, S, vecGX, rowvecWX, matrixFX);
    }
    if(boundsTestResult!=0){
        return wrap(boundsTestResult);
    }

    return wrap(occurrenceOptimizer(matrixVt, matrixF, rowvecW, vecG, vecOt,
                                    lags, E, T, S, O,
                                    matrixXt, matrixAt, matrixFX, vecGX));
}


// # Fitter for the occurrence part of the univariate models. This does not include iETS_G model
List occurenceGeneralFitter(arma::vec const &vecOt,
                            arma::uvec &lagsA, char const &EA, char const &TA, char const &SA,
                            arma::mat &matrixVtA, arma::mat const &matrixFA, arma::rowvec const &rowvecWA, arma::vec const &vecGA,
                            arma::mat const &matrixXtA, arma::mat &matrixAtA, arma::mat const &matrixFXA, arma::vec const &vecGXA,
                            arma::uvec &lagsB, char const &EB, char const &TB, char const &SB,
                            arma::mat &matrixVtB, arma::mat const &matrixFB, arma::rowvec const &rowvecWB, arma::vec const &vecGB,
                            arma::mat const &matrixXtB, arma::mat &matrixAtB, arma::mat const &matrixFXB, arma::vec const &vecGXB){

    /* # matrixVt should have a length of obs + maxlag.
    * # rowvecW should have 1 row.
    * # vecG should be a vector
    * # lags is a vector of lags
    * # matrixXt is the matrix with the exogenous variables
    * # matrixAt is the matrix with the parameters for the exogenous
    */

    arma::mat matrixXtATrans = matrixXtA.t();
    arma::mat matrixXtBTrans = matrixXtB.t();

    // The parameters for the fitter
    int obs = vecOt.n_rows;
    int obsallA = matrixVtA.n_cols;
    int obsallB = matrixVtB.n_cols;
    unsigned int nComponentsA = matrixVtA.n_rows;
    unsigned int nComponentsB = matrixVtB.n_rows;
    arma::uvec lagsInternalA = lagsA;
    arma::uvec lagsInternalB = lagsB;
    unsigned int maxlagA = max(lagsInternalA);
    unsigned int maxlagB = max(lagsInternalB);
    int lagslengthA = lagsInternalA.n_rows;
    int lagslengthB = lagsInternalB.n_rows;

    lagsInternalA = lagsInternalA * nComponentsA;
    lagsInternalB = lagsInternalB * nComponentsB;

    for(int i=0; i<lagslengthA; i=i+1){
        lagsInternalA(i) = lagsInternalA(i) + (lagslengthA - i - 1);
    }
    for(int i=0; i<lagslengthB; i=i+1){
        lagsInternalB(i) = lagsInternalB(i) + (lagslengthB - i - 1);
    }

    arma::uvec lagrowsA(lagslengthA, arma::fill::zeros);
    arma::uvec lagrowsB(lagslengthB, arma::fill::zeros);

    // The vector for the fitted values of A
    arma::vec vecAfit(obs, arma::fill::zeros);
    // The vector for the fitted values of B
    arma::vec vecBfit(obs, arma::fill::zeros);
    // The vector for the fitted probabilities
    arma::vec vecPfit(obs, arma::fill::zeros);
    // The vector of errors of the model A
    arma::vec vecErrorsA(obs, arma::fill::zeros);
    // The vector of errors of the model B
    arma::vec vecErrorsB(obs, arma::fill::zeros);

    // The buffer for the calculation of transition for explanatory part
    arma::vec bufferforat(vecGXA.n_rows);
    // The warning level:
    // true - the states became negative or the value was rediculous, so we substituted it with the previous one;
    bool warning = false;

    //
    std::vector<double> bufferForErrors;

    for(int i=0; i<obs; i=i+1){
        lagrowsA = (i+maxlagA+1) * nComponentsA - lagsInternalA - 1;
        lagrowsB = (i+maxlagB+1) * nComponentsB - lagsInternalB - 1;

/* # Measurement equation and the error term */
        vecAfit(i) = wvalue(matrixVtA(lagrowsA), rowvecWA, EA, TA, SA,
                                   matrixXtA.row(i), matrixAtA.col(i));
        vecBfit(i) = wvalue(matrixVtB(lagrowsB), rowvecWB, EB, TB, SB,
                                   matrixXtB.row(i), matrixAtB.col(i));

        // This is a failsafe for cases of ridiculously high and ridiculously low values
        if(vecAfit(i) > 1e+100){
            warning = true;
            vecAfit(i) = vecAfit(i-1);
        }
        if(vecBfit(i) > 1e+100){
            warning = true;
            vecBfit(i) = vecBfit(i-1);
        }

        // Generate occurrence error for the model
        bufferForErrors = occurrenceError(vecOt(i), vecAfit(i), vecBfit(i), EA, EB, 'g');
        vecErrorsA(i) = bufferForErrors[0];
        vecErrorsB(i) = bufferForErrors[1];

/* # Transition equation */
        matrixVtA.col(i+maxlagA) = fvalue(matrixVtA(lagrowsA), matrixFA, TA, SA) +
                          gvalue(matrixVtA(lagrowsA), matrixFA, rowvecWA, EA, TA, SA) % vecGA * vecErrorsA(i);
        matrixVtB.col(i+maxlagB) = fvalue(matrixVtB(lagrowsB), matrixFB, TB, SB) +
                          gvalue(matrixVtB(lagrowsB), matrixFB, rowvecWB, EB, TB, SB) % vecGB * vecErrorsB(i);


/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVtA.col(i+maxlagA).is_finite()){
            warning = true;
            matrixVtA.col(i+maxlagA) = matrixVtA(lagrowsA);
        }
        if(!matrixVtB.col(i+maxlagB).is_finite()){
            warning = true;
            matrixVtB.col(i+maxlagB) = matrixVtB(lagrowsB);
        }
        if((SA=='M') && (matrixVtA(nComponentsA-1,i+maxlagA) <= 0)){
            warning = true;
            matrixVtA(nComponentsA-1,i+maxlagA) = arma::as_scalar(matrixVtA(lagrowsA.row(nComponentsA-1)));
        }
        if((SB=='M') && (matrixVtB(nComponentsB-1,i+maxlagB) <= 0)){
            warning = true;
            matrixVtB(nComponentsB-1,i+maxlagB) = arma::as_scalar(matrixVtB(lagrowsB.row(nComponentsB-1)));
        }
        if(TA=='M' && ((matrixVtA(0,i+maxlagA) <= 0) || (matrixVtA(1,i+maxlagA) <= 0))){
                warning = true;
                matrixVtA(0,i+maxlagA) = arma::as_scalar(matrixVtA(lagrowsA.row(0)));
                matrixVtA(1,i+maxlagA) = arma::as_scalar(matrixVtA(lagrowsA.row(1)));
        }
        if(TB=='M' && ((matrixVtB(0,i+maxlagB) <= 0) || (matrixVtB(1,i+maxlagB) <= 0))){
                warning = true;
                matrixVtB(0,i+maxlagB) = arma::as_scalar(matrixVtB(lagrowsB.row(0)));
                matrixVtB(1,i+maxlagB) = arma::as_scalar(matrixVtB(lagrowsB.row(1)));
        }
        if(any(matrixVtA.col(i+maxlagA)>1e+100)){
            warning = true;
            matrixVtA.col(i+maxlagA) = matrixVtA(lagrowsA);
        }
        if(any(matrixVtB.col(i+maxlagB)>1e+100)){
            warning = true;
            matrixVtB.col(i+maxlagB) = matrixVtB(lagrowsB);
        }

/* Renormalise components if the seasonal model is chosen */
        if(SA!='N'){
            if(double(i+1) / double(maxlagA) == double((i+1) / maxlagA)){
                matrixVtA.cols(i-maxlagA+1,i) = normaliser(matrixVtA.cols(i-maxlagA+1,i), obsallA, maxlagA, SA, TA);
            }
        }
        if(SB!='N'){
            if(double(i+1) / double(maxlagB) == double((i+1) / maxlagB)){
                matrixVtB.cols(i-maxlagB+1,i) = normaliser(matrixVtB.cols(i-maxlagB+1,i), obsallB, maxlagB, SB, TB);
            }
        }

/* # Transition equation for xreg */
        bufferforat = gXvalue(matrixXtATrans.col(i), vecGXA, vecErrorsA.row(i), EA);
        matrixAtA.col(i+maxlagA) = matrixFXA * matrixAtA.col(i) + bufferforat;
        bufferforat = gXvalue(matrixXtBTrans.col(i), vecGXB, vecErrorsB.row(i), EB);
        matrixAtB.col(i+maxlagB) = matrixFXB * matrixAtB.col(i) + bufferforat;
    }

    // The tail for the states of the model A
    for(int i=obs+maxlagA; i<obsallA; i=i+1){
        lagrowsA = (i+1) * nComponentsA - lagsInternalA - 1;
        matrixVtA.col(i) = fvalue(matrixVtA(lagrowsA), matrixFA, TA, SA);
        matrixAtA.col(i) = matrixFXA * matrixAtA.col(i-1);

/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVtA.col(i).is_finite()){
            warning = true;
            matrixVtA.col(i) = matrixVtA(lagrowsA);
        }
        if((SA=='M') && (matrixVtA(matrixVtA.n_rows-1,i) <= 0)){
            warning = true;
            matrixVtA(matrixVtA.n_rows-1,i) = arma::as_scalar(matrixVtA(lagrowsA.row(matrixVtA.n_rows-1)));
        }
        if(TA=='M' && ((matrixVtA(0,i) <= 0) || (matrixVtA(1,i) <= 0))){
                warning = true;
                matrixVtA(0,i) = arma::as_scalar(matrixVtA(lagrowsA.row(0)));
                matrixVtA(1,i) = arma::as_scalar(matrixVtA(lagrowsA.row(1)));
        }
    }

    // The tail for the states of the model B
    for(int i=obs+maxlagB; i<obsallB; i=i+1){
        lagrowsB = (i+1) * nComponentsB - lagsInternalB - 1;
        matrixVtB.col(i) = fvalue(matrixVtB(lagrowsB), matrixFB, TB, SB);
        matrixAtB.col(i) = matrixFXB * matrixAtB.col(i-1);

/* Failsafe for cases when unreasonable value for state vector was produced */
        if(!matrixVtB.col(i).is_finite()){
            warning = true;
            matrixVtB.col(i) = matrixVtB(lagrowsB);
        }
        if((SB=='M') && (matrixVtB(matrixVtB.n_rows-1,i) <= 0)){
            warning = true;
            matrixVtB(matrixVtB.n_rows-1,i) = arma::as_scalar(matrixVtB(lagrowsB.row(matrixVtB.n_rows-1)));
        }
        if(TB=='M' && ((matrixVtB(0,i) <= 0) || (matrixVtB(1,i) <= 0))){
                warning = true;
                matrixVtB(0,i) = arma::as_scalar(matrixVtB(lagrowsB.row(0)));
                matrixVtB(1,i) = arma::as_scalar(matrixVtB(lagrowsB.row(1)));
        }
    }

    // Check the fitted values and produce the probability
    if(EA=='M' && any(vecAfit<0)){
        warning = true;
        vecAfit.elem(find(vecAfit<0)).fill(1E-10);
    }
    if(EB=='M' && any(vecBfit<0)){
        warning = true;
        vecBfit.elem(find(vecBfit<0)).fill(1E-10);
    }

    switch(EA){
        case 'A':
            switch(EB){
                case 'A':
                    vecPfit = exp(vecAfit) / (exp(vecAfit) + exp(vecBfit));
                break;
                case 'M':
                    vecPfit = exp(vecAfit) / (exp(vecAfit) + vecBfit);
                break;
            }
        break;
        case 'M':
            switch(EB){
                case 'A':
                    vecPfit = vecAfit / (vecAfit + exp(vecBfit));
                break;
                case 'M':
                    vecPfit = vecAfit / (vecAfit + vecBfit);
                break;
            }
        break;
    }
    // This is needed for cases when huge numbers were generated
    // vecPfit.elem(find_nonfinite(vecPfit)).replace(1.0-1E-10);
    vecPfit.replace(NA_REAL, 1.0-1E-10);

    return List::create(Named("pfit") = vecPfit,
                        Named("afit") = vecAfit, Named("matvtA") = matrixVtA,
                        Named("errorsA") = vecErrorsA, Named("matatA") = matrixAtA,
                        Named("bfit") = vecBfit, Named("matvtB") = matrixVtB,
                        Named("errorsB") = vecErrorsB, Named("matatB") = matrixAtB,
                        Named("warning") = warning);
}

/* # Wrapper for fitter */
// [[Rcpp::export]]
RcppExport SEXP occurenceGeneralFitterWrap(SEXP ot,
                                           SEXP modellagsA, SEXP EtypeA, SEXP TtypeA, SEXP StypeA,
                                           SEXP matvtA, SEXP matFA, SEXP matwA, SEXP vecgA,
                                           SEXP matxtA, SEXP matatA, SEXP matFXA, SEXP vecgXA,
                                           SEXP modellagsB, SEXP EtypeB, SEXP TtypeB, SEXP StypeB,
                                           SEXP matvtB, SEXP matFB, SEXP matwB, SEXP vecgB,
                                           SEXP matxtB, SEXP matatB, SEXP matFXB, SEXP vecgXB){

    NumericMatrix ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.nrow(), false);

    //// #### The model A #### ////
    IntegerVector modellagsA_n(modellagsA);
    arma::uvec lagsA = as<arma::uvec>(modellagsA_n);

    char EA = as<char>(EtypeA);
    char TA = as<char>(TtypeA);
    char SA = as<char>(StypeA);

    NumericMatrix matvtA_n(matvtA);
    arma::mat matrixVtA(matvtA_n.begin(), matvtA_n.nrow(), matvtA_n.ncol());

    NumericMatrix matFA_n(matFA);
    arma::mat matrixFA(matFA_n.begin(), matFA_n.nrow(), matFA_n.ncol(), false);

    NumericMatrix matwA_n(matwA);
    arma::rowvec rowvecWA(matwA_n.begin(), matwA_n.ncol(), false);

    NumericMatrix vecgA_n(vecgA);
    arma::vec vecGA(vecgA_n.begin(), vecgA_n.nrow(), false);

    NumericMatrix matxtA_n(matxtA);
    arma::mat matrixXtA(matxtA_n.begin(), matxtA_n.nrow(), matxtA_n.ncol(), false);

    NumericMatrix matatA_n(matatA);
    arma::mat matrixAtA(matatA_n.begin(), matatA_n.nrow(), matatA_n.ncol());

    NumericMatrix matFXA_n(matFXA);
    arma::mat matrixFXA(matFXA_n.begin(), matFXA_n.nrow(), matFXA_n.ncol(), false);

    NumericMatrix vecgXA_n(vecgXA);
    arma::vec vecGXA(vecgXA_n.begin(), vecgXA_n.nrow(), false);

    //// #### The model B #### ////
    IntegerVector modellagsB_n(modellagsB);
    arma::uvec lagsB = as<arma::uvec>(modellagsB_n);

    char EB = as<char>(EtypeB);
    char TB = as<char>(TtypeB);
    char SB = as<char>(StypeB);

    NumericMatrix matvtB_n(matvtB);
    arma::mat matrixVtB(matvtB_n.begin(), matvtB_n.nrow(), matvtB_n.ncol());

    NumericMatrix matFB_n(matFB);
    arma::mat matrixFB(matFB_n.begin(), matFB_n.nrow(), matFB_n.ncol(), false);

    NumericMatrix matwB_n(matwB);
    arma::rowvec rowvecWB(matwB_n.begin(), matwB_n.ncol(), false);

    NumericMatrix vecgB_n(vecgB);
    arma::vec vecGB(vecgB_n.begin(), vecgB_n.nrow(), false);

    NumericMatrix matxtB_n(matxtB);
    arma::mat matrixXtB(matxtB_n.begin(), matxtB_n.nrow(), matxtB_n.ncol(), false);

    NumericMatrix matatB_n(matatB);
    arma::mat matrixAtB(matatB_n.begin(), matatB_n.nrow(), matatB_n.ncol());

    NumericMatrix matFXB_n(matFXB);
    arma::mat matrixFXB(matFXB_n.begin(), matFXB_n.nrow(), matFXB_n.ncol(), false);

    NumericMatrix vecgXB_n(vecgXB);
    arma::vec vecGXB(vecgXB_n.begin(), vecgXB_n.nrow(), false);

    return wrap(occurenceGeneralFitter(vecOt,
                                       lagsA, EA, TA, SA,
                                       matrixVtA, matrixFA, rowvecWA, vecGA,
                                       matrixXtA, matrixAtA, matrixFXA, vecGXA,
                                       lagsB, EB, TB, SB,
                                       matrixVtB, matrixFB, rowvecWB, vecGB,
                                       matrixXtB, matrixAtB, matrixFXB, vecGXB));
}

/* # Function returns the chosen Cost Function based on the chosen model and produced errors */
double occurrenceGeneralOptimizer(arma::vec const &vecOt,
                                  arma::uvec &lagsA, char const &EA, char const &TA, char const &SA,
                                  arma::mat &matrixVtA, arma::mat const &matrixFA, arma::rowvec const &rowvecWA, arma::vec const &vecGA,
                                  arma::mat const &matrixXtA, arma::mat &matrixAtA, arma::mat const &matrixFXA, arma::vec const &vecGXA,
                                  arma::uvec &lagsB, char const &EB, char const &TB, char const &SB,
                                  arma::mat &matrixVtB, arma::mat const &matrixFB, arma::rowvec const &rowvecWB, arma::vec const &vecGB,
                                  arma::mat const &matrixXtB, arma::mat &matrixAtB, arma::mat const &matrixFXB, arma::vec const &vecGXB){

    List fitting = occurenceGeneralFitter(vecOt,
                                          lagsA, EA, TA, SA,
                                          matrixVtA, matrixFA, rowvecWA, vecGA,
                                          matrixXtA, matrixAtA, matrixFXA, vecGXA,
                                          lagsB, EB, TB, SB,
                                          matrixVtB, matrixFB, rowvecWB, vecGB,
                                          matrixXtB, matrixAtB, matrixFXB, vecGXB);

    NumericMatrix pfitfromfit = as<NumericMatrix>(fitting["pfit"]);
    arma::vec vecPfit = as<arma::vec>(pfitfromfit);

    // 0.5 is needed for cases, when the variable is continuous in (0, 1)
    double CFres = -sum(log(vecPfit.elem(find(vecOt>=0.5)))) - sum(log(1-vecPfit.elem(find(vecOt<0.5))));

    return CFres;
}

/* # Function is used for the occurrence model of type "G"
# If bounds are violated, it returns variance of ot. */
// [[Rcpp::export]]
RcppExport SEXP occurrenceGeneralOptimizerWrap(SEXP ot, SEXP bounds,
                                               SEXP modellagsA, SEXP EtypeA, SEXP TtypeA, SEXP StypeA,
                                               SEXP matvtA, SEXP matFA, SEXP matwA, SEXP vecgA,
                                               SEXP matxtA, SEXP matatA, SEXP matFXA, SEXP vecgXA,
                                               SEXP modellagsB, SEXP EtypeB, SEXP TtypeB, SEXP StypeB,
                                               SEXP matvtB, SEXP matFB, SEXP matwB, SEXP vecgB,
                                               SEXP matxtB, SEXP matatB, SEXP matFXB, SEXP vecgXB){

    NumericMatrix ot_n(ot);
    arma::vec vecOt(ot_n.begin(), ot_n.nrow(), false);

    char boundtype = as<char>(bounds);

    //// #### The model A #### ////
    IntegerVector modellagsA_n(modellagsA);
    arma::uvec lagsA = as<arma::uvec>(modellagsA_n);

    char EA = as<char>(EtypeA);
    char TA = as<char>(TtypeA);
    char SA = as<char>(StypeA);

    NumericMatrix matvtA_n(matvtA);
    arma::mat matrixVtA(matvtA_n.begin(), matvtA_n.nrow(), matvtA_n.ncol());

    NumericMatrix matFA_n(matFA);
    arma::mat matrixFA(matFA_n.begin(), matFA_n.nrow(), matFA_n.ncol(), false);

    NumericMatrix matwA_n(matwA);
    arma::rowvec rowvecWA(matwA_n.begin(), matwA_n.ncol(), false);

    NumericMatrix vecgA_n(vecgA);
    arma::vec vecGA(vecgA_n.begin(), vecgA_n.nrow(), false);

    NumericMatrix matxtA_n(matxtA);
    arma::mat matrixXtA(matxtA_n.begin(), matxtA_n.nrow(), matxtA_n.ncol(), false);

    NumericMatrix matatA_n(matatA);
    arma::mat matrixAtA(matatA_n.begin(), matatA_n.nrow(), matatA_n.ncol());

    NumericMatrix matFXA_n(matFXA);
    arma::mat matrixFXA(matFXA_n.begin(), matFXA_n.nrow(), matFXA_n.ncol(), false);

    NumericMatrix vecgXA_n(vecgXA);
    arma::vec vecGXA(vecgXA_n.begin(), vecgXA_n.nrow(), false);

    // Test the bounds for the ETS elements
    double boundsTestResult = boundsTester(boundtype, TA, SA, vecGA, rowvecWA, matrixFA);
    if(boundsTestResult!=0){
        return wrap(boundsTestResult);
    }

    if(matrixAtA(0,0)!=0){
        // Test the bounds for the explanatory part
        arma::rowvec rowvecWX(matFXA_n.nrow(), arma::fill::ones);
        boundsTestResult = boundsTester(boundtype,'N', 'N', vecGXA, rowvecWX, matrixFXA);
        if(boundsTestResult!=0){
            return wrap(boundsTestResult);
        }
    }


    //// #### The model B #### ////
    IntegerVector modellagsB_n(modellagsB);
    arma::uvec lagsB = as<arma::uvec>(modellagsB_n);

    char EB = as<char>(EtypeB);
    char TB = as<char>(TtypeB);
    char SB = as<char>(StypeB);

    NumericMatrix matvtB_n(matvtB);
    arma::mat matrixVtB(matvtB_n.begin(), matvtB_n.nrow(), matvtB_n.ncol());

    NumericMatrix matFB_n(matFB);
    arma::mat matrixFB(matFB_n.begin(), matFB_n.nrow(), matFB_n.ncol(), false);

    NumericMatrix matwB_n(matwB);
    arma::rowvec rowvecWB(matwB_n.begin(), matwB_n.ncol(), false);

    NumericMatrix vecgB_n(vecgB);
    arma::vec vecGB(vecgB_n.begin(), vecgB_n.nrow(), false);

    NumericMatrix matxtB_n(matxtB);
    arma::mat matrixXtB(matxtB_n.begin(), matxtB_n.nrow(), matxtB_n.ncol(), false);

    NumericMatrix matatB_n(matatB);
    arma::mat matrixAtB(matatB_n.begin(), matatB_n.nrow(), matatB_n.ncol());

    NumericMatrix matFXB_n(matFXB);
    arma::mat matrixFXB(matFXB_n.begin(), matFXB_n.nrow(), matFXB_n.ncol(), false);

    NumericMatrix vecgXB_n(vecgXB);
    arma::vec vecGXB(vecgXB_n.begin(), vecgXB_n.nrow(), false);

    // Test the bounds for the ETS elements
    boundsTestResult = boundsTester(boundtype, TB, SB, vecGB, rowvecWB, matrixFB);
    if(boundsTestResult!=0){
        return wrap(boundsTestResult);
    }

    if(matrixAtB(0,0)!=0){
        // Test the bounds for the explanatory part
        arma::rowvec rowvecWX(matFXB_n.nrow(), arma::fill::ones);
        boundsTestResult = boundsTester(boundtype, 'N', 'N', vecGXB, rowvecWX, matrixFXB);
        if(boundsTestResult!=0){
            return wrap(boundsTestResult);
        }
    }

    return wrap(occurrenceGeneralOptimizer(vecOt,
                                           lagsA, EA, TA, SA,
                                           matrixVtA, matrixFA, rowvecWA, vecGA,
                                           matrixXtA, matrixAtA, matrixFXA, vecGXA,
                                           lagsB, EB, TB, SB,
                                           matrixVtB, matrixFB, rowvecWB, vecGB,
                                           matrixXtB, matrixAtB, matrixFXB, vecGXB));
}
