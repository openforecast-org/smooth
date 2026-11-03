#include "headers/fpContract.h"
#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
#include "headers/ssGeneral.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/* # Function produces the point forecasts for the specified model */
arma::mat forecaster(arma::mat matrixVt, arma::mat const &matrixF, arma::rowvec const &rowvecW,
                     unsigned int const &hor, char const &E, char const &T, char const &S, arma::uvec lags,
                     arma::mat matrixXt, arma::mat matrixAt, arma::mat const &matrixFX){
    int lagslength = lags.n_rows;
    unsigned int lagsModelMax = max(lags);
    unsigned int hh = hor + lagsModelMax;

    arma::uvec lagrows(lagslength, arma::fill::zeros);
    arma::vec matyfor(hor, arma::fill::zeros);
    arma::mat matrixVtnew(hh, matrixVt.n_cols, arma::fill::zeros);
    arma::mat matrixAtnew(hh, matrixAt.n_cols, arma::fill::zeros);

    lags = lagsModelMax - lags;
    for(int i=1; i<lagslength; i=i+1){
        lags(i) = lags(i) + hh * i;
    }

    matrixVtnew.submat(0,0,lagsModelMax-1,matrixVtnew.n_cols-1) = matrixVt.submat(0,0,lagsModelMax-1,matrixVtnew.n_cols-1);
    matrixAtnew.submat(0,0,lagsModelMax-1,matrixAtnew.n_cols-1) = matrixAtnew.submat(0,0,lagsModelMax-1,matrixAtnew.n_cols-1);

/* # Fill in the new xt matrix using F. Do the forecasts. */
    for (unsigned int i=lagsModelMax; i<(hor+lagsModelMax); i=i+1) {
        lagrows = lags - lagsModelMax + i;
        matrixVtnew.row(i) = arma::trans(fvalue(matrixVtnew(lagrows), matrixF, T, S));
        matrixAtnew.row(i) = matrixAtnew.row(i-1) * matrixFX;

        matyfor.row(i-lagsModelMax) = (wvalue(matrixVtnew(lagrows), rowvecW, E, T, S, matrixXt.row(i-lagsModelMax), trans(matrixAt.row(i-lagsModelMax))));
    }

    return matyfor;
}

/* # Wrapper for forecaster */
// [[Rcpp::export]]
RcppExport SEXP forecasterwrap(SEXP matvt, SEXP matF, SEXP matw,
                               SEXP h, SEXP Etype, SEXP Ttype, SEXP Stype, SEXP lagsModel,
                               SEXP matxt, SEXP matat, SEXP matFX){
    NumericMatrix matvt_n(matvt);
    arma::mat matrixVt(matvt_n.begin(), matvt_n.nrow(), matvt_n.ncol(), false);

    NumericMatrix matF_n(matF);
    arma::mat matrixF(matF_n.begin(), matF_n.nrow(), matF_n.ncol(), false);

    NumericMatrix matw_n(matw);
    arma::mat rowvecW(matw_n.begin(), matw_n.nrow(), matw_n.ncol(), false);

    unsigned int hor = as<int>(h);
    char E = as<char>(Etype);
    char T = as<char>(Ttype);
    char S = as<char>(Stype);

    IntegerVector lagsModel_n(lagsModel);
    arma::uvec lags = as<arma::uvec>(lagsModel_n);

    NumericMatrix matxt_n(matxt);
    arma::mat matrixXt(matxt_n.begin(), matxt_n.nrow(), matxt_n.ncol(), false);

    NumericMatrix matat_n(matat);
    arma::mat matrixAt(matat_n.begin(), matat_n.nrow(), matat_n.ncol());

    NumericMatrix matFX_n(matFX);
    arma::mat matrixFX(matFX_n.begin(), matFX_n.nrow(), matFX_n.ncol(), false);

    return wrap(forecaster(matrixVt, matrixF, rowvecW, hor, E, T, S, lags, matrixXt, matrixAt, matrixFX));
}
