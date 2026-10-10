#ifndef COVARANALCORE_H
#define COVARANALCORE_H

#include <armadillo>

// The analytical covariance matrix of the 1..h steps ahead errors of a pure additive
// model, s2 times the matrix of the coefficients c_j = w F^(j-1) g over the lags of the
// components (R's covarAnal()). It is used in the analytical intervals, multicov() and
// the variance of the errors after the periods without an observed value.
inline arma::mat covarAnalCore(const arma::vec& lagsModel, const unsigned int h,
                               const arma::rowvec& measurement, const arma::mat& transition,
                               const arma::vec& persistence, const double s2){
    arma::mat covarMat = arma::eye(h, h);
    if(h <= lagsModel.min()){
        return covarMat * s2;
    }

    arma::vec lagsUnique = arma::unique(lagsModel);
    arma::vec steps = lagsUnique.elem(arma::find(lagsUnique <= h));
    unsigned int stepsNumber = steps.n_elem;
    unsigned int nComponents = transition.n_rows;
    arma::mat identity = arma::eye(nComponents, nComponents);

    // The transition and the measurement of the components of each lag
    arma::cube arrayTransition(nComponents, nComponents, stepsNumber, arma::fill::zeros);
    arma::mat arrayMeasurement(stepsNumber, nComponents, arma::fill::zeros);
    for(unsigned int i=0; i<stepsNumber; i++){
        arma::uvec columns = arma::find(lagsModel == steps(i));
        arrayTransition.slice(i).cols(columns) = transition.cols(columns);
        for(arma::uword column : columns){
            arrayMeasurement(i, column) = measurement(column);
        }
    }

    // The powers of the transition matrix, identities up to the lowest lag
    unsigned int minStep = steps(0);
    arma::field<arma::mat> transitionPowered(h, stepsNumber);
    for(unsigned int i=0; i<h; i++){
        for(unsigned int k=0; k<stepsNumber; k++){
            transitionPowered(i, k) = (i < minStep) ? identity : arma::mat(nComponents, nComponents, arma::fill::zeros);
        }
    }

    // The values of c_j, with i, j and k as R's indices
    arma::vec cValues(h, arma::fill::zeros);
    for(unsigned int i=minStep+1; i<=h; i++){
        unsigned int stepsBelow = arma::accu(steps < i);
        for(unsigned int k=1; k<=stepsBelow; k++){
            // This is produced only for the lowest lag and reused for the higher ones
            if(k==1){
                for(unsigned int j=1; j<=stepsBelow; j++){
                    const arma::mat& transitionNew = ((i - steps(k-1)) / steps(j-1) > 1) ?
                        arrayTransition.slice(j-1) : identity;
                    arma::mat product = transitionNew * transitionPowered(i - steps(j-1) - 1, k-1);
                    if(arma::all(arma::vectorise(transitionPowered(i-1, k-1)) == 0)){
                        transitionPowered(i-1, k-1) = product;
                    }
                    else if(!arma::all(arma::vectorise(product == identity))){
                        transitionPowered(i-1, k-1) += product;
                    }
                }
            }
            else{
                transitionPowered(i-1, k-1) = transitionPowered(i - steps(k-1), 0);
            }
            arma::rowvec measurementPowered = arrayMeasurement.row(k-1) * transitionPowered(i-1, k-1);
            cValues(i-1) += arma::as_scalar(measurementPowered * persistence);
        }
    }

    // The diagonal and the off-diagonals
    for(unsigned int i=1; i<h; i++){
        covarMat(i, i) = covarMat(i-1, i-1) + cValues(i) * cValues(i);
    }
    for(unsigned int i=0; i<h; i++){
        for(unsigned int j=0; j<h; j++){
            if(i==j){
                continue;
            }
            else if(i==0){
                covarMat(i, j) = cValues(j);
            }
            else if(i>j){
                covarMat(i, j) = covarMat(j, i);
            }
            else{
                covarMat(i, j) = covarMat(i-1, j-1) + covarMat(0, j) * covarMat(0, i);
            }
        }
    }
    return covarMat * s2;
}

#endif
