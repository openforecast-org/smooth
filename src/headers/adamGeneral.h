#pragma once

/* # Function returns value of w() -- y-fitted -- used in the measurement equation */
inline double adamWvalue(arma::vec const &vecVt, arma::rowvec const &rowvecW,
                         char const &E, char const &T, char const &S,
                         unsigned int const &nETS, unsigned int const &nNonSeasonal,
                         unsigned int const &nSeasonal, unsigned int const &nArima,
                         unsigned int const &nXreg, unsigned int const &nComponents,
                         bool const &constant){
    // vecVt is a vector here!
    double yfit = 0;
    if(E=='M'){
        yfit = 1;
    }
    arma::mat vecYfit;

    // If there is ETS, calculate the measurement
    if(nETS>0){
        switch(S){
        // ZZN
        case 'N':
            switch(T){
            case 'N':
                vecYfit = vecVt.row(0);
                break;
            case 'A':
                vecYfit = rowvecW.cols(0,1) * vecVt.rows(0,1);
                break;
            case 'M':
                // vecYfit = exp(rowvecW.cols(0,1) * log(vecVt.rows(0,1)));
                vecYfit = arma::real(exp(rowvecW.cols(0,1) *
                                               log(arma::conv_to<arma::cx_vec>::from(vecVt.rows(0,1)))));
                break;
            }
            break;
            // ZZA
        case 'A':
            switch(T){
            case 'N':
            case 'A':
                vecYfit = rowvecW.cols(0,nETS-1) * vecVt.rows(0,nETS-1);
                break;
            case 'M':
                // vecYfit = exp(rowvecW.cols(0,1) * log(vecVt.rows(0,1))) + rowvecW.cols(2,2+nSeasonal-1) * vecVt.rows(2,2+nSeasonal-1);
                vecYfit = arma::real(exp(rowvecW.cols(0,1) *
                                               log(arma::conv_to<arma::cx_vec>::from(vecVt.rows(0,1))))) +
                          rowvecW.cols(2,2+nSeasonal-1) * vecVt.rows(2,2+nSeasonal-1);
                break;
            }
            break;
            // ZZM
        case 'M':
            switch(T){
            case 'N':
            case 'M':
                switch(E){
                    case 'A':
                        // Use complex numbers to avoid issues with negative states
                        vecYfit = arma::real(exp(rowvecW.cols(0,nETS-1) *
                                             log(arma::conv_to<arma::cx_vec>::from(vecVt.rows(0,nETS-1)))));
                    break;
                    case 'M':
                        vecYfit = exp(rowvecW.cols(0,nETS-1) * log(vecVt.rows(0,nETS-1)));
                }
                break;
            case 'A':
                vecYfit = rowvecW.cols(0,1) * vecVt.rows(0,1) * exp(rowvecW.cols(2,2+nSeasonal-1) * log(vecVt.rows(2,2+nSeasonal-1)));
                break;
            }
            break;
        }
        yfit = as_scalar(vecYfit);
    }

    // ARIMA components
    if(nArima > 0){
        // If error is additive, add explanatory variables. Otherwise multiply by exp(ax)
        switch(E){
        case 'A':
            yfit += as_scalar(rowvecW.cols(nETS,nETS+nArima-1) *
                vecVt.rows(nETS,nETS+nArima-1));
            break;
        case 'M':
            yfit = yfit * as_scalar(exp(rowvecW.cols(nETS,nETS+nArima-1) *
                log(vecVt.rows(nETS,nETS+nArima-1))));
            break;
        }
    }

    // Explanatory variables, without the constant that follows them
    if(nXreg > 0){
        unsigned int xregLast = nETS+nArima+nXreg-1;
        // If error is additive, add explanatory variables. Otherwise multiply by exp(ax)
        switch(E){
        case 'A':
            yfit += as_scalar(rowvecW.cols(nETS+nArima,xregLast) *
                vecVt.rows(nETS+nArima,xregLast));
            break;
        case 'M':
            yfit = yfit * as_scalar(exp(rowvecW.cols(nETS+nArima,xregLast) *
                vecVt.rows(nETS+nArima,xregLast)));
            break;
        }
    }

    // The constant: a summand with the additive error, a ratio with the multiplicative one
    if(constant){
        switch(E){
        case 'A':
            yfit += vecVt(nComponents-1);
            break;
        case 'M':
            yfit = yfit * vecVt(nComponents-1);
            break;
        }
    }

    return yfit;
}

/* # Function returns value of r() -- additive or multiplicative error -- used in the error term of measurement equation.
 This is mainly needed by sim.ets */
inline double adamRvalue(arma::vec const &vecVt, arma::rowvec const &rowvecW,
                         char const &E, char const &T, char const &S,
                         unsigned int const &nETS, unsigned int const &nNonSeasonal,
                         unsigned int const &nSeasonal, unsigned int const &nArima,
                         unsigned int const &nXreg, unsigned int const &nComponents,
                         bool const &constant){

    switch(E){
    // MZZ
    case 'M':
        return adamWvalue(vecVt, rowvecW, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
        break;
        // AZZ
    case 'A':
    default:
        return 1.0;
    }
}

/* # Function returns value of f() -- new states without the update -- used in the transition equation */
inline arma::vec adamFvalue(arma::vec const &matrixVt, arma::mat const &matrixF,
                            char const E, char const T, char const S,
                            unsigned int const &nETS, unsigned int const &nNonSeasonal,
                            unsigned int const &nSeasonal, unsigned int const &nArima,
                            unsigned int const &nComponents, bool const &constant){
    arma::vec matrixVtnew = matrixVt;

    switch(T){
    case 'N':
    case 'A':
        matrixVtnew = matrixF * matrixVt;
        break;
    case 'M':
        if(nETS>0){
            // Use complex numbers to avoid issues in mixed models
            // matrixVtnew.rows(0,1) = arma::real(exp(matrixF.submat(0,0,1,1) *
            //                                    log(arma::conv_to<arma::cx_vec>::from(matrixVt.rows(0,1)))));
            // The new code is just for completeness to reflect the potential change in weird model cases
            matrixVtnew.rows(0,nETS-1) = arma::abs(exp(matrixF.submat(0,0,nETS-1,nETS-1) *
                log(arma::conv_to<arma::cx_vec>::from(matrixVt.rows(0,nETS-1)))));
        }
        break;
    }

    // If there is ARIMA, fix the states for E='M'. They are multiplicative, so F acts on
    // their logarithms, and the constant (a ratio) enters through its column in logs, as
    // it does in levels with E='A': without it, the drift was absorbed by the states
    if(nArima>0 && E=='M'){
        arma::vec logStatesNew = matrixF.submat(nETS,nETS,nETS+nArima-1,nETS+nArima-1) *
            log(matrixVt.rows(nETS,nETS+nArima-1));
        if(constant){
            logStatesNew += matrixF.submat(nETS,nComponents-1,nETS+nArima-1,nComponents-1) *
                std::log(matrixVt(nComponents-1));
        }
        matrixVtnew.rows(nETS,nETS+nArima-1) = exp(logStatesNew);
    }

    // If there is a constant, fix the first state of ETS(M,*,*)
    if(constant && nETS>0 && E=='M'){
        matrixVtnew.row(0) = (matrixVtnew.row(0)-matrixVt.row(nComponents-1)) % matrixVt.row(nComponents-1);
    }

    return matrixVtnew;
}

/* # Function returns value of g() -- the update of states -- used in components estimation for the persistence */
inline arma::vec adamGvalue(arma::vec const &matrixVt, arma::mat const &matrixF, arma::mat const &rowvecW,
                            char const &E, char const &T, char const &S,
                            unsigned int const &nETS, unsigned int const &nNonSeasonal,
                            unsigned int const &nSeasonal, unsigned int const &nArima,
                            unsigned int const &nXreg, unsigned int const &nComponents,
                            bool const &constant, arma::vec const &vectorG, double const error,
                            double const fitted, bool const &adamETS){
    arma::vec g(matrixVt.n_rows, arma::fill::ones);

    if(nETS>0){
    // If this is ADAM ETS, use different functions
        if(adamETS){
            g = vectorG * error;
            switch(E){
            // AZZ
            case 'A':
                // ANZ
                switch(T){
                case 'N':
                    switch(S){
                    case 'M':
                        g.row(0) = vectorG.row(0) * error / as_scalar(rowvecW.cols(1,nSeasonal) * matrixVt.rows(1,nSeasonal));
                        // !!! This sort of thing can be written as a function not to duplicate the principle...
                        g.rows(1,nSeasonal) = matrixVt.rows(1,nSeasonal) %
                            (abs(exp(vectorG.rows(1,nSeasonal) * log(std::complex<double>(1+error/fitted)))) - 1);
                        break;
                        // Nothing to do in the other cases
                    }
                    break;
                // AAZ
                case 'A':
                    switch(S){
                    case 'M':
                        g.rows(0,1) = vectorG.rows(0,1) * error / as_scalar(abs(exp(rowvecW.cols(2,2+nSeasonal-1) *
                            log(arma::conv_to<arma::cx_vec>::from(matrixVt.rows(2,2+nSeasonal-1))))));
                    // Complex is needed to avoid issues with mixed models
                        g.rows(2,2+nSeasonal-1) = matrixVt.rows(2,2+nSeasonal-1) %
                            (abs(exp(vectorG.rows(2,2+nSeasonal-1) * log(std::complex<double>(1+error/fitted)))) - 1);
                        break;
                        // Nothing to do in the other cases
                    }
                    break;
                // AMZ
                case 'M':
                    // Complex is needed to avoid issues with mixed models
                    g.row(1) = exp(matrixF(1,1) * log(matrixVt.row(1))) *
                        (abs(exp(vectorG.row(1) * log(std::complex<double>(1+error/fitted)))) - 1);
                    switch(S){
                    case 'N':
                    case 'A':
                        g.row(1) = vectorG.row(1) * error / matrixVt.row(0);
                        break;
                    case 'M':
                        g.row(0) = vectorG.row(0) * error / as_scalar(rowvecW.cols(2,2+nSeasonal-1) * matrixVt.rows(2,2+nSeasonal-1));
                    // Complex is needed to avoid issues with mixed models
                        g.rows(2,2+nSeasonal-1) = matrixVt.rows(2,2+nSeasonal-1) %
                            (abs(exp(vectorG.rows(2,2+nSeasonal-1) * log(std::complex<double>(1+error/fitted)))) - 1);
                        break;
                    }
                    break;
                }
                break;
            // MZZ
            case 'M':
                // MNZ
                switch(T){
                case 'N':
                    g.row(0) = matrixVt.row(0) * (abs(exp(vectorG.row(0) * log(std::complex<double>(1+error)))) - 1);
                    switch(S){
                    case 'A':
                        g.rows(1,nSeasonal) = vectorG.rows(1,nSeasonal) * fitted * error;
                        break;
                    case 'M':
                        g.rows(1,nSeasonal) = matrixVt.rows(1,nSeasonal) %
                            (abs(exp(vectorG.rows(1,nSeasonal) * log(std::complex<double>(1+error)))) - 1);
                        break;
                    }
                    break;
                // MAZ
                case 'A':
                    // Complex is needed to avoid issues with mixed models
                    g.row(0) = matrixF.submat(0,0,0,1) * matrixVt.rows(0,1) *
                        (abs(exp(vectorG.row(0) * log(std::complex<double>(1+error)))) - 1);
                    switch(S){
                    case 'N':
                        g.row(1) = vectorG.row(1) * fitted * error;
                        break;
                    case 'A':
                        g.row(1) = vectorG.row(1) * fitted * error;
                        g.rows(2,2+nSeasonal-1) = vectorG.rows(2,2+nSeasonal-1) * error;
                        break;
                    case 'M':
                        g.row(1) = matrixF.submat(0,0,0,1) * matrixVt.rows(0,1) * vectorG.row(1) * error;
                        // Complex is needed to avoid issues with mixed models
                        g.rows(2,2+nSeasonal-1) = matrixVt.rows(2,2+nSeasonal-1) %
                            (arma::abs(exp(vectorG.rows(2,2+nSeasonal-1) * log(std::complex<double>(1+error)))) - 1);
                        break;
                    }
                    break;
                // MMZ
                case 'M':
                    // Complex is needed to avoid issues with mixed models
                    g.row(0) = arma::abs(exp(matrixF.submat(0,0,0,1) *
                                               log(arma::conv_to<arma::cx_vec>::from(matrixVt.rows(0,1))))) *
                                               (abs(exp(vectorG.row(0) * log(std::complex<double>(1+error)))) - 1);
                    g.row(1) = exp(matrixF(1,1) * log(matrixVt.row(1))) *
                        (abs(exp(vectorG.row(1) * log(std::complex<double>(1+error)))) - 1);
                    switch(S){
                    case 'A':
                        g.rows(2,2+nSeasonal-1) = vectorG.rows(2,2+nSeasonal-1) * fitted * error;
                        break;
                    case 'M':
                        g.rows(2,2+nSeasonal-1) = matrixVt.rows(2,2+nSeasonal-1) %
                            (exp(vectorG.rows(2,2+nSeasonal-1) * log(1+error)) - 1);

                        break;
                    }
                    break;
                }
                break;
            }
        }
    // If this is the conventional ETS...
        else{
            // AZZ
            switch(E){
            case 'A':
                // ANZ
                switch(T){
                case 'N':
                    switch(S){
                    case 'M':
                        g(0) = 1 / as_scalar(rowvecW.cols(1,nSeasonal) * matrixVt.rows(1,nSeasonal));
                        g.rows(1,nSeasonal).fill(1 / matrixVt(0));
                        // // Explanatory variables
                        // if(nComponents > (nETS)){
                        //     /* g.rows(1,nSeasonal) = 1 / (1/g(1) +
                        //      as_scalar(rowvecW.cols(nSeasonal,nComponents-1) * matrixVt.rows(nSeasonal,nComponents))); */
                        //     g.rows(nETS,nComponents-1) = 1/matrixVt.rows(nETS,nComponents-1);
                        // }
                        break;
                    }
                    break;
                    // AAZ
                case 'A':
                    switch(S){
                    case 'M':
                        g.rows(0,1) = g.rows(0,1) / as_scalar(exp(rowvecW.cols(2,2+nSeasonal-1) * log(matrixVt.rows(2,2+nSeasonal-1))));
                        g.rows(2,2+nSeasonal-1).fill(1 / as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1)));
                        // // Explanatory variables
                        // if(nComponents > (nETS)){
                        //     /*g.rows(2,2+nSeasonal-1) = 1 / (1/g(1) +
                        //       as_scalar(rowvecW.cols(nSeasonal,nComponents) * matrixVt.rows(nSeasonal,nComponents)));*/
                        //     g.rows(nETS,nComponents-1) = 1/matrixVt.rows(nETS,nComponents-1);
                        // }
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
                        g(0) = g(0) / as_scalar(rowvecW.cols(2,2+nSeasonal-1) * matrixVt.rows(2,2+nSeasonal-1));
                        g(1) = g(1) / (matrixVt(0) * as_scalar(exp(rowvecW.cols(2,2+nSeasonal-1) * log(matrixVt.rows(2,2+nSeasonal-1)))));
                        g.rows(2,2+nSeasonal-1) = g.rows(2,2+nSeasonal-1) / as_scalar(exp(rowvecW.cols(0,1) * log(matrixVt.rows(0,1))));
                        break;
                    }
                    break;
                }
                break;
                // MZZ
            case 'M':
                // MNZ
                switch(T){
                case 'N':
                    switch(S){
                    case 'N':
                        g = matrixVt;
                        break;
                    case 'A':
                        g.rows(0,nSeasonal) = matrixVt.rows(0,nSeasonal);
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
                        g.rows(0,1).fill(as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1)));
                        break;
                    case 'A':
                        g.fill(as_scalar(rowvecW * matrixVt));
                        // Explanatory variables
                        // if(nComponents > (nETS)){
                        //     g.rows(nETS,nComponents-1) = matrixVt.rows(nETS,nComponents-1);
                        // }
                        break;
                    case 'M':
                        g.rows(0,1).fill(as_scalar(rowvecW.cols(0,1) * matrixVt.rows(0,1)));
                        g.rows(2,nComponents-1) = matrixVt.rows(2,nComponents-1);
                        break;
                    }
                    break;
                    // MMZ
                case 'M':
                    switch(S){
                    case 'N':
                        g.rows(0,1) = exp(matrixF.submat(0,0,1,1) * log(matrixVt.rows(0,nNonSeasonal-1)));
                        break;
                    case 'A':
                        g.rows(0,nComponents-1).fill(as_scalar(exp(rowvecW.cols(0,1) * log(matrixVt.rows(0,1))) +
                            rowvecW.cols(2,nETS-1) * matrixVt.rows(2,nETS-1)));
                        // g(0) = g(0) / matrixVt(1);
                        g(1) = g(1) / matrixVt(0);
                        break;
                    case 'M':
                        g.rows(0,1) = exp(matrixF.submat(0,0,1,1) * log(matrixVt.rows(0,nNonSeasonal-1)));
                        g.rows(2,nComponents-1) = matrixVt.rows(2,nComponents-1);
                        break;
                    }
                    break;
                }
                break;
            }
        }
    }

    // Explanatory variables. Needed in order to update the parameters
    if(nXreg>0){
        arma::vec vecWtxreg(1/rowvecW.cols(nETS+nArima,nComponents-1).t());
        vecWtxreg.rows(find_nonfinite(vecWtxreg)).fill(0);
        // If there are xreg components, make this: delta * log(1+e)/x from this: delta * e / x
        switch(E){
        case 'M':
            g.rows(nETS+nArima,nComponents-1) = vecWtxreg * log(1+error) / error;
            break;
        case 'A':
            g.rows(nETS+nArima,nComponents-1) = vecWtxreg;
            break;
        }
        g.rows(find_nonfinite(g)).fill(0);
    }

    if(!adamETS){
        // Do the multiplication in order to get the correct g(v) value
        g = g % vectorG * error;
    }
    else{
        if(nXreg>0){
            // For adam, only the xreg part needs g values
            g.rows(nETS+nArima,nComponents-1) = g.rows(nETS+nArima,nComponents-1) % vectorG.rows(nETS+nArima,nComponents-1) * error;
        }
    }

    // If there are arima components, make sure that the g is correct for multiplicative error type
    // This goes last not to mess g with additional values from vectorG (line 432)
    if(nArima>0 && E=='M'){
        g.rows(nETS,nETS+nArima-1) =
            adamFvalue(matrixVt, matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal,
                       nArima, nComponents, constant).rows(nETS, nETS+nArima-1) %
            (exp(vectorG.rows(nETS,nETS+nArima-1)*log(1+error)) - 1);
    }

    return g;
}

// /* # Function is needed for the renormalisation of seasonal components. It should be done seasonal-wise.*/
// inline arma::mat normaliser(arma::mat Vt, int &obsall, unsigned int &maxlag, char const &S, char const &T){
//
//     unsigned int nComponents = Vt.n_rows;
//     double meanseason = 0;
//
//     switch(S){
//     case 'A':
//         meanseason = mean(Vt.row(nComponents-1));
//         Vt.row(nComponents-1) = Vt.row(nComponents-1) - meanseason;
//         switch(T){
//         case 'N':
//         case 'A':
//             Vt.row(0) = Vt.row(0) + meanseason;
//             break;
//         case 'M':
//             Vt.row(0) = Vt.row(0) + meanseason / Vt.row(1);
//             break;
//         }
//         break;
//     case 'M':
//         meanseason = exp(mean(log(Vt.row(nComponents-1))));
//         Vt.row(nComponents-1) = Vt.row(nComponents-1) / meanseason;
//         switch(T){
//         case 'N':
//         case 'M':
//             Vt.row(0) = Vt.row(0) / meanseason;
//             break;
//         case 'A':
//             Vt.row(0) = Vt.row(0) * meanseason;
//             Vt.row(1) = Vt.row(1) * meanseason;
//             break;
//         }
//         break;
//     }
//
//     return(Vt);
// }
