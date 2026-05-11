#include "ssGeneral.h"
#include "adamGeneral.h"
#include "ssOccurrence.h"

// ============================================================================
// STRUCTURE DEFINITIONS
// ============================================================================

// Result structure for polynomialise
struct PolyResult {
    arma::vec arPolynomial;
    arma::vec iPolynomial;
    arma::vec ariPolynomial;
    arma::vec maPolynomial;
};

// Result structure for fitter
struct FitResult {
    arma::mat states;
    arma::vec fitted;
    arma::vec errors;
    arma::mat profile;
};

// Result structure for the general occurrence model fitter (two parallel models)
struct OmFitGeneralResult {
    arma::mat statesA;
    arma::vec fittedA;
    arma::vec errorsA;
    arma::mat profileA;
    arma::mat statesB;
    arma::vec fittedB;
    arma::vec errorsB;
    arma::mat profileB;
};

// Result structure for forecaster
struct ForecastResult {
    arma::vec forecast;
};

// Result structure for ferrors
struct ErrorResult {
    arma::mat errors;
};

// Result structure for simulator
struct SimulateResult {
    arma::cube states;
    arma::cube profile;
    arma::mat data;
};

// Result structure for refitter/reapply
struct ReapplyResult {
    arma::cube states;
    arma::mat fitted;
    arma::cube profile;
};

// Result structure for reforecaster
struct ReforecastResult {
    arma::cube data;
};

// ============================================================================
// ADAMCORE CLASS
// ============================================================================

class adamCore {
private:
    arma::uvec lags;
    char E;
    char T;
    char S;
    unsigned int nNonSeasonal;
    unsigned int nSeasonal;
    unsigned int nETS;
    unsigned int nArima;
    unsigned int nXreg;
    // Overall number of components
    unsigned int nComponents;
    bool constant;
    bool adamETS;

    // Private helper: shared loop control for fit(), omfit(), omfitGeneral()
    template<typename ForwardFn, typename BackwardFn,
             typename HeadFillFwdFn, typename HeadFillBwdFn,
             typename TrendReversalFn>
    void fitLoopImpl(int obs, int lagsModelMax,
                     bool backcast, unsigned int nIterations, bool refineHead,
                     ForwardFn forwardStep,
                     BackwardFn backwardStep,
                     HeadFillFwdFn headFillFwd,
                     HeadFillBwdFn headFillBwd,
                     TrendReversalFn trendReversal) {
        // Loop for the backcast
        for (unsigned int j=1; j<=nIterations; j=j+1) {
            // Refine the head (in order for it to make sense)
            // This is only needed for ETS(*,Z,Z) models, with trend.
            // This is not needed for lagsMax=1, because there is nothing to fill in
            if(lagsModelMax > 1 && refineHead) {
                headFillFwd();
            }
            ////// Run forward
            // Loop for the model construction
            for (int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
                forwardStep(i);
            }
            ////// Backwards run
            if(backcast && j<nIterations) {
                // Change the specific element in the state vector to negative/inverse
                trendReversal();
                for (int i=obs+lagsModelMax-1; i>=lagsModelMax; i=i-1) {
                    backwardStep(i);
                }
                // Fill in the head of the series.
                if(refineHead) {
                    headFillBwd();
                }
                // Restore the specific element in the state vector
                trendReversal();
            }
        }
    }

public:
    // Constructor
    adamCore(arma::uvec lags_, char E_, char T_, char S_,
             unsigned int nNonSeasonal_, unsigned int nSeasonal_,
             unsigned int nETS_, unsigned int nArima_, unsigned int nXreg_,
             unsigned int nComponents_,
             bool constant_, bool adamETS_) :
    lags(lags_), E(E_), T(T_), S(S_),
    nNonSeasonal(nNonSeasonal_), nSeasonal(nSeasonal_),
    nETS(nETS_), nArima(nArima_), nXreg(nXreg_),
    nComponents(nComponents_),
    constant(constant_), adamETS(adamETS_) {}

public:
    // Method 1: polynomialiser - returns polynomials for ARIMA
    PolyResult polynomialise(arma::vec const &B,
                             arma::uvec const &arOrders, arma::uvec const &iOrders, arma::uvec const &maOrders,
                             bool const &arEstimate, bool const &maEstimate,
                             arma::vec armaParameters, arma::uvec const &lagsARIMA){

        // Sometimes armaParameters is NULL. Treat this correctly
        arma::vec armaParametersValue;
        if(armaParameters.n_elem != 0){
            armaParametersValue = armaParameters;
        }

        // Form matrices with parameters, that are then used for polynomial multiplication
        arma::mat arParameters(max(arOrders % lagsARIMA)+1, arOrders.n_elem, arma::fill::zeros);
        arma::mat iParameters(max(iOrders % lagsARIMA)+1, iOrders.n_elem, arma::fill::zeros);
        arma::mat maParameters(max(maOrders % lagsARIMA)+1, maOrders.n_elem, arma::fill::zeros);

        arParameters.row(0).fill(1);
        iParameters.row(0).fill(1);
        maParameters.row(0).fill(1);

        int nParam = 0;
        int armanParam = 0;
        for(unsigned int i=0; i<lagsARIMA.n_rows; ++i){
            if(arOrders(i) * lagsARIMA(i) != 0){
                for(unsigned int j=0; j<arOrders(i); ++j){
                    if(arEstimate){
                        arParameters((j+1)*lagsARIMA(i),i) = -B(nParam);
                        nParam += 1;
                    }
                    else{
                        arParameters((j+1)*lagsARIMA(i),i) = -armaParametersValue(armanParam);
                        armanParam += 1;
                    }
                }
            }

            if(iOrders(i) * lagsARIMA(i) != 0){
                iParameters(lagsARIMA(i),i) = -1;
            }

            if(maOrders(i) * lagsARIMA(i) != 0){
                for(unsigned int j=0; j<maOrders(i); ++j){
                    if(maEstimate){
                        maParameters((j+1)*lagsARIMA(i),i) = B(nParam);
                        nParam += 1;
                    }
                    else{
                        maParameters((j+1)*lagsARIMA(i),i) = armaParametersValue(armanParam);
                        armanParam += 1;
                    }
                }
            }
        }

        // Prepare vectors with coefficients for polynomials
        arma::vec arPolynomial(sum(arOrders % lagsARIMA)+1, arma::fill::zeros);
        arma::vec iPolynomial(sum(iOrders % lagsARIMA)+1, arma::fill::zeros);
        arma::vec maPolynomial(sum(maOrders % lagsARIMA)+1, arma::fill::zeros);
        arma::vec ariPolynomial(sum(arOrders % lagsARIMA)+sum(iOrders % lagsARIMA)+1, arma::fill::zeros);
        arma::vec bufferPolynomial;

        arPolynomial.rows(0,arOrders(0)*lagsARIMA(0)) = arParameters.submat(0,0,arOrders(0)*lagsARIMA(0),0);
        iPolynomial.rows(0,iOrders(0)*lagsARIMA(0)) = iParameters.submat(0,0,iOrders(0)*lagsARIMA(0),0);
        maPolynomial.rows(0,maOrders(0)*lagsARIMA(0)) = maParameters.submat(0,0,maOrders(0)*lagsARIMA(0),0);

        for(unsigned int i=0; i<lagsARIMA.n_rows; ++i){
            // Form polynomials
            if(i!=0){
                bufferPolynomial = polyMult(arPolynomial, arParameters.col(i));
                arPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;

                bufferPolynomial = polyMult(maPolynomial, maParameters.col(i));
                maPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;

                bufferPolynomial = polyMult(iPolynomial, iParameters.col(i));
                iPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;
            }
            if(iOrders(i)>1){
                for(unsigned int j=1; j<iOrders(i); ++j){
                    bufferPolynomial = polyMult(iPolynomial, iParameters.col(i));
                    iPolynomial.rows(0,bufferPolynomial.n_rows-1) = bufferPolynomial;
                }
            }

        }
        // ariPolynomial contains 1 in the first place
        ariPolynomial = polyMult(arPolynomial, iPolynomial);

        // Check if the length of polynomials is correct. Fix if needed
        // This might happen if one of parameters became equal to zero
        if(maPolynomial.n_rows!=sum(maOrders % lagsARIMA)+1){
            maPolynomial.resize(sum(maOrders % lagsARIMA)+1);
        }
        if(ariPolynomial.n_rows!=sum(arOrders % lagsARIMA)+sum(iOrders % lagsARIMA)+1){
            ariPolynomial.resize(sum(arOrders % lagsARIMA)+sum(iOrders % lagsARIMA)+1);
        }
        if(arPolynomial.n_rows!=sum(arOrders % lagsARIMA)+1){
            arPolynomial.resize(sum(arOrders % lagsARIMA)+1);
        }

        PolyResult result;
        result.arPolynomial = arPolynomial;
        result.iPolynomial = iPolynomial;
        result.ariPolynomial = ariPolynomial;
        result.maPolynomial = maPolynomial;
        return result;
    }

    // Method 2: Fitter - fits SSOE model to the data.
    // For demand models (O='n', default): vectorYt is the demand series; vectorOt is the
    // occurrence multiplier (1, fractional, or 0 for intermittent).
    // For occurrence models (O='d'/'o'/'i'): pass vectorOt as both vectorYt and vectorOt;
    // the raw state-space output is transformed to a probability post-loop.
    FitResult fit(arma::mat matrixVt, arma::mat const &matrixWt,
                  arma::mat &matrixF, arma::vec const &vectorG,
                  arma::umat const &indexLookupTable, arma::mat profilesRecent,
                  arma::vec const &vectorYt, arma::vec const &vectorOt,
                  bool const &backcast, unsigned int const &nIterations,
                  bool const &refineHead, char const &O = 'n') {
        /* # matrixVt should have a length of obs + lagsModelMax.
         * # matrixWt is a matrix with nrows = obs
         * # vecG should be a vector
         * # lags is a vector of lags
         */

        int obs = vectorYt.n_rows;
        int lagsModelMax = max(lags);

        // Fitted values and the residuals
        arma::vec vecYfit(obs, arma::fill::zeros);
        arma::vec vecErrors(obs, arma::fill::zeros);

        // What to do in the forward pass
        auto forwardStep = [&](int i) {
            int idx = i - lagsModelMax;
            /* # Measurement equation and the error term */
            vecYfit(idx) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                    matrixWt.row(idx), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
            // We need this multiplication for cases, when occurrence is fractional
            if(vectorOt(idx) != 0) {
                vecYfit(idx) = vectorOt(idx) * vecYfit(idx);
            }
            // errorf() returns 0 when ot==0; dispatches to occurrenceError() when O!='n'
            vecErrors(idx) = errorf(vectorYt(idx), vecYfit(idx), E, vectorOt(idx), O);
            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) =
                adamFvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                           vectorG, vecErrors(idx), vecYfit(idx), adamETS);
            matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
        };

        // What to do in the backward pass
        auto backwardStep = [&](int i) {
            int idx = i - lagsModelMax;
            /* # Measurement equation and the error term */
            vecYfit(idx) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                    matrixWt.row(idx), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
            if(vectorOt(idx) != 0) {
                vecYfit(idx) = vectorOt(idx) * vecYfit(idx);
            }
            vecErrors(idx) = errorf(vectorYt(idx), vecYfit(idx), E, vectorOt(idx), O);
            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) =
                adamFvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                           vectorG, vecErrors(idx), vecYfit(idx), adamETS);
        };

        // How to fill in the head before the forward pass
        auto headFillFwd = [&]() {
            if(T != 'N') {
                // Record the initial profile to the first column
                matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
                // Update the head, but only for the trend component
                for (int i=1; i<lagsModelMax; i=i+1) {
                    profilesRecent(indexLookupTable.col(i).rows(0,1)) =
                        adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                   matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                                   nComponents, constant).rows(0,1);
                    matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                }
            } else {
                // Record the profile to the head of time series to fill in the state matrix
                for (int i=0; i<lagsModelMax; i=i+1) {
                    matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                }
            }
        };

        // How to fix the head after the bakwards pass
        auto headFillBwd = [&]() {
            for (int i=lagsModelMax-1; i>=0; i=i-1) {
                profilesRecent(indexLookupTable.col(i)) =
                    adamFvalue(profilesRecent(indexLookupTable.col(i)),
                               matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
            }
        };

        // How to revert the trend component for backcasting
        auto trendReversal = [&]() {
            if(T == 'A')      { profilesRecent(1) = -profilesRecent(1); }
            else if(T == 'M') { profilesRecent(1) = 1/profilesRecent(1); }
        };

        // Do the fit!
        fitLoopImpl(obs, lagsModelMax, backcast, nIterations, refineHead,
                    forwardStep, backwardStep, headFillFwd, headFillBwd, trendReversal);

        FitResult result;
        result.states = matrixVt;
        result.fitted = vecYfit;
        result.errors = vecErrors;
        result.profile = profilesRecent;
        return result;
    }

    // Method 2b: General occurrence model fitter (type 'g')
    // Fits two parallel ETS models (A and B) simultaneously.
    // this = model A; model B structural params are explicit.
    OmFitGeneralResult omfitGeneral(
            arma::mat matrixVtA, arma::mat const &matrixWtA,
            arma::mat &matrixFA, arma::vec const &vectorGA,
            arma::umat const &indexLookupTableA, arma::mat profilesRecentA,
            char const &EB, char const &TB, char const &SB,
            unsigned int const &nNonSeasonalB, unsigned int const &nSeasonalB,
            unsigned int const &nETSB, unsigned int const &nArimaB,
            unsigned int const &nXregB, unsigned int const &nComponentsB,
            bool const &constantB, bool const &adamETSB,
            arma::mat matrixVtB, arma::mat const &matrixWtB,
            arma::mat &matrixFB, arma::vec const &vectorGB,
            arma::umat const &indexLookupTableB, arma::mat profilesRecentB,
            arma::vec const &vectorOt,
            bool const &backcast, unsigned int const &nIterations,
            bool const &refineHead) {
        int obs = vectorOt.n_rows;
        int lagsModelMaxA = max(lags);
        // Model B may have different lags so we use indexLookupTableB.n_cols
        // lagsModelMax for the loop is taken from model A (same obs)
        arma::vec vecAfit(obs, arma::fill::zeros);
        arma::vec vecBfit(obs, arma::fill::zeros);
        arma::vec vecErrorsA(obs, arma::fill::zeros);
        arma::vec vecErrorsB(obs, arma::fill::zeros);

        auto forwardStep = [&](int i) {
            int idx = i - lagsModelMaxA;
            /* # Measurement equations for models A and B */
            vecAfit(idx) = adamWvalue(profilesRecentA(indexLookupTableA.col(i)),
                    matrixWtA.row(idx), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
            vecBfit(idx) = adamWvalue(profilesRecentB(indexLookupTableB.col(i)),
                    matrixWtB.row(idx), EB, TB, SB,
                    nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nXregB, nComponentsB, constantB);
            // Compute errors for both models jointly via occurrenceError(O='g')
            auto errs = occurrenceError(vectorOt(idx), vecAfit(idx), vecBfit(idx), E, EB, 'g');
            vecErrorsA(idx) = errs[0];
            vecErrorsB(idx) = errs[1];
            /* # Transition equations for models A and B */
            profilesRecentA(indexLookupTableA.col(i)) =
                adamFvalue(profilesRecentA(indexLookupTableA.col(i)),
                           matrixFA, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                adamGvalue(profilesRecentA(indexLookupTableA.col(i)), matrixFA, matrixWtA.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                           vectorGA, vecErrorsA(idx), vecAfit(idx), adamETS);
            profilesRecentB(indexLookupTableB.col(i)) =
                adamFvalue(profilesRecentB(indexLookupTableB.col(i)),
                           matrixFB, EB, TB, SB, nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nComponentsB, constantB) +
                adamGvalue(profilesRecentB(indexLookupTableB.col(i)), matrixFB, matrixWtB.row(idx), EB, TB, SB,
                           nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nXregB, nComponentsB, constantB,
                           vectorGB, vecErrorsB(idx), vecBfit(idx), adamETSB);
            matrixVtA.col(i) = profilesRecentA(indexLookupTableA.col(i));
            matrixVtB.col(i) = profilesRecentB(indexLookupTableB.col(i));
        };

        auto backwardStep = [&](int i) {
            int idx = i - lagsModelMaxA;
            /* # Measurement equations for models A and B */
            vecAfit(idx) = adamWvalue(profilesRecentA(indexLookupTableA.col(i)),
                    matrixWtA.row(idx), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
            vecBfit(idx) = adamWvalue(profilesRecentB(indexLookupTableB.col(i)),
                    matrixWtB.row(idx), EB, TB, SB,
                    nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nXregB, nComponentsB, constantB);
            auto errs = occurrenceError(vectorOt(idx), vecAfit(idx), vecBfit(idx), E, EB, 'g');
            vecErrorsA(idx) = errs[0];
            vecErrorsB(idx) = errs[1];
            /* # Transition equations for models A and B */
            profilesRecentA(indexLookupTableA.col(i)) =
                adamFvalue(profilesRecentA(indexLookupTableA.col(i)),
                           matrixFA, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                adamGvalue(profilesRecentA(indexLookupTableA.col(i)), matrixFA, matrixWtA.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                           vectorGA, vecErrorsA(idx), vecAfit(idx), adamETS);
            profilesRecentB(indexLookupTableB.col(i)) =
                adamFvalue(profilesRecentB(indexLookupTableB.col(i)),
                           matrixFB, EB, TB, SB, nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nComponentsB, constantB) +
                adamGvalue(profilesRecentB(indexLookupTableB.col(i)), matrixFB, matrixWtB.row(idx), EB, TB, SB,
                           nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nXregB, nComponentsB, constantB,
                           vectorGB, vecErrorsB(idx), vecBfit(idx), adamETSB);
        };

        auto headFillFwd = [&]() {
            if(T != 'N') {
                // Record the initial profile to the first column (model A)
                matrixVtA.col(0) = profilesRecentA(indexLookupTableA.col(0));
                // Update the head, but only for the trend component
                for (int i=1; i<lagsModelMaxA; i=i+1) {
                    profilesRecentA(indexLookupTableA.col(i).rows(0,1)) =
                        adamFvalue(profilesRecentA(indexLookupTableA.col(i)),
                                   matrixFA, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                                   nComponents, constant).rows(0,1);
                    matrixVtA.col(i) = profilesRecentA(indexLookupTableA.col(i));
                }
            } else {
                // Record the profile to the head of time series to fill in the state matrix
                for (int i=0; i<lagsModelMaxA; i=i+1) {
                    matrixVtA.col(i) = profilesRecentA(indexLookupTableA.col(i));
                }
            }
            // Model B head fill (always non-trend style for simplicity; B typically has no trend)
            for (int i=0; i<lagsModelMaxA; i=i+1) {
                matrixVtB.col(i) = profilesRecentB(indexLookupTableB.col(i));
            }
        };

        auto headFillBwd = [&]() {
            // Fill in the head of the series for both models
            for (int i=lagsModelMaxA-1; i>=0; i=i-1) {
                profilesRecentA(indexLookupTableA.col(i)) =
                    adamFvalue(profilesRecentA(indexLookupTableA.col(i)),
                               matrixFA, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
                profilesRecentB(indexLookupTableB.col(i)) =
                    adamFvalue(profilesRecentB(indexLookupTableB.col(i)),
                               matrixFB, EB, TB, SB, nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nComponentsB, constantB);
            }
        };

        // Reverse/restore both models' trends symmetrically
        auto trendReversal = [&]() {
            if(T == 'A')       { profilesRecentA(1) = -profilesRecentA(1); }
            else if(T == 'M')  { profilesRecentA(1) = 1/profilesRecentA(1); }
            if(TB == 'A')      { profilesRecentB(1) = -profilesRecentB(1); }
            else if(TB == 'M') { profilesRecentB(1) = 1/profilesRecentB(1); }
        };

        fitLoopImpl(obs, lagsModelMaxA, backcast, nIterations, refineHead,
                    forwardStep, backwardStep, headFillFwd, headFillBwd, trendReversal);

        OmFitGeneralResult result;
        result.statesA   = matrixVtA;
        result.fittedA   = vecAfit;
        result.errorsA   = vecErrorsA;
        result.profileA  = profilesRecentA;
        result.statesB   = matrixVtB;
        result.fittedB   = vecBfit;
        result.errorsB   = vecErrorsB;
        result.profileB  = profilesRecentB;
        return result;
    }

    // Method 3: Forecaster - produces forecasts for the adam
    ForecastResult forecast(arma::mat const &matrixWt, arma::mat const &matrixF,
                            arma::umat const &indexLookupTable, arma::mat profilesRecent,
                            unsigned int const &horizon) {

        arma::vec vecYfor(horizon, arma::fill::zeros);

        /* # Fill in the new xt matrix using F. Do the forecasts. */
        for (unsigned int i=0; i<horizon; i=i+1) {
            vecYfor.row(i) = adamWvalue(profilesRecent(indexLookupTable.col(i)), matrixWt.row(i), E, T, S,
                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

            profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
        }

        ForecastResult result;
        result.forecast = vecYfor;
        return result;
    }

    // Method 4: Forecast Errors - generates in-sample multistep forecasts error matrix
    ErrorResult ferrors(arma::mat matrixVt, arma::mat matrixWt,
                        arma::mat matrixF,
                        arma::umat const &indexLookupTable, arma::mat profilesRecent,
                        unsigned int const &horizon, arma::vec vectorYt) {
        unsigned int obs = vectorYt.n_rows;
        unsigned int lagsModelMax = max(lags);
        // This is needed for cases, when hor>obs
        unsigned int hh = 0;
        arma::mat matErrors(horizon, obs, arma::fill::zeros);

        // Fill in the head, similar to how it's done in the fitter
        for (unsigned int i=0; i<lagsModelMax; i=i+1) {
            profilesRecent(indexLookupTable.col(i)) = matrixVt.col(i);
        }

        for(unsigned int i = 0; i < (obs-horizon); i=i+1){
            hh = std::min(horizon, obs-i);
            // Update the profile to get the recent value from the state matrix
            // lagsModelMax moves the thing to the next obs. This way, we have the structure
            // similar to the fitter
            profilesRecent(indexLookupTable.col(i+lagsModelMax)) = matrixVt.col(i+lagsModelMax);
            // This needs to take probability of occurrence into account in order to deal with intermittent models
            // The problem is that the probability needs to be a matrix, i.e. to reflect multistep from each point
            matErrors.submat(0, i, hh-1, i) =
                errorvf(vectorYt.rows(i, i+hh-1),
                        forecast(matrixWt.rows(i,i+hh-1), matrixF,
                                 indexLookupTable.cols(i+lagsModelMax,i+lagsModelMax+hh-1), profilesRecent,
                                 hh).forecast,
                                 // vectorPt.rows(i, i+hh-1),
                                 E);
        }

        // Cut-off the redundant last part
        if(obs>horizon){
            matErrors = matErrors.cols(0,obs-horizon-1);
        }

        ErrorResult result;
        result.errors = matErrors.t();
        return result;
    }

    // Method 5: Simulator - creates the simulated data based on the SSOE matrices
    SimulateResult simulate(arma::mat const &matrixErrors, arma::mat const &matrixOt,
                            arma::cube &arrayVt, arma::mat const &matrixWt,
                            arma::cube const &arrayF, arma::mat const &matrixG,
                            arma::umat const &indexLookupTable, arma::cube arrayProfile, char const &E){

        unsigned int obs = matrixErrors.n_rows;
        unsigned int nSeries = matrixErrors.n_cols;

        int lagsModelMax = max(lags);
        int obsAll = obs + lagsModelMax;

        double yFitted;

        arma::mat matrixVt(nComponents, obsAll, arma::fill::zeros);
        arma::mat matrixF(arrayF.n_rows, arrayF.n_cols, arma::fill::zeros);
        arma::mat profilesRecent(arrayProfile.n_rows, arrayProfile.n_cols, arma::fill::zeros);

        arma::mat matY(obs, nSeries);

        for(unsigned int i=0; i<nSeries; i=i+1){
            matrixVt = arrayVt.slice(i);
            matrixF = arrayF.slice(i);
            profilesRecent = arrayProfile.slice(i);
            for(int j=lagsModelMax; j<obsAll; j=j+1) {
                /* # Measurement equation and the error term */
                yFitted = adamWvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                     matrixWt.row(j-lagsModelMax), E, T, S,
                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                     nComponents, constant);
                matY(j-lagsModelMax,i) = matrixOt(j-lagsModelMax,i) *
                    (yFitted +
                    adamRvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                               matrixWt.row(j-lagsModelMax), E, T, S,
                               nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                   matrixErrors(j-lagsModelMax,i));

                /* # Transition equation */
                profilesRecent(indexLookupTable.col(j-lagsModelMax)) =
                (adamFvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                            matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                            nComponents, constant) +
                                adamGvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                           matrixF, matrixWt.row(j-lagsModelMax),
                                           E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                           nComponents, constant, matrixG.col(i),
                                           matrixErrors(j-lagsModelMax,i), yFitted, adamETS));

                matrixVt.col(j) = profilesRecent(indexLookupTable.col(j-lagsModelMax));
            }
            arrayVt.slice(i) = matrixVt;
            arrayProfile.slice(i) = profilesRecent;
        }

        SimulateResult result;
        result.states = arrayVt;
        result.profile = arrayProfile;
        result.data = matY;
        return result;
    }

    // Method 6: Refit - function reapplies ADAM to the data with different parameters
    ReapplyResult reapply(arma::mat const &matrixYt, arma::mat const &matrixOt,
                          arma::cube &arrayVt, arma::cube const &arrayWt,
                          arma::cube const &arrayF, arma::mat const &matrixG,
                          arma::umat const &indexLookupTable, arma::cube arrayProfilesRecent,
                          bool const &backcast, bool const &refineHead){

        int obs = matrixYt.n_rows;
        unsigned int nSeries = matrixG.n_cols;

        // nIterations=1 means that we don't do backcasting
        // It doesn't seem to matter anyway...
        unsigned int nIterations = 1;
        if(backcast){
            nIterations = 2;
        }

        int lagsModelMax = max(lags);

        arma::mat matYfit(obs, nSeries, arma::fill::zeros);
        arma::vec vecErrors(obs, arma::fill::zeros);

        for(unsigned int k=0; k<nSeries; k=k+1){
            // Loop for the backcasting
            for (unsigned int j=1; j<=nIterations; j=j+1) {
                // Refine the head (in order for it to make sense)
                if(lagsModelMax>1){
                    if(refineHead && (T!='N')){
                        // Record the initial profile to the first column
                        arrayVt.slice(k).col(0) = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(0));

                        for(int i=1; i<lagsModelMax; i=i+1) {
                            arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                                adamFvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                           arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
                            arrayVt.slice(k).col(i) = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i));
                        }
                    }
                    else if(refineHead){
                        // Record the profile to the head of time series to fill in the state matrix
                        for (int i=0; i<lagsModelMax; i=i+1) {
                            arrayVt.slice(k).col(i) = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i));
                        }
                    }
                }
                // Loop for the model construction
                for(int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
                    /* # Measurement equation and the error term */
                    matYfit(i-lagsModelMax,k) = adamWvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                            arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                            nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                    // Fix potential issue with negatives in mixed models
                    if((E=='M' || T=='M' || S=='M') && (matYfit(i-lagsModelMax,k)<=0)){
                        matYfit(i-lagsModelMax,k) = 1;
                    }

                    // We need this multiplication for cases, when occurrence is fractional
                    if(matrixOt(i-lagsModelMax)!=0){
                        matYfit(i-lagsModelMax,k) = matrixOt(i-lagsModelMax) * matYfit(i-lagsModelMax,k);
                    }
                    // errorf() returns 0 immediately when ot==0
                    vecErrors(i-lagsModelMax) = errorf(matrixYt(i-lagsModelMax), matYfit(i-lagsModelMax,k), E,
                                                       matrixOt(i-lagsModelMax));

                    /* # Transition equation */
                    arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                    adamFvalue(arrayProfilesRecent.slice(k)(indexLookupTable.col(i)),
                               arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                   adamGvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                              arrayF.slice(k), arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                              matrixG.col(k), vecErrors(i-lagsModelMax), matYfit(i-lagsModelMax,k), adamETS);

                    arrayVt.slice(k).col(i) = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i));
                }

                ////// Backwards run
                if(backcast && j<(nIterations)){
                    // Change the specific element in the state vector to negative
                    if(T=='A'){
                        arrayProfilesRecent.slice(k)(1) = -arrayProfilesRecent.slice(k)(1);
                    }
                    else if(T=='M'){
                        arrayProfilesRecent.slice(k)(1) = 1/arrayProfilesRecent.slice(k)(1);
                    }

                    for(int i=obs+lagsModelMax-1; i>=lagsModelMax; i=i-1) {
                        /* # Measurement equation and the error term */
                        matYfit(i-lagsModelMax,k) = adamWvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                                nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                        // Fix potential issue with negatives in mixed models
                        if((E=='M' || T=='M' || S=='M') && (matYfit(i-lagsModelMax,k)<=0)){
                            matYfit(i-lagsModelMax,k) = 1;
                        }

                        if(matrixOt(i-lagsModelMax)!=0){
                            matYfit(i-lagsModelMax,k) = matrixOt(i-lagsModelMax) * matYfit(i-lagsModelMax,k);
                        }
                        vecErrors(i-lagsModelMax) = errorf(matrixYt(i-lagsModelMax), matYfit(i-lagsModelMax,k), E,
                                                           matrixOt(i-lagsModelMax));

                        /* # Transition equation */
                        arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                        adamFvalue(arrayProfilesRecent.slice(k)(indexLookupTable.col(i)),
                                   arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                       adamGvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                                  arrayF.slice(k), arrayWt.slice(k).row(i-lagsModelMax), E, T, S,
                                                  nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                                  matrixG.col(k), vecErrors(i-lagsModelMax), matYfit(i-lagsModelMax,k), adamETS);
                    }

                    if(refineHead){
                        // Fill in the head of the series.
                        for(int i=lagsModelMax-1; i>=0; i=i-1) {
                            arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                                adamFvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                           arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
                        }
                    }

                    // Change the specific element in the state vector to negative
                    if(T=='A'){
                        arrayProfilesRecent.slice(k)(1) = -arrayProfilesRecent.slice(k)(1);
                    }
                    else if(T=='M'){
                        arrayProfilesRecent.slice(k)(1) = 1/arrayProfilesRecent.slice(k)(1);
                    }
                }
            }
        }

        ReapplyResult result;
        result.states = arrayVt;
        result.fitted = matYfit;
        result.profile = arrayProfilesRecent;
        return result;
    }

    // Method 7: Reforecast - produce many forecasts given the matrices
    ReforecastResult reforecast(arma::cube const &arrayErrors, arma::cube const &arrayOt,
                                arma::cube const &arrayWt,
                                arma::cube const &arrayF, arma::mat const &matrixG,
                                arma::umat const &indexLookupTable, arma::cube arrayProfileRecent,
                                char const &E){

        unsigned int obs = arrayErrors.n_rows;
        unsigned int nSeries = arrayErrors.n_cols;
        unsigned int nsim = arrayErrors.n_slices;

        unsigned int lagsModelMax = max(lags);

        double yFitted;

        arma::cube arrY(obs, nSeries, nsim);

        for(unsigned int j=0; j<nsim; j=j+1){
            for(unsigned int k=0; k<nSeries; k=k+1){
                for(unsigned int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
                    /* # Measurement equation and the error term */
                    yFitted = adamWvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                         arrayWt.slice(j).row(i-lagsModelMax), E, T, S,
                                         nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                    arrY(i-lagsModelMax,k,j) = arrayOt(i-lagsModelMax,k,j) *
                        (yFitted + adamRvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                              arrayWt.slice(j).row(i-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                                  arrayErrors.slice(j)(i-lagsModelMax,k));

                    // Fix potential issue with negatives in mixed models
                    if((E=='M' || T=='M' || S=='M') && (arrY(i-lagsModelMax,k,j)<0)){
                        arrY(i-lagsModelMax,k,j) = 0;
                    }

                    /* # Transition equation */
                    arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)) =
                    (adamFvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                arrayF.slice(j), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                    adamGvalue(arrayProfileRecent.slice(j).elem(indexLookupTable.col(i-lagsModelMax)),
                                               arrayF.slice(j), arrayWt.slice(j).row(i-lagsModelMax),
                                               E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                               nComponents, constant, matrixG.col(k),
                                               arrayErrors.slice(j)(i-lagsModelMax,k), yFitted, adamETS));
                }
            }
        }

        ReforecastResult result;
        result.data = arrY;
        return result;
    }
};
