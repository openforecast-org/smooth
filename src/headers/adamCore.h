#include "ssGeneral.h"
#include "adamGeneral.h"
#include "ssOccurrence.h"
#include "olsCore.h"
#include "adamGradient.h"

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
public:
    // Whether to flip the sign of the constant (drift) during the backward
    // pass of backcasting. Time reversal changes the drift of an integrated
    // series by (-1)^(d+D), so the flip is needed when the total number of
    // differences (non-seasonal + seasonal) is odd. Set from R/Python after
    // construction; defaults to false (no ARIMA differencing).
    bool flipConstant = false;

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

    // Private helper for gradientSolve(): one in-sample forward pass from the
    // given recent profile (mirrors the in-sample slice loop of reapply() with
    // backcast=false, including the head refinement). Writes the residuals into
    // vecErrors and returns false when they go non-finite.
    bool gradientPass(arma::mat profilesRecent,
                      arma::mat const &matrixYt, arma::mat const &matrixOt,
                      arma::mat const &matrixWt, arma::mat const &matrixF,
                      arma::vec const &vectorG, arma::umat const &indexLookupTable,
                      arma::vec &vecErrors) {
        int obs = matrixYt.n_rows;
        int lagsModelMax = max(lags);
        if(lagsModelMax > 1) {
            arma::mat scratchVt(profilesRecent.n_rows, lagsModelMax);
            refineHeadFwd(scratchVt, profilesRecent, matrixF,
                          indexLookupTable, lagsModelMax);
        }
        double yFit;
        for(int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
            int idx = i - lagsModelMax;
            yFit = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                              matrixWt.row(idx), E, T, S,
                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                              nComponents, constant);
            // Fix potential issue with negatives in mixed models
            if((E=='M' || T=='M' || S=='M') && (yFit<=0)){
                yFit = 1;
            }
            // Multiplication needed for cases when occurrence is fractional
            if(matrixOt(idx)!=0){
                yFit = matrixOt(idx) * yFit;
            }
            // errorf() returns 0 immediately when ot==0
            vecErrors(idx) = errorf(matrixYt(idx), yFit, E, matrixOt(idx));

            profilesRecent(indexLookupTable.col(i)) =
                adamFvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal,
                           nArima, nComponents, constant) +
                adamGvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, matrixWt.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                           nComponents, constant, vectorG, vecErrors(idx),
                           yFit, adamETS);
        }
        return vecErrors.is_finite();
    }

    // Private helper for gradientSolve(): one forward pass that also propagates
    // the sensitivities of every profile cell to the free initial parameters
    // (chain rule through the analytic derivatives in adamGradient.h), producing
    // the residuals AND the full Jacobian d(residual)/d(theta) in a single pass —
    // the analytic replacement for the nFree finite-difference probe passes.
    // sensitivities starts as the probe basis (d(profile cell)/d(theta)).
    // Returns false when the residuals or the Jacobian go non-finite.
    bool gradientPassJacobian(arma::mat profilesRecent, arma::mat sensitivities,
                              arma::mat const &matrixYt, arma::mat const &matrixOt,
                              arma::mat const &matrixWt, arma::mat const &matrixF,
                              arma::vec const &vectorG, arma::umat const &indexLookupTable,
                              arma::vec &vecErrors, arma::mat &jacobian) {
        int obs = matrixYt.n_rows;
        int lagsModelMax = max(lags);
        unsigned int nFree = sensitivities.n_cols;

        // Head refinement: refineHeadFwd walks the level/trend rows via
        // adamFvalue; the sensitivities follow through its Jacobian.
        if(lagsModelMax > 1 && T != 'N') {
            for(int i=1; i<lagsModelMax; i=i+1){
                arma::uvec cells = indexLookupTable.col(i);
                arma::vec vNew = adamFvalue(profilesRecent(cells), matrixF, E, T, S,
                                            nETS, nNonSeasonal, nSeasonal, nArima,
                                            nComponents, constant);
                arma::mat sensNew = adamFvalueJac(profilesRecent(cells), matrixF,
                                                  T, nComponents) *
                                    sensitivities.rows(cells);
                profilesRecent(cells.rows(0,1)) = vNew.rows(0,1);
                sensitivities.rows(cells.rows(0,1)) = sensNew.rows(0,1);
            }
        }

        arma::mat jacGv(nComponents, nComponents);
        arma::vec jacGe(nComponents), jacGy(nComponents);
        for(int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
            int idx = i - lagsModelMax;
            arma::uvec cells = indexLookupTable.col(i);
            arma::vec vCurrent = profilesRecent(cells);
            arma::mat sensCurrent = sensitivities.rows(cells);
            arma::rowvec wRow = matrixWt.row(idx);

            double yFit = adamWvalue(vCurrent, wRow, E, T, S,
                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                     nComponents, constant);
            arma::rowvec dyFit = adamWvalueJac(vCurrent, wRow, E, T, S,
                                               nComponents, nSeasonal) * sensCurrent;
            // The mixed-model clamp pins yhat to a constant, so its derivative
            // vanishes there (finite differences see the same flat spot).
            if((E=='M' || T=='M' || S=='M') && (yFit<=0)){
                yFit = 1;
                dyFit.zeros();
            }
            if(matrixOt(idx)!=0){
                yFit = matrixOt(idx) * yFit;
                dyFit = matrixOt(idx) * dyFit;
            }
            vecErrors(idx) = errorf(matrixYt(idx), yFit, E, matrixOt(idx));
            arma::rowvec dError(nFree, arma::fill::zeros);
            if(matrixOt(idx)!=0){
                // E='A': e = y - yhat; E='M': e = y/yhat - 1
                if(E=='A'){
                    dError = -dyFit;
                }
                else{
                    dError = -(matrixYt(idx) / (yFit*yFit)) * dyFit;
                }
            }
            jacobian.row(idx) = dError;

            adamGvalueJac(vCurrent, matrixF, wRow, E, T, S,
                          nComponents, nSeasonal, vectorG,
                          vecErrors(idx), yFit, adamETS,
                          jacGv, jacGe, jacGy);
            profilesRecent(cells) =
                adamFvalue(vCurrent, matrixF, E, T, S, nETS, nNonSeasonal,
                           nSeasonal, nArima, nComponents, constant) +
                adamGvalue(vCurrent, matrixF, wRow, E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                           nComponents, constant, vectorG, vecErrors(idx),
                           yFit, adamETS);
            sensitivities.rows(cells) =
                (adamFvalueJac(vCurrent, matrixF, T, nComponents) + jacGv) * sensCurrent +
                jacGe * dError + jacGy * dyFit;
        }
        return vecErrors.is_finite() && jacobian.is_finite();
    }

    // Private helper for gradientSolve() in occurrence mode (O = 'd'/'o'/'i'):
    // one forward pass of the occurrence recursion (mirrors fit()'s O-path:
    // vectorYt is the binary occurrence, errorf dispatches to
    // occurrenceError()). Writes the PROBABILITY residuals r = o - p into
    // vecResiduals (the om losses are separable in r) while the state update
    // consumes the occurrence error. Returns false on non-finite output.
    bool gradientPassOccurrence(arma::mat profilesRecent,
                                arma::mat const &matrixYt,
                                arma::mat const &matrixWt, arma::mat const &matrixF,
                                arma::vec const &vectorG, arma::umat const &indexLookupTable,
                                char const &O, arma::vec &vecResiduals) {
        int obs = matrixYt.n_rows;
        int lagsModelMax = max(lags);
        if(lagsModelMax > 1) {
            arma::mat scratchVt(profilesRecent.n_rows, lagsModelMax);
            refineHeadFwd(scratchVt, profilesRecent, matrixF,
                          indexLookupTable, lagsModelMax);
        }
        double p, dpdy;
        for(int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
            int idx = i - lagsModelMax;
            double yFit = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                                     matrixWt.row(idx), E, T, S,
                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                     nComponents, constant);
            double const error = errorf(matrixYt(idx), yFit, E, matrixYt(idx), O);
            occurrenceLinkJac(yFit, E, O, p, dpdy);
            vecResiduals(idx) = matrixYt(idx) - p;

            profilesRecent(indexLookupTable.col(i)) =
                adamFvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal,
                           nArima, nComponents, constant) +
                adamGvalue(profilesRecent(indexLookupTable.col(i)),
                           matrixF, matrixWt.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                           nComponents, constant, vectorG, error,
                           yFit, adamETS);
        }
        return vecResiduals.is_finite();
    }

    // Occurrence analog of gradientPassJacobian(): one pass producing the
    // probability residuals AND their exact Jacobian d(r)/d(theta). The
    // sensitivities chain through the occurrence link (dr = -dp/dyhat * dyhat)
    // for the design rows and through occurrenceErrorJac (de = de/dyhat * dyhat)
    // for the state-update channel.
    bool gradientPassJacobianOccurrence(arma::mat profilesRecent, arma::mat sensitivities,
                                        arma::mat const &matrixYt,
                                        arma::mat const &matrixWt, arma::mat const &matrixF,
                                        arma::vec const &vectorG, arma::umat const &indexLookupTable,
                                        char const &O,
                                        arma::vec &vecResiduals, arma::mat &jacobian) {
        int obs = matrixYt.n_rows;
        int lagsModelMax = max(lags);

        if(lagsModelMax > 1 && T != 'N') {
            for(int i=1; i<lagsModelMax; i=i+1){
                arma::uvec cells = indexLookupTable.col(i);
                arma::vec vNew = adamFvalue(profilesRecent(cells), matrixF, E, T, S,
                                            nETS, nNonSeasonal, nSeasonal, nArima,
                                            nComponents, constant);
                arma::mat sensNew = adamFvalueJac(profilesRecent(cells), matrixF,
                                                  T, nComponents) *
                                    sensitivities.rows(cells);
                profilesRecent(cells.rows(0,1)) = vNew.rows(0,1);
                sensitivities.rows(cells.rows(0,1)) = sensNew.rows(0,1);
            }
        }

        arma::mat jacGv(nComponents, nComponents);
        arma::vec jacGe(nComponents), jacGy(nComponents);
        double p, dpdy;
        for(int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
            int idx = i - lagsModelMax;
            arma::uvec cells = indexLookupTable.col(i);
            arma::vec vCurrent = profilesRecent(cells);
            arma::mat sensCurrent = sensitivities.rows(cells);
            arma::rowvec wRow = matrixWt.row(idx);

            double yFit = adamWvalue(vCurrent, wRow, E, T, S,
                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                     nComponents, constant);
            arma::rowvec const dyFit = adamWvalueJac(vCurrent, wRow, E, T, S,
                                                     nComponents, nSeasonal) * sensCurrent;
            double const error = errorf(matrixYt(idx), yFit, E, matrixYt(idx), O);
            occurrenceLinkJac(yFit, E, O, p, dpdy);
            vecResiduals(idx) = matrixYt(idx) - p;
            jacobian.row(idx) = -dpdy * dyFit;
            arma::rowvec const dError = occurrenceErrorJac(matrixYt(idx), yFit, E, O) * dyFit;

            adamGvalueJac(vCurrent, matrixF, wRow, E, T, S,
                          nComponents, nSeasonal, vectorG,
                          error, yFit, adamETS,
                          jacGv, jacGe, jacGy);
            profilesRecent(cells) =
                adamFvalue(vCurrent, matrixF, E, T, S, nETS, nNonSeasonal,
                           nSeasonal, nArima, nComponents, constant) +
                adamGvalue(vCurrent, matrixF, wRow, E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                           nComponents, constant, vectorG, error,
                           yFit, adamETS);
            sensitivities.rows(cells) =
                (adamFvalueJac(vCurrent, matrixF, T, nComponents) + jacGv) * sensCurrent +
                jacGe * dError + jacGy * dyFit;
        }
        return vecResiduals.is_finite() && jacobian.is_finite();
    }

    // Private helper: shared loop control for fit(), omfit(), omfitGeneral()
    template<typename ForwardFn, typename BackwardFn,
             typename HeadFillFwdFn, typename HeadFillBwdFn,
             typename TrendReversalFn>
    void fitLoopImpl(int obs, int lagsModelMax,
                     bool backcast, unsigned int nIterations,
                     ForwardFn forwardStep,
                     BackwardFn backwardStep,
                     HeadFillFwdFn headFillFwd,
                     HeadFillBwdFn headFillBwd,
                     TrendReversalFn trendReversal) {
        // Loop for the backcast
        for (unsigned int j=1; j<=nIterations; j=j+1) {
            // Refine the head so the initial level/trend land at position -lagsModelMax+1
            // and walk forward across the head cycle. Skip when lagsModelMax=1 (nothing to fill).
            if(lagsModelMax > 1) {
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
                headFillBwd();
                // Restore the specific element in the state vector
                trendReversal();
            }
        }
    }

    // Private helper: refine the head of a state matrix so that the initial
    // level/trend are placed at column 0 (observation -lagsModelMax+1) and the
    // trend rows are walked forward one step per column across the head cycle.
    // Uses this instance's T/E/S/nETS/... members. For the T=='N' case it just
    // copies the profile into the head columns (no trend to walk).
    void refineHeadFwd(arma::mat &matVt, arma::mat &profile,
                       arma::mat const &matF,
                       arma::umat const &lookup, int lagsModelMax) {
        if(T != 'N') {
            // Record the initial profile to the first column
            matVt.col(0) = profile(lookup.col(0));
            // Update the head, but only for the trend component
            for (int i=1; i<lagsModelMax; i=i+1) {
                profile(lookup.col(i).rows(0,1)) =
                    adamFvalue(profile(lookup.col(i)),
                               matF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                               nComponents, constant).rows(0,1);
                matVt.col(i) = profile(lookup.col(i));
            }
        } else {
            // No trend to walk; just seed the head columns from the profile
            for (int i=0; i<lagsModelMax; i=i+1) {
                matVt.col(i) = profile(lookup.col(i));
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
                  char const &O = 'n') {
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
            refineHeadFwd(matrixVt, profilesRecent, matrixF,
                          indexLookupTable, lagsModelMax);
        };

        // How to fix the head after the bakwards pass
        auto headFillBwd = [&]() {
            for (int i=lagsModelMax-1; i>=0; i=i-1) {
                profilesRecent(indexLookupTable.col(i)) =
                    adamFvalue(profilesRecent(indexLookupTable.col(i)),
                               matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
            }
        };

        // How to revert the trend component for backcasting.
        // The constant (drift) flips sign when the total order of differencing
        // is odd — the direct ARIMA analog of the ETS trend reversal.
        auto trendReversal = [&]() {
            if(T == 'A')      { profilesRecent(1) = -profilesRecent(1); }
            else if(T == 'M') { profilesRecent(1) = 1/profilesRecent(1); }
            if(constant && flipConstant) {
                profilesRecent(nComponents-1) = -profilesRecent(nComponents-1);
            }
        };

        // Do the fit!
        fitLoopImpl(obs, lagsModelMax, backcast, nIterations,
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
            bool const &backcast, unsigned int const &nIterations) {
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
            // Model A: full trend-aware refinement via the shared helper
            refineHeadFwd(matrixVtA, profilesRecentA, matrixFA,
                          indexLookupTableA, lagsModelMaxA);
            // Model B: plain copy (B typically has no trend; keep original behaviour)
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

        fitLoopImpl(obs, lagsModelMaxA, backcast, nIterations,
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

    // Method 5: Simulator - creates the simulated data based on the SSOE matrices.
    // ``refineHead=true`` walks the initial level/trend forward across the seasonal
    // head so the simulator's first observation reads the same state the fitter
    // would (used by ``sim.es`` / ``simulate.adam``). ``refineHead=false`` treats
    // the caller-supplied head columns as already positioned — that path is used
    // by the forecast-interval simulator, which feeds the fitted tail of the
    // state matrix rather than a raw initialiser output.
    SimulateResult simulate(arma::mat const &matrixErrors, arma::mat const &matrixOt,
                            arma::cube &arrayVt, arma::mat const &matrixWt,
                            arma::cube const &arrayF, arma::mat const &matrixG,
                            arma::umat const &indexLookupTable, arma::cube arrayProfile,
                            char const &E, bool const &refineHead){

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
            // Walk the initial level/trend forward across the head cycle so the
            // simulator's first observation reads the same state the fitter would.
            if(lagsModelMax > 1 && refineHead) {
                refineHeadFwd(matrixVt, profilesRecent, matrixF,
                              indexLookupTable, lagsModelMax);
            }
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
                          bool const &backcast){

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
                // Refine the head via the shared helper so it is walked one step
                // per column across the head cycle (or copied verbatim when T=='N').
                if(lagsModelMax > 1) {
                    // Bind slice views so refineHeadFwd can mutate them via references.
                    arma::mat sliceVt = arrayVt.slice(k);
                    arma::mat sliceProfile = arrayProfilesRecent.slice(k);
                    arma::mat sliceF = arrayF.slice(k);
                    // Note: reapply's original branch used the full profile column,
                    // not just the trend rows, so we call refineHeadFwd once here to
                    // match — the helper writes the trend-walked value into the
                    // level+trend rows and preserves the seasonal via the profile.
                    refineHeadFwd(sliceVt, sliceProfile, sliceF,
                                  indexLookupTable, lagsModelMax);
                    arrayVt.slice(k) = sliceVt;
                    arrayProfilesRecent.slice(k) = sliceProfile;
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
                    // The constant (drift) flips sign when the total order of
                    // differencing is odd — ARIMA analog of the trend reversal
                    if(constant && flipConstant){
                        arrayProfilesRecent.slice(k)(nComponents-1) =
                            -arrayProfilesRecent.slice(k)(nComponents-1);
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

                    // Fill in the head of the series (backward pass).
                    for(int i=lagsModelMax-1; i>=0; i=i-1) {
                        arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                            adamFvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                       arrayF.slice(k), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
                    }

                    // Change the specific element in the state vector to negative
                    if(T=='A'){
                        arrayProfilesRecent.slice(k)(1) = -arrayProfilesRecent.slice(k)(1);
                    }
                    else if(T=='M'){
                        arrayProfilesRecent.slice(k)(1) = 1/arrayProfilesRecent.slice(k)(1);
                    }
                    // Restore the constant sign after the backward pass
                    if(constant && flipConstant){
                        arrayProfilesRecent.slice(k)(nComponents-1) =
                            -arrayProfilesRecent.slice(k)(nComponents-1);
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

    // Method 6b: gradientSolve - loss-profiling solve for the initial recent
    // profile (initial="gradient"). Given fixed persistence, finds the initial
    // state minimising the estimation loss. probeBasis maps nFree free parameters
    // to profile cells (column j holds 1s on the cells that move together as one
    // parameter; profile candidates are profile + reshape(probeBasis*theta)).
    //
    // lossType and lossParams select the profiled loss (codes and rho documented
    // in adamGradient.h; 'S' is the SSE default, exact for likelihood+dnorm).
    // lossParams(0) carries the dgnorm beta for 'G' and the horizon for the
    // multistep codes.
    //
    // For additive ETS the residuals are affine in theta, so the design matrix
    // (the residual sensitivities) is propagated analytically alongside a single
    // forward pass; SSE is one pivoted-QR least squares, the separable robust
    // losses are IRLS sweeps on the affine surrogate, and the multistep losses
    // are solved on the affine multistep design replicated from the ferrors()
    // recursion. Otherwise Gauss-Newton (analytic or finite-difference Jacobian)
    // with loss-aware steps, a backtracking line search and a Levenberg-Marquardt
    // fallback is used; multistep losses fall back to the one-step SSE there.
    // The solve is stateless: the result is a deterministic function of the
    // inputs, which the optimiser requires and which keeps the R and Python
    // builds bit-identical.
    //
    // Returns the solved recent profile, or an empty matrix on failure (the
    // caller then falls back to the backcasting fit).
    arma::mat gradientSolve(arma::mat const &matrixYt, arma::mat const &matrixOt,
                            arma::mat const &matrixWt, arma::mat const &matrixF,
                            arma::vec const &vectorG, arma::umat const &indexLookupTable,
                            arma::mat const &profile, arma::mat const &probeBasis,
                            unsigned int const &nIterations, bool const &analytic,
                            char const &lossType, arma::vec const &lossParams,
                            char const &O){

        int obs = matrixYt.n_rows;
        int lagsModelMax = max(lags);
        unsigned int nFree = probeBasis.n_cols;
        arma::mat const failure(0, 0);
        if(nFree == 0){
            return failure;
        }

        // The solve is defined over the same forward pass the final fit runs,
        // so both branches share gradientPass(). Occurrence mode (O != 'n')
        // always goes through the Gauss-Newton branch: the probability link
        // makes the residuals nonlinear in the states even for E='A'.
        bool additive = (E=='A') && (T!='M') && (S!='M') && (O=='n');

        if(additive){
            // Resolve the effective loss for this branch. The multiplicative
            // likelihood losses cannot reach it (the wrappers map them for
            // E='M' only) -- treated defensively as SSE. Multistep losses need
            // a valid horizon and at least one forecast origin.
            char L = lossType;
            double const beta = (lossParams.n_elem>0) ? lossParams(0) : 2.0;
            int hor = 0;
            if(L=='l' || L=='g' || L=='i'){
                L = 'S';
            }
            if(gradientLossMultistep(L)){
                hor = (lossParams.n_elem>0) ? (int)lossParams(0) : 0;
                if(hor < 1 || obs - hor < 1){
                    L = 'S';
                }
            }
            bool const multistep = gradientLossMultistep(L);
            int const nOrigins = multistep ? (obs - hor) : 0;
            arma::mat msDesign(multistep ? nOrigins*hor : 0, nFree);
            arma::vec msResiduals(multistep ? nOrigins*hor : 0);

            // Affine case: one forward pass propagating the residual sensitivities.
            // S holds d(profile cell)/d(theta) for every profile cell (rows follow
            // the profile's column-major linear indices, which is exactly what
            // indexLookupTable contains), so S.rows(lookup) are the sensitivities
            // of the states involved at a given observation. All the involved maps
            // are linear here: measurement w'v, transition F*v, update g*e.
            arma::mat prof = profile;
            arma::mat sens = probeBasis;
            arma::vec residuals(obs);
            arma::mat design(obs, nFree);

            // Head refinement: refineHeadFwd walks the level/trend rows across the
            // head columns via adamFvalue (= F*v here); sensitivities follow.
            if(lagsModelMax > 1 && T != 'N'){
                for(int i=1; i<lagsModelMax; i=i+1){
                    arma::uvec cells = indexLookupTable.col(i);
                    arma::vec profNew = adamFvalue(prof(cells), matrixF, E, T, S,
                                                   nETS, nNonSeasonal, nSeasonal,
                                                   nArima, nComponents, constant);
                    arma::mat sensNew = matrixF * sens.rows(cells);
                    prof(cells.rows(0,1)) = profNew.rows(0,1);
                    sens.rows(cells.rows(0,1)) = sensNew.rows(0,1);
                }
            }

            for(int i=lagsModelMax; i<obs+lagsModelMax; i=i+1){
                int idx = i - lagsModelMax;
                arma::uvec cells = indexLookupTable.col(i);
                double yFit = adamWvalue(prof(cells), matrixWt.row(idx), E, T, S,
                                         nETS, nNonSeasonal, nSeasonal, nArima,
                                         nXreg, nComponents, constant);
                arma::rowvec dyFit = matrixWt.row(idx) * sens.rows(cells);
                if(matrixOt(idx)!=0){
                    yFit = matrixOt(idx) * yFit;
                    dyFit = matrixOt(idx) * dyFit;
                }
                residuals(idx) = errorf(matrixYt(idx), yFit, E, matrixOt(idx));
                // d(residual)/d(theta) = -dyFit when the point is observed;
                // errorf returns a constant 0 when ot==0.
                arma::rowvec dError = (matrixOt(idx)!=0) ? arma::rowvec(-dyFit)
                                                         : arma::rowvec(nFree, arma::fill::zeros);
                // The design column j is e0 - e_j (residual drop per unit probe),
                // i.e. minus the residual sensitivity.
                design.row(idx) = -dError;

                prof(cells) = adamFvalue(prof(cells), matrixF, E, T, S,
                                         nETS, nNonSeasonal, nSeasonal, nArima,
                                         nComponents, constant) +
                              adamGvalue(prof(cells), matrixF, matrixWt.row(idx),
                                         E, T, S, nETS, nNonSeasonal, nSeasonal,
                                         nArima, nXreg, nComponents, constant,
                                         vectorG, residuals(idx), yFit, adamETS);
                sens.rows(cells) = matrixF * sens.rows(cells) + vectorG * dError;

                // Multistep losses: replicate the ferrors() recursion from the
                // just-updated buffers -- h no-update forecast steps from every
                // origin, with the sensitivities following through F. Like
                // ferrors(), the multistep errors ignore the occurrence.
                if(multistep && idx < nOrigins){
                    arma::mat profStep = prof;
                    arma::mat sensStep = sens;
                    for(int k=0; k<hor; k=k+1){
                        arma::uvec const cellsStep = indexLookupTable.col(i+k);
                        double const yStep =
                            adamWvalue(profStep(cellsStep), matrixWt.row(idx+k),
                                       E, T, S, nETS, nNonSeasonal, nSeasonal,
                                       nArima, nXreg, nComponents, constant);
                        msResiduals(idx*hor+k) = matrixYt(idx+k) - yStep;
                        msDesign.row(idx*hor+k) =
                            matrixWt.row(idx+k) * sensStep.rows(cellsStep);
                        profStep(cellsStep) =
                            adamFvalue(profStep(cellsStep), matrixF, E, T, S,
                                       nETS, nNonSeasonal, nSeasonal, nArima,
                                       nComponents, constant);
                        sensStep.rows(cellsStep) = matrixF * sensStep.rows(cellsStep);
                    }
                }
            }

            if(!residuals.is_finite() || !design.is_finite()){
                return failure;
            }

            arma::vec theta;
            if(multistep){
                if(!msResiduals.is_finite() || !msDesign.is_finite()){
                    return failure;
                }
                theta = gradientLossMultistepSolve(msDesign, msResiduals,
                                                   L, hor, nOrigins);
            }
            else{
                // The SSE solution; for the separable robust losses it seeds
                // the IRLS sweeps on the same affine surrogate.
                theta = olsCore(design, residuals, 1e-7);
                theta.elem(arma::find_nonfinite(theta)).zeros();
                if(L != 'S'){
                    theta = gradientLossIrls(design, residuals, theta, L, beta);
                }
            }
            theta.elem(arma::find_nonfinite(theta)).zeros();
            return profile + arma::reshape(probeBasis * theta,
                                           profile.n_rows, profile.n_cols);
        }

        // Nonlinear case: Gauss-Newton with an analytic (or finite-difference)
        // Jacobian, loss-aware steps, a backtracking line search and a
        // Levenberg-Marquardt fallback. The solve is deliberately stateless
        // (always starts from the seed profile): the objective must be a
        // deterministic function of the inputs, both for the optimiser and for
        // exact parity between the R and Python builds.
        //
        // The multistep losses are out of scope here (their errors are not
        // affine and would need h extra passes per origin) -- they fall back to
        // the one-step SSE profile, as does everything the wrappers map to 'S'.
        // In occurrence mode only the probability-residual losses ('B' and the
        // power family) are defined.
        char L = lossType;
        if(gradientLossMultistep(L)){
            L = 'S';
        }
        if(O!='n' && L!='B' && L!='A' && L!='H' && L!='G'){
            L = 'S';
        }
        double const beta = (lossParams.n_elem>0) ? lossParams(0) : 2.0;
        double lossScale = 0;

        // One pass computing the solver residuals: errorf residuals on the
        // demand path, probability residuals r = o - p in occurrence mode.
        auto passResiduals = [&](arma::mat const &profCurrent, arma::vec &res) {
            if(O=='n'){
                return gradientPass(profCurrent, matrixYt, matrixOt, matrixWt,
                                    matrixF, vectorG, indexLookupTable, res);
            }
            return gradientPassOccurrence(profCurrent, matrixYt, matrixWt,
                                          matrixF, vectorG, indexLookupTable,
                                          O, res);
        };

        arma::mat prof = profile;
        arma::vec residuals(obs);
        if(!passResiduals(prof, residuals)){
            return failure;
        }
        double lossCurrent = gradientLossSum(residuals, L, beta, lossScale);
        // A saturated seed (e.g. 'B' with |r| >= 1) cannot be line-searched
        // against: hand the model back to the backcasting fallback.
        if(!std::isfinite(lossCurrent)){
            return failure;
        }

        // First profile cell of each probe: the finite-difference step is scaled
        // by the current value there, matching the original R implementation.
        arma::uvec firstCell(nFree);
        for(unsigned int j=0; j<nFree; j=j+1){
            arma::uvec nonZero = arma::find(probeBasis.col(j));
            firstCell(j) = nonZero(0);
        }

        arma::mat jacobian(obs, nFree);
        arma::vec residualsProbe(obs);
        double lossBeforeIteration;
        for(unsigned int iter=0; iter<nIterations; iter=iter+1){
            // Refresh the concentrated scale from the current residuals; the
            // objective is held at this fixed scale within the iteration.
            if(L=='l' || L=='i'){
                lossScale = gradientLossScale(residuals, L);
                lossCurrent = gradientLossSum(residuals, L, beta, lossScale);
            }
            lossBeforeIteration = lossCurrent;
            bool jacobianOk = true;
            if(analytic){
                // One pass propagating the exact sensitivities (adamGradient.h)
                // instead of nFree probe passes. The residuals it returns equal
                // the ones already held (same forward recursion).
                if(O=='n'){
                    jacobianOk = gradientPassJacobian(prof, probeBasis,
                                                      matrixYt, matrixOt, matrixWt,
                                                      matrixF, vectorG, indexLookupTable,
                                                      residualsProbe, jacobian);
                }
                else{
                    jacobianOk = gradientPassJacobianOccurrence(prof, probeBasis,
                                                                matrixYt, matrixWt,
                                                                matrixF, vectorG,
                                                                indexLookupTable, O,
                                                                residualsProbe, jacobian);
                }
            }
            else{
                // Finite-difference fallback (also the validation oracle).
                for(unsigned int j=0; j<nFree; j=j+1){
                    double h = 1e-4 * std::max(1.0, std::abs(prof(firstCell(j))));
                    arma::mat profProbe = prof + arma::reshape(h * probeBasis.col(j),
                                                               profile.n_rows, profile.n_cols);
                    if(!passResiduals(profProbe, residualsProbe)){
                        jacobianOk = false;
                        break;
                    }
                    jacobian.col(j) = (residualsProbe - residuals) / h;
                }
            }
            if(!jacobianOk){
                break;
            }

            // The step system: plain (J, e) for SSE; rows scaled by the IRLS /
            // Newton weights with the matching right-hand side otherwise
            // (gradientLossStepRow). For 'S' this is bit-identical to the
            // previous SSE-only implementation.
            arma::mat jacobianStep;
            arma::vec targetStep;
            if(L=='S'){
                jacobianStep = jacobian;
                targetStep = residuals;
            }
            else{
                arma::vec rowScale(obs);
                targetStep.set_size(obs);
                double a, b;
                for(int t=0; t<obs; t=t+1){
                    gradientLossStepRow(residuals(t), L, beta, lossScale, a, b);
                    rowScale(t) = a;
                    targetStep(t) = b;
                }
                jacobianStep = jacobian.each_col() % rowScale;
            }

            arma::vec step = -olsCore(jacobianStep, targetStep, 1e-7);
            step.elem(arma::find_nonfinite(step)).zeros();
            if(std::sqrt(arma::dot(step, step)) < 1e-8){
                break;
            }

            // Try the full step with a backtracking line search on the loss.
            bool improved = false;
            double stepSize = 1;
            for(int half=0; half<6; half=half+1){
                arma::mat profCandidate = prof + arma::reshape(probeBasis * (stepSize * step),
                                                               profile.n_rows, profile.n_cols);
                if(passResiduals(profCandidate, residualsProbe) &&
                   gradientLossSum(residualsProbe, L, beta, lossScale) < lossCurrent){
                    prof = profCandidate;

                    residuals = residualsProbe;
                    lossCurrent = gradientLossSum(residuals, L, beta, lossScale);
                    improved = true;
                    break;
                }
                stepSize = stepSize / 2;
            }

            // Levenberg-Marquardt fallback: when the raw step fails (typically
            // because the Jacobian is ill-conditioned and the step is wildly
            // oversized), solve the damped normal equations
            // [Jw; sqrt(lambda) I] step = [bw; 0] with increasing damping until
            // a step improves the loss. This keeps the descent property without
            // clipping anything: pure gradient-descent behaviour as lambda grows.
            if(!improved){
                double lambdaScale = arma::accu(arma::square(jacobianStep)) / nFree;
                arma::mat jacobianDamped(obs + nFree, nFree, arma::fill::zeros);
                jacobianDamped.rows(0, obs-1) = jacobianStep;
                arma::vec residualsDamped(obs + nFree, arma::fill::zeros);
                residualsDamped.rows(0, obs-1) = targetStep;
                for(int power=-2; power<=4 && !improved; power=power+2){
                    double lambda = lambdaScale * std::pow(10.0, power);
                    jacobianDamped.rows(obs, obs+nFree-1) =
                        std::sqrt(lambda) * arma::eye(nFree, nFree);
                    step = -olsCore(jacobianDamped, residualsDamped, 1e-7);
                    step.elem(arma::find_nonfinite(step)).zeros();
                    if(std::sqrt(arma::dot(step, step)) < 1e-8){
                        break;
                    }
                    arma::mat profCandidate = prof + arma::reshape(probeBasis * step,
                                                                   profile.n_rows, profile.n_cols);
                    if(passResiduals(profCandidate, residualsProbe) &&
                       gradientLossSum(residualsProbe, L, beta, lossScale) < lossCurrent){
                        prof = profCandidate;

                        residuals = residualsProbe;
                        lossCurrent = gradientLossSum(residuals, L, beta, lossScale);
                        improved = true;
                    }
                }
            }
            if(!improved){
                break;
            }
            // Diminishing returns: once an iteration improves the loss by less
            // than a relative 1e-4, further iterations are not worth their
            // Jacobian cost — the warm start carries the remaining convergence
            // across the optimiser's evaluations.
            if(lossCurrent > lossBeforeIteration * (1 - 1e-4)){
                break;
            }
        }


        return prof;
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
