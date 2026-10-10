#include "ssGeneral.h"
#include "adamGeneral.h"
#include "ssOccurrence.h"
#include "olsCore.h"
#include "adamGradient.h"
#include "arimaBounds.h"

// ============================================================================
// STRUCTURE DEFINITIONS
// ============================================================================

// Result structure for polynomialise
struct PolyResult {
    arma::vec arPolynomial;
    arma::vec iPolynomial;
    arma::vec ariPolynomial;
    arma::vec maPolynomial;
    // The largest reflection coefficients of the AR / MA factors (arimaBounds.h):
    // below one exactly when the AR part is stationary / the MA one invertible
    double arReflection;
    double maReflection;
};

// Result structure for fitter
struct FitResult {
    arma::mat states;
    arma::vec fitted;
    arma::vec errors;
    arma::mat profile;
    // The profile the fit started from (the solved one of initial="gradient")
    arma::mat profileInitial;
};

// Result structure for the filter (filter()): the fitter's fields, and for each period the
// variance of the one-step error relative to sigma^2 (or to s in logs) and, in logs, the
// derivative of the innovation in the scale s
struct FilterResult {
    arma::mat states;
    arma::vec fitted;
    arma::vec errors;
    arma::mat profile;
    arma::mat profileInitial;
    arma::vec variances;
    arma::vec innovationsScale;
};

// The state of the filter over the profile cells the model uses: the means of the states
// (in logs for the ETS components of a pure multiplicative ADAM ETS), their derivative in
// the scale, and the covariance of the states relative to sigma^2 (or s)
struct FilterState {
    bool logs;
    double scale;
    arma::uvec cells;
    arma::uvec compact;
    arma::vec means;
    arma::vec meansScale;
    arma::mat covariance;
    bool linear = false;
    bool covarianceActive = false;
    arma::vec variances;
    arma::vec innovationsScale;
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

// Result structure for the coupled (general) occurrence gradient solve: the
// two solved recent profiles. An empty pair signals failure (caller falls back
// to backcasting).
struct GradientSolveGeneralResult {
    arma::mat profileA;
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

    // Length of the zero-error head that headFillBwd() produces and that the final forward pass
    // filters over. 0 means "use lagsModelMax", i.e. the behaviour before this change.
    unsigned int headLength = 0;

    // The transition of fit() and reapply() as a sparse product (SparseTransition), for the large and
    // mostly zero matrices of tbats() and ssarima(). Set from R/Python; false otherwise.
    bool sparseTransition = false;

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
            // A missing observation does not enter the least squares
            vecResiduals(idx) = std::isnan(matrixYt(idx)) ? 0 : matrixYt(idx) - p;

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
            // A missing observation does not enter the least squares
            bool const missing = std::isnan(matrixYt(idx));
            vecResiduals(idx) = missing ? 0 : matrixYt(idx) - p;
            jacobian.row(idx) = missing ? arma::rowvec(dyFit.n_elem, arma::fill::zeros) : arma::rowvec(-dpdy * dyFit);
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

    // Linear-state occurrence Jacobian: for an additive-error occurrence
    // sub-model (E='A', no multiplicative trend/season) the whole state -> yhat
    // map is linear (yhat = w'v, transition F*v, update F*v + g*e) over the FULL
    // component vector -- ETS, ARIMA, xreg and constant alike. So the exact
    // sensitivities need no per-branch ETS companions (which cover ETS columns
    // only): d(yhat) = w'*S, and the update Jacobian is F*S + g*d(e). Only the
    // occurrence link/error stay nonlinear (occurrenceLinkJac / occurrenceErrorJac).
    // This replaces the finite-difference Jacobian for occurrence models with
    // ARIMA / xreg, and is bit-identical to gradientPassJacobianOccurrence for
    // pure additive ETS (there jacGv = jacGy = 0, jacGe = g). Multiplicative-error
    // occurrence still uses the companion pass (pure ETS) or FD (with extras).
    bool gradientPassJacobianOccurrenceLinear(arma::mat profilesRecent, arma::mat sensitivities,
                                              arma::mat const &matrixYt,
                                              arma::mat const &matrixWt, arma::mat const &matrixF,
                                              arma::vec const &vectorG,
                                              arma::umat const &indexLookupTable, char const &O,
                                              arma::vec &vecResiduals, arma::mat &jacobian) {
        int obs = matrixYt.n_rows;
        int lagsModelMax = max(lags);

        // Head refinement: walk the level/trend rows across the head columns
        // (linear transition => sensitivities follow through F).
        if(lagsModelMax > 1 && T != 'N') {
            for(int i=1; i<lagsModelMax; i=i+1){
                arma::uvec cells = indexLookupTable.col(i);
                arma::vec vNew = adamFvalue(profilesRecent(cells), matrixF, E, T, S,
                                            nETS, nNonSeasonal, nSeasonal, nArima,
                                            nComponents, constant);
                arma::mat sensNew = matrixF * sensitivities.rows(cells);
                profilesRecent(cells.rows(0,1)) = vNew.rows(0,1);
                sensitivities.rows(cells.rows(0,1)) = sensNew.rows(0,1);
            }
        }

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
            arma::rowvec const dyFit = wRow * sensCurrent;   // linear: d(yhat) = w'S
            double const error = errorf(matrixYt(idx), yFit, E, matrixYt(idx), O);
            occurrenceLinkJac(yFit, E, O, p, dpdy);
            // A missing observation does not enter the least squares
            bool const missing = std::isnan(matrixYt(idx));
            vecResiduals(idx) = missing ? 0 : matrixYt(idx) - p;
            jacobian.row(idx) = missing ? arma::rowvec(dyFit.n_elem, arma::fill::zeros) : arma::rowvec(-dpdy * dyFit);
            arma::rowvec const dError = occurrenceErrorJac(matrixYt(idx), yFit, E, O) * dyFit;

            profilesRecent(cells) =
                adamFvalue(vCurrent, matrixF, E, T, S, nETS, nNonSeasonal,
                           nSeasonal, nArima, nComponents, constant) +
                adamGvalue(vCurrent, matrixF, wRow, E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                           nComponents, constant, vectorG, error, yFit, adamETS);
            sensitivities.rows(cells) = matrixF * sensCurrent + vectorG * dError;
        }
        return vecResiduals.is_finite() && jacobian.is_finite();
    }

    // Private helper: shared loop control for fit(), omfit(), omfitGeneral()
    template<typename ForwardFn, typename BackwardFn,
             typename HeadFillFwdFn, typename HeadFillBwdFn,
             typename TrendReversalFn, typename HeadForwardFn, typename TailTurnFn>
    void fitLoopImpl(int obs, int H,
                     bool backcast, unsigned int nIterations,
                     ForwardFn forwardStep,
                     BackwardFn backwardStep,
                     HeadFillFwdFn headFillFwd,
                     HeadFillBwdFn headFillBwd,
                     TrendReversalFn trendReversal,
                     HeadForwardFn headForwardStep,
                     bool useHeadFilter,
                     TailTurnFn tailTurn) {
        // Without backcasting there is nothing to iterate: one forward pass, no backward run.
        if(!backcast) { nIterations = 1; }
        // Loop for the backcast
        for (unsigned int j=1; j<=nIterations; j=j+1) {
            if(j == 1 || !useHeadFilter) {
                // Refine the head so the initial level/trend land at position -H+1
                // and walk forward across the head cycle. With H=1 this only writes
                // the head column, which after a backcast holds the backcasted states.
                headFillFwd();
            } else {
                headForwardStep();
            }
            ////// Run forward
            // Loop for the model construction
            for (int i=H; i<obs+H; i=i+1) {
                forwardStep(i);
            }
            ////// Backwards run
            if(backcast && j<nIterations) {
                // Move the states past the end of the sample, where the backward pass reads them
                tailTurn();
                // Change the specific element in the state vector to negative/inverse
                trendReversal();
                for (int i=obs+H-1; i>=H; i=i-1) {
                    backwardStep(i);
                }
                // Move the states into the head of the series.
                headFillBwd();
                // Restore the specific element in the state vector
                trendReversal();
            }
        }
    }

    // Overload for omfitGeneral (no head filter)
    template<typename ForwardFn, typename BackwardFn,
             typename HeadFillFwdFn, typename HeadFillBwdFn,
             typename TrendReversalFn, typename TailTurnFn>
    void fitLoopImpl(int obs, int lagsModelMax,
                     bool backcast, unsigned int nIterations,
                     ForwardFn forwardStep,
                     BackwardFn backwardStep,
                     HeadFillFwdFn headFillFwd,
                     HeadFillBwdFn headFillBwd,
                     TrendReversalFn trendReversal,
                     TailTurnFn tailTurn) {
        auto noopHeadForward = [&]() {
            if(lagsModelMax > 1) {
                headFillFwd();
            }
        };
        fitLoopImpl(obs, lagsModelMax, backcast, nIterations,
                    forwardStep, backwardStep, headFillFwd, headFillBwd,
                    trendReversal, noopHeadForward, false, tailTurn);
    }

    // Private helper: one turn of the backcast. The backward pass reads a state with lag L
    // at t+L, the forward pass at t-L, so at a turn each state is moved exactly L zero-error
    // steps with the model's own transition: forward from time s0=obs+1 (dir=+1) before the
    // backward pass, backward from s0=0 (dir=-1) after it, plus `extra` steps when the head
    // is filtered. The slope states (trend, drift) are flipped around the backward pass by
    // trendReversal(), as before. Optionally records, for the first nCasts times, the
    // model's predictions (the pseudo-observations of the head filter) and the states after
    // each step (the head of matVt, from time 0 backwards). rowLags is passed explicitly so
    // that model B of omfitGeneral, with its own lags, can use the same helper.
    void turnPerLag(arma::mat &profile, arma::mat const &matrixF, arma::uvec const &rowLags,
                    long s0, int dir, unsigned int extra, arma::rowvec const &rowW,
                    char const E_, char const T_, char const S_,
                    unsigned int const nETS_, unsigned int const nNonSeasonal_,
                    unsigned int const nSeasonal_, unsigned int const nArima_,
                    unsigned int const nXreg_, unsigned int const nComponents_,
                    bool const constant_, arma::vec *casts=nullptr,
                    arma::mat *trajectory=nullptr) const {
        unsigned int nCasts = (casts == nullptr) ? 0 : casts->n_elem;
        unsigned int nTrajectory = (trajectory == nullptr) ? 0 : trajectory->n_cols;
        unsigned int nSteps = std::max(extra + (unsigned int) rowLags.max(), nTrajectory);
        arma::mat scratch = profile;
        arma::uvec cells(nComponents_);
        for(unsigned int k=1; k<=nSteps; ++k) {
            // The profile cells read at time t: the lookup table's phase of each row
            long t = s0 + dir * (long)(k - 1);
            for(unsigned int r=0; r<nComponents_; ++r) {
                long L = rowLags(r);
                cells(r) = r + nComponents_ * (((t - 1) % L + L) % L);
            }
            if(k <= nCasts) {
                (*casts)(dir < 0 ? nCasts - k : k - 1) =
                    adamWvalue(scratch(cells), rowW, E_, T_, S_, nETS_, nNonSeasonal_,
                               nSeasonal_, nArima_, nXreg_, nComponents_, constant_);
            }
            scratch(cells) = adamFvalue(scratch(cells), matrixF, E_, T_, S_, nETS_, nNonSeasonal_,
                                        nSeasonal_, nArima_, nComponents_, constant_);
            if(k <= nTrajectory) {
                trajectory->col(nTrajectory - k) = scratch(cells);
            }
            for(unsigned int r=0; r<nComponents_; ++r) {
                if(extra + rowLags(r) == k) {
                    profile.row(r) = scratch.row(r);
                }
            }
        }
    }

    // Private helper: the lag of each row of a lookup table, as the number of distinct
    // profile cells it uses over the sample (the calendar shifts reuse existing cells)
    static arma::uvec lookupLags(arma::umat const &indexLookupTable, unsigned int H, int obs) {
        arma::uvec rowLags(indexLookupTable.n_rows);
        for(unsigned int r=0; r<indexLookupTable.n_rows; ++r) {
            rowLags(r) = arma::unique(indexLookupTable.row(r).cols(H, H + obs - 1)).eval().n_elem;
        }
        return rowLags;
    }

    // Private helper: undo the backward-pass flip of the trend in recorded states
    static void unflipTrend(arma::mat &states, char const T_) {
        if(T_ == 'A') { states.row(1) = -states.row(1); }
        else if(T_ == 'M') { states.row(1) = 1/states.row(1); }
    }

    // Private helper: the constant (drift) of the time-reversed series. It is additive
    // with an additive error and a ratio with a multiplicative one, so time reversal
    // takes minus it in the first case and its inverse in the second, as for the trend
    void flipDrift(double &drift) const {
        drift = (E == 'M') ? 1/drift : -drift;
    }

    // Private helper: undo the backward-pass flips of the slope states in recorded states
    void unflipSlopes(arma::mat &states) const {
        unflipTrend(states, T);
        if(constant && flipConstant) {
            states.row(nComponents-1).transform([this](double drift){ flipDrift(drift); return drift; });
        }
    }

    // Private helper: is the backcast turn already the exact time reversal? It is for a
    // pure ETS model without a damped trend (the per-lag turns plus the trend flip), so
    // filtering the head there cannot change the result and is pure cost. ARIMA states,
    // regressors and a damped trend are not exactly reversed by the turns, so the head is
    // filtered for them.
    bool headFlipIsExact(arma::mat const &matrixF) const {
        if(nArima != 0 || nXreg != 0) { return false; }
        if(T == 'N') { return true; }
        return (matrixF.n_rows > 1) && (matrixF(1,1) == 1.0);
    }

    // Private helper: refine the head of a state matrix so that the initial
    // level/trend are placed at column 0 (observation -headLength+1) and the
    // trend rows are walked forward one step per column across the head cycle.
    // Uses this instance's T/E/S/nETS/... members. For the T=='N' case it just
    // copies the profile into the head columns (no trend to walk).
    void refineHeadFwd(arma::mat &matVt, arma::mat &profile,
                       arma::mat const &matF,
                       arma::umat const &lookup, int lagsModelMax, int H) {
        int headOffset = H - lagsModelMax;
        if(T != 'N') {
            for (int i=0; i<headOffset; i=i+1) {
                matVt.col(i) = profile(lookup.col(i));
            }
            // Record the initial profile to the start of the seasonal cycle
            profile(lookup.col(headOffset).rows(0,1)) = profile(lookup.col(0).rows(0,1));
            matVt.col(headOffset) = profile(lookup.col(headOffset));
            // Update the head, but only for the trend component
            for (int i=headOffset+1; i<H; i=i+1) {
                profile(lookup.col(i).rows(0,1)) =
                    adamFvalue(profile(lookup.col(i)),
                               matF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                               nComponents, constant).rows(0,1);
                matVt.col(i) = profile(lookup.col(i));
            }
        } else {
            // No trend to walk; just seed the head columns from the profile
            for (int i=0; i<H; i=i+1) {
                matVt.col(i) = profile(lookup.col(i));
            }
        }
    }

    void refineHeadFwd(arma::mat &matVt, arma::mat &profile,
                       arma::mat const &matF,
                       arma::umat const &lookup, int lagsModelMax) {
        refineHeadFwd(matVt, profile, matF, lookup, lagsModelMax, lagsModelMax);
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

        // Stationarity / invertibility, factor by factor (c = -phi and theta)
        double arReflection = 0, maReflection = 0;
        for(unsigned int i=0; i<lagsARIMA.n_rows; ++i){
            if(arOrders(i) * lagsARIMA(i) != 0){
                arma::uvec positions = arma::regspace<arma::uvec>(1, arOrders(i)) * lagsARIMA(i);
                arma::vec factor = arParameters.col(i);
                arReflection = std::max(arReflection, arimaReflection(factor.elem(positions)));
            }
            if(maOrders(i) * lagsARIMA(i) != 0){
                arma::uvec positions = arma::regspace<arma::uvec>(1, maOrders(i)) * lagsARIMA(i);
                arma::vec factor = maParameters.col(i);
                maReflection = std::max(maReflection, arimaReflection(factor.elem(positions)));
            }
        }

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
        result.arReflection = arReflection;
        result.maReflection = maReflection;
        return result;
    }

    // Method 2: Fitter - fits SSOE model to the data.
    // For demand models (O='n', default): vectorYt is the demand series; vectorOt is the
    // occurrence multiplier (1, fractional, or 0 for intermittent).
    // For occurrence models (O='d'/'o'/'i'): pass vectorOt as both vectorYt and vectorOt;
    // the raw state-space output is transformed to a probability post-loop.
    // The transition of the forward and backward passes: the sparse product where
    // SparseTransition took it, adamFvalue() otherwise
    arma::vec transitionValue(SparseTransition const &sparseF, arma::vec const &v,
                              arma::mat const &matrixF) const {
        if(sparseF.use) {
            return sparseF.matrixF * v;
        }
        return adamFvalue(v, matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                          nComponents, constant);
    }

    FitResult fit(arma::mat matrixVt, arma::mat const &matrixWt,
                  arma::mat &matrixF, arma::vec const &vectorG,
                  arma::umat const &indexLookupTable, arma::mat profilesRecent,
                  arma::vec const &vectorYt, arma::vec const &vectorOt,
                  bool const &backcast, unsigned int const &nIterations,
                  char const &O = 'n') {
        return fitImpl(matrixVt, matrixWt, matrixF, vectorG, indexLookupTable, profilesRecent,
                       vectorYt, vectorOt, backcast, nIterations, O, nullptr);
    }

    // The filter of a linear Gaussian model with periods without an observed value (vectorOt
    // zero or NaN): a pure additive model, or a pure multiplicative ADAM ETS in logs (logs=true,
    // with the log-normal error log(1+e) ~ N(-scale/2, scale)). The backcasting passes are those
    // of fit(); the last forward pass carries the covariance of the states, which is zero until
    // the first period without an observed value (the pass is then fit()'s own), builds up over
    // such periods and sets the gain of the update. It returns, for each period, the variance of
    // the one-step error relative to sigma^2 (or to scale) and, in logs, the derivative of the
    // innovation in the scale: neither depends on the data.
    FilterResult filter(arma::mat matrixVt, arma::mat const &matrixWt,
                        arma::mat &matrixF, arma::vec const &vectorG,
                        arma::umat const &indexLookupTable, arma::mat profilesRecent,
                        arma::vec const &vectorYt, arma::vec const &vectorOt,
                        bool const &backcast, unsigned int const &nIterations,
                        bool const &logs, double const &scale) {
        FilterState state;
        state.logs = logs;
        state.scale = scale;
        state.cells = arma::unique(arma::vectorise(indexLookupTable));
        state.compact = arma::uvec(profilesRecent.n_elem, arma::fill::zeros);
        for(arma::uword k=0; k<state.cells.n_elem; k++) {
            state.compact(state.cells(k)) = k;
        }
        FitResult fitted = fitImpl(matrixVt, matrixWt, matrixF, vectorG, indexLookupTable,
                                   profilesRecent, vectorYt, vectorOt, backcast, nIterations,
                                   'n', &state);
        FilterResult result;
        result.states = fitted.states;
        result.fitted = fitted.fitted;
        result.errors = fitted.errors;
        result.profile = fitted.profile;
        result.profileInitial = fitted.profileInitial;
        result.variances = state.variances;
        result.innovationsScale = state.innovationsScale;
        return result;
    }

    FitResult fitImpl(arma::mat matrixVt, arma::mat const &matrixWt,
                      arma::mat &matrixF, arma::vec const &vectorG,
                      arma::umat const &indexLookupTable, arma::mat profilesRecent,
                      arma::vec const &vectorYt, arma::vec const &vectorOt,
                      bool const &backcast, unsigned int const &nIterations,
                      char const &O, FilterState *filterState) {
        /* # matrixVt should have a length of obs + lagsModelMax.
         * # matrixWt is a matrix with nrows = obs
         * # vecG should be a vector
         * # lags is a vector of lags
         */

        const arma::mat profileInitial = profilesRecent;
        int obs = vectorYt.n_rows;
        int lagsModelMax = max(lags);
        unsigned int H = (headLength == 0 ? lagsModelMax : headLength);
        if(H < (unsigned int)lagsModelMax) {
            H = lagsModelMax;
        }
        // Skip the head filtering where the flip is already exact and the head has the
        // default length: there it cannot change the result.
        bool useHeadFilter = (headLength > 0) &&
            !(headFlipIsExact(matrixF) && H == (unsigned int)lagsModelMax);

        // Fitted values and the residuals
        arma::vec vecYfit(obs, arma::fill::zeros);
        arma::vec vecErrors(obs, arma::fill::zeros);
        arma::vec backcasts(H, arma::fill::zeros);
        // The transition of the forward and backward passes, sparse where requested
        const SparseTransition sparseF(sparseTransition, matrixF, E, T, nETS, nArima, constant);
        // The head steps all measure with the same regressor row; hoist it out of the loops
        const arma::rowvec wHead = matrixWt.row(0);

        // The filter on the last forward pass (filter()); the passes are counted at their heads
        unsigned int pass = 0;
        const unsigned int lastPass = backcast ? nIterations : 1;
        if(filterState != nullptr) {
            filterState->variances = arma::vec(obs, arma::fill::ones);
            filterState->innovationsScale = arma::vec(obs);
            filterState->innovationsScale.fill(filterState->logs ? 0.5 : 0.0);
        }

        // The ETS components are in logs in the filter of a pure multiplicative ADAM ETS
        auto cellNative = [&](arma::uword cell, double value) {
            return (filterState->logs && (cell % nComponents) < nETS) ? std::exp(value) : value;
        };
        auto cellLinear = [&](arma::uword cell, double value) {
            return (filterState->logs && (cell % nComponents) < nETS) ? std::log(value) : value;
        };

        // One step of the filter, from its first period without an observed value on
        auto filterStep = [&](int i) {
            FilterState &fs = *filterState;
            int idx = i - H;
            arma::uvec cellsNow = indexLookupTable.col(i);
            arma::uvec c = fs.compact(cellsNow);
            arma::rowvec w = matrixWt.row(idx);
            bool observed = vectorOt(idx) != 0 && !std::isnan(vectorOt(idx));
            // The persistence of the linear model: the adaptive regressors move by delta/x
            arma::vec gEff = vectorG;
            for(unsigned int r=nETS+nArima; r<nETS+nArima+nXreg; r++) {
                gEff(r) = (w(r) != 0 && std::isfinite(1/w(r))) ? vectorG(r)/w(r) : 0;
            }
            if(!fs.linear) {
                fs.means = arma::vec(fs.cells.n_elem);
                for(arma::uword k=0; k<fs.cells.n_elem; k++) {
                    fs.means(k) = cellLinear(fs.cells(k), profilesRecent(fs.cells(k)));
                }
                fs.meansScale = arma::vec(fs.cells.n_elem, arma::fill::zeros);
                fs.covariance = arma::mat(fs.cells.n_elem, fs.cells.n_elem, arma::fill::zeros);
                fs.linear = true;
            }
            arma::vec xc = fs.means(c);
            arma::vec xsc = fs.meansScale(c);
            // The variance of the one-step error and the derivative of the innovation in the scale
            arma::vec covarW;
            double f = 1;
            if(fs.covarianceActive) {
                covarW = fs.covariance.cols(c) * w.t();
                f = arma::as_scalar(w * covarW(c)) + 1;
            }
            double bScale = fs.logs ? 0.5 - arma::as_scalar(w * xsc) : 0;
            fs.variances(idx) = f;
            fs.innovationsScale(idx) = bScale;
            // The fitted value and the error as the fitter defines them
            arma::vec native(xc.n_elem);
            for(arma::uword k=0; k<xc.n_elem; k++) {
                native(k) = cellNative(cellsNow(k), xc(k));
            }
            vecYfit(idx) = adamWvalue(native, w, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                                      nXreg, nComponents, constant);
            vecErrors(idx) = errorf(vectorYt(idx), vecYfit(idx), E, vectorOt(idx), 'n');
            // The mean of the error in the linear model
            double mu = fs.logs ? -fs.scale/2 : 0;
            arma::vec meansNew = fs.means;
            arma::vec meansScaleNew = fs.meansScale;
            arma::mat covarianceNew = fs.covariance;
            if(fs.covarianceActive) {
                covarianceNew.rows(c) = matrixF * fs.covariance.rows(c);
                covarianceNew.cols(c) = covarianceNew.cols(c) * matrixF.t();
            }
            covarianceNew(c, c) += gEff * gEff.t();
            meansNew(c) = matrixF * xc + gEff * mu;
            meansScaleNew(c) = matrixF * xsc - gEff * 0.5;
            if(observed) {
                double z = fs.logs ? std::log(vectorYt(idx)) : vectorYt(idx);
                double v = z - (fs.logs ? std::log(vecYfit(idx)) : vecYfit(idx)) - mu;
                arma::vec gain(fs.cells.n_elem, arma::fill::zeros);
                if(fs.covarianceActive) {
                    gain = covarW;
                    gain(c) = matrixF * covarW(c);
                }
                gain(c) += gEff;
                gain /= f;
                meansNew += gain * v;
                meansScaleNew += gain * bScale;
                covarianceNew -= gain * gain.t() * f;
            }
            fs.means = meansNew;
            fs.meansScale = meansScaleNew;
            fs.covariance = covarianceNew;
            // The covariance that has vanished leaves the means to the gain of the model
            fs.covarianceActive = arma::abs(fs.covariance).max() > 1e-15;
            if(!fs.covarianceActive) {
                fs.covariance.zeros();
            }
            for(arma::uword k=0; k<fs.cells.n_elem; k++) {
                profilesRecent(fs.cells(k)) = cellNative(fs.cells(k), fs.means(k));
            }
            matrixVt.col(i) = profilesRecent(cellsNow);
        };

        // What to do in the forward pass
        auto forwardStep = [&](int i) {
            int idx = i - H;
            if(filterState != nullptr && pass == lastPass &&
               (filterState->linear || vectorOt(idx) == 0 || std::isnan(vectorOt(idx)))) {
                filterStep(i);
                return;
            }
            /* # Measurement equation and the error term */
            vecYfit(idx) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                    matrixWt.row(idx), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
            // We need this multiplication for cases, when occurrence is fractional (NaN
            // marks a missing occurrence, which has no error)
            if(vectorOt(idx) != 0 && !std::isnan(vectorOt(idx))) {
                vecYfit(idx) = vectorOt(idx) * vecYfit(idx);
            }
            // errorf() returns 0 when ot==0; dispatches to occurrenceError() when O!='n'
            vecErrors(idx) = errorf(vectorYt(idx), vecYfit(idx), E, vectorOt(idx), O);
            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) =
                transitionValue(sparseF, profilesRecent(indexLookupTable.col(i)), matrixF) +
                adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                           vectorG, vecErrors(idx), vecYfit(idx), adamETS);
            matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
        };

        // What to do in the backward pass
        auto backwardStep = [&](int i) {
            int idx = i - H;
            /* # Measurement equation and the error term */
            vecYfit(idx) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                    matrixWt.row(idx), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
            if(vectorOt(idx) != 0 && !std::isnan(vectorOt(idx))) {
                vecYfit(idx) = vectorOt(idx) * vecYfit(idx);
            }
            vecErrors(idx) = errorf(vectorYt(idx), vecYfit(idx), E, vectorOt(idx), O);
            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) =
                transitionValue(sparseF, profilesRecent(indexLookupTable.col(i)), matrixF) +
                adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(idx), E, T, S,
                           nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                           vectorG, vecErrors(idx), vecYfit(idx), adamETS);
        };

        // The states of the head as the backward pass leaves them, from time -H+1 to 0
        arma::mat headStates(nComponents, H, arma::fill::zeros);
        bool afterBackcast = false;

        // How to fill in the head before the forward pass: from the seed on the first pass,
        // from the head left by the backward pass afterwards (the states already sit at time 0)
        auto headFillFwd = [&]() {
            if(afterBackcast) {
                matrixVt.cols(0, H-1) = headStates;
            }
            else {
                refineHeadFwd(matrixVt, profilesRecent, matrixF,
                              indexLookupTable, lagsModelMax, H);
            }
        };

        // How to revert the trend component for backcasting.
        // The constant (drift) flips sign when the total order of differencing
        // is odd — the direct ARIMA analog of the ETS trend reversal.
        auto trendReversal = [&]() {
            if(T == 'A')      { profilesRecent(1) = -profilesRecent(1); }
            else if(T == 'M') { profilesRecent(1) = 1/profilesRecent(1); }
            if(constant && flipConstant) {
                flipDrift(profilesRecent(nComponents-1));
            }
        };

        // Before the backward pass: each state moves its lag past the end of the sample
        auto tailTurn = [&]() {
            turnPerLag(profilesRecent, matrixF, lags, obs+1, 1, 0, wHead, E, T, S,
                       nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
        };

        // After the backward pass: each state moves its lag (plus the filtered head) before
        // the sample, recording the backward predictions there as the head's backcasts
        auto headFillBwd = [&]() {
            afterBackcast = true;
            turnPerLag(profilesRecent, matrixF, lags, 0, -1, useHeadFilter ? H : 0, wHead,
                       E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                       useHeadFilter ? &backcasts : nullptr, &headStates);
            unflipSlopes(headStates);
        };

        // How to filter over the backcast head on subsequent forward passes
        auto headForwardStep = [&]() {
            for (unsigned int i=0; i<H; i=i+1) {
                // Gather the profile cells for this head step once and reuse them
                arma::vec vCur = profilesRecent(indexLookupTable.col(i));
                double yFitHead = adamWvalue(vCur, wHead, E, T, S,
                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
                // The head's pseudo-observation is the model's own predicted value, not an
                // occurrence indicator, so the occurrence dispatch does not apply here.
                double errHead = errorf(backcasts(i), yFitHead, E, 1.0);
                arma::vec vNew =
                    adamFvalue(vCur, matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal,
                               nArima, nComponents, constant) +
                    adamGvalue(vCur, matrixF, wHead, E, T, S,
                               nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                               vectorG, errHead, yFitHead, adamETS);
                profilesRecent(indexLookupTable.col(i)) = vNew;
                matrixVt.col(i) = vNew;
            }
        };

        // Count the forward passes at their heads, for the filter on the last one
        auto headFillFwdCounted = [&]() {
            pass++;
            headFillFwd();
        };
        auto headForwardStepCounted = [&]() {
            pass++;
            headForwardStep();
        };

        // Do the fit!
        fitLoopImpl(obs, H, backcast, nIterations,
                    forwardStep, backwardStep, headFillFwdCounted, headFillBwd, trendReversal,
                    headForwardStepCounted, useHeadFilter, tailTurn);

        FitResult result;
        result.states = matrixVt;
        result.fitted = vecYfit;
        result.errors = vecErrors;
        result.profile = profilesRecent;
        result.profileInitial = profileInitial;
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

        // The lags of model B come from its own lookup table
        const arma::uvec lagsB = lookupLags(indexLookupTableB, lagsModelMaxA, obs);
        // The head states as the backward pass leaves them, from time -H+1 to 0
        arma::mat headStatesA(nComponents, lagsModelMaxA, arma::fill::zeros);
        arma::mat headStatesB(nComponentsB, lagsModelMaxA, arma::fill::zeros);
        bool afterBackcast = false;

        auto headFillFwd = [&]() {
            if(afterBackcast) {
                matrixVtA.cols(0, lagsModelMaxA-1) = headStatesA;
                matrixVtB.cols(0, lagsModelMaxA-1) = headStatesB;
                return;
            }
            // Model A: full trend-aware refinement via the shared helper
            refineHeadFwd(matrixVtA, profilesRecentA, matrixFA,
                          indexLookupTableA, lagsModelMaxA);
            // Model B: plain copy (B typically has no trend; keep original behaviour)
            for (int i=0; i<lagsModelMaxA; i=i+1) {
                matrixVtB.col(i) = profilesRecentB(indexLookupTableB.col(i));
            }
        };

        // Each state of both models moves its lag past the end of the sample
        auto tailTurn = [&]() {
            turnPerLag(profilesRecentA, matrixFA, lags, obs+1, 1, 0, matrixWtA.row(0),
                       E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
            turnPerLag(profilesRecentB, matrixFB, lagsB, obs+1, 1, 0, matrixWtB.row(0),
                       EB, TB, SB, nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nXregB, nComponentsB,
                       constantB);
        };

        // ... and its lag before the sample after the backward pass
        auto headFillBwd = [&]() {
            afterBackcast = true;
            turnPerLag(profilesRecentA, matrixFA, lags, 0, -1, 0, matrixWtA.row(0),
                       E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                       nullptr, &headStatesA);
            turnPerLag(profilesRecentB, matrixFB, lagsB, 0, -1, 0, matrixWtB.row(0),
                       EB, TB, SB, nETSB, nNonSeasonalB, nSeasonalB, nArimaB, nXregB, nComponentsB,
                       constantB, nullptr, &headStatesB);
            unflipTrend(headStatesA, T);
            unflipTrend(headStatesB, TB);
        };

        // Reverse/restore both models' trends symmetrically
        auto trendReversal = [&]() {
            if(T == 'A')       { profilesRecentA(1) = -profilesRecentA(1); }
            else if(T == 'M')  { profilesRecentA(1) = 1/profilesRecentA(1); }
            if(TB == 'A')      { profilesRecentB(1) = -profilesRecentB(1); }
            else if(TB == 'M') { profilesRecentB(1) = 1/profilesRecentB(1); }
        };

        fitLoopImpl(obs, lagsModelMaxA, backcast, nIterations,
                    forwardStep, backwardStep, headFillFwd, headFillBwd, trendReversal, tailTurn);

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

    // Method 6c: gradientSolveGeneral - the coupled ("general") occurrence
    // analog of gradientSolve. The two occurrence sub-models A and B share a
    // single fitted probability p = aFit/(aFit+bFit) (occurrenceError O='g'),
    // so their initial profiles are cross-coupled and the probability is a
    // nonlinear logistic of both fitted values. The solve minimises the loss on
    // the probability residual r = o - p jointly over both initial profiles by
    // Gauss-Newton with a finite-difference Jacobian (the coupled analytic
    // sensitivities are not available; FD drives the same loop through the fully
    // general omfitGeneral residual pass, so it covers ETS / ARIMA / xreg on
    // either side). Line search + Levenberg-Marquardt fallback mirror
    // gradientSolve. Stateless: always starts from the seed profiles, so the
    // objective is a deterministic function of the inputs (R/Python parity).
    // A's structural params come from this instance's members (as in
    // omfitGeneral); B's are passed explicitly.
    GradientSolveGeneralResult gradientSolveGeneral(
            arma::mat const &matrixVtA, arma::mat const &matrixWtA,
            arma::mat &matrixFA, arma::vec const &vectorGA,
            arma::umat const &indexLookupTableA, arma::mat const &profileA,
            arma::mat const &probeBasisA,
            char const &EB, char const &TB, char const &SB,
            unsigned int const &nNonSeasonalB, unsigned int const &nSeasonalB,
            unsigned int const &nETSB, unsigned int const &nArimaB,
            unsigned int const &nXregB, unsigned int const &nComponentsB,
            bool const &constantB, bool const &adamETSB,
            arma::mat const &matrixVtB, arma::mat const &matrixWtB,
            arma::mat &matrixFB, arma::vec const &vectorGB,
            arma::umat const &indexLookupTableB, arma::mat const &profileB,
            arma::mat const &probeBasisB,
            arma::vec const &vectorOt,
            unsigned int const &nIterations, char const &lossType){

        int obs = vectorOt.n_rows;
        unsigned int nFreeA = probeBasisA.n_cols;
        unsigned int nFreeB = probeBasisB.n_cols;
        unsigned int nFree = nFreeA + nFreeB;
        GradientSolveGeneralResult failure;   // empty profiles => caller falls back
        if(nFree == 0){
            return failure;
        }

        // Only the probability-residual losses are defined here; anything else
        // (including the multistep codes) collapses to the SSE surrogate on r.
        char L = lossType;
        if(L!='B' && L!='A' && L!='H' && L!='G'){
            L = 'S';
        }
        double const beta = 2.0;
        double const lossScale = 0;

        // r = o - p from one coupled forward pass (backcast=false, one sweep):
        // exactly the recursion the final fit runs from the solved initials.
        auto combinedResiduals = [&](arma::mat const &profACur,
                                     arma::mat const &profBCur,
                                     arma::vec &res) -> bool {
            OmFitGeneralResult fg = omfitGeneral(
                matrixVtA, matrixWtA, matrixFA, vectorGA,
                indexLookupTableA, profACur,
                EB, TB, SB, nNonSeasonalB, nSeasonalB, nETSB, nArimaB,
                nXregB, nComponentsB, constantB, adamETSB,
                matrixVtB, matrixWtB, matrixFB, vectorGB,
                indexLookupTableB, profBCur, vectorOt, false, 1);
            for(int idx=0; idx<obs; idx=idx+1){
                double aFit = (E=='A')  ? std::exp(fg.fittedA(idx)) : fg.fittedA(idx);
                double bFit = (EB=='A') ? std::exp(fg.fittedB(idx)) : fg.fittedB(idx);
                double p = aFit / (aFit + bFit);
                // A missing observation does not enter the least squares
                res(idx) = std::isnan(vectorOt(idx)) ? 0 : vectorOt(idx) - p;
                if(!std::isfinite(p) || p<=0 || p>=1){
                    return false;
                }
            }
            return res.is_finite();
        };

        // Analytic coupled Jacobian: when both sub-models are additive-error and
        // linear-state (E='A', no multiplicative trend/season), one forward pass
        // propagates the exact joint sensitivities instead of nFree finite-
        // difference passes. Each side's state map is linear (yhat = w'v,
        // v' = F v + g e); the coupling enters only through the shared
        // probability p = aFit/(aFit+bFit) and the occurrenceError('g') split
        // (errorA/errorB), whose derivatives w.r.t. both fitted values are known
        // in closed form. Mirrors omfitGeneral's forward recursion (side A head
        // refined, side B plain) so its residuals match combinedResiduals.
        bool const bothLinear = (E=='A' && T!='M' && S!='M') &&
                                (EB=='A' && TB!='M' && SB!='M');
        auto combinedJacobian = [&](arma::mat const &profACur, arma::mat const &profBCur,
                                    arma::vec &res, arma::mat &jac) -> bool {
            int lagsModelMaxA = max(lags);
            arma::mat profA = profACur, profB = profBCur;
            arma::mat sensA(profA.n_elem, nFree, arma::fill::zeros);
            arma::mat sensB(profB.n_elem, nFree, arma::fill::zeros);
            if(nFreeA > 0){ sensA.cols(0, nFreeA-1) = probeBasisA; }
            if(nFreeB > 0){ sensB.cols(nFreeA, nFree-1) = probeBasisB; }

            // Head refinement: side A walks level/trend through F (sensitivities
            // follow); side B is a plain copy in omfitGeneral, so it is untouched.
            if(lagsModelMaxA > 1 && T != 'N'){
                for(int i=1; i<lagsModelMaxA; i=i+1){
                    arma::uvec cA = indexLookupTableA.col(i);
                    arma::vec vNew = adamFvalue(profA(cA), matrixFA, E, T, S,
                                                nETS, nNonSeasonal, nSeasonal, nArima,
                                                nComponents, constant);
                    arma::mat sNew = matrixFA * sensA.rows(cA);
                    profA(cA.rows(0,1)) = vNew.rows(0,1);
                    sensA.rows(cA.rows(0,1)) = sNew.rows(0,1);
                }
            }

            for(int i=lagsModelMaxA; i<obs+lagsModelMaxA; i=i+1){
                int idx = i - lagsModelMaxA;
                arma::uvec cA = indexLookupTableA.col(i), cB = indexLookupTableB.col(i);
                arma::vec vA = profA(cA), vB = profB(cB);
                arma::mat sA = sensA.rows(cA), sB = sensB.rows(cB);
                arma::rowvec wA = matrixWtA.row(idx), wB = matrixWtB.row(idx);

                double fitA = adamWvalue(vA, wA, E, T, S, nETS, nNonSeasonal,
                                         nSeasonal, nArima, nXreg, nComponents, constant);
                double fitB = adamWvalue(vB, wB, EB, TB, SB, nETSB, nNonSeasonalB,
                                         nSeasonalB, nArimaB, nXregB, nComponentsB, constantB);
                arma::rowvec dfitA = wA * sA;
                arma::rowvec dfitB = wB * sB;

                double aFit = (E=='A')  ? std::exp(fitA) : fitA;
                double bFit = (EB=='A') ? std::exp(fitB) : fitB;
                double denom = aFit + bFit;
                double p = aFit / denom;
                // A missing observation does not enter the least squares
                res(idx) = std::isnan(vectorOt(idx)) ? 0 : vectorOt(idx) - p;
                if(!std::isfinite(p) || p<=0 || p>=1){
                    return false;
                }
                // dp/dfit = [+/- aFit*bFit / denom^2] * d(fit-transform)
                double daA = (E=='A')  ? aFit : 1.0;
                double dbB = (EB=='A') ? bFit : 1.0;
                arma::rowvec dp = (bFit/(denom*denom)*daA) * dfitA
                                - (aFit/(denom*denom)*dbB) * dfitB;
                bool const missingObs = std::isnan(vectorOt(idx));
                jac.row(idx) = -dp;

                // occurrenceError('g'): u=(1+o-p)/2; errorA/errorB per side type.
                double u = (1.0 + vectorOt(idx) - p) / 2.0;
                double eA = (E=='A')  ? std::log(u/(1-u)) : (u/(1-u) - 1);
                double eB = (EB=='A') ? std::log((1-u)/u) : ((1-u)/u - 1);
                double dEAdu = (E=='A')  ?  1.0/(u*(1-u)) : 1.0/((1-u)*(1-u));
                double dEBdu = (EB=='A') ? -1.0/(u*(1-u)) : -1.0/(u*u);
                arma::rowvec dEA = (dEAdu * -0.5) * dp;
                arma::rowvec dEB = (dEBdu * -0.5) * dp;
                // A missing observation: no error, no residual, the states move only
                if(missingObs){
                    jac.row(idx).zeros();
                    eA = 0;
                    eB = 0;
                    dEA.zeros();
                    dEB.zeros();
                }

                profA(cA) = adamFvalue(vA, matrixFA, E, T, S, nETS, nNonSeasonal,
                                       nSeasonal, nArima, nComponents, constant) +
                            adamGvalue(vA, matrixFA, wA, E, T, S, nETS, nNonSeasonal,
                                       nSeasonal, nArima, nXreg, nComponents, constant,
                                       vectorGA, eA, fitA, adamETS);
                sensA.rows(cA) = matrixFA * sA + vectorGA * dEA;
                profB(cB) = adamFvalue(vB, matrixFB, EB, TB, SB, nETSB, nNonSeasonalB,
                                       nSeasonalB, nArimaB, nComponentsB, constantB) +
                            adamGvalue(vB, matrixFB, wB, EB, TB, SB, nETSB, nNonSeasonalB,
                                       nSeasonalB, nArimaB, nXregB, nComponentsB, constantB,
                                       vectorGB, eB, fitB, adamETSB);
                sensB.rows(cB) = matrixFB * sB + vectorGB * dEB;
            }
            return res.is_finite() && jac.is_finite();
        };

        // Apply a joint step: its first nFreeA entries move profileA along
        // probeBasisA, the rest move profileB along probeBasisB.
        auto applyStep = [&](arma::vec const &step, arma::mat &profAOut,
                             arma::mat &profBOut){
            if(nFreeA > 0){
                profAOut = profAOut + arma::reshape(probeBasisA * step.head(nFreeA),
                                                    profAOut.n_rows, profAOut.n_cols);
            }
            if(nFreeB > 0){
                profBOut = profBOut + arma::reshape(probeBasisB * step.tail(nFreeB),
                                                    profBOut.n_rows, profBOut.n_cols);
            }
        };

        arma::mat profA = profileA;
        arma::mat profB = profileB;
        arma::vec residuals(obs);
        if(!combinedResiduals(profA, profB, residuals)){
            return failure;
        }
        double lossCurrent = gradientLossSum(residuals, L, beta, lossScale);
        if(!std::isfinite(lossCurrent)){
            return failure;
        }

        // Per-free-parameter finite-difference step, scaled by the current value
        // of the first perturbed profile cell (A cells for j<nFreeA, else B).
        arma::uvec firstCellA(nFreeA), firstCellB(nFreeB);
        for(unsigned int j=0; j<nFreeA; j=j+1){
            arma::uvec nz = arma::find(probeBasisA.col(j));
            firstCellA(j) = nz(0);
        }
        for(unsigned int j=0; j<nFreeB; j=j+1){
            arma::uvec nz = arma::find(probeBasisB.col(j));
            firstCellB(j) = nz(0);
        }

        arma::mat jacobian(obs, nFree);
        arma::vec residualsProbe(obs);
        double lossBeforeIteration;
        for(unsigned int iter=0; iter<nIterations; iter=iter+1){
            lossBeforeIteration = lossCurrent;
            bool jacobianOk = true;
            if(bothLinear){
                // One analytic pass (residualsProbe equals the held residuals).
                jacobianOk = combinedJacobian(profA, profB, residualsProbe, jacobian);
            }
            else{
                // Finite-difference Jacobian: nFree coupled residual passes.
                for(unsigned int j=0; j<nFree && jacobianOk; j=j+1){
                    arma::mat profAProbe = profA;
                    arma::mat profBProbe = profB;
                    double h;
                    if(j < nFreeA){
                        h = 1e-4 * std::max(1.0, std::abs(profA(firstCellA(j))));
                        profAProbe = profA + arma::reshape(h * probeBasisA.col(j),
                                                           profA.n_rows, profA.n_cols);
                    }
                    else{
                        unsigned int jb = j - nFreeA;
                        h = 1e-4 * std::max(1.0, std::abs(profB(firstCellB(jb))));
                        profBProbe = profB + arma::reshape(h * probeBasisB.col(jb),
                                                           profB.n_rows, profB.n_cols);
                    }
                    if(!combinedResiduals(profAProbe, profBProbe, residualsProbe)){
                        jacobianOk = false;
                        break;
                    }
                    jacobian.col(j) = (residualsProbe - residuals) / h;
                }
            }
            if(!jacobianOk || !jacobian.is_finite()){
                break;
            }

            // Loss-aware step system (mirrors gradientSolve): plain (J, r) for
            // SSE, row-scaled otherwise.
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

            bool improved = false;
            double stepSize = 1;
            for(int half=0; half<6; half=half+1){
                arma::mat profACand = profA, profBCand = profB;
                applyStep(stepSize * step, profACand, profBCand);
                if(combinedResiduals(profACand, profBCand, residualsProbe) &&
                   gradientLossSum(residualsProbe, L, beta, lossScale) < lossCurrent){
                    profA = profACand;
                    profB = profBCand;
                    residuals = residualsProbe;
                    lossCurrent = gradientLossSum(residuals, L, beta, lossScale);
                    improved = true;
                    break;
                }
                stepSize = stepSize / 2;
            }

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
                    arma::mat profACand = profA, profBCand = profB;
                    applyStep(step, profACand, profBCand);
                    if(combinedResiduals(profACand, profBCand, residualsProbe) &&
                       gradientLossSum(residualsProbe, L, beta, lossScale) < lossCurrent){
                        profA = profACand;
                        profB = profBCand;
                        residuals = residualsProbe;
                        lossCurrent = gradientLossSum(residuals, L, beta, lossScale);
                        improved = true;
                    }
                }
            }
            if(!improved){
                break;
            }
            if(lossCurrent > lossBeforeIteration * (1 - 1e-4)){
                break;
            }
        }

        GradientSolveGeneralResult result;
        result.profileA = profA;
        result.profileB = profB;
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
        unsigned int lagsModelMax = max(lags);
        // The windows with all their targets in the sample
        int nWindows = std::max((int)vectorYt.n_rows - (int)horizon + 1, 0);
        arma::mat matErrors(horizon, nWindows, arma::fill::zeros);

        // Fill in the head, similar to how it's done in the fitter
        for (unsigned int i=0; i<lagsModelMax; i=i+1) {
            profilesRecent(indexLookupTable.col(i)) = matrixVt.col(i);
        }

        // Row i forecasts the targets i..i+h-1 from the states after the observation
        // i-1 (the head for i=0), which the fitter stored in the column i+lagsModelMax-1
        for(int i = 0; i < nWindows; i=i+1){
            profilesRecent(indexLookupTable.col(i+lagsModelMax-1)) = matrixVt.col(i+lagsModelMax-1);
            matErrors.col(i) =
                errorvf(vectorYt.rows(i, i+horizon-1),
                        forecast(matrixWt.rows(i, i+horizon-1), matrixF,
                                 indexLookupTable.cols(i+lagsModelMax, i+lagsModelMax+horizon-1),
                                 profilesRecent, horizon).forecast,
                        E);
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
        unsigned int H = (headLength == 0 ? lagsModelMax : headLength);
        if(H < (unsigned int)lagsModelMax) {
            H = lagsModelMax;
        }

        arma::mat matYfit(obs, nSeries, arma::fill::zeros);
        arma::vec vecErrors(obs, arma::fill::zeros);

        for(unsigned int k=0; k<nSeries; k=k+1){
            // The data of each series, when they differ (the Box-Cox transforms of tbats)
            arma::vec const vectorYt = matrixYt.col(matrixYt.n_cols==nSeries ? k : 0);
            const SparseTransition sparseF(sparseTransition, arrayF.slice(k), E, T, nETS, nArima, constant);
            // Loop for the backcasting
            arma::vec backcasts(H, arma::fill::zeros);
            // The head states as the backward pass leaves them, from time -H+1 to 0
            arma::mat headStates(nComponents, H, arma::fill::zeros);
            // Same gate as in fit(): no filtering where the flip is already exact
            bool useHeadFilter = (headLength > 0) &&
                !(headFlipIsExact(arrayF.slice(k)) && H == (unsigned int)lagsModelMax);
            const arma::rowvec wHead = arrayWt.slice(k).row(0);
            for (unsigned int j=1; j<=nIterations; j=j+1) {
                if(j > 1 && !useHeadFilter) {
                    // After the backward pass the states already sit at time 0
                    arrayVt.slice(k).cols(0, H-1) = headStates;
                }
                else if(j == 1 || !useHeadFilter) {
                    // Refine the head via the shared helper so it is walked one step
                    // per column across the head cycle (or copied verbatim when T=='N').
                    // As in fit(), this also writes the initial states of a head of one
                    // column, which otherwise kept those of the original model.
                    // Bind slice views so refineHeadFwd can mutate them via references.
                    arma::mat sliceVt = arrayVt.slice(k);
                    arma::mat sliceProfile = arrayProfilesRecent.slice(k);
                    arma::mat sliceF = arrayF.slice(k);
                    // Note: reapply's original branch used the full profile column,
                    // not just the trend rows, so we call refineHeadFwd once here to
                    // match — the helper writes the trend-walked value into the
                    // level+trend rows and preserves the seasonal via the profile.
                    refineHeadFwd(sliceVt, sliceProfile, sliceF,
                                  indexLookupTable, lagsModelMax, H);
                    arrayVt.slice(k) = sliceVt;
                    arrayProfilesRecent.slice(k) = sliceProfile;
                } else {
                    for(unsigned int i=0; i<H; i=i+1) {
                        // Gather the profile cells for this head step once and reuse them
                        arma::vec vCur = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i));
                        double yFitHead = adamWvalue(vCur, wHead, E, T, S,
                                nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
                        double errHead = errorf(backcasts(i), yFitHead, E, 1.0);
                        arma::vec vNew =
                            adamFvalue(vCur, arrayF.slice(k), E, T, S, nETS, nNonSeasonal,
                                       nSeasonal, nArima, nComponents, constant) +
                            adamGvalue(vCur, arrayF.slice(k), wHead, E, T, S,
                                       nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                       matrixG.col(k), errHead, yFitHead, adamETS);
                        arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) = vNew;
                        arrayVt.slice(k).col(i) = vNew;
                    }
                }
                // Loop for the model construction
                for(int i=H; i<obs+(int)H; i=i+1) {
                    /* # Measurement equation and the error term */
                    matYfit(i-H,k) = adamWvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                            arrayWt.slice(k).row(i-H), E, T, S,
                            nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                    // Fix potential issue with negatives in mixed models
                    if((E=='M' || T=='M' || S=='M') && (matYfit(i-H,k)<=0)){
                        matYfit(i-H,k) = 1;
                    }

                    // We need this multiplication for cases, when occurrence is fractional
                    if(matrixOt(i-H)!=0){
                        matYfit(i-H,k) = matrixOt(i-H) * matYfit(i-H,k);
                    }
                    // errorf() returns 0 immediately when ot==0
                    vecErrors(i-H) = errorf(vectorYt(i-H), matYfit(i-H,k), E,
                                                       matrixOt(i-H));

                    /* # Transition equation */
                    arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                    transitionValue(sparseF, arrayProfilesRecent.slice(k)(indexLookupTable.col(i)),
                                    arrayF.slice(k)) +
                                   adamGvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                              arrayF.slice(k), arrayWt.slice(k).row(i-H), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                              matrixG.col(k), vecErrors(i-H), matYfit(i-H,k), adamETS);

                    arrayVt.slice(k).col(i) = arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i));
                }

                ////// Backwards run
                if(backcast && j<(nIterations)){
                    // Each state moves its lag past the end of the sample
                    arma::mat profile = arrayProfilesRecent.slice(k);
                    turnPerLag(profile, arrayF.slice(k), lags, obs+1, 1, 0, wHead, E, T, S,
                               nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);
                    arrayProfilesRecent.slice(k) = profile;
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
                        flipDrift(arrayProfilesRecent.slice(k)(nComponents-1));
                    }

                    for(int i=obs+H-1; i>=(int)H; i=i-1) {
                        /* # Measurement equation and the error term */
                        matYfit(i-H,k) = adamWvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                arrayWt.slice(k).row(i-H), E, T, S,
                                nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                        // Fix potential issue with negatives in mixed models
                        if((E=='M' || T=='M' || S=='M') && (matYfit(i-H,k)<=0)){
                            matYfit(i-H,k) = 1;
                        }

                        if(matrixOt(i-H)!=0){
                            matYfit(i-H,k) = matrixOt(i-H) * matYfit(i-H,k);
                        }
                        vecErrors(i-H) = errorf(vectorYt(i-H), matYfit(i-H,k), E,
                                                           matrixOt(i-H));

                        /* # Transition equation */
                        arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)) =
                        transitionValue(sparseF, arrayProfilesRecent.slice(k)(indexLookupTable.col(i)),
                                        arrayF.slice(k)) +
                                       adamGvalue(arrayProfilesRecent.slice(k).elem(indexLookupTable.col(i)),
                                                  arrayF.slice(k), arrayWt.slice(k).row(i-H), E, T, S,
                                                  nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                                  matrixG.col(k), vecErrors(i-H), matYfit(i-H,k), adamETS);
                    }

                    // Each state moves its lag (plus the filtered head) before the sample,
                    // recording the backward predictions there as the head's backcasts
                    profile = arrayProfilesRecent.slice(k);
                    turnPerLag(profile, arrayF.slice(k), lags, 0, -1, useHeadFilter ? H : 0, wHead,
                               E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents,
                               constant, useHeadFilter ? &backcasts : nullptr, &headStates);
                    unflipSlopes(headStates);
                    arrayProfilesRecent.slice(k) = profile;

                    // Change the specific element in the state vector to negative
                    if(T=='A'){
                        arrayProfilesRecent.slice(k)(1) = -arrayProfilesRecent.slice(k)(1);
                    }
                    else if(T=='M'){
                        arrayProfilesRecent.slice(k)(1) = 1/arrayProfilesRecent.slice(k)(1);
                    }
                    // Restore the constant sign after the backward pass
                    if(constant && flipConstant){
                        flipDrift(arrayProfilesRecent.slice(k)(nComponents-1));
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
            int const nOrigins = multistep ? (obs - hor + 1) : 0;
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

                // Multistep losses: replicate the ferrors() recursion from the
                // buffers before the update with the observation idx -- h no-update
                // forecast steps of the targets idx..idx+h-1, with the sensitivities
                // following through F. Like ferrors(), the multistep errors ignore
                // the occurrence.
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

                prof(cells) = adamFvalue(prof(cells), matrixF, E, T, S,
                                         nETS, nNonSeasonal, nSeasonal, nArima,
                                         nComponents, constant) +
                              adamGvalue(prof(cells), matrixF, matrixWt.row(idx),
                                         E, T, S, nETS, nNonSeasonal, nSeasonal,
                                         nArima, nXreg, nComponents, constant,
                                         vectorG, residuals(idx), yFit, adamETS);
                sens.rows(cells) = matrixF * sens.rows(cells) + vectorG * dError;
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
        // The analytic Jacobian companions (adamGradient.h) cover pure ETS only.
        // For occurrence models with ARIMA / xreg / constant components, the
        // exact sensitivities are not available, so fall back to the
        // finite-difference Jacobian, which drives the same Gauss-Newton loop
        // through the fully general residual pass (adamW/F/Gvalue). The demand
        // path (O=='n') keeps its affine/analytic branches unchanged.
        // Additive-error occurrence (E='A', no multiplicative trend/season) is
        // linear in the full component vector, so its exact Jacobian is
        // available for ARIMA / xreg / constant too (gradientPassJacobianOccurrenceLinear).
        // Only multiplicative-error occurrence WITH extra components has no
        // analytic companion and drops to finite differences.
        bool const occurrenceLinear = (E=='A') && (T!='M') && (S!='M');
        bool useAnalytic = analytic;
        if(O!='n' && !occurrenceLinear && (nArima>0 || nXreg>0 || constant)){
            useAnalytic = false;
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
            if(useAnalytic){
                // One pass propagating the exact sensitivities (adamGradient.h)
                // instead of nFree probe passes. The residuals it returns equal
                // the ones already held (same forward recursion).
                if(O=='n'){
                    jacobianOk = gradientPassJacobian(prof, probeBasis,
                                                      matrixYt, matrixOt, matrixWt,
                                                      matrixF, vectorG, indexLookupTable,
                                                      residualsProbe, jacobian);
                }
                else if(occurrenceLinear){
                    // Additive-error occurrence: exact linear-state Jacobian,
                    // covering ARIMA / xreg / constant (bit-identical to the
                    // companion pass for pure additive ETS).
                    jacobianOk = gradientPassJacobianOccurrenceLinear(prof, probeBasis,
                                                                      matrixYt, matrixWt,
                                                                      matrixF, vectorG,
                                                                      indexLookupTable, O,
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
                                arma::umat const &indexLookupTable, arma::cube const &arrayProfileRecent,
                                char const &E){

        unsigned int obs = arrayErrors.n_rows;
        unsigned int nSeries = arrayErrors.n_cols;
        unsigned int nsim = arrayErrors.n_slices;

        unsigned int lagsModelMax = max(lags);

        double yFitted;

        arma::cube arrY(obs, nSeries, nsim);

        for(unsigned int j=0; j<nsim; j=j+1){
            for(unsigned int k=0; k<nSeries; k=k+1){
                // Every path starts from the profile at the end of the sample
                arma::mat profile = arrayProfileRecent.slice(j);
                for(unsigned int i=lagsModelMax; i<obs+lagsModelMax; i=i+1) {
                    /* # Measurement equation and the error term */
                    yFitted = adamWvalue(profile.elem(indexLookupTable.col(i-lagsModelMax)),
                                         arrayWt.slice(j).row(i-lagsModelMax), E, T, S,
                                         nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                    arrY(i-lagsModelMax,k,j) = arrayOt(i-lagsModelMax,k,j) *
                        (yFitted + adamRvalue(profile.elem(indexLookupTable.col(i-lagsModelMax)),
                                              arrayWt.slice(j).row(i-lagsModelMax), E, T, S,
                                              nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                                  arrayErrors.slice(j)(i-lagsModelMax,k));

                    // Fix potential issue with negatives in mixed models
                    if((E=='M' || T=='M' || S=='M') && (arrY(i-lagsModelMax,k,j)<0)){
                        arrY(i-lagsModelMax,k,j) = 0;
                    }

                    /* # Transition equation */
                    profile.elem(indexLookupTable.col(i-lagsModelMax)) =
                    (adamFvalue(profile.elem(indexLookupTable.col(i-lagsModelMax)),
                                arrayF.slice(j), E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                    adamGvalue(profile.elem(indexLookupTable.col(i-lagsModelMax)),
                                               arrayF.slice(j), arrayWt.slice(j).row(i-lagsModelMax),
                                               E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                               nComponents, constant, matrixG.col(j),
                                               arrayErrors.slice(j)(i-lagsModelMax,k), yFitted, adamETS));
                }
            }
        }

        ReforecastResult result;
        result.data = arrY;
        return result;
    }
};
