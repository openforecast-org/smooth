#pragma once

// Assumes armadillo is included by the wrapper.
//
// Hannan-Rissanen starting values for the AR / MA parameters of the
// multiplicative (multiple) seasonal ARMA used in ADAM:
//     prod_i phi_i(B^{s_i}) w_t = prod_i theta_i(B^{s_i}) e_t,
// with phi_i(z) = 1 - sum_k phi_{ik} z^k and theta_i(z) = 1 + sum_j theta_{ij} z^j.
//
// Each seasonal level is estimated separately, from the largest lag down:
//   1. a short AR in B^s gives the innovations a_t of that level;
//   2. OLS of w_t on w_{t-ks} and a_{t-js} gives phi and theta;
//   3. the estimated factor is filtered out and the result goes to the next level.
// One backfitting pass then re-estimates each level on w with all the other
// levels filtered out. A level that cannot be estimated (too few seasons) keeps
// the defaults 0.1 (AR) and -0.1 (MA).
//
// A non-invertible MA factor is replaced by its invertible counterpart with the
// same autocorrelations, reflecting the inverse roots outside the unit circle
// (hrReflect). The estimates are otherwise returned as they are, unless the cost
// function would reject them (hrFeasible); then only the
// offending factors are moved inside the boundary. The filtering between the levels uses an invertible copy
// of each MA factor, as a non-invertible one makes the recursion explode.

#include "olsCore.h"
#include "arimaBounds.h"

struct HRLevel {
    arma::uword lag, arOrder, maOrder;
    bool arEstimate, maEstimate;
    arma::vec ar, ma;
};

// Least squares via the normal equations: cheap for the tall, narrow regressions
// here (olsCore forms the full n x n Q). Falls back to olsCore if ill-conditioned
inline arma::vec hrOLS(const arma::mat &X, const arma::vec &y) {
    arma::mat XtX = X.t() * X;
    if(arma::rcond(XtX) > 1e-10) {
        return arma::solve(XtX, X.t() * y, arma::solve_opts::fast + arma::solve_opts::likely_sympd);
    }
    return olsCore(X, y);
}

// Inverse roots (companion eigenvalues) of 1 - sum_k (sign * c_k) z^k: largest modulus
inline double hrModulus(const arma::vec &coefs, double sign) {
    arma::uword k = coefs.n_elem;
    if(k == 0) {
        return 0;
    }
    arma::mat companion(k, k, arma::fill::zeros);
    companion.row(0) = sign * coefs.t();
    if(k > 1) {
        companion.submat(1, 0, k - 1, k - 2) = arma::eye(k - 1, k - 1);
    }
    return arma::max(arma::abs(arma::eig_gen(companion)));
}

// Scale c_k by lambda^k, so that the largest inverse root is at most target
inline arma::vec hrScale(arma::vec coefs, double sign, double target) {
    double modulus = hrModulus(coefs, sign);
    if(modulus > target) {
        for(arma::uword i = 0; i < coefs.n_elem; ++i) {
            coefs(i) *= std::pow(target / modulus, i + 1);
        }
    }
    return coefs;
}

// Reflect the inverse roots of 1 + sum_j theta_j z^j outside the unit circle,
// lambda -> 1 / conj(lambda): the invertible MA with the same autocorrelations
inline arma::vec hrReflect(const arma::vec &ma) {
    arma::uword k = ma.n_elem;
    if(k == 0 || hrModulus(ma, -1) <= 1) {
        return ma;
    }
    arma::mat companion(k, k, arma::fill::zeros);
    companion.row(0) = -ma.t();
    if(k > 1) {
        companion.submat(1, 0, k - 1, k - 2) = arma::eye(k - 1, k - 1);
    }
    arma::cx_vec roots = arma::eig_gen(companion);
    arma::cx_vec poly(1, arma::fill::ones);
    for(arma::uword i = 0; i < k; ++i) {
        std::complex<double> root = (std::abs(roots(i)) > 1) ? 1.0 / std::conj(roots(i)) : roots(i);
        arma::cx_vec factor = {1.0, -root};
        poly = arma::conv(poly, factor);
    }
    return arma::real(poly.tail(k));
}

// The inverse roots are moved to this modulus when a start is infeasible
const double hrInside = 0.99;

// u = theta(B^s)^{-1} phi(B^s) w with zero pre-sample values
inline arma::vec hrFilter(const arma::vec &w, const arma::vec &ar, arma::vec ma, arma::uword s) {
    ma = hrScale(ma, -1, hrInside);
    arma::uword n = w.n_elem;
    arma::vec u = w;
    for(arma::uword t = 0; t < n; ++t) {
        for(arma::uword k = 1; k <= ar.n_elem && k * s <= t; ++k) {
            u(t) -= ar(k - 1) * w(t - k * s);
        }
        for(arma::uword j = 1; j <= ma.n_elem && j * s <= t; ++j) {
            u(t) -= ma(j - 1) * u(t - j * s);
        }
    }
    return u;
}

// Estimate one level on the series w; false if the sample is too short
inline bool hrLevelEstimate(const arma::vec &w, HRLevel &level) {
    long n = w.n_elem, s = level.lag, P = level.arOrder, Q = level.maOrder;
    long nAR = level.arEstimate * P, nMA = level.maEstimate * Q;
    if(nAR + nMA == 0) {
        return true;
    }

    // Stage 1: innovations of this level from an AR of order m in B^s
    arma::vec a(n, arma::fill::zeros);
    long start = P * s;
    if(Q > 0) {
        long m = (s == 1) ? std::min(std::max(P + Q + 1, (long) std::floor(10 * std::log10((double) n))), n / 4) :
                            std::min(5L, n / s - P - Q - 2);
        if(m < std::max(P, Q) + 1) {
            return false;
        }
        arma::mat X(n - m * s, m);
        for(long j = 1; j <= m; ++j) {
            X.col(j - 1) = w.subvec(m * s - j * s, n - 1 - j * s);
        }
        arma::vec wRows = w.subvec(m * s, n - 1);
        a.subvec(m * s, n - 1) = wRows - X * hrOLS(X, wRows);
        start = m * s + std::max(P, Q) * s;
    }
    long nRows = n - start;
    if(nRows <= nAR + nMA + 2) {
        return false;
    }

    // Stage 2: regression on the lags of w and a, known parameters moved to the left
    arma::vec target = w.subvec(start, n - 1);
    arma::mat X(nRows, nAR + nMA);
    for(long k = 1; k <= P; ++k) {
        arma::vec lagged = w.subvec(start - k * s, n - 1 - k * s);
        if(level.arEstimate) {
            X.col(k - 1) = lagged;
        }
        else {
            target -= level.ar(k - 1) * lagged;
        }
    }
    for(long j = 1; j <= Q; ++j) {
        arma::vec lagged = a.subvec(start - j * s, n - 1 - j * s);
        if(level.maEstimate) {
            X.col(nAR + j - 1) = lagged;
        }
        else {
            target -= level.ma(j - 1) * lagged;
        }
    }
    arma::vec b = hrOLS(X, target);

    if(nAR > 0) {
        level.ar = b.head(nAR);
    }
    if(nMA > 0) {
        level.ma = hrReflect(b.tail(nMA));
    }
    return true;
}

// The largest reflection coefficient of the AR (sign 1) or MA (sign -1) factor of a
// level, in the form 1 + c z of arimaBounds.h: c = -phi or theta
inline double hrReflection(const HRLevel &level, double sign) {
    const arma::vec &coefs = (sign > 0) ? level.ar : level.ma;
    return (coefs.n_elem == 0) ? 0 : arimaReflection(-sign * coefs);
}

// Move the estimated factors that the cost function would reject, the not
// stationary AR and not invertible MA ones (arimaBounds.h), inside the boundary
inline void hrFeasible(std::vector<HRLevel> &levels, double sign) {
    for(HRLevel &level : levels) {
        bool estimate = (sign > 0) ? level.arEstimate : level.maEstimate;
        if(estimate && hrReflection(level, sign) >= 1) {
            arma::vec &coefs = (sign > 0) ? level.ar : level.ma;
            coefs = hrScale(coefs, sign, hrInside);
        }
    }
}

// Filter out all the levels in the list except the one with index skip
inline arma::vec hrFilterLevels(arma::vec w, const std::vector<HRLevel> &levels,
                                const std::vector<arma::uword> &indices, long skip) {
    for(arma::uword idx : indices) {
        if((long) idx != skip) {
            w = hrFilter(w, levels[idx].ar, levels[idx].ma, levels[idx].lag);
        }
    }
    return w;
}

// The levels with their estimates, behind arimaHRCore and arimaHRSelectCore
inline std::vector<HRLevel> arimaHRLevels(arma::vec w, const arma::uvec &arOrders, const arma::uvec &maOrders,
                                          const arma::uvec &lags, bool arEstimate, bool maEstimate,
                                          const arma::vec &armaParameters, const arma::uvec &useLevel,
                                          bool bounded) {
    arma::uword nLevels = lags.n_elem, nProvided = 0;
    std::vector<HRLevel> levels(nLevels);
    for(arma::uword i = 0; i < nLevels; ++i) {
        HRLevel &level = levels[i];
        level.lag = lags(i);
        level.arOrder = arOrders(i);
        level.maOrder = maOrders(i);
        level.arEstimate = arEstimate;
        level.maEstimate = maEstimate;
        level.ar = arma::vec(arOrders(i)).fill(0.1);
        level.ma = arma::vec(maOrders(i)).fill(-0.1);
        if(!arEstimate && arOrders(i) > 0) {
            level.ar = armaParameters.subvec(nProvided, nProvided + arOrders(i) - 1);
            nProvided += arOrders(i);
        }
        if(!maEstimate && maOrders(i) > 0) {
            level.ma = armaParameters.subvec(nProvided, nProvided + maOrders(i) - 1);
            nProvided += maOrders(i);
        }
    }

    bool usable = w.n_elem > 3 && w.is_finite();
    if(usable) {
        w -= arma::mean(w);

        // Levels with nothing to estimate are known: filter them out first
        std::vector<arma::uword> known, active;
        arma::uvec order = arma::sort_index(lags, "descend");
        for(arma::uword idx : order) {
            bool toEstimate = (arEstimate * arOrders(idx) + maEstimate * maOrders(idx)) > 0;
            if(toEstimate && useLevel(idx)) {
                active.push_back(idx);
            }
            else if(!toEstimate && (arOrders(idx) + maOrders(idx)) > 0) {
                known.push_back(idx);
            }
        }
        w = hrFilterLevels(w, levels, known, -1);

        // Top-down pass. Failed levels keep the defaults and are not filtered out
        std::vector<arma::uword> estimated;
        arma::vec u = w;
        for(arma::uword idx : active) {
            if(hrLevelEstimate(u, levels[idx])) {
                u = hrFilter(u, levels[idx].ar, levels[idx].ma, levels[idx].lag);
                estimated.push_back(idx);
            }
        }

        // Backfitting pass
        if(estimated.size() > 1) {
            for(arma::uword idx : estimated) {
                HRLevel candidate = levels[idx];
                if(hrLevelEstimate(hrFilterLevels(w, levels, estimated, idx), candidate)) {
                    levels[idx] = candidate;
                }
            }
        }
        if(bounded) {
            hrFeasible(levels, 1);
            hrFeasible(levels, -1);
        }
    }
    return levels;
}

// The parameters of the levels in the order of B in ADAM
inline arma::vec hrParameters(const std::vector<HRLevel> &levels, bool arEstimate, bool maEstimate) {
    std::vector<double> result;
    for(const HRLevel &level : levels) {
        if(arEstimate) {
            result.insert(result.end(), level.ar.begin(), level.ar.end());
        }
        if(maEstimate) {
            result.insert(result.end(), level.ma.begin(), level.ma.end());
        }
    }
    return arma::vec(result);
}

// Returns the estimated AR / MA parameters in the order of B in ADAM:
// for each lag, the AR parameters (if arEstimate) and then the MA ones (if maEstimate).
// armaParameters holds the provided values in the same traversal for the parts
// that are not estimated. useLevel switches the estimation of a level off.
// bounded switches the stationarity / invertibility checks of the cost function on.
inline arma::vec arimaHRCore(arma::vec w, const arma::uvec &arOrders, const arma::uvec &maOrders,
                             const arma::uvec &lags, bool arEstimate, bool maEstimate,
                             const arma::vec &armaParameters, const arma::uvec &useLevel,
                             bool bounded) {
    return hrParameters(arimaHRLevels(w, arOrders, maOrders, lags, arEstimate, maEstimate,
                                      armaParameters, useLevel, bounded),
                        arEstimate, maEstimate);
}

struct HRSelectResult {
    arma::umat orders;
    arma::mat parameters, innovations;
};

// Hannan-Rissanen screen of the orders of the level with index screen, the other
// levels at the orders given: for every AR order up to arMax and MA order up to maMax,
// the parameters (a row padded with NaN, in the order of B in ADAM) and the
// innovations, w with all the levels filtered out
inline HRSelectResult arimaHRSelectCore(arma::vec w, arma::uvec arOrders, arma::uvec maOrders,
                                        const arma::uvec &lags, arma::uword screen,
                                        arma::uword arMax, arma::uword maMax, bool bounded) {
    arma::uword nCandidates = (arMax + 1) * (maMax + 1), nLevels = lags.n_elem;
    arOrders(screen) = arMax;
    maOrders(screen) = maMax;
    HRSelectResult result;
    result.orders.set_size(nCandidates, 2);
    result.parameters.set_size(nCandidates, arma::accu(arOrders + maOrders));
    result.parameters.fill(arma::datum::nan);
    result.innovations.set_size(w.n_elem, nCandidates);

    w -= arma::mean(w);
    arma::uvec order = arma::sort_index(lags, "descend");
    std::vector<arma::uword> indices(order.begin(), order.end());
    arma::uword i = 0;
    for(arma::uword p = 0; p <= arMax; ++p) {
        for(arma::uword q = 0; q <= maMax; ++q, ++i) {
            arOrders(screen) = p;
            maOrders(screen) = q;
            std::vector<HRLevel> levels = arimaHRLevels(w, arOrders, maOrders, lags, true, true,
                                                        arma::vec(), arma::ones<arma::uvec>(nLevels),
                                                        bounded);
            arma::vec parameters = hrParameters(levels, true, true);
            result.orders(i, 0) = p;
            result.orders(i, 1) = q;
            if(parameters.n_elem > 0) {
                result.parameters.row(i).head(parameters.n_elem) = parameters.t();
            }
            result.innovations.col(i) = hrFilterLevels(w, levels, indices, -1);
        }
    }
    return result;
}
