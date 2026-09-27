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
// levels filtered out. Estimated factors are shrunk so that their roots have
// modulus at least 1/cap (stationary / invertible with a margin).
// A level that cannot be estimated (too few seasons) keeps the defaults
// 0.1 (AR) and -0.1 (MA).

#include "olsCore.h"

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

// u = theta(B^s)^{-1} phi(B^s) w with zero pre-sample values
inline arma::vec hrFilter(const arma::vec &w, const arma::vec &ar, const arma::vec &ma, arma::uword s) {
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

// Scale coefficients c_k by lambda^k so that the companion eigenvalues of the
// polynomial 1 - sum_k (sign * c_k) z^k have modulus at most cap
inline arma::vec hrStabilise(arma::vec coefs, double sign, double cap) {
    arma::uword k = coefs.n_elem;
    if(k == 0) {
        return coefs;
    }
    arma::mat companion(k, k, arma::fill::zeros);
    companion.row(0) = sign * coefs.t();
    if(k > 1) {
        companion.submat(1, 0, k - 1, k - 2) = arma::eye(k - 1, k - 1);
    }
    double modulus = arma::max(arma::abs(arma::eig_gen(companion)));
    if(modulus > cap) {
        double lambda = cap / modulus;
        for(arma::uword i = 0; i < k; ++i) {
            coefs(i) *= std::pow(lambda, i + 1);
        }
    }
    return coefs;
}

// Estimate one level on the series w; false if the sample is too short
inline bool hrLevelEstimate(const arma::vec &w, HRLevel &level, double cap) {
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
        level.ar = hrStabilise(b.head(nAR), 1, cap);
    }
    if(nMA > 0) {
        level.ma = hrStabilise(b.tail(nMA), -1, cap);
    }
    return true;
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

// Returns the estimated AR / MA parameters in the order of B in ADAM:
// for each lag, the AR parameters (if arEstimate) and then the MA ones (if maEstimate).
// armaParameters holds the provided values in the same traversal for the parts
// that are not estimated. useLevel switches the estimation of a level off.
inline arma::vec arimaHRCore(arma::vec w, const arma::uvec &arOrders, const arma::uvec &maOrders,
                             const arma::uvec &lags, bool arEstimate, bool maEstimate,
                             const arma::vec &armaParameters, const arma::uvec &useLevel,
                             double cap = 0.9) {
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
            if(hrLevelEstimate(u, levels[idx], cap)) {
                u = hrFilter(u, levels[idx].ar, levels[idx].ma, levels[idx].lag);
                estimated.push_back(idx);
            }
        }

        // Backfitting pass
        if(estimated.size() > 1) {
            for(arma::uword idx : estimated) {
                HRLevel candidate = levels[idx];
                if(hrLevelEstimate(hrFilterLevels(w, levels, estimated, idx), candidate, cap)) {
                    levels[idx] = candidate;
                }
            }
        }
    }

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
