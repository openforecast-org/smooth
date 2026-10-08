#pragma once

// Assumes armadillo is included by the wrapper.
//
// Least-squares solution to X * b = y via pivoted QR with a scale-invariant
// rank cutoff. The QR runs on X as-is (no equilibration), so well-conditioned
// inputs go through the same floating-point path on the R and Python builds.
// Rank determination uses the ratio |R(i,i)| / ||X.col(P(i))||, which matches
// LINPACK dqrls' criterion and stays well-defined under wild column scaling
// (e.g. raw Vandermonde basis on long t-vectors).
//
// Aliased columns get a zero coefficient. Caller computes fitted = X * b.
inline arma::vec olsCore(const arma::mat& X, const arma::vec& y, double tol = 1e-7) {
    const arma::uword p = X.n_cols;

    arma::vec colNorms(p);
    for(arma::uword j = 0; j < p; j++) {
        double s = arma::norm(X.col(j), 2);
        colNorms(j) = (s > 0.0) ? s : 1.0;
    }

    arma::mat Q, R;
    arma::uvec P;
    arma::qr(Q, R, P, X, "vector");

    arma::uword maxRank = std::min(X.n_rows, p);
    arma::uword rank = 0;
    for(arma::uword i = 0; i < maxRank; i++) {
        double ratio = std::abs(R(i, i)) / colNorms(P(i));
        if(ratio > tol) {
            rank++;
        }
        else {
            break;
        }
    }

    arma::vec b(p, arma::fill::zeros);
    if(rank > 0) {
        arma::vec rhs = Q.cols(0, rank - 1).t() * y;
        arma::mat Rsub = R.submat(0, 0, rank - 1, rank - 1);
        // pinv handles near-singular Rsub gracefully (SVD-based, no warning),
        // matching the LINPACK dqrls fallback semantics. For well-conditioned
        // Rsub the result is identical (up to floating point) to a triangular
        // back-substitution.
        arma::vec z = arma::pinv(Rsub) * rhs;
        for(arma::uword i = 0; i < rank; i++) {
            b(P(i)) = z(i);
        }
    }
    return b;
}

// Householder QR of a fixed design, for the least squares of many responses (the
// global model of tbats()). The loops are written out, without BLAS or LAPACK, so the
// R and Python builds round identically whatever library each links. The compact form
// is LINPACK's: the reflections in and below the diagonal of qr, with the leading
// element of each in qraux, and R above the diagonal, its diagonal in rDiag.
struct HouseholderQR {
    arma::mat qr;
    arma::vec qraux;
    arma::vec rDiag;
};

inline HouseholderQR householderQR(arma::mat X) {
    const arma::uword n = X.n_rows, k = X.n_cols;
    arma::vec qraux(k, arma::fill::zeros), rDiag(k, arma::fill::zeros);
    for(arma::uword j = 0; j < k && j < n; j++) {
        double norm = 0.0;
        for(arma::uword i = j; i < n; i++) {
            norm += X(i, j) * X(i, j);
        }
        norm = std::sqrt(norm);
        if(norm == 0.0) {
            continue;
        }
        if(X(j, j) < 0.0) {
            norm = -norm;
        }
        for(arma::uword i = j; i < n; i++) {
            X(i, j) /= norm;
        }
        X(j, j) += 1.0;
        for(arma::uword l = j + 1; l < k; l++) {
            double t = 0.0;
            for(arma::uword i = j; i < n; i++) {
                t -= X(i, j) * X(i, l);
            }
            t /= X(j, j);
            for(arma::uword i = j; i < n; i++) {
                X(i, l) += t * X(i, j);
            }
        }
        qraux(j) = X(j, j);
        rDiag(j) = -norm;
    }
    return HouseholderQR{X, qraux, rDiag};
}

// Applies the reflection j to y in place
inline void householderReflect(const HouseholderQR &d, arma::uword j, arma::vec &y) {
    if(d.qraux(j) == 0.0) {
        return;
    }
    double t = -d.qraux(j) * y(j);
    for(arma::uword i = j + 1; i < y.n_elem; i++) {
        t -= d.qr(i, j) * y(i);
    }
    t /= d.qraux(j);
    y(j) += t * d.qraux(j);
    for(arma::uword i = j + 1; i < y.n_elem; i++) {
        y(i) += t * d.qr(i, j);
    }
}

// The coefficients, as R's qr.coef(): Q'y, then the back substitution on R
inline arma::vec householderCoef(const HouseholderQR &d, arma::vec y) {
    const arma::uword k = d.qraux.n_elem;
    for(arma::uword j = 0; j < k; j++) {
        householderReflect(d, j, y);
    }
    // A column that depends on the previous ones (a diagonal of R at zero, relative to the
    // largest) gets a zero coefficient, as an aliased one in R's lm()
    const double tol = 1e-10 * arma::max(arma::abs(d.rDiag));
    arma::vec b(k);
    for(arma::uword jj = k; jj-- > 0;) {
        if(std::abs(d.rDiag(jj)) <= tol) {
            b(jj) = 0.0;
            continue;
        }
        double s = y(jj);
        for(arma::uword l = jj + 1; l < k; l++) {
            s -= d.qr(jj, l) * b(l);
        }
        b(jj) = s / d.rDiag(jj);
    }
    return b;
}

// The residuals, as R's qr.resid(): Q'y with the first k elements set to zero, then Q
inline arma::vec householderResid(const HouseholderQR &d, arma::vec y) {
    const arma::uword k = d.qraux.n_elem;
    for(arma::uword j = 0; j < k; j++) {
        householderReflect(d, j, y);
    }
    y.head(k).zeros();
    for(arma::uword j = k; j-- > 0;) {
        householderReflect(d, j, y);
    }
    return y;
}

// sin(pi*x) as R's sinpi(): exact at the multiples of 1/2
inline double sinpiCore(double x) {
    x = std::fmod(x, 2.0);
    if(x <= -1.0) {
        x += 2.0;
    }
    else if(x > 1.0) {
        x -= 2.0;
    }
    if(x == 0.0 || x == 1.0) {
        return 0.0;
    }
    if(x == 0.5) {
        return 1.0;
    }
    if(x == -0.5) {
        return -1.0;
    }
    return std::sin(arma::datum::pi * x);
}

// The missing values (NaN) of a series filled for the initialisation of a model, which
// skips them in its fit: the least squares of the observed values (of their logarithms for
// positive data) on a polynomial of the time and harmonics of lagMax, through the
// Householder QR above, so that R and Python fill identically
inline arma::vec naFillCore(arma::vec y, unsigned int lagMax) {
    const arma::uword n = y.n_elem;
    arma::uvec observed = arma::find_finite(y);
    arma::uvec missing = arma::find_nonfinite(y);
    if(missing.n_elem == 0 || observed.n_elem == 0) {
        return y;
    }
    bool positive = true;
    for(arma::uword i : observed) {
        if(y(i) <= 0) {
            positive = false;
            break;
        }
    }
    const arma::uword degree = std::min<arma::uword>(std::max<arma::uword>(n / 10, 1), 5);
    // The time scaled into [-1, 1], so that its powers stay comparable
    arma::mat X(n, 1 + degree + (lagMax > 1 ? lagMax - 1 : 0), arma::fill::ones);
    for(arma::uword t = 0; t < n; t++) {
        double scaled = (n > 1) ? 2.0 * t / (n - 1.0) - 1.0 : 0.0;
        double power = 1.0;
        for(arma::uword k = 1; k <= degree; k++) {
            power *= scaled;
            X(t, k) = power;
        }
        // sinpi((t+1)*lagMax/lagMax) is zero at every integer time, so k stops before it
        for(arma::uword k = 1; k < lagMax; k++) {
            X(t, degree + k) = sinpiCore(static_cast<double>((t + 1) * k) / lagMax);
        }
    }
    arma::vec target = y.elem(observed);
    if(positive) {
        target = arma::log(target);
    }
    HouseholderQR d = householderQR(X.rows(observed));
    arma::vec b = householderCoef(d, target);
    for(arma::uword i : missing) {
        double value = 0.0;
        for(arma::uword j = 0; j < X.n_cols; j++) {
            value += X(i, j) * b(j);
        }
        y(i) = positive ? std::exp(value) : value;
    }
    return y;
}
