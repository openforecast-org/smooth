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
    arma::vec b(k);
    for(arma::uword jj = k; jj-- > 0;) {
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
