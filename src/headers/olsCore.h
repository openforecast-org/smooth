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
