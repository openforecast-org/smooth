#pragma once

// Assumes armadillo is included by the wrapper.
//
// Stationarity / invertibility of a factor 1 + c_1 z + ... + c_q z^q of the
// multiplicative (multiple) seasonal ARIMA, with c = -phi for the AR factors and
// c = theta for the MA ones. The polynomial of a lag s is c(B^s), whose roots are
// the s-th roots of those of c(z), so it is stationary / invertible exactly when
// c(z) is, and the product of the factors exactly when all of them are.
//
// Schur-Cohn step-down: with k = c_q, the roots of c(z) lie outside the unit
// circle if and only if |k| < 1 and those of the polynomial of order q-1 with
// c'_i = (c_i - k c_{q-i}) / (1 - k^2) do. The k are the partial autocorrelations
// of an AR with these coefficients. Returns the largest |k| up to the first one
// that is not below one: below one exactly when the factor is stationary /
// invertible, and growing as it moves away, which gives the cost function a slope.
inline double arimaReflection(arma::vec c) {
    double largest = 0;
    for(arma::uword q = c.n_elem; q > 0; --q) {
        double k = c(q - 1);
        largest = std::max(largest, std::abs(k));
        if(std::abs(k) >= 1) {
            break;
        }
        if(q > 1) {
            arma::vec head = c.head(q - 1);
            c.head(q - 1) = (head - k * arma::reverse(head)) / (1 - k * k);
        }
    }
    return largest;
}
