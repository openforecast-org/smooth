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

// The interval of the j-th (from 0) of the parameters of one factor, the others
// fixed, on which the factor stays stationary / invertible, around their current
// values; sign is -1 for AR (c = -phi) and 1 for MA (c = theta). The coefficient
// of z^(j+1) of a factor with all roots outside the unit circle is below the
// binomial coefficient C(q, j+1) in absolute value, so stepping out from the
// current value by step finds the first crossing, which is then bisected. NaNs if
// the factor is not stationary / invertible at the current values.
inline arma::vec arimaParameterBounds(arma::vec values, arma::uword j, double sign,
                                      double step = 0.01) {
    arma::vec bounds(2);
    bounds.fill(arma::datum::nan);
    auto stable = [&](double value) {
        values(j) = value;
        return arimaReflection(sign * values) < 1;
    };
    double current = values(j);
    if(!stable(current)) {
        return bounds;
    }
    arma::uword q = values.n_elem;
    double limit = 1;
    for(arma::uword i = 0; i <= j; ++i) {
        limit *= double(q - i) / double(i + 1);
    }
    for(int side = 0; side < 2; ++side) {
        double direction = side == 0 ? -1 : 1, inside = current, outside = current;
        do {
            inside = outside;
            outside += direction * step;
        } while(std::abs(outside) < limit && stable(outside));
        if(std::abs(outside) > limit) {
            outside = direction * limit;
        }
        for(int i = 0; i < 40; ++i) {
            double middle = (inside + outside) / 2;
            (stable(middle) ? inside : outside) = middle;
        }
        bounds(side) = inside;
    }
    return bounds;
}
