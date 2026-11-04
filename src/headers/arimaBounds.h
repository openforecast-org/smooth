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
        // c_i and c_{q-2-i} in pairs, in place: (c_i - k c_{q-2-i}) / (1 - k^2) and back
        const double scale = 1 / (1 - k * k);
        for(arma::uword i = 0, j = q - 1; i + 1 <= j; ++i) {
            --j;
            double a = c(i), b = c(j);
            c(i) = (a - k * b) * scale;
            if(j != i) {
                c(j) = (b - k * a) * scale;
            }
        }
    }
    return largest;
}

// The interval around current on which stable(value) holds: stepping out from current by
// step until stable() fails or |value| passes limit, then bisecting that crossing 40 times
// and keeping the stable side. NaNs if current is not stable. The bounds of the ARMA
// parameters below, and those of the smoothing parameters of ETS and CES under the
// forecastability condition (eigenBounds() and cesBounds() in R, eigen_bounds() and
// ces_bounds() in Python)
template <typename Stable>
inline arma::vec stabilityInterval(Stable stable, double current, double step, double limit) {
    arma::vec bounds(2);
    bounds.fill(arma::datum::nan);
    if(!stable(current)) {
        return bounds;
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

// The interval of the j-th (from 0) of the parameters of one factor, the others
// fixed, on which the factor stays stationary / invertible, around their current
// values; sign is -1 for AR (c = -phi) and 1 for MA (c = theta). The coefficient
// of z^(j+1) of a factor with all roots outside the unit circle is below the
// binomial coefficient C(q, j+1) in absolute value, which limits the search. NaNs
// if the factor is not stationary / invertible at the current values.
inline arma::vec arimaParameterBounds(arma::vec values, arma::uword j, double sign,
                                      double step = 0.01) {
    arma::uword q = values.n_elem;
    double limit = 1;
    for(arma::uword i = 0; i <= j; ++i) {
        limit *= double(q - i) / double(i + 1);
    }
    return stabilityInterval([&](double value) {
        values(j) = value;
        return arimaReflection(sign * values) < 1;
    }, values(j), step, limit);
}
