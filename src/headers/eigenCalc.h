#include <map>
#include <vector>
#include "arimaBounds.h"

// Helper: Invert measurement matrix (1/x, with inf->0)
arma::mat measurementInverterCpp(const arma::mat& measurement) {
    arma::mat result = 1.0 / measurement;
    result.replace(arma::datum::inf, 0.0);
    result.replace(-arma::datum::inf, 0.0);
    return result;
}

// The moduli of the eigenvalues of a real square matrix, written out without LAPACK so
// that the R and Python builds round identically whatever library each links (the
// admissible bounds of tbats() sit on the boundary, where a last-bit difference between
// two LAPACK builds decided which parameters were admissible). EISPACK's balancing,
// orthogonal reduction to Hessenberg form and shifted QR (balanc, orthes and hqr, with
// the iteration limit and exceptional shifts of LAPACK's dlahqr), with 1-based indices
// as there. A matrix whose QR does not converge gets infinite moduli, so that it is
// never admissible.
inline arma::vec eigenModuliCore(const arma::mat& A) {
    const int n = A.n_rows;
    arma::vec moduli(n);
    if(n == 0) {
        return moduli;
    }
    arma::mat a(n + 1, n + 1, arma::fill::zeros);
    a.submat(1, 1, n, n) = A;

    // Balancing by powers of two
    bool done = false;
    while(!done) {
        done = true;
        for(int i = 1; i <= n; i++) {
            double r = 0.0, c = 0.0;
            for(int j = 1; j <= n; j++) {
                if(j != i) {
                    c += std::fabs(a(j, i));
                    r += std::fabs(a(i, j));
                }
            }
            if(c != 0.0 && r != 0.0) {
                double g = r / 2.0, f = 1.0, s = c + r;
                while(c < g) {
                    f *= 2.0;
                    c *= 4.0;
                }
                g = r * 2.0;
                while(c > g) {
                    f /= 2.0;
                    c /= 4.0;
                }
                if((c + r) / f < 0.95 * s) {
                    done = false;
                    g = 1.0 / f;
                    for(int j = 1; j <= n; j++) {
                        a(i, j) *= g;
                    }
                    for(int j = 1; j <= n; j++) {
                        a(j, i) *= f;
                    }
                }
            }
        }
    }

    // Reduction to upper Hessenberg form by Householder reflections (EISPACK's orthes)
    arma::vec ort(n + 1, arma::fill::zeros);
    for(int m = 2; m < n; m++) {
        double scale = 0.0;
        for(int i = m; i <= n; i++) {
            scale += std::fabs(a(i, m - 1));
        }
        if(scale == 0.0) {
            continue;
        }
        double h = 0.0;
        for(int i = n; i >= m; i--) {
            ort(i) = a(i, m - 1) / scale;
            h += ort(i) * ort(i);
        }
        double g = ort(m) >= 0.0 ? -std::sqrt(h) : std::sqrt(h);
        h -= ort(m) * g;
        ort(m) -= g;
        for(int j = m; j <= n; j++) {
            double f = 0.0;
            for(int i = n; i >= m; i--) {
                f += ort(i) * a(i, j);
            }
            f /= h;
            for(int i = m; i <= n; i++) {
                a(i, j) -= f * ort(i);
            }
        }
        for(int i = 1; i <= n; i++) {
            double f = 0.0;
            for(int j = n; j >= m; j--) {
                f += ort(j) * a(i, j);
            }
            f /= h;
            for(int j = m; j <= n; j++) {
                a(i, j) -= f * ort(j);
            }
        }
        a(m, m - 1) = scale * g;
    }
    for(int i = 3; i <= n; i++) {
        for(int j = 1; j < i - 1; j++) {
            a(i, j) = 0.0;
        }
    }

    // The shifted QR on the Hessenberg matrix
    arma::vec wr(n + 1, arma::fill::zeros), wi(n + 1, arma::fill::zeros);
    double anorm = 0.0;
    for(int i = 1; i <= n; i++) {
        for(int j = std::max(i - 1, 1); j <= n; j++) {
            anorm += std::fabs(a(i, j));
        }
    }
    // LAPACK's dlahqr: up to 30*max(10, n) iterations per eigenvalue, with an exceptional
    // shift every tenth (hqr's 30 and two shifts fail on the clustered eigenvalues near
    // the unit circle of harmonics with small smoothing parameters)
    const int maxIterations = 30 * std::max(10, n);
    int nn = n, l = 1, m = 1;
    double t = 0.0, p = 0.0, q = 0.0, r = 0.0, s, w, x, y, z, u, v;
    while(nn >= 1) {
        int its = 0;
        do {
            for(l = nn; l >= 2; l--) {
                s = std::fabs(a(l - 1, l - 1)) + std::fabs(a(l, l));
                if(s == 0.0) {
                    s = anorm;
                }
                if(std::fabs(a(l, l - 1)) + s == s) {
                    a(l, l - 1) = 0.0;
                    break;
                }
            }
            x = a(nn, nn);
            if(l == nn) {
                wr(nn) = x + t;
                wi(nn--) = 0.0;
            }
            else {
                y = a(nn - 1, nn - 1);
                w = a(nn, nn - 1) * a(nn - 1, nn);
                if(l == nn - 1) {
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    z = std::sqrt(std::fabs(q));
                    x += t;
                    if(q >= 0.0) {
                        z = p + (p >= 0.0 ? std::fabs(z) : -std::fabs(z));
                        wr(nn - 1) = wr(nn) = x + z;
                        if(z != 0.0) {
                            wr(nn) = x - w / z;
                        }
                        wi(nn - 1) = wi(nn) = 0.0;
                    }
                    else {
                        wr(nn - 1) = wr(nn) = x + p;
                        wi(nn - 1) = -(wi(nn) = z);
                    }
                    nn -= 2;
                }
                else {
                    if(its == maxIterations) {
                        moduli.fill(arma::datum::inf);
                        return moduli;
                    }
                    if(its > 0 && its % 10 == 0) {
                        t += x;
                        for(int i = 1; i <= nn; i++) {
                            a(i, i) -= x;
                        }
                        s = std::fabs(a(nn, nn - 1)) + std::fabs(a(nn - 1, nn - 2));
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    for(m = nn - 2; m >= l; m--) {
                        z = a(m, m);
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / a(m + 1, m) + a(m, m + 1);
                        q = a(m + 1, m + 1) - z - r - s;
                        r = a(m + 2, m + 1);
                        s = std::fabs(p) + std::fabs(q) + std::fabs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if(m == l) {
                            break;
                        }
                        u = std::fabs(a(m, m - 1)) * (std::fabs(q) + std::fabs(r));
                        v = std::fabs(p) * (std::fabs(a(m - 1, m - 1)) + std::fabs(z) + std::fabs(a(m + 1, m + 1)));
                        if(u + v == v) {
                            break;
                        }
                    }
                    for(int i = m + 2; i <= nn; i++) {
                        a(i, i - 2) = 0.0;
                        if(i != m + 2) {
                            a(i, i - 3) = 0.0;
                        }
                    }
                    for(int k = m; k <= nn - 1; k++) {
                        if(k != m) {
                            p = a(k, k - 1);
                            q = a(k + 1, k - 1);
                            r = 0.0;
                            if(k != nn - 1) {
                                r = a(k + 2, k - 1);
                            }
                            if((x = std::fabs(p) + std::fabs(q) + std::fabs(r)) != 0.0) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        double root = std::sqrt(p * p + q * q + r * r);
                        if((s = (p >= 0.0 ? root : -root)) != 0.0) {
                            if(k == m) {
                                if(l != m) {
                                    a(k, k - 1) = -a(k, k - 1);
                                }
                            }
                            else {
                                a(k, k - 1) = -s * x;
                            }
                            p += s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q /= p;
                            r /= p;
                            for(int j = k; j <= nn; j++) {
                                p = a(k, j) + q * a(k + 1, j);
                                if(k != nn - 1) {
                                    p += r * a(k + 2, j);
                                    a(k + 2, j) -= p * z;
                                }
                                a(k + 1, j) -= p * y;
                                a(k, j) -= p * x;
                            }
                            int mmin = nn < k + 3 ? nn : k + 3;
                            for(int i = l; i <= mmin; i++) {
                                p = x * a(i, k) + y * a(i, k + 1);
                                if(k != nn - 1) {
                                    p += z * a(i, k + 2);
                                    a(i, k + 2) -= p * r;
                                }
                                a(i, k + 1) -= p * q;
                                a(i, k) -= p;
                            }
                        }
                    }
                }
            }
        } while(l < nn - 1);
    }
    for(int i = 1; i <= n; i++) {
        moduli(i - 1) = std::hypot(wr(i), wi(i));
    }
    return moduli;
}

// ---------------------------------------------------------------------------------------
// The admissibility of a lagged SSOE model through its ARIMA form. With the lag matrix
// L(B) = diag(B^l_1, ..., B^l_k), the model y_t = w' v_{t-l} + e_t, v_t = F v_{t-l} + g e_t
// satisfies phi(B) y_t = theta(B) e_t with phi(B) = det(I - F L(B)) and
// theta(B) = det(I - D L(B)), D = F - g w'. The model forgets its initial states when the
// roots of theta, after removing the factors it shares with phi, lie outside the unit
// circle. The shared factors that matter are unit roots: the redundancies between
// components (level and seasonals, seasonals of nested periods) and the components that the
// error does not drive (a constant, static regressors, persistence fixed at zero), all
// products of cyclotomic polynomials Phi_n(B) with n dividing a lag. The determinant and
// the FFT are written out, as in eigenModuliCore(), so that the R and Python builds round
// identically whatever LAPACK each links.

// The determinant of a complex square matrix by LU with partial pivoting; A is overwritten
inline std::complex<double> complexDeterminant(arma::cx_mat& A) {
    const arma::uword n = A.n_rows;
    std::complex<double> determinant(1.0, 0.0);
    for(arma::uword j = 0; j < n; j++) {
        arma::uword pivot = j;
        for(arma::uword i = j + 1; i < n; i++) {
            if(std::abs(A(i, j)) > std::abs(A(pivot, j))) {
                pivot = i;
            }
        }
        if(std::abs(A(pivot, j)) == 0.0) {
            return std::complex<double>(0.0, 0.0);
        }
        if(pivot != j) {
            A.swap_rows(pivot, j);
            determinant = -determinant;
        }
        determinant *= A(j, j);
        for(arma::uword i = j + 1; i < n; i++) {
            std::complex<double> factor = A(i, j) / A(j, j);
            for(arma::uword c = j + 1; c < n; c++) {
                A(i, c) -= factor * A(j, c);
            }
        }
    }
    return determinant;
}

// The discrete Fourier transform sum_j x_j exp(-2 pi i j m / n) of a vector whose length
// is a power of 2: iterative radix-2 Cooley-Tukey
inline arma::cx_vec fftCore(arma::cx_vec x) {
    const arma::uword n = x.n_elem;
    for(arma::uword i = 1, j = 0; i < n; i++) {
        arma::uword bit = n >> 1;
        for(; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if(i < j) {
            std::swap(x(i), x(j));
        }
    }
    for(arma::uword length = 2; length <= n; length <<= 1) {
        const std::complex<double> step = std::polar(1.0, -2.0 * arma::datum::pi / double(length));
        for(arma::uword start = 0; start < n; start += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for(arma::uword k = 0; k < length / 2; k++) {
                std::complex<double> even = x(start + k), odd = twiddle * x(start + k + length / 2);
                x(start + k) = even + odd;
                x(start + k + length / 2) = even - odd;
                twiddle *= step;
            }
        }
    }
    return x;
}

// Coefficients (increasing powers of B) of det(I - M L(B)) by the expansion in principal
// minors: the sum over the subsets S of the components of (-1)^|S| det(M_SS) B^(sum of the
// lags in S). 2^k minors, so for a few components only.
inline arma::vec minorPolynomial(const arma::mat& M, const arma::ivec& lags) {
    const arma::uword k = lags.n_elem;
    arma::vec coefficients(arma::accu(lags) + 1, arma::fill::zeros);
    arma::uvec idx(k);
    arma::cx_mat A;
    for(arma::uword subset = 0; subset < (arma::uword(1) << k); subset++) {
        arma::uword size = 0, power = 0;
        for(arma::uword c = 0; c < k; c++) {
            if(subset & (arma::uword(1) << c)) {
                idx(size++) = c;
                power += arma::uword(lags(c));
            }
        }
        if(size == 0) {
            coefficients(0) += 1.0;
            continue;
        }
        A = arma::conv_to<arma::cx_mat>::from(M.submat(idx.head(size), idx.head(size)));
        coefficients(power) += (size % 2 == 0 ? 1.0 : -1.0) * std::real(complexDeterminant(A));
    }
    return coefficients;
}

// Coefficients (increasing powers of B) of det(I - M L(B)), from its values at the N-th
// roots of unity, N the smallest power of 2 above the degree sum(lags); by the principal
// minors when there are no more of them than determinants to evaluate
inline arma::vec lagPolynomial(const arma::mat& M, const arma::ivec& lags) {
    const arma::uword k = lags.n_elem, degree = arma::accu(lags);
    arma::uword N = 1;
    while(N < degree + 1) {
        N <<= 1;
    }
    if(k < 20 && (arma::uword(1) << k) <= N / 2 + 1) {
        return minorPolynomial(M, lags);
    }
    // The N-th roots of unity, kept once computed
    static std::map<arma::uword, arma::cx_vec> unityRoots;
    arma::cx_vec& roots = unityRoots[N];
    if(roots.n_elem != N) {
        roots.set_size(N);
        for(arma::uword t = 0; t < N; t++) {
            roots(t) = std::polar(1.0, 2.0 * arma::datum::pi * double(t) / double(N));
        }
    }
    arma::cx_vec values(N);
    // The coefficients are real, so the values at conjugate roots of unity are conjugate
    const arma::cx_mat Mc = arma::conv_to<arma::cx_mat>::from(M);
    arma::cx_mat A(k, k);
    for(arma::uword j = 0; j <= N / 2; j++) {
        A = -Mc;
        for(arma::uword c = 0; c < k; c++) {
            A.col(c) *= roots((j * arma::uword(lags(c))) % N);
        }
        A.diag() += 1.0;
        values(j) = complexDeterminant(A);
        if(j > 0 && j < N - j) {
            values(N - j) = std::conj(values(j));
        }
    }
    const arma::vec coefficients = arma::real(fftCore(values));
    return coefficients.head(degree + 1) / double(N);
}

// Divide p by q (both in increasing powers); true, with the quotient, if the remainder
// is at most tolerance times the sum of the absolute coefficients of p
inline bool polynomialDivide(const arma::vec& p, const arma::vec& q, arma::vec& quotient,
                             double tolerance = 1e-9) {
    const arma::uword m = q.n_elem - 1;
    if(p.n_elem <= m) {
        return false;
    }
    arma::vec r = p;
    quotient.zeros(p.n_elem - m);
    for(arma::uword i = p.n_elem - m; i-- > 0;) {
        quotient(i) = r(i + m) / q(m);
        r.subvec(i, i + m) -= quotient(i) * q;
    }
    return arma::max(arma::abs(r.head(m))) <= tolerance * arma::accu(arma::abs(p));
}

// Integer coefficients of the n-th cyclotomic polynomial, (B^n - 1) divided by those of the
// proper divisors of n; kept once computed
inline const arma::vec& cyclotomicPolynomial(arma::uword n) {
    static std::map<arma::uword, arma::vec> computed;
    auto found = computed.find(n);
    if(found != computed.end()) {
        return found->second;
    }
    arma::vec p(n + 1, arma::fill::zeros), quotient;
    p(0) = -1;
    p(n) = 1;
    for(arma::uword d = 1; d < n; d++) {
        if(n % d == 0) {
            polynomialDivide(p, cyclotomicPolynomial(d), quotient);
            p = arma::round(quotient);
        }
    }
    return computed.emplace(n, p).first->second;
}

// Remove from theta and phi the cyclotomic factors Phi_n (n dividing a lag) they share, with
// their multiplicity. A root of Phi_n is tried first: a remainder within the tolerance of
// polynomialDivide() leaves |p(zeta)| below (n - 1) times it, so the division is attempted
// only when both polynomials are that small there. phi depends on F alone, which often stays
// the same from one call to the next (ETS); when the same phi comes a second time, its own
// cyclotomic factors are tabulated, and from then on only those are tried on theta, and the
// cancelled phi of the last call is reused when the same factors are removed.
inline void cancelUnitRoots(arma::vec& theta, arma::vec& phi, const arma::ivec& lags,
                            double tolerance = 1e-9) {
    struct PhiFactors {
        arma::vec phi, phiCancelled;
        arma::ivec lags;
        bool tabulated = false, cancelled = false;
        std::vector<std::pair<arma::uword, arma::uword>> factors;
        std::vector<arma::uword> removed;
    };
    static PhiFactors last;
    auto small = [&](const arma::vec& p, arma::uword n) {
        const std::complex<double> zeta = std::polar(1.0, 2.0 * arma::datum::pi / double(n));
        std::complex<double> value(0.0, 0.0);
        double size = 0.0;
        for(arma::uword i = p.n_elem; i-- > 0;) {
            value = value * zeta + p(i);
            size += std::abs(p(i));
        }
        return std::abs(value) <= std::max(1.0, double(n) - 1.0) * tolerance * size;
    };
    // Divide p by Phi_n while it is small at a root of Phi_n and divides, at most limit times
    arma::vec quotient;
    auto removeFactor = [&](arma::vec& p, arma::uword n, arma::uword limit) {
        arma::uword times = 0;
        while(times < limit && small(p, n) && polynomialDivide(p, cyclotomicPolynomial(n), quotient, tolerance)) {
            p = quotient;
            times++;
        }
        return times;
    };
    std::vector<arma::uword> candidates;
    for(arma::uword n = 1; n <= arma::uword(arma::max(lags)); n++) {
        bool divides = false;
        for(arma::uword i = 0; i < lags.n_elem && !divides; i++) {
            divides = arma::uword(lags(i)) % n == 0;
        }
        if(divides) {
            candidates.push_back(n);
        }
    }

    const bool same = last.phi.n_elem == phi.n_elem && arma::all(last.phi == phi) &&
                      last.lags.n_elem == lags.n_elem && arma::all(last.lags == lags);
    if(!same) {
        last.phi = phi;
        last.lags = lags;
        last.tabulated = false;
        last.cancelled = false;
        arma::vec thetaNew, phiNew;
        for(arma::uword n : candidates) {
            while(small(theta, n) && small(phi, n)) {
                const arma::vec& c = cyclotomicPolynomial(n);
                if(!polynomialDivide(theta, c, thetaNew, tolerance) || !polynomialDivide(phi, c, phiNew, tolerance)) {
                    break;
                }
                theta = thetaNew;
                phi = phiNew;
            }
        }
        return;
    }
    if(!last.tabulated) {
        arma::vec rest = phi;
        last.factors.clear();
        for(arma::uword n : candidates) {
            const arma::uword times = removeFactor(rest, n, arma::uword(-1));
            if(times > 0) {
                last.factors.emplace_back(n, times);
            }
        }
        last.tabulated = true;
    }
    std::vector<arma::uword> removed;
    for(const auto& factor : last.factors) {
        removed.push_back(removeFactor(theta, factor.first, factor.second));
    }
    if(!last.cancelled || removed != last.removed) {
        last.phiCancelled = phi;
        for(arma::uword i = 0; i < removed.size(); i++) {
            for(arma::uword times = 0; times < removed[i]; times++) {
                polynomialDivide(last.phiCancelled, cyclotomicPolynomial(last.factors[i].first), quotient, tolerance);
                last.phiCancelled = quotient;
            }
        }
        last.removed = removed;
        last.cancelled = true;
    }
    phi = last.phiCancelled;
}

// The largest |lambda| = 1 / min|B| over the roots B of the polynomial c, by bisection on r
// with the step-down test of arimaReflection() applied to c(B / r), which passes exactly when
// all roots of c have |lambda| < r. With belowOne false, a polynomial with all |lambda| < 1
// returns the largest reflection coefficient of its step-down instead, also below one, which
// spares the bisection when only the admissibility is needed.
inline double largestRootModulus(arma::vec c, double tolerance = 1e-6, bool belowOne = true) {
    const double scale = arma::max(arma::abs(c));
    arma::uword degree = c.n_elem - 1;
    while(degree > 0 && std::abs(c(degree)) <= 1e-12 * scale) {
        degree--;
    }
    if(degree == 0) {
        return 0.0;
    }
    c = c.head(degree + 1) / c(0);
    arma::vec powers(degree);
    auto below = [&](double r) {
        double power = 1.0;
        for(arma::uword i = 0; i < degree; i++) {
            power /= r;
            powers(i) = c(i + 1) * power;
        }
        return arimaReflection(powers) < 1;
    };
    double lower = 0.0, upper = 1.0;
    if(!belowOne) {
        const double reflection = arimaReflection(c.tail(degree));
        if(reflection < 1) {
            return reflection;
        }
    }
    if(!below(upper)) {
        do {
            lower = upper;
            upper *= 2.0;
        } while(!below(upper));
    }
    while(upper - lower > tolerance) {
        const double middle = (lower + upper) / 2.0;
        (below(middle) ? upper : lower) = middle;
    }
    return upper;
}

struct SmoothEigensResult {
    double modulus;
    arma::vec ariPolynomial;
    arma::vec maPolynomial;
};

// The admissibility of the model: the modulus is below one exactly when the components
// without adaptive regressors forget their initial states and, with adaptive regressors, the
// condition on average holds for them (the regressors vary over time, so the discount matrix
// of their block is averaged over the in-sample observations). Below one, it is the largest
// reflection coefficient of the cancelled MA polynomial, from one pass of the step-down; at
// or above one, the largest modulus of its roots (or of the eigenvalues of the averaged
// block), which grows as the parameters move away from the admissible region.
inline SmoothEigensResult smoothEigensCpp(const arma::mat& persistence,
                                          const arma::mat& transition,
                                          const arma::mat& measurement,
                                          const arma::ivec& lagsModelAll,
                                          bool& xregModel,
                                          int& obsInSample,
                                          bool& hasDelta,
                                          int xregNumber = 0,
                                          bool constantRequired = false) {
    const bool adaptive = xregModel && hasDelta;
    const int nComponents = int(lagsModelAll.n_elem) - (adaptive ? int(constantRequired) + xregNumber : 0);
    SmoothEigensResult result;
    result.modulus = 0.0;
    if(nComponents > 0) {
        const arma::uvec idx = arma::regspace<arma::uvec>(0, nComponents - 1);
        const arma::ivec lags = lagsModelAll.elem(idx);
        const arma::mat F = transition.submat(idx, idx);
        const arma::rowvec measurementRow = measurement.row(obsInSample - 1);
        result.maPolynomial = lagPolynomial(F - persistence.rows(idx) * measurementRow.cols(idx), lags);
        result.ariPolynomial = lagPolynomial(F, lags);
        cancelUnitRoots(result.maPolynomial, result.ariPolynomial, lags);
        result.modulus = largestRootModulus(result.maPolynomial, 1e-6, false);
    }
    if(adaptive && xregNumber > 0) {
        const arma::uvec xregIdx = arma::regspace<arma::uvec>(nComponents, nComponents + xregNumber - 1);
        const arma::mat measurementXreg = measurement.submat(arma::regspace<arma::uvec>(0, obsInSample - 1), xregIdx);
        const arma::mat discountXreg = transition.submat(xregIdx, xregIdx) -
            arma::diagmat(persistence.rows(xregIdx)) * measurementInverterCpp(measurementXreg).t() *
            measurementXreg / obsInSample;
        result.modulus = std::max(result.modulus, arma::max(arma::abs(arma::eig_gen(discountXreg))));
    }
    return result;
}
