// Helper: Invert measurement matrix (1/x, with inf->0)
arma::mat measurementInverterCpp(const arma::mat& measurement) {
    arma::mat result = 1.0 / measurement;
    result.replace(arma::datum::inf, 0.0);
    result.replace(-arma::datum::inf, 0.0);
    return result;
}

// Function to calculate eigenvalues
arma::vec smoothEigensCpp(const arma::mat& persistence,
                          const arma::mat& transition,
                          const arma::mat& measurement,
                          const arma::ivec& lagsModelAll,
                          bool& xregModel,
                          int& obsInSample,
                          bool& hasDelta,
                          int xregNumber = 0,
                          bool constantRequired = false) {

    int nComponents = lagsModelAll.n_elem;

    if (xregModel && hasDelta) {
        // Handle xreg with adaptive regressors
        // Non-xreg components use loop approach, xreg components use averaging

        int effectiveComponents = nComponents;
        // Drop constant if present (last element)
        if (constantRequired) {
            effectiveComponents -= 1;
        }

        arma::vec eigenValues(effectiveComponents, arma::fill::zeros);
        int nonXregEnd = effectiveComponents - xregNumber;

        // Part 1: Non-xreg components (loop approach)
        if (nonXregEnd > 0) {
            arma::ivec lagsNonXreg = lagsModelAll.head(nonXregEnd);
            arma::ivec lagsUniqueNonXreg = arma::unique(lagsNonXreg);
            arma::uvec nonXregIdx = arma::regspace<arma::uvec>(0, nonXregEnd - 1);

            for (arma::uword i = 0; i < lagsUniqueNonXreg.n_elem; i++) {
                // Find indices where lagsNonXreg == lagsUniqueNonXreg[i]
                arma::uvec idx = arma::find(lagsNonXreg == lagsUniqueNonXreg[i]);

                // Extract submatrices for non-xreg portion
                arma::mat transSub = transition.submat(idx, idx);
                arma::mat persSub = persistence.rows(idx);
                arma::rowvec measRow = measurement.row(obsInSample - 1);
                arma::mat measSub = measRow.cols(idx);

                // Compute: transition_sub - persistence_sub * measurement_sub
                arma::mat matToDecomp = transSub - persSub * measSub;

                // Get eigenvalues
                arma::cx_vec eigVals = arma::eig_gen(matToDecomp);
                arma::vec absEigVals = arma::abs(eigVals);

                // Assign to result
                for (arma::uword j = 0; j < idx.n_elem; j++) {
                    eigenValues(idx(j)) = absEigVals(j);
                }
            }
        }

        // Part 2: Xreg components (averaging approach)
        if (xregNumber > 0) {
            arma::uvec xregIdx = arma::regspace<arma::uvec>(nonXregEnd, effectiveComponents - 1);
            arma::mat transSub = transition.submat(xregIdx, xregIdx);
            arma::mat persSub = persistence.rows(xregIdx);
            arma::mat measSub = measurement.submat(
                arma::regspace<arma::uvec>(0, obsInSample - 1), xregIdx);
            arma::mat measInv = measurementInverterCpp(measSub);
            arma::mat matToDecomp = transSub -
                arma::diagmat(persSub) * measInv.t() * measSub / obsInSample;

            arma::cx_vec eigVals = arma::eig_gen(matToDecomp);
            arma::vec absEigVals = arma::abs(eigVals);

            for (int j = 0; j < xregNumber; j++) {
                eigenValues(nonXregEnd + j) = absEigVals(j);
            }
        }

        return eigenValues;
    }
    else {
        // Normal case: loop through unique lags
        arma::ivec lagsUnique = arma::unique(lagsModelAll);
        int lagsUniqueLength = lagsUnique.n_elem;
        arma::vec eigenValues(nComponents, arma::fill::zeros);

        for (int i = 0; i < lagsUniqueLength; i++) {
            // Find indices where lagsModelAll == lagsUnique[i]
            arma::uvec idx = arma::find(lagsModelAll == lagsUnique[i]);

            // Extract submatrices
            arma::mat transSub = transition.submat(idx, idx);
            arma::mat persSub = persistence.rows(idx);
            arma::rowvec measRow = measurement.row(obsInSample - 1);
            arma::mat measSub = measRow.cols(idx);

            // Compute: transition_sub - persistence_sub * measurement_sub
            arma::mat matToDecomp = transSub - persSub * measSub;

            // Get eigenvalues
            arma::cx_vec eigVals = arma::eig_gen(matToDecomp);
            arma::vec absEigVals = arma::abs(eigVals);

            // Assign to result
            for (arma::uword j = 0; j < idx.n_elem; j++) {
                eigenValues(idx(j)) = absEigVals(j);
            }
        }

        return eigenValues;
    }
}

// The moduli of the eigenvalues of a real square matrix, written out without LAPACK so
// that the R and Python builds round identically whatever library each links (the
// admissible bounds of tbats() sit on the boundary, where a last-bit difference between
// two LAPACK builds decided which parameters were admissible). EISPACK's balancing,
// orthogonal reduction to Hessenberg form and shifted QR (balanc, orthes and hqr), with
// 1-based indices as there. A matrix whose QR
// does not converge gets infinite moduli, so that it is never admissible.
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
                    if(its == 30) {
                        moduli.fill(arma::datum::inf);
                        return moduli;
                    }
                    if(its == 10 || its == 20) {
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
