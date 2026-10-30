// Long-run mean of the occurrence probability produced by an occurrence filter on
// o_t ~ Bernoulli(p) with constant p, with its Monte Carlo standard error (batch means).
// Filters: iETS_I ('i') and iETS_O ('o') with the proxy smooth uses, occurrenceError()
// from src/headers/ssOccurrence.h ('old'), the score proxy ('score', derivations.md §4) or
// the exponential score proxy, 1 + e = exp(score) ('expscore', derivations.md §5);
// Croston's 1/q_hat ('c').
// Error types: 'M' conventional multiplicative ETS, mu_t = mu_t-1 (1 + alpha e_t);
// 'L' ADAM multiplicative ETS (ets="adam"), mu_t = mu_t-1 (1 + e_t)^alpha, as adamGvalue()
// in src/headers/adamGeneral.h (case MNN); 'A' additive ETS on log(mu).
// Usage: filter <model: i|o|c> <error: M|L|A> <proxy: old|score|expscore> <p> <alpha> <T>
//               <burnin> <seed> <batches>
// Prints: mean of p_hat after burn-in, its standard error, number of non-finite states.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../../../src/headers/ssOccurrence.h"

// Score proxy: d log-likelihood / d log(state) = o - p_hat for mu_a ('o'), p_hat - o for mu_b ('i')
double scoreError(double o, double pHat, char model) {
    return (model == 'o') ? o - pHat : pHat - o;
}

int main(int argc, char **argv) {
    if (argc != 10) {
        std::fprintf(stderr, "usage: filter model error proxy p alpha T burnin seed batches\n");
        return 1;
    }
    char model = argv[1][0], error = argv[2][0];
    bool score = std::strcmp(argv[3], "score") == 0;
    bool expscore = std::strcmp(argv[3], "expscore") == 0;
    double p = std::atof(argv[4]), alpha = std::atof(argv[5]);
    long T = std::atol(argv[6]), burnin = std::atol(argv[7]), batches = std::atol(argv[9]);
    std::mt19937_64 rng(std::strtoull(argv[8], nullptr, 10));
    std::bernoulli_distribution occurrence(p);

    // State starts at the true value: mu_b = (1-p)/p for 'i', mu_a = p/(1-p) for 'o',
    // on the log scale for the additive error; Croston's interval starts at 1/p.
    double odds = (model == 'i') ? (1 - p) / p : p / (1 - p);
    double start = (model == 'c') ? 1 / p : (error == 'A' ? std::log(odds) : odds);
    double state = start, interval = 0;
    long nonfinite = 0, batchSize = (T - burnin) / batches;
    std::vector<double> batchSum(batches, 0.0);

    for (long t = 0; t < burnin + batchSize * batches; t++) {
        double level = (error == 'A') ? std::exp(state) : state;
        double pHat = (model == 'c') ? 1 / state
                    : (model == 'i') ? 1 / (1 + level) : level / (level + 1);
        bool o = occurrence(rng);
        if (t >= burnin) {
            batchSum[(t - burnin) / batchSize] += pHat;
        }

        if (model == 'c') {
            interval += 1;
            if (o) {
                state += alpha * (interval - state);
                interval = 0;
            }
        } else {
            double y = o ? 1.0 : 0.0;
            // smooth's old proxy: multiplicative form for 'M' and 'L', log form for 'A'
            char oldType = (error == 'A') ? 'A' : 'M';
            double e = score ? scoreError(y, pHat, model)
                     : expscore ? std::exp(scoreError(y, pHat, model)) - 1
                     : occurrenceError(y, state, 0, oldType, 'M', model)[0];
            state = (error == 'M') ? state * (1 + alpha * e)
                  : (error == 'L') ? state * std::pow(1 + e, alpha) : state + alpha * e;
        }
        if (!std::isfinite(state)) {
            nonfinite++;
            state = start;
        }
    }

    double mean = 0, var = 0;
    for (double s : batchSum) {
        mean += s / batchSize / batches;
    }
    for (double s : batchSum) {
        var += std::pow(s / batchSize - mean, 2) / (batches - 1);
    }
    std::printf("%.12g %.6g %ld\n", mean, std::sqrt(var / batches), nonfinite);
    return 0;
}
