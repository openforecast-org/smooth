// Long-run mean of the occurrence probability produced by smooth's occurrence filters
// (occurrenceError() from src/headers/ssOccurrence.h, the code smooth runs) and by
// Croston's probability 1/q_hat, on o_t ~ Bernoulli(p) with constant p.
// Usage: filter <model: i|o|c> <error: M|A> <p> <alpha> <T> <burnin> <seed>
// Prints: mean of p_hat after burn-in, number of non-finite states.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "../../../src/headers/ssOccurrence.h"

int main(int argc, char **argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: filter model error p alpha T burnin seed\n");
        return 1;
    }
    char model = argv[1][0], error = argv[2][0];
    double p = std::atof(argv[3]), alpha = std::atof(argv[4]);
    long T = std::atol(argv[5]), burnin = std::atol(argv[6]);
    std::mt19937_64 rng(std::strtoull(argv[7], nullptr, 10));
    std::bernoulli_distribution occurrence(p);

    // State starts at the true value: mu_b = (1-p)/p for 'i', mu_a = p/(1-p) for 'o',
    // on the log scale for the additive error; Croston's interval starts at 1/p.
    double odds = (model == 'i') ? (1 - p) / p : p / (1 - p);
    double state = (model == 'c') ? 1 / p : (error == 'A' ? std::log(odds) : odds);
    double sum = 0, interval = 0;
    long nonfinite = 0;

    for (long t = 0; t < T; t++) {
        double level = (error == 'A') ? std::exp(state) : state;
        double pHat = (model == 'c') ? 1 / state
                    : (model == 'i') ? 1 / (1 + level) : level / (level + 1);
        bool o = occurrence(rng);
        if (t >= burnin) {
            sum += pHat;
        }

        if (model == 'c') {
            interval += 1;
            if (o) {
                state += alpha * (interval - state);
                interval = 0;
            }
        } else {
            double e = occurrenceError(o ? 1.0 : 0.0, state, 0, error, 'M', model)[0];
            state = (error == 'M') ? state * (1 + alpha * e) : state + alpha * e;
        }
        if (!std::isfinite(state)) {
            nonfinite++;
            state = (error == 'A') ? std::log(odds) : odds;
        }
    }
    std::printf("%.10g %ld\n", sum / (T - burnin), nonfinite);
    return 0;
}
