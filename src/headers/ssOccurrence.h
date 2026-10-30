#pragma once

#include <cmath>
#include <vector>

/* Function returns a and b errors as a vector, depending on the types of E, O and the others.
 * In case of O=="p", the error is provided in the first element of the vector.
 * In case of O=="i", the error is moved in the first element.
 * With score=true (proxyOccurrence="score"), 'o', 'i' and 'g' use the score of the Bernoulli
 * log-likelihood with respect to the log of the state, s = o - p for a and p - o for b: the error
 * is s for an additive error and for conventional multiplicative ETS, and exp(s) - 1 for ADAM
 * multiplicative ETS (adamA / adamB), whose update (1 + e)^alpha then moves log(state) by alpha*s. */
inline std::vector<double> occurrenceError(
        double const &yAct, double aFit, double bFit,
        char const &EA, char const &EB, char const &O,
        bool const &score = false, bool const &adamA = false, bool const &adamB = false) {
// aFit is the fitted value of a, bFit is the same for b. O is the type of occurrence.
// In cases of O!="g", aFit is the variable under consideration; bFit is unused.

    double pfit = 0;
    double error = 0;
    double kappa = 1E-10;
    std::vector<double> output(2);

    // A missing observation (NaN): no error, so the states move with the transition only
    if(std::isnan(yAct)){
        output[0] = 0;
        output[1] = 0;
        return output;
    }

    switch(O){
        // The direct probability model
        case 'd':
            pfit = std::max(std::min(aFit, 1.0), 0.0);
            switch(EA){
                case 'M':
                    output[0] = (yAct * (1 - 2 * kappa) + kappa - pfit) / pfit;
                    break;
                case 'A':
                    output[0] = yAct - pfit;
                    break;
            }
            break;
        // The odds-ratio probability model
        case 'o':
            switch(EA){
                case 'A':
                    aFit = exp(aFit);
                    break;
            }
            pfit = aFit / (aFit + 1);
            if(score){
                output[0] = (EA=='M' && adamA) ? exp(yAct - pfit) - 1 : yAct - pfit;
                break;
            }
            error = (1 + yAct - pfit) / 2;
            output[0] = error / (1 - error);
            switch(EA){
                case 'M':
                    output[0] = output[0] - 1;
                    break;
                case 'A':
                    output[0] = log(output[0]);
                    break;
            }
            break;
        // The inverse-odds-ratio probability model
        case 'i':
            switch(EA){
                case 'A':
                    aFit = exp(aFit);
                    break;
            }
            pfit = 1 / (1 + aFit);
            if(score){
                output[0] = (EA=='M' && adamA) ? exp(pfit - yAct) - 1 : pfit - yAct;
                break;
            }
            error = (1 + yAct - pfit) / 2;
            output[0] = (1 - error) / error;
            switch(EA){
                case 'M':
                    output[0] = output[0] - 1;
                    break;
                case 'A':
                    output[0] = log(output[0]);
                    break;
            }
            break;
        // The general model
        case 'g':
            switch(EA){
                case 'A':
                    aFit = exp(aFit);
                    break;
            }
            switch(EB){
                case 'A':
                    bFit = exp(bFit);
                    break;
            }
            pfit = aFit / (aFit + bFit);
            if(score){
                output[0] = (EA=='M' && adamA) ? exp(yAct - pfit) - 1 : yAct - pfit;
                output[1] = (EB=='M' && adamB) ? exp(pfit - yAct) - 1 : pfit - yAct;
                break;
            }
            error = (1 + yAct - pfit) / 2;
            output[0] = error / (1 - error);
            output[1] = (1 - error) / error;
            switch(EA){
                case 'M':
                    output[0] = output[0] - 1;
                    break;
                case 'A':
                    output[0] = log(output[0]);
                    break;
            }
            switch(EB){
                case 'M':
                    output[1] = output[1] - 1;
                    break;
                case 'A':
                    output[1] = log(output[1]);
                    break;
            }
            break;
    }

    return output;
}
