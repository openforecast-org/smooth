"""Completes the draft claimset that `claimset lab export` writes for this study: the
decisions and wording the export leaves to the authors (claim roles, statements of negative
results, scopes, strengths, novelty and relations, references, formal evidence, artefact
locations and licences, contributors). Usage:
    claimset_finalise.py <exported draft.json> <aggregate_measures.csv> <output.json>
Everything the record states (hypotheses, outcomes, planning, tests, lab runs, results) is
kept as exported.
"""

import csv
import json
import sys

REPO = "https://github.com/openforecast-org/smooth/blob"
CODE = "c8cad1d053d6fe5ec830faae867235cf67a58787"  # code branch iets-occurrence-bias
M5_URI = ("https://github.com/Nixtla/m5-forecasts/blob/72b8e7fd3b565b3c538adcb1d1a05117d8562d7e/"
          "datasets/m5.zip")
LEVELS = ["0.8", "0.9", "0.95", "0.99"]

REFERENCES = [
    ("B1", "10.1016/j.ijpe.2023.109013", "doi"),
    ("B2", "10.2307/3007885", "doi"),
    ("B3", "10.1002/jae.1279", "doi"),
    ("B4", "10.1017/CBO9781139540933", "doi"),
    ("B6", "10.1007/b97441", "doi"),
    ("B7", "10.1016/S0925-5273(00)00143-2", "doi"),
    ("B8", "10.1016/j.ijforecast.2004.10.001", "doi"),
    ("B9", "10.1201/9781003452652", "doi"),
    ("B10", "10.1016/j.ijforecast.2021.11.013", "doi"),
    ("B11", "10.1016/j.ejor.2011.05.018", "doi"),
    ("B12", "10.1016/j.ijforecast.2011.03.009", "doi"),
    ("B13", "10.1016/j.ijforecast.2004.10.003", "doi"),
    ("B14", "10.1016/j.ijpe.2014.06.007", "doi"),
]
DEF_SOURCE = {"D1": "B1", "D2": "B1", "D3": "B1", "D5": "B2", "D8": "B3", "D10": "B4",
              "D11": "B6", "D12": "B7", "D16": "B9", "D19": "B10", "D20": "B1", "D21": "B8",
              "D22": "B11", "D23": "B12", "D24": "B12", "D25": "B10", "D26": "B14",
              "D27": "B14", "D28": "B10", "D31": "B13", "D33": "B1", "D36": "B3"}

PRINCIPAL = ["C8", "C10", "C27", "C15"]
DEPENDS = {"C8": ["C6", "C1", "C2", "C3"], "C6": ["C5"], "C10": ["C7"], "C15": ["C13"],
           "C12": ["C4"], "C27": ["C8", "C12"]}
ASSUMPTIONS = {"C8": ["C26"], "C10": ["C26"], "C15": ["C26"], "C12": ["C26"]}
SECONDARY = ["C9", "C11", "C14", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
             "C24", "C25"]
LOCAL = {"C14": ("supports", "C13"), "C16": ("qualifies", "C11"), "C17": ("qualifies", "C11"),
         "C18": ("refines", "C9")}
FORMAL = {"C5": "E5", "C6": "E6", "C7": "E7", "C13": "E13"}
SIMULATION = ["E1", "E2", "E3", "E4", "E8", "E9", "E10", "E11", "E12", "E14", "E15", "E16",
              "E17", "E18"]
M5 = ["E19", "E20", "E21", "E22", "E23", "E24", "E25"]

SB23_NOTE = ("Svetunkov and Boylan (2023) define the iETS occurrence models and the ratio "
             "proxies of their unobservable occurrence error (eqs. 24-25), the old error proxy "
             "of this paper.")
SB23_M5_NOTE = ("Svetunkov and Boylan (2023, section 5) compare iETS models with Croston's "
                "method, TSB, NegBin and HSP on the M5 data, with the upper quantiles rounded up "
                "(eq. 35).")
GAS_NOTE = ("Creal, Koopman and Lucas (2013) define score-driven filters, which update a "
            "time-varying parameter by the scaled score of the observation density.")
ADAM_NOTE = ("Svetunkov (2023) defines the ADAM form of multiplicative ETS, which updates a "
             "level as mu_t = mu_{t-1} (1 + e_t)^alpha.")
SB01_NOTE = ("Syntetos and Boylan (2001) show that Croston's estimator is biased upwards by "
             "the inversion of the smoothed inter-demand interval, by an amount that grows "
             "with the smoothing parameter.")


def rel(kind, ref, note, self_cite=False, locator=None):
    out = {"type": kind, "reference": ref, "note": note, "self": self_cite}
    if locator:
        out["locator"] = locator
    return out


NOVELTY = {
    **{c: ("no_prior_found", [rel("extends", "B1", SB23_NOTE, True, "eqs. 24-25")])
       for c in ["C1", "C2", "C3", "C5", "C6", "C8", "C9", "C11", "C16", "C17", "C18"]},
    **{c: ("no_prior_found", [rel("extends", "B3", GAS_NOTE)]) for c in ["C7", "C10"]},
    **{c: ("no_prior_found", [rel("extends", "B9", ADAM_NOTE, True)])
       for c in ["C13", "C14", "C15"]},
    **{c: ("replication", [rel("replicates", "B7", SB01_NOTE)]) for c in ["C4", "C12"]},
    "C27": ("no_prior_found", [rel("extends", "B7", SB01_NOTE)]),
    **{c: ("no_prior_found", [rel("extends", "B1", SB23_M5_NOTE, True, "section 5")])
       for c in ["C19", "C20", "C21", "C22", "C23", "C24", "C25", "C28", "C29", "C30", "C31"]},
}

SIM_SCOPE = {"data_type": "simulated Bernoulli occurrence series with a constant probability",
             "excludes": ["time-varying occurrence probabilities",
                          "smoothing parameters estimated from data"]}
M5_SCOPE = {"data_type": "daily unit sales of retail products",
            "setting": "Walmart stores in the M5 competition data",
            "geography": "United States (California, Texas, Wisconsin)",
            "period": "1,969 days from 29 January 2011",
            "conditions": ["last 14 days held out; horizons 1 to 14",
                           "iETS(M,N,N) with Gamma demand sizes, parameters estimated by "
                           "likelihood",
                           "series with at least 5 non-zero in-sample observations"]}


def sim_scope(*conditions):
    return {**SIM_SCOPE, "conditions": list(conditions)}


SCOPE = {
    "C1": sim_scope("multiplicative occurrence error, conventional ETS update",
                    "limit as alpha tends to 0; simulated at alpha = 0.002"),
    "C2": sim_scope("multiplicative occurrence error, conventional ETS update",
                    "limit as alpha tends to 0; simulated at alpha = 0.002"),
    "C3": sim_scope("additive occurrence error (state on the log scale)",
                    "limit as alpha tends to 0; simulated at alpha = 0.002"),
    "C4": sim_scope("alpha = 0.002"),
    "C5": {"conditions": ["o ~ Bernoulli(p), p in (0,1)", "one-step-ahead probability equal "
                          "to the true probability"]},
    "C6": {"conditions": ["o ~ Bernoulli(p), p in (0,1)", "expected update of the state "
                          "given the one-step-ahead probability"]},
    "C7": {"conditions": ["o ~ Bernoulli(p), p in (0,1)", "alpha in (0,1]"]},
    "C8": sim_scope("both occurrence error types, conventional ETS update",
                    "limit as alpha tends to 0, extrapolated from alpha in {0.004, 0.002, "
                    "0.001}"),
    "C9": sim_scope("200 random (p, alpha) pairs, p in [0.02, 0.98], log alpha uniform on "
                    "[log 0.001, 0]",
                    "the wrong-direction cells have multiplicative error and alpha between "
                    "0.88 and 0.96"),
    "C10": sim_scope("both occurrence error types, conventional ETS update",
                     "limit as alpha tends to 0, extrapolated from alpha in {0.004, 0.002, "
                     "0.001}"),
    "C11": sim_scope("200 random (p, alpha) pairs, p in [0.02, 0.98], log alpha uniform on "
                     "[log 0.001, 0]"),
    "C12": sim_scope("all alpha in (0,1] for the direction; limit as alpha tends to 0"),
    "C13": {"conditions": ["o ~ Bernoulli(p), p in (0,1)", "ADAM multiplicative ETS update "
                           "(ets='adam')"]},
    "C14": sim_scope("ADAM multiplicative ETS update (ets='adam')",
                     "limit as alpha tends to 0, extrapolated from alpha in {0.004, 0.002, "
                     "0.001}"),
    "C15": sim_scope("ADAM multiplicative ETS update (ets='adam')",
                     "limit as alpha tends to 0, extrapolated from alpha in {0.004, 0.002, "
                     "0.001}"),
    "C16": sim_scope("additive occurrence error",
                     "300 random (p, alpha) pairs, alpha uniform on [0.001, 1]"),
    "C17": sim_scope("conventional multiplicative occurrence error",
                     "300 random (p, alpha) pairs, alpha uniform on [0.001, 0.5]"),
    "C18": sim_scope("additive error: alpha in (0,1]; multiplicative error: alpha in (0, 0.8]"),
    "C27": sim_scope("limit as alpha tends to 0"),
}
M5_FOCUS = {
    "C19": "measure: sum of the 14 point forecasts, score proxy minus old proxy",
    "C20": "measure: RMSSE, score proxy minus old proxy",
    "C21": "measure: scaled pinball of the upper quantiles, score proxy minus old proxy",
    "C22": "the first MCB evaluation, whose ranks were attached to the wrong methods",
    "C23": "MCB at 99% over all 23 methods for RMSSE and the 20 methods with quantiles for "
           "the pinball",
    "C24": "measure: scaled pinball, rounded-up quantile minus raw quantile",
    "C25": "measure: coverage deviation, rounded-up quantile minus raw quantile",
    "C28": "iETS_G under conventional ETS",
    "C29": "iETS_I under conventional ETS, level 0.99",
    "C30": "iETS models other than iETS_O and iETS_I under conventional ETS",
    "C31": "levels 0.80, 0.90 and 0.95",
}
for c, focus in M5_FOCUS.items():
    SCOPE[c] = {**M5_SCOPE, "conditions": M5_SCOPE["conditions"] + [focus]}

STATEMENTS = {
    "C8": "On [[D7:Bernoulli occurrence with constant probability]] p, for every p in (0,1), as "
          "alpha tends to 0, the stationary mean probability of [[D1:iETS_I]] and [[D2:iETS_O]] "
          "with the [[D3:old error proxy]], under either occurrence error type, converges to the "
          "[[D9:zero-drift point]] of the filter, not to p: the bias of the old filters does not "
          "vanish with a small smoothing parameter.",
    "C3": "For [[D1:iETS_I]] and [[D2:iETS_O]] with additive occurrence error, applied to "
          "[[D7:Bernoulli occurrence with constant probability]] p, the [[D6:stationary mean "
          "probability]] as alpha tends to 0 is the same for both models and equals the root x "
          "of $p\\log\\frac{x}{2-x} + (1-p)\\log\\frac{1+x}{1-x} = 0$, a value strictly "
          "between p and one half for p other than one half and equal to one half at p = one "
          "half.",
    "C5": "For every p in (0,1), with o ~ Bernoulli(p) and the one-step-ahead probability equal "
          "to p, the [[D3:old error proxy]] has a non-zero expected value: $E[1+e_a] = 2 - p + "
          "(1-p)^2/(1+p) > 1$ and $E[1+e_b] = 1 + p + p^2/(2-p) > 1$ for the multiplicative "
          "error, and $E[\\log(1+e_a)] = -E[\\log(1+e_b)]$ is positive below one half, zero "
          "at one half and negative above it for the additive error.",
    "C6": "For every p in (0,1), the [[D4:occurrence filters]] with the [[D3:old error "
          "proxy]] have a unique stable [[D9:zero-drift point]]: $1 - \\sqrt{1-p} < p$ for "
          "[[D1:iETS_I]] and $\\sqrt{p} > p$ for [[D2:iETS_O]] with multiplicative error, and "
          "for both models with additive error the root of $p\\log\\frac{x}{2-x} + "
          "(1-p)\\log\\frac{1+x}{1-x} = 0$, strictly between p and one half for p other than one "
          "half and equal to one half at p = one half.",
    "C7": "The [[D8:score proxy]], $o - \\hat p$ for [[D2:iETS_O]] and $\\hat p - o$ for "
          "[[D1:iETS_I]], is the score of the Bernoulli occurrence log-likelihood with respect "
          "to the log of the state, has the true probability p as its unique stable "
          "[[D9:zero-drift point]] for every p in (0,1), and keeps a multiplicatively updated "
          "state positive for every alpha in (0,1].",
    "C9": "On [[D7:Bernoulli occurrence with constant probability]], the [[D13:bias of the "
          "occurrence probability]] of the [[D3:old error proxy]] filters does not keep a "
          "fixed direction at every alpha in (0,1]: with multiplicative error the bias has "
          "the opposite sign in 1 simulated cell for [[D1:iETS_I]] and 2 for [[D2:iETS_O]], "
          "and with additive error in 0.",
    "C11": "On [[D7:Bernoulli occurrence with constant probability]], the [[D8:score proxy]] "
           "is not less biased than the [[D3:old error proxy]] at every alpha: with "
           "multiplicative error its absolute [[D13:bias of the occurrence probability]] is "
           "larger in 10 simulated cells for [[D1:iETS_I]] and 12 for [[D2:iETS_O]], and with "
           "additive error in 0.",
    "C13": "Under the [[D16:ADAM multiplicative ETS update]] the occurrence filter is the "
           "additive filter with error $\\log(1+e)$, so that its [[D9:zero-drift point]] is "
           "that of the additive old filter with the [[D3:old error proxy]], differs from p "
           "for every p in (0,1) with the [[D18:multiplicative score proxy]] (below p for "
           "[[D2:iETS_O]], above p for [[D1:iETS_I]]), and equals p with the "
           "[[D17:exponential score proxy]].",
    "C14": "On [[D7:Bernoulli occurrence with constant probability]], under the "
           "[[D16:ADAM multiplicative ETS update]] the [[D18:multiplicative score proxy]] does "
           "not give an alpha-to-0 limit of the stationary mean probability equal to p: its "
           "largest relative deviation from p is 1.69 for [[D1:iETS_I]] and 0.30 for "
           "[[D2:iETS_O]].",
    "C17": "On [[D7:Bernoulli occurrence with constant probability]], with conventional "
           "multiplicative occurrence error and alpha up to one half, the [[D18:multiplicative "
           "score proxy]] is not always less biased than the [[D3:old error proxy]]: its "
           "absolute [[D13:bias of the occurrence probability]] is larger in 2 simulated cells "
           "for [[D1:iETS_I]] and 0 for [[D2:iETS_O]].",
    "C19": "On the [[D19:M5 subset]], replacing the [[D3:old error proxy]] with the "
           "[[D8:score proxy]] in the [[D20:iETS(M,N,N) flavours]] raises the "
           "[[D32:sum of point forecasts]] of iETS_I under conventional ETS and lowers it for "
           "iETS_O under conventional ETS and for iETS_I and iETS_O under ADAM ETS, the "
           "directions predicted by the zero-drift points of the old proxy.",
    "C20": "On the [[D19:M5 subset]], each of the eight [[D20:iETS(M,N,N) flavours]] with a "
           "proxy, iETS_O, iETS_I, iETS_G and iETS_A under conventional and ADAM ETS, has a "
           "lower [[D25:RMSSE]] with the [[D8:score proxy]] than with the [[D3:old error "
           "proxy]].",
    "C21": "On the [[D19:M5 subset]], the [[D8:score proxy]] does not give a lower "
           "[[D28:scaled pinball]] than the [[D3:old error proxy]] for every one of the eight "
           "[[D20:iETS(M,N,N) flavours]] with a proxy at every level from 0.80 to 0.99: 11 "
           "comparisons are statistically significantly lower and 1, iETS_I under "
           "conventional ETS at 0.99, is statistically significantly higher.",
    "C22": "The first [[D31:MCB]] evaluation of the M5 forecasts attached each method's mean "
           "rank to another method, so its outcome of 0 comparisons in favour of the best "
           "[[D8:score proxy]] model and 7 against carries no information on how the methods "
           "rank on the [[D19:M5 subset]].",
    "C23": "On the [[D19:M5 subset]], the best-ranked [[D20:iETS(M,N,N) flavour]] with the "
           "[[D8:score proxy]] does not rank statistically significantly better than every "
           "benchmark in the [[D31:MCB]] test: it ranks better on [[D25:RMSSE]] than "
           "[[D5:Croston's method]], [[D21:SBA]], [[D22:TSB]] and [[D23:HSP]] and on the "
           "[[D28:scaled pinball]] at 0.99 than [[D24:NegBin]] (5 comparisons), but worse on "
           "the scaled pinball than NegBin and HSP at 0.90 and 0.95, than NegBin at 0.80 and "
           "than HSP at 0.99 (6 comparisons).",
    "C24": "On the [[D19:M5 subset]], rounding the upper quantiles of the [[D20:iETS(M,N,N) "
           "flavours]] up ([[D33:rounded-up quantile]]) does not lower the [[D28:scaled "
           "pinball]]: it raises it statistically significantly in 53 model-level comparisons "
           "and lowers it in 0.",
    "C25": "On the [[D19:M5 subset]], rounding the upper quantiles of the [[D20:iETS(M,N,N) "
           "flavours]] up ([[D33:rounded-up quantile]]) does not bring the mean coverage "
           "closer to the level: the [[D35:coverage deviation]] grows statistically "
           "significantly in 34 model-level pairs and shrinks in 18.",
    "C26": "For a fixed p, the stationary mean of the one-step-ahead probability of an "
           "occurrence filter with a unique stable zero-drift point converges to that point "
           "as alpha tends to 0, as [[D11:constant-gain stochastic approximation]] gives under "
           "regularity conditions that are not verified for these filters.",
}
RESULTS = {"C9": ["E9.R801", "E9.R803", "E9.R802", "E9.R804"],
           "C11": ["E11.R1605", "E11.R1607", "E11.R1606", "E11.R1608"],
           "C14": ["E14.R181", "E14.R182"], "C17": ["E17.R1203", "E17.R1204"],
           "C21": ["E21.R490", "E21.R491"], "C22": ["E22.R502", "E22.R503"],
           "C23": ["E23.R502", "E23.R503"], "C24": ["E24.R610", "E24.R611"],
           "C25": ["E25.R466", "E25.R467"]}

STRENGTH = {
    "C1": ("high", "The simulated means at five new values of p lie within 0.5% of the derived "
                   "limit, and the limit follows from the zero-drift point derived in C6."),
    "C2": ("high", "The simulated means at five new values of p lie within 0.4% of the derived "
                   "limit, and the limit follows from the zero-drift point derived in C6."),
    "C3": ("high", "The simulated means at five new values of p lie within 0.4% of the derived "
                   "root for both models."),
    "C4": ("high", "Long simulations at five values of p agree with the known order-alpha "
                   "inversion bias of Croston's method."),
    "C8": ("high", "Thirty random values of p at three small alphas agree with the derived "
                   "zero-drift points to within 0.3% for both models and error types, with "
                   "Monte Carlo errors estimated by batch means; the step from zero-drift "
                   "point to limit rests on the unchecked regularity conditions of C26."),
    "C9": ("moderate", "Three cells out of 800 decide the outcome, each with the bias more than "
                       "four Monte Carlo standard errors on the wrong side, all at large alpha."),
    "C10": ("high", "Thirty random values of p at three small alphas agree with p to within "
                    "0.3% for both models and error types, as the derivation in C7 predicts."),
    "C11": ("moderate", "The comparisons rest on 800 simulated cells with a four-standard-"
                        "error rule; the cells against the score proxy all have "
                        "multiplicative error."),
    "C12": ("high", "Long simulations over random p and alpha agree with the known "
                    "inversion bias of Croston's method in direction and limit."),
    "C14": ("high", "The deviation from p is far beyond the 5% refutation bound and matches "
                    "the zero-drift points derived in C13."),
    "C15": ("high", "Thirty random values of p at three small alphas agree with p to within "
                    "0.5% for both models, as the derivation in C13 predicts."),
    "C16": ("moderate", "No cell of 300 random (p, alpha) pairs shows the additive score proxy "
                        "more biased than the old proxy beyond four standard errors."),
    "C17": ("moderate", "Two cells of 300 random (p, alpha) pairs, both for iETS_I, decide the "
                        "outcome under a four-standard-error rule."),
    "C18": ("moderate", "No wrong-direction cell appears in two runs over random (p, alpha) "
                        "pairs restricted to the stated ranges, after an exploratory run "
                        "suggested the restriction."),
    "C19": ("low", "Only two of the four predicted shifts reached the 1% level after Holm's "
                   "correction, although none was significant in the opposite direction."),
    "C20": ("low", "Six of the eight comparisons were significant in the predicted direction "
                   "and none against it, so the test did not meet its rule for every model."),
    "C21": ("moderate", "A pre-registered test on 3,000 series found one comparison "
                        "significant against the prediction at the 1% level after Holm's "
                        "correction."),
    "C22": ("very_low", "The run's mean ranks were attached to the wrong methods (L1), so its "
                        "outcome is not evidence about the methods."),
    "C23": ("moderate", "A pre-registered MCB test on 3,000 series of one retailer gave "
                        "significant results in both directions at the 1% level; the series "
                        "come from one retailer (L2)."),
    "C24": ("moderate", "A pre-registered test on 3,000 series found 53 significant "
                        "increases of the pinball and no decrease; the series come from one "
                        "retailer (L2)."),
    "C25": ("moderate", "A pre-registered bootstrap test on 3,000 series found significant "
                        "changes in both directions; the series come from one retailer (L2)."),
    "C27": ("high", "An exploratory conclusion drawn without a planned test of its own from two "
                    "confirmatory claims whose derivations long simulations confirmed."),
    "C28": ("low", "An untested explanation drawn from the M5 results and the derived "
                   "zero-drift points."),
    "C29": ("low", "An untested explanation drawn from the M5 results and the derived bias "
                   "direction."),
    "C30": ("low", "An untested explanation; the difference in reactivity is derived, its "
                   "effect on accuracy is not measured."),
    "C31": ("low", "An untested explanation; no iETS variant with count demand sizes was "
                   "evaluated."),
}

NEW_CLAIMS = [
    {"id": "C27",
     "statement": "The [[D13:bias of the occurrence probability]] of [[D1:iETS_I]] and "
                  "[[D2:iETS_O]] with the [[D3:old error proxy]] does not vanish as the "
                  "smoothing parameter tends to 0, unlike the [[D12:inversion bias]] of "
                  "[[D5:Croston's method]].",
     "about": {"subject": "D3", "relation": "has a bias that does not vanish as alpha tends to "
                                            "0, unlike that of", "object": "D13",
               "comparator": "D5", "conditions": ["limit as alpha tends to 0"]},
     "type": "descriptive", "role": "principal",
     "support": [{"kind": "argument", "warrant": "W1"}], "warrant": "W1",
     "planning": "exploratory", "provenance": "author_stated"},
    {"id": "C28",
     "statement": "On the [[D19:M5 subset]], the small effect of the [[D8:score proxy]] on "
                  "iETS_G under conventional ETS is explained by the [[D3:old error proxy]] "
                  "biasing the two states of iETS_G in opposite directions, so that their "
                  "effects on its probability $\\mu_a/(\\mu_a+\\mu_b)$ largely offset.",
     "about": {"subject": "D3", "relation": "has offsetting biases in the two states of",
               "object": "iETS_G", "comparator": None, "conditions": ["D19"]},
     "type": "interpretive", "role": "secondary",
     "support": [{"kind": "argument", "warrant": "W2"}], "warrant": "W2",
     "relations": [{"type": "explains", "target": "C20", "id_type": "local"}],
     "planning": "exploratory", "provenance": "author_stated"},
    {"id": "C29",
     "statement": "On the [[D19:M5 subset]], the higher [[D28:scaled pinball]] at the 0.99 level "
                  "of iETS_I under conventional ETS with the [[D8:score proxy]] than with the "
                  "[[D3:old error proxy]] is explained by the downward bias of the old proxy's "
                  "probability lowering the upper quantiles of [[D1:iETS_I]].",
     "about": {"subject": "D3", "relation": "lowers the upper quantiles of", "object": "D1",
               "comparator": "D8", "conditions": ["D19"]},
     "type": "interpretive", "role": "secondary",
     "support": [{"kind": "argument", "warrant": "W3"}], "warrant": "W3",
     "relations": [{"type": "explains", "target": "C21", "id_type": "local"}],
     "planning": "exploratory", "provenance": "author_stated"},
    {"id": "C30",
     "statement": "On the [[D19:M5 subset]], the small [[D25:RMSSE]] gains of the [[D8:score "
                  "proxy]] outside the conventional iETS_O and iETS_I may be explained by its "
                  "weaker reaction to a demand that follows a run of zeros, offsetting the gain "
                  "from removing the bias of the [[D3:old error proxy]].",
     "about": {"subject": "D8", "relation": "reacts more weakly to an occurrence after a run of "
                                            "zeros than", "object": "D3", "comparator": None,
               "conditions": ["D19"]},
     "type": "interpretive", "role": "secondary",
     "support": [{"kind": "argument", "warrant": "W4"}], "warrant": "W4",
     "relations": [{"type": "explains", "target": "C20", "id_type": "local"}],
     "planning": "exploratory", "provenance": "author_stated"},
    {"id": "C31",
     "statement": "On the [[D19:M5 subset]], the worse [[D31:MCB]] ranks of the [[D28:scaled "
                  "pinball]] of the [[D20:iETS(M,N,N) flavours]] than of [[D24:NegBin]] and "
                  "[[D23:HSP]] at the levels 0.80 to 0.95 are explained by their continuous "
                  "Gamma demand sizes rather than by the occurrence proxy.",
     "about": {"subject": "D20", "relation": "ranks worse on the scaled pinball, through its "
                                             "demand-size model, than", "object": "D28",
               "comparator": "D24", "conditions": ["D19"]},
     "type": "interpretive", "role": "secondary",
     "support": [{"kind": "argument", "warrant": "W5"}], "warrant": "W5",
     "relations": [{"type": "explains", "target": "C23", "id_type": "local"}],
     "planning": "exploratory", "provenance": "author_stated"},
]

WARRANTS = [
    {"id": "W1", "reasoning": "deductive", "premises": ["C8", "C12"],
     "text": "C12 shows the stationary mean of Croston's 1/q_hat converging to p as alpha "
             "tends to 0, and C8 shows that of iETS_I and iETS_O with the old error proxy "
             "converging to a zero-drift point different from p; the two biases therefore "
             "differ in kind: one vanishes with the smoothing parameter, the other does not."},
    {"id": "W2", "reasoning": "abductive", "premises": ["C6", "C20"],
     "text": "By C6 the old proxy pushes the probability of iETS_O up and that of iETS_I down. "
             "iETS_G updates mu_a as iETS_O does and mu_b as iETS_I does, and its probability "
             "is mu_a/(mu_a+mu_b), so the two upward pushes on its states act on the "
             "probability in opposite directions. The M5 RMSSE of iETS_G under conventional "
             "ETS changed little with the score proxy (C20: not significant at the 1% level), "
             "unlike that of iETS_O and iETS_I. iETS_A selects among the iETS "
             "models by information criterion and inherits the same offset."},
    {"id": "W3", "reasoning": "abductive", "premises": ["C1", "C21", "C19"],
     "text": "By C1 the old proxy biases the probability of iETS_I under conventional ETS "
             "downwards, and on M5 the forecast sum of iETS_I rises when the score proxy "
             "replaces it (C19). A lower probability puts more mass at zero and lowers the "
             "upper quantiles. At 0.99 the lower quantiles of the old proxy gave a lower scaled "
             "pinball on most series and a better MCB rank, while their mean scaled pinball was "
             "higher and their coverage further below 0.99 (table T1 of E23)."},
    {"id": "W4", "reasoning": "abductive", "premises": ["C7", "C20"],
     "text": "After a demand that follows a run of zeros, with a small one-step-ahead "
             "probability p_hat, the old proxy of iETS_O gives 1 + e_a = (2 - p_hat)/p_hat, a "
             "jump in the state of about 2/p_hat, close to Croston's interval update; the score "
             "proxy moves the log of the state by o - p_hat, at most 1 (C7). A filter that "
             "reacts strongly after long runs of zeros adapts faster to changes in the "
             "probability, which may be worth more on non-stationary retail series than the "
             "removal of the bias; the RMSSE gains of the score proxy were large only where the "
             "old proxy's bias was large (C20)."},
    {"id": "W5", "reasoning": "abductive", "premises": ["C23", "C21", "C24", "C25"],
     "text": "NegBin and HSP rank better than every iETS model on the scaled pinball at 0.80, "
             "0.90 and 0.95 whichever proxy the iETS model uses (C23, C21), and rounding the "
             "iETS quantiles up does not close the gap (C24, C25). The iETS models model demand "
             "sizes with a continuous Gamma distribution, NegBin and HSP with count "
             "distributions; the occurrence proxy changes the probability, not the shape of "
             "the size distribution."},
]

OPEN_QUESTIONS = [
    {"id": "O1", "question": "Whether the score proxy scaled by the inverse Fisher information "
                             "of the Bernoulli log-odds, (o - p_hat)/(p_hat (1 - p_hat)), keeps "
                             "the zero drift of the score proxy and the reactivity of the old "
                             "proxy, and so improves the accuracy of iETS models, is not "
                             "established.", "claims": ["C10", "C20", "C30"]},
    {"id": "O2", "question": "How much of the bias of the old error proxy the likelihood "
                             "estimation of the occurrence smoothing parameter and initial "
                             "level absorbs on real data is not established.",
     "claims": ["C8", "C19"]},
]

LIMITATIONS = [
    {"id": "L2", "category": "sampling",
     "statement": "All 3,000 M5 series come from the stores of one retailer, Walmart, in three "
                  "US states, over one period of 1,969 days.",
     "affects": ["C19", "C20", "C21", "C23", "C24", "C25"],
     "impact": "reduces_transferability", "mitigation": "none"},
    {"id": "L3", "category": "model_specification",
     "statement": "The convergence of the stationary mean to the zero-drift point as alpha "
                  "tends to 0 is taken from constant-gain stochastic approximation without "
                  "verifying its regularity conditions for the iETS filters (C26).",
     "affects": ["C8", "C10", "C15"], "impact": "could_bias_magnitude", "mitigation": "partial",
     "mitigation_evidence": "E8"},
]

FORMAL_STEPS = {
    "E5": [
        ("s1", "u = \\frac{1 + o - \\hat p}{2};\\ \\hat p = p \\Rightarrow u = \\frac{2-p}{2} "
               "\\text{ w.p. } p,\\ u = \\frac{1-p}{2} \\text{ w.p. } 1-p", ["D3"], "definition"),
        ("s2", "E[1+e_a] = p\\frac{2-p}{p} + (1-p)\\frac{1-p}{1+p} = 2 - p + "
               "\\frac{(1-p)^2}{1+p}", ["s1"], "substitution"),
        ("s3", "E[1+e_b] = p\\frac{p}{2-p} + (1-p)\\frac{1+p}{1-p} = 1 + p + \\frac{p^2}{2-p}",
         ["s1"], "substitution"),
        ("s4", "1 - p > 0,\\ \\frac{(1-p)^2}{1+p} > 0,\\ p > 0,\\ \\frac{p^2}{2-p} > 0 "
               "\\Rightarrow E[1+e_a] > 1,\\ E[1+e_b] > 1", ["s2", "s3"], "algebra"),
        ("s5", "g(p) := E[\\log(1+e_a)] = p\\log\\frac{2-p}{p} + (1-p)\\log\\frac{1-p}{1+p} = "
               "-E[\\log(1+e_b)]", ["s1"], "substitution"),
        ("s6", "g(1-p) = -g(p) \\Rightarrow g(1/2) = 0", ["s5"], "algebra"),
        ("s7", "g(p) > 0 \\text{ on } (0, 1/2)", ["s5", "s6"],
         "numerical verification on 1,999 grid values with 50-digit arithmetic"),
    ],
    "E6": [
        ("s1", "h_b(x) = p\\frac{x}{2-x} + (1-p)\\frac{1+x}{1-x} - 1 = "
               "\\frac{-2(x^2 - 2x + p)}{(2-x)(1-x)}", ["D3", "D9"], "substitution"),
        ("s2", "h_b(x) = 0,\\ x \\in (0,1) \\iff x = 1 - \\sqrt{1-p};\\ \\sqrt{1-p} > 1 - p "
               "\\Rightarrow 1 - \\sqrt{1-p} < p", ["s1"], "quadratic formula"),
        ("s3", "h_b < 0 \\text{ for } x < x^*,\\ h_b > 0 \\text{ for } x > x^*: x^* "
               "\\text{ stable}", ["s1", "s2"], "sign argument"),
        ("s4", "h_a(x) = p\\frac{2-x}{x} + (1-p)\\frac{1-x}{1+x} - 1 = "
               "\\frac{2(p - x^2)}{x(1+x)} = 0 \\iff x = \\sqrt{p} > p,\\ \\text{stable}",
         ["D3", "D9"], "substitution and sign argument"),
        ("s5", "\\text{additive error: } h(x) = p\\log\\frac{x}{2-x} + (1-p)\\log\\frac{1+x}{1-x}"
               " \\text{ for both models}", ["D3", "D9"], "substitution"),
        ("s6", "h \\text{ has a unique root in } (0,1), \\text{ stable, equal to } 1/2 \\text{ at"
               " } p = 1/2 \\text{ and strictly between } p \\text{ and } 1/2 \\text{ otherwise}",
         ["s5"], "numerical verification on 1,999 grid values with 50-digit arithmetic"),
    ],
    "E7": [
        ("s1", "\\ell = o\\log p + (1-o)\\log(1-p),\\ \\frac{\\partial\\ell}{\\partial p} = "
               "\\frac{o-p}{p(1-p)}", ["D1", "D2"], "differentiation"),
        ("s2", "\\frac{\\partial p}{\\partial\\log\\mu_a} = p(1-p),\\ "
               "\\frac{\\partial p}{\\partial\\log\\mu_b} = -p(1-p)", ["D1", "D2"],
         "differentiation"),
        ("s3", "\\frac{\\partial\\ell}{\\partial\\log\\mu_a} = o - p,\\ "
               "\\frac{\\partial\\ell}{\\partial\\log\\mu_b} = p - o", ["s1", "s2"],
         "chain rule"),
        ("s4", "E[e \\mid \\hat p = x] = \\pm(p - x) \\Rightarrow \\text{unique stable zero-drift "
               "point } x = p", ["s3", "D9"], "linearity of expectation and sign argument"),
        ("s5", "e \\in (-1, 1) \\Rightarrow 1 + \\alpha e > 0 \\text{ for } \\alpha \\in (0,1]",
         ["s3"], "bound"),
    ],
    "E13": [
        ("s1", "\\mu_t = \\mu_{t-1}(1+e_t)^{\\alpha} \\Rightarrow \\log\\mu_t = \\log\\mu_{t-1} "
               "+ \\alpha\\log(1+e_t)", ["D16"], "logarithm"),
        ("s2", "\\text{old proxy: } \\log(1+e) = \\log\\frac{u}{1-u}: \\text{ the additive old "
               "filter}", ["s1", "D3"], "substitution"),
        ("s3", "1+e = 1+s,\\ E[s] = 0 \\text{ at } x = p \\Rightarrow E[\\log(1+s)] < 0 "
               "\\text{ at } x = p", ["s1", "D18"], "Jensen's inequality"),
        ("s4", "\\text{iETS\\_O: } p\\log(2-x) + (1-p)\\log(1-x) = 0,\\ x < p;\\ "
               "\\text{iETS\\_I: } p\\log x + (1-p)\\log(1+x) = 0,\\ x > p", ["s3"],
         "substitution; numerical verification on 1,999 grid values"),
        ("s5", "1+e = \\exp(s) \\Rightarrow \\log(1+e) = s: \\text{ the additive score filter, "
               "zero-drift point } p", ["s1", "D17"], "substitution"),
    ],
}


DEFINITIONS = {
    "D30": "Two-sided Wilcoxon signed-rank test on the per-series differences, score proxy "
           "minus old proxy, zero differences dropped (scipy.stats.wilcoxon, "
           "zero_method='wilcox'), with Holm's correction over the comparisons of each test at "
           "the 1% level; the direction of a difference is the sign of the median non-zero "
           "difference.",
    "D31": "Multiple comparisons with the best (Nemenyi test on ranks) by rmcb() of the Python "
           "greybox at the 99% level: mean ranks of the methods over series, with intervals; two "
           "methods differ statistically significantly when their intervals do not overlap.",
    "D4": "The recursion that produces the one-step-ahead probability $\\hat p_t$ from the "
          "occurrences $o_1, \\dots, o_{t-1}$, the initial state and the smoothing parameter "
          "$\\alpha$: $\\hat p_t = f(\\mu_{t-1})$ with $f(\\mu) = \\mu/(1+\\mu)$ for "
          "[[D2:iETS_O]] and $f(\\mu) = 1/(1+\\mu)$ for [[D1:iETS_I]]; the state is updated "
          "on every observation through an error proxy $e_t$ computed from $o_t$ and "
          "$\\hat p_t$, as $\\mu_t = \\mu_{t-1}(1 + \\alpha e_t)$ (conventional "
          "multiplicative error), $\\log\\mu_t = \\log\\mu_{t-1} + \\alpha e_t$ "
          "(additive error) or $\\mu_t = \\mu_{t-1}(1 + e_t)^{\\alpha}$ "
          "([[D16:ADAM multiplicative ETS update]]).",
    "D9": "For an [[D4:occurrence filter]] applied to $o_t \\sim \\text{Bernoulli}(p)$, a value "
          "$x^*$ of the one-step-ahead probability at which the expected update of the state is "
          "zero: $h(x^*) = 0$ with $h(x) = E[e_t \\mid \\hat p_t = x]$ (additive error) or "
          "$h(x) = E[\\log(1 + e_t) \\mid \\hat p_t = x]$ (ADAM multiplicative error); for the "
          "conventional multiplicative update the expected change of the state is "
          "$\\alpha\\mu_{t-1}h(x)$. It is stable when $h$ moves $\\hat p_t$ towards $x^*$ on "
          "both sides.",
    "D17": "Proxy proposed in this study for a multiplicative occurrence error under the "
           "[[D16:ADAM multiplicative ETS update]]: $1 + e_t = \\exp(s_t)$, with $s_t = o_t - "
           "\\hat p_t$ for $\\mu_a$ ([[D2:iETS_O]]) and $s_t = \\hat p_t - o_t$ for $\\mu_b$ "
           "([[D1:iETS_I]]), so that $\\log\\mu_t = \\log\\mu_{t-1} + \\alpha s_t$; in smooth "
           "proxyOccurrence='score' with ets='adam'.",
    "D18": "The [[D8:score proxy]] used with a multiplicative occurrence error in the form "
           "$1 + e_t = 1 + s_t$, with $s_t = o_t - \\hat p_t$ for $\\mu_a$ and $s_t = \\hat p_t "
           "- o_t$ for $\\mu_b$; under the conventional ETS update this gives $\\mu_t = "
           "\\mu_{t-1}(1 + \\alpha s_t)$ (in smooth proxyOccurrence='score' with "
           "ets='conventional'), under the ADAM update $\\mu_t = \\mu_{t-1}(1 + s_t)^{\\alpha}$.",
}


def by_id(items):
    return {x["id"]: x for x in items}


def finalise_paper(d):
    p = d["paper"]
    p["title"] = ("Bias of the occurrence probability in iETS odds-ratio and inverse-odds-ratio "
                  "models, and a score-based occurrence error proxy")
    p["licence"] = "CC-BY-4.0"
    p["contributions"] = PRINCIPAL
    p["autonomy"] = "human_supervised"
    p["declarations"] = {"funding": [], "competing_interests": "None declared."}
    p["classification"] = {"keywords": [
        "intermittent demand forecasting", "iETS occurrence model", "occurrence probability bias",
        "score-driven filter", "exponential smoothing", "Croston's method", "M5 competition"]}
    d["paper"] = {k: p[k] for k in ["title", "licence", "classification", "declarations", "contributions",
                                    "autonomy", "contribution_record", "origin", "exploration",
                                    "study"] if k in p}


def finalise_people(d):
    people = by_id(d["contributors"])
    people["P1"]["positions"] = ["supervisor", "guarantor"]
    people["P1"]["credit_roles"] = ["conceptualization", "supervision"]
    agent = people["P2"]
    agent["name"] = "Claude Code research agent"
    agent["credit_roles"] = ["methodology", "software", "formal_analysis", "investigation",
                             "data_curation", "visualization", "writing_original_draft"]
    agent["ai_tasks"] = [{"task": t, "extent": e} for t, e in
                         [("analysis", "full"), ("code", "full"), ("writing", "full"),
                          ("json_preparation", "full"), ("literature_search", "assisted")]]
    agent["tools"] = ["code_execution", "file_access", "web_search"]
    agent["record_source"] = "harness"
    d["models"][0]["weights"] = "api_only"


def finalise_definitions(d):
    for x in d["definitions"]:
        if x["id"] in DEFINITIONS:
            x["definition"] = DEFINITIONS[x["id"]]
        if x["id"] in DEF_SOURCE:
            x["source"] = DEF_SOURCE[x["id"]]
    d["references"] = [{"id": i, "target": t, "id_type": k} for i, t, k in REFERENCES]


def formal_evidence(d):
    ev = by_id(d["evidence"])
    for eid, steps in FORMAL_STEPS.items():
        new = f"E{len(d['evidence']) + 1}"
        source = ev[eid]
        d["evidence"].append({
            "id": new, "profile": "formal",
            "description": source["description"].split(" Analysis:")[0].replace(
                "Lab runs", "Derivation checked by lab runs"),
            "foundation": "real analysis and the Kolmogorov axioms of probability",
            "steps": [{"id": i, "statement": s, "by": b, "rule": r} for i, s, b, r in steps],
            "availability": {"level": "public"}, "artefacts": ["A1"]})
        FORMAL[next(c for c, e in FORMAL.items() if e == eid)] = (eid, new)


def finalise_evidence(d):
    for e in d["evidence"]:
        if e["id"] in SIMULATION:
            e["profile"] = "simulation"
            e.pop("design", None)
            e["model"] = "occurrence filter of D4 driven by o_t ~ Bernoulli(p), as coded in " \
                         "stationary_probability.py, general_design.py and filter.cpp"
            e["parameters"] = "A17"
            e["runs"] = len(e.get("lab_runs", []))
            e["artefacts"] = ["A17"]
        elif e["id"] in M5:
            e["design"] = "benchmark"
            e["artefacts"] = ["A2", "A5", "A17"]
            e["description"] = e["description"].replace(
                "on the two forecast files produced by runs 1 and 2 of PL25 (lab/runs/<run>/"
                "iets_forecasts.csv and benchmark_forecasts.csv)",
                "on the forecast files of lab runs RN32 (iETS models, Python smooth) and RN33 "
                "(benchmarks, R)")
    m5 = by_id(d["evidence"])
    m5["E23"]["artefacts"] = ["A2", "A5", "A17", "A13", "A14", "A15"]


def aggregate_table(path):
    rows = list(csv.DictReader(open(path)))
    cols = [c for c in rows[0] if c != "method"]
    variable = {"RMSSE": "D25", "sCE": "D26", "sAPIS": "D27"}
    columns = [variable[c] if c in variable else
               {"variable": "D28" if c.startswith("pinball") else "D29",
                "conditions": {"level": float(c.replace("pinball", "").replace("coverage", ""))}}
               for c in cols]
    return {"id": "T1", "artefact": "A15",
            "description": "Mean over the 3,000 M5 series of each forecast error measure per "
                           "method: RMSSE, sCE, sAPIS, and the scaled pinball and coverage of "
                           "the upper quantile at 0.80, 0.90, 0.95 and 0.99 (benchmarks without "
                           "quantiles have none); horizons 1 to 14.",
            "row_labels": ["method"], "columns": columns, "units": ["1"] * len(cols),
            "rows": [{"id": f"r{i + 1}", "label": [r["method"]],
                      "values": [float(r[c]) if r[c] else None for c in cols]}
                     for i, r in enumerate(rows)]}


def finalise_artefacts(d, table_path):
    art = by_id(d["artefacts"])
    art["A1"]["uri"] = f"{REPO}/{CODE}/claimset/code/iets-occurrence-bias/derivations.md"
    art["A2"].update({"uri": M5_URI, "reference": "B10"})
    for a in ["A3", "A4"]:
        art[a].update({"uri": M5_URI, "reference": "B10"})
        art[a]["description"] += " The file is inside the archive at the URI."
    art["A5"]["uri"] = f"{REPO}/{CODE}/claimset/code/iets-occurrence-bias/m5/subset.csv"
    art["A5"]["licence"] = "CC-BY-4.0"
    study = art["A16"]["uri"].split("/blob/")[1].split("/")[0]  # anchored study commit
    for a in [f"A{i}" for i in range(6, 17)]:
        if a != "A16":
            art[a]["uri"] = art[a]["uri"].replace("claimset/iets-occurrence-bias/",
                                                  f"{REPO}/{study}/")
        art[a]["licence"] = "CC-BY-4.0"
    ranks = [f"E23.{r['id']}" for r in by_id(d["evidence"])["E23"]["results"]
             if r["variable"] == "mean rank, rmcb"]
    art["A14"].update({"function": "data_plot", "shows": ranks, "underlying_data": "A13",
                       "encoding": {"x": "method", "y": "mean rank", "panel": "measure"}})
    art["A15"]["description"] = ("Mean of each forecast error measure per method over the "
                                 "3,000 M5 series (RN39, from RN38's summary_measures.csv); "
                                 "the values of table T1 of E23.")
    d["artefacts"].append({
        "id": "A17", "kind": "code", "licence": "LGPL-2.1-only",
        "uri": f"{REPO}/{CODE}/claimset/code/iets-occurrence-bias",
        "version": CODE, "availability": {"level": "public"},
        "description": "Study code: the occurrence-filter simulations (filter.cpp, "
                       "stationary_probability.py, general_design.py with their design files), "
                       "the symbolic checks (derivations_check.py) and the M5 forecasting and "
                       "evaluation scripts (m5/)."})
    d["artefacts"].append({
        "id": "A18", "kind": "software", "licence": "LGPL-2.1-only",
        "uri": f"{REPO}/{CODE}/src/headers/ssOccurrence.h",
        "version": CODE, "availability": {"level": "public"},
        "description": "The smooth package at the commit of the study, with the argument "
                       "proxyOccurrence ('ratio', the old error proxy, or 'score') of om(), "
                       "adam() and the Python OM and ADAM classes; occurrenceError() in this "
                       "file implements both proxies."})
    by_id(d["evidence"])["E23"]["tables"] = [aggregate_table(table_path)]


def support_for(c, ev):
    out = []
    for s in c["support"]:
        if s.get("evidence") in SIMULATION:
            s = {"kind": "simulation", "evidence": s["evidence"], "artefacts": ["A17"]}
        elif s.get("evidence") in M5:
            s = {"kind": "empirical_quant", "evidence": s["evidence"]}
        else:
            s = {k: v for k, v in s.items() if k != "results"}
        if c["id"] in RESULTS:
            s["results"] = RESULTS[c["id"]]
        out.append(s)
    return out


def finalise_claims(d):
    claims = by_id(d["claims"])
    for c in d["claims"]:
        cid = c["id"]
        if cid in STATEMENTS:
            c["statement"] = STATEMENTS[cid]
        if cid == "C26":
            c["check"], c["check_evidence"] = "tested", "E8"
            continue
        c["support"] = support_for(c, by_id(d["evidence"]))
        c["role"] = ("principal" if cid in PRINCIPAL else
                     "secondary" if cid in SECONDARY else "supporting")
        if cid in FORMAL:
            check, proof = FORMAL[cid]
            c["type"] = "formal"
            c["support"] = [{"kind": "formal_proof", "evidence": proof},
                            {"kind": "empirical_quant", "evidence": check}]
            checks = c.get("checks", [])
            confirmed = any(x.get("outcome") == "confirmed" and x.get("extent") == "full"
                            for x in checks)
            disputed = any(x.get("outcome") == "disputed" and not x.get("resolution")
                           for x in checks)
            c["verification"] = ("disputed" if disputed else
                                 "agent_checked" if confirmed else "unchecked")
    claims["C22"]["limitations"] = ["L1"]
    claims["C9"]["about"]["object"] = "D13"
    claims["C8"]["about"].update({"subject": "D3", "object": "D9"})
    for c in NEW_CLAIMS:
        d["claims"].append(c)
    for c in d["claims"]:
        cid = c["id"]
        if cid == "C26":
            continue
        if cid in DEPENDS:
            c["depends_on"] = DEPENDS[cid]
        if cid in ASSUMPTIONS:
            c["assumptions"] = ASSUMPTIONS[cid]
        novelty, relations = NOVELTY[cid]
        c["novelty"] = novelty
        c["relations"] = c.get("relations", []) + relations
        if cid in LOCAL:
            kind, target = LOCAL[cid]
            c["relations"].append({"type": kind, "target": target, "id_type": "local"})
        if cid in SCOPE:
            c["scope"] = SCOPE[cid]
        if c["type"] != "formal":
            strength, why = STRENGTH[cid]
            c["assessments"] = [{"by": "author", "strength": strength, "rationale": why}]
    order = PRINCIPAL + [c["id"] for c in d["claims"] if c["id"] not in PRINCIPAL]
    claims = by_id(d["claims"])
    d["claims"] = [claims[i] for i in order]


def main(draft, table_path, out):
    d = json.load(open(draft))
    finalise_paper(d)
    finalise_people(d)
    finalise_definitions(d)
    formal_evidence(d)
    finalise_evidence(d)
    finalise_artefacts(d, table_path)
    finalise_claims(d)
    d["warrants"] = WARRANTS
    d["limitations"][0]["statement"] = (
        "In the first MCB evaluation of the M5 forecasts, evaluate.py matched the mean ranks and "
        "intervals returned by greybox rmcb(), which are sorted by mean rank, to the method "
        "names in alphabetical order, so each method's rank and interval belong to another "
        "method and the counts of significant comparisons are not those of the planned "
        "analysis; the labelling was fixed in code commit 237c71d2 and the evaluation rerun on "
        "the same forecasts.")
    d["limitations"] += LIMITATIONS
    d["open_questions"] = OPEN_QUESTIONS
    json.dump(d, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
