"""Why reforecast() blows up on ar1 and sar12: the covariance of the parameters, the width
per horizon against predict() and the same in R on the same series."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import subprocess
import tempfile
import warnings

import numpy as np

from q1_simulate import DGPS, HMAX, simulate, spec_of, fit_model

warnings.filterwarnings("ignore")
ROOT = os.environ["SMOOTH_ROOT"]
R = """
suppressMessages(pkgload::load_all("{root}", quiet=TRUE))
a <- jsonlite::fromJSON("{path}", simplifyVector=FALSE)
for(i in seq_along(a)){{a[[i]]$x <- unlist(a[[i]]$x); a[[i]]$lags <- unlist(a[[i]]$lags); for(k in c("ar","i","ma")){{a[[i]]$orders[[k]] <- unlist(a[[i]]$orders[[k]])}}}}
out <- list()
for(e in a){{
  m <- msarima(e$x, orders=list(ar=e$orders$ar, i=e$orders$i, ma=e$orders$ma), lags=e$lags, constant=e$constant, initial=e$initial, h=18, holdout=FALSE)
  f0 <- forecast(m, h=18, interval="prediction", level=0.8, side="both")
  set.seed(1)
  f1 <- tryCatch(reforecast(m, h=18, nsim=300, level=0.8, interval="prediction"), error=function(err) NULL)
  out[[length(out)+1]] <- list(name=e$name, initial=e$initial, coef=as.numeric(coef(m)), sd=as.numeric(sqrt(diag(vcov(m)))),
     w0=as.numeric(f0$upper-f0$lower)[c(1,6,18)], w1=if(is.null(f1)) NA else as.numeric(f1$upper-f1$lower)[c(1,6,18)])
}}
jsonlite::write_json(out, "{res}", auto_unbox=TRUE, digits=NA)
"""
results, inputs = [], []
for name in ("ar1", "sar12", "airline12", "ima"):
    for initial in ("backcasting", "optimal"):
        rng = np.random.default_rng(np.random.SeedSequence([21, len(name)]))
        d = DGPS[name]
        x = simulate(d, 100, rng)[0]
        lags, orders, _, const = spec_of(d)
        inputs.append(dict(name=name, initial=initial, x=x.tolist(), orders=orders, lags=lags, constant=const))
        m = fit_model("E", initial, name, x)
        v = m.vcov().values
        f0 = m.predict(h=HMAX, interval="prediction", level=0.8, side="both")
        f1 = m.reforecast(h=HMAX, interval="prediction", level=0.8, side="both", nsim=300, seed=1)
        w0 = (f0.upper.values - f0.lower.values).ravel()[[0, 5, 17]]
        w1 = (f1.upper.values - f1.lower.values).ravel()[[0, 5, 17]]
        d1 = (f1.mean.values - f0.mean.values)[[0, 5, 17]]
        corr = np.corrcoef(np.linalg.cholesky(v + 0 * np.eye(len(v))).T) if False else v / np.outer(np.sqrt(np.diag(v)), np.sqrt(np.diag(v)))
        print("corr first row", np.round(corr[0], 4))
        draws = m.reapply(nsim=300, seed=1)
        print("reapply keys", [k for k in dir(draws) if not k.startswith("_")][:14] if not isinstance(draws, dict) else list(draws))
        print("PY", name, initial, "coef", np.round(m.coef, 3)[:4], "sd", np.round(np.sqrt(np.diag(v)), 3)[:4],
              "w0", np.round(w0, 1), "w1", np.round(w1, 1), "mean diff", np.round(d1, 2))
        for h, a, b in zip((1, 6, 18), w0, w1):
            results.append(dict(quantity="reforecast over predict 80% width, one series", value=float(b / a),
                                conditions=dict(dgp=name, initial=initial, h=h, language="python")))
with tempfile.TemporaryDirectory() as tmp:
    path, res = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
    json.dump(inputs, open(path, "w"))
    run = subprocess.run(["Rscript", "-e", R.format(root=ROOT, path=path, res=res)], capture_output=True, text=True, cwd=tmp)
    print(run.stderr[-600:])
    for e in json.load(open(res)):
        print("R ", e["name"], e["initial"], "coef", np.round(e["coef"], 3)[:4], "sd", np.round(e["sd"], 3)[:4], "w0", np.round(e["w0"], 1), "w1", np.round(e["w1"], 1) if e["w1"] != ["NA"] else e["w1"])
        if isinstance(e["w1"], list) and e["w1"][0] != "NA":
            for h, a, b in zip((1, 6, 18), e["w0"], e["w1"]):
                results.append(dict(quantity="reforecast over predict 80% width, one series", value=float(b) / float(a),
                                    conditions=dict(dgp=e["name"], initial=e["initial"], h=h, language="R")))
if os.environ.get("CLAIMSET_RESULTS"):
    json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))
