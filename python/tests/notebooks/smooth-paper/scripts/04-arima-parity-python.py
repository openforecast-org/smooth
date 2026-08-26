"""Per-series AutoMSARIMA diagnostics (Python) for the R-vs-Python comparison."""
import warnings, multiprocessing, csv
warnings.filterwarnings("ignore")
import numpy as np

def task(a):
    i, s, init = a
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    from smooth import AutoMSARIMA
    lags = [1, s.period] if s.period > 1 else [1]
    try:
        m = AutoMSARIMA(lags=lags, initial=init); m.fit(np.asarray(s.x, float))
        fc = np.asarray(m.predict(h=s.h)["mean"]).ravel()
        y = np.asarray(s.xx, float); tr = np.asarray(s.x, float)
        sc = np.mean(np.diff(tr) ** 2)
        rmsse = float(np.sqrt(np.mean((y - fc) ** 2) / sc)) if sc else float("nan")
        o = m.orders
        spec = "ar%s_i%s_ma%s" % (o.get("ar"), o.get("i"), o.get("ma"))
        return (i, init, m.model_name, spec, float(m.loglik), float(m.aicc),
                int(m.nparam), rmsse, float(fc.sum()))
    except Exception as e:
        return (i, init, "ERROR", str(type(e).__name__), float("nan"), float("nan"), -1,
                float("nan"), float("nan"))

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from fcompdata import M1, M3, Tourism
    data = [M1[k] for k in M1.keys()] + [M3[k] for k in M3.keys()] + [Tourism[k] for k in Tourism.keys()]
    tasks = [(i, s, init) for init in ("backcasting", "optimal") for i, s in enumerate(data)]
    out = []
    with ProcessPoolExecutor(max_workers=30, mp_context=multiprocessing.get_context("fork")) as ex:
        futs = [ex.submit(task, t) for t in tasks]
        for n, f in enumerate(as_completed(futs)):
            out.append(f.result())
            if (n + 1) % 2500 == 0: print("  %d/%d" % (n + 1, len(tasks)), flush=True)
    with open("py_arima.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx","init","model","spec","loglik","aicc","nparam","rmsse","fsum"])
        for r in sorted(out): w.writerow(r)
    print("wrote py_arima.csv", len(out))
