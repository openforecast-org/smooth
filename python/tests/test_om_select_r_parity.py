"""OM / OMG ETS model selection against R's ``om()`` / ``omg()``.

The same pool, branch-and-bound and estimator run on both sides, so the
selected model, the information criteria, the logLik and the forecasts match
to the last bits. Skipped by default (``r_parity`` marker).
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import OM, OMG

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

DATA = (
    "set.seed(41); "
    "y <- rbinom(364, 1, plogis(-0.3 + 1.2*sin(2*pi*(1:364)/7)))*rpois(364, 3);"
)

OM_CASES = [
    ("ZZN", [1], "odds-ratio"),
    ("ZXZ", [1, 7], "odds-ratio"),
    ("ZZZ", [1, 7], "inverse-odds-ratio"),
    ("ZXN", [1], "direct"),
    ("ZXN", [1], "auto"),
    ("ZXN", [1], "general"),
]


def _r_lags(lags):
    return "c(" + ",".join(map(str, lags)) + ")"


def _r_fit(call):
    return r_dict(
        f"{{ {DATA} m <- {call}; "
        "list(y=y, model=m$model, logLik=as.numeric(logLik(m)), "
        "ICs=if(is.null(m$ICs)) NULL else unname(m$ICs), "
        "ICnames=if(is.null(m$ICs)) NULL else names(m$ICs), "
        "f=as.numeric(forecast(m, h=7)$mean)) }"
    )


def _compare(py, r):
    assert py.model == r["model"][0]
    assert py.loglik == pytest.approx(r["logLik"][0], rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(py.predict(h=7).mean), r["f"], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "model,lags,occurrence", OM_CASES, ids=[f"{c[0]}-{c[2]}" for c in OM_CASES]
)
def test_om_select_matches_r(model, lags, occurrence):
    r = _r_fit(
        f"om(y, model='{model}', lags={_r_lags(lags)}, occurrence='{occurrence}')"
    )
    # fit() returns the selected object for occurrence="auto", as auto.om() does
    py = OM(model=model, lags=lags, occurrence=occurrence).fit(
        np.asarray(r["y"], dtype=float)
    )
    _compare(py, r)
    if r.get("ICnames") and getattr(py, "ICs", None) is not None:
        assert list(py.ICs) == r["ICnames"]
        np.testing.assert_allclose(list(py.ICs.values()), r["ICs"], rtol=1e-12)


@pytest.mark.parametrize(
    "model_a,model_b", [("ZXZ", "ANN"), ("ANA", "ANN"), ("ANN", "ANA")]
)
def test_omg_mixed_sides_match_r(model_a, model_b):
    r = _r_fit(f"omg(y, modelA='{model_a}', modelB='{model_b}', lags=c(1,7))")
    py = OMG(model_a=model_a, model_b=model_b, lags=[1, 7])
    py.fit(np.asarray(r["y"], dtype=float))
    _compare(py, r)


REDIRECT_CASES = [
    ("general", "ets='adam'", {"ets": "adam"}),
    ("general", "orders=list(ar=1,i=0,ma=0)", {"ar_order": 1}),
    ("general", "loss='MSE'", {"loss": "MSE"}),
    ("general", "loss='RIDGE', lambda=0.3", {"loss": "RIDGE", "lambda_param": 0.3}),
    ("auto", "ets='adam'", {"ets": "adam"}),
    ("auto", "loss='MSE'", {"loss": "MSE"}),
]


@pytest.mark.parametrize(
    "occurrence,r_args,py_args",
    REDIRECT_CASES,
    ids=[f"{c[0]}-{c[1]}" for c in REDIRECT_CASES],
)
def test_om_redirect_passes_arguments_as_r(occurrence, r_args, py_args):
    """om() hands its arguments to omg() / auto.om(), as R."""
    r = _r_fit(f"om(y, model='MNN', occurrence='{occurrence}', {r_args})")
    py = OM(model="MNN", occurrence=occurrence, **py_args)
    _compare(py.fit(np.asarray(r["y"], dtype=float)), r)


NAME_CASES = [
    ("om(d, 'MNN', occurrence='odds-ratio')", "OM", {"occurrence": "odds-ratio"}),
    ("om(d, 'MNN', occurrence='fixed')", "OM", {"occurrence": "fixed"}),
    (
        "om(d, 'NNN', orders=c(1,0,0), occurrence='direct')",
        "OM",
        {"model": "NNN", "ar_order": 1, "occurrence": "direct"},
    ),
    ("omg(d, 'MNN', 'MNN')", "OMG", {"model_a": "MNN"}),
    ("adam(d, 'MNN', occurrence='odds-ratio')", "ADAM", {"occurrence": "odds-ratio"}),
]


@pytest.mark.parametrize("r_call,py_class,py_args", NAME_CASES)
def test_names_with_regressors_match_r(r_call, py_class, py_args):
    """The X of the regressors in the names, as R's adam_model_name."""
    import smooth

    r = r_dict(
        "{ set.seed(3); x <- rnorm(200); y <- rbinom(200,1,plogis(x))*exp(x+2);"
        f" d <- data.frame(y=y, x=x); list(y=y, x=x, name=modelName({r_call})) }}"
    )
    if py_class != "OMG":
        py_args = {"model": "MNN", **py_args}
    X = np.asarray(r["x"], dtype=float).reshape(-1, 1)
    fit = getattr(smooth, py_class)(**py_args).fit(np.asarray(r["y"], dtype=float), X)
    assert fit.model_name == r["name"][0]


@pytest.mark.parametrize(
    "r_call,py_class,py_args",
    [
        ("om(y, 'MNN', occurrence='odds-ratio', h=10, holdout=TRUE)", "OM", {}),
        ("om(y, 'MNN', occurrence='fixed', h=10, holdout=TRUE)", "OM", {}),
        ("omg(y, 'MNN', 'MNN', h=10, holdout=TRUE)", "OMG", {"model_a": "MNN"}),
    ],
)
def test_holdout_accuracy_matches_r(r_call, py_class, py_args):
    """accuracy of om() / omg() with a holdout, as R's measures()."""
    import smooth

    r = r_dict(
        "{ set.seed(3); y <- rbinom(120,1,0.4)*exp(rnorm(120,2,0.3));"
        f" a <- ({r_call})$accuracy; a <- a[is.finite(a)];"
        " list(y=y, names=names(a), a=unname(a)) }"
    )
    if py_class == "OM":
        occurrence = r_call.split("occurrence='")[1].split("'")[0]
        py_args = {"model": "MNN", "occurrence": occurrence}
    fit = getattr(smooth, py_class)(h=10, holdout=True, **py_args)
    accuracy = fit.fit(np.asarray(r["y"], dtype=float)).accuracy
    # MPE and MAPE divide by the zeros of the holdout: R's measures() returns
    # -Inf / Inf, which JSON does not carry, and greybox's in Python finite values
    for name, value in zip(r["names"], r["a"]):
        assert accuracy[name] == pytest.approx(value, rel=1e-6), name


AUTO_DATA = (
    "set.seed(5); z <- as.numeric(arima.sim(list(ar=0.8), 300));"
    " y <- rbinom(300, 1, plogis(z))*rpois(300, 3);"
)


@pytest.mark.parametrize(
    "r_args,py_args",
    [
        ("model='NNN'", {"model": "NNN"}),
        ("model='MNN'", {"model": "MNN"}),
        ("model='ANN', lags=7", {"model": "ANN", "lags": [7]}),
        (
            "model='MNN', occurrence=c('odds-ratio','direct')",
            {"model": "MNN", "occurrence": ["odds-ratio", "direct"]},
        ),
    ],
)
def test_auto_om_selects_the_orders_as_r(r_args, py_args):
    """auto.om() selects the ARIMA orders by default."""
    from smooth import AutoOM

    r = r_dict(
        f"{{ {AUTO_DATA} m <- auto.om(y, {r_args}); "
        "list(y=y, model=m$model, logLik=as.numeric(logLik(m)), "
        "f=as.numeric(forecast(m, h=7)$mean)) }"
    )
    _compare(AutoOM(**py_args).fit(np.asarray(r["y"], dtype=float)), r)


@pytest.mark.parametrize(
    "r_call,py_class", [("om(d, occurrence='fixed')", "OM"), ("auto.om(d)", "AutoOM")]
)
def test_fixed_drops_the_regressors_as_r(r_call, py_class):
    """A constant probability has no regressors, so auto.om() takes them."""
    import smooth

    r = r_dict(
        "{ set.seed(4); d <- data.frame(y=rbinom(120,1,0.5), a=rnorm(120));"
        f" m <- {r_call}; list(y=d$y, a=d$a, model=m$model,"
        " logLik=as.numeric(logLik(m)), nparam=nparam(m),"
        " f=as.numeric(forecast(m, h=7)$mean)) }"
    )
    X = np.asarray(r["a"], dtype=float).reshape(-1, 1)
    kwargs = {"occurrence": "fixed"} if py_class == "OM" else {}
    fit = getattr(smooth, py_class)(**kwargs).fit(np.asarray(r["y"], dtype=float), X)
    _compare(fit, r)
    assert fit.nparam == r["nparam"][0]
