"""AC3: (b) the worked example, (a) ADAM's ARIMA part and (d) the shortcut of DC6 on matrices built by adam()."""
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import *  # noqa
from smooth.adam_general._eigenCalc import smooth_eigens

res = []


def add(q, v, **c):
    res.append({"quantity": q, "value": float(v), "conditions": c})


def package_modulus(F, g, w, lags, xreg=False, nx=0, const=False):
    ev = smooth_eigens(persistence=np.asfortranarray(np.array(g, float).reshape(-1, 1)),
                       transition=np.asfortranarray(np.array(F, float)),
                       measurement=np.asfortranarray(np.array([w], float)),
                       lags_model_all=np.asarray(lags, dtype=np.int32), xreg_model=xreg, obs_in_sample=1,
                       has_delta=False, xreg_number=nx, constant_required=const)
    return float(np.max(ev))


def minroot(c):
    c = np.array(c, float)
    c = np.trim_zeros(c, "b")
    return float(np.abs(np.roots(c[::-1])).min()) if len(c) > 1 else np.inf


# ---- (b) ETS(A,N,N) alpha = 1.9 + MA(1) theta = 0.1
alpha, th = Fr(19, 10), Fr(1, 10)
etsP = ([[Fr(1)]], [alpha], [Fr(1)], [1])
ari, ma = arima_polys([], [], [(1, [th])], True)
arP = arima_matrices(ari, ma, True)
F, g, w, lags = blockdiag([etsP, arP])
theta = detpoly(Dmat(F, g, w), lags, True)
phi = detpoly(F, lags, True)
thEts = detpoly(Dmat(*etsP[:3]), [1], True)
prodParts = pmul(thEts, ma)
for i, c in enumerate(theta):
    add("coefficient of B^%d in theta (MA polynomial) of the whole ETS(A,N,N)+MA(1)" % i, c, alpha=1.9, theta=0.1)
for i, c in enumerate(prodParts):
    add("coefficient of B^%d in the product of the parts' MA polynomials" % i, c, alpha=1.9, theta=0.1)
add("coefficient of B^1 in phi of the whole model", phi[1], alpha=1.9, theta=0.1)
add("smallest root modulus of theta of the whole model", minroot(theta), alpha=1.9, theta=0.1)
add("smallest root modulus of the product of the parts' MA polynomials", minroot(prodParts), alpha=1.9, theta=0.1)
add("package smooth_eigens modulus of the whole model", package_modulus(F, g, w, lags), alpha=1.9, theta=0.1)
add("package smooth_eigens modulus of the ETS part alone", package_modulus(*etsP), alpha=1.9)
add("max |eigenvalue| of D of the whole model (lag 1, so the expanded D)",
    np.abs(np.linalg.eigvals(np.array(Dmat(F, g, w), float))).max(), alpha=1.9, theta=0.1)

# ---- (a) and (d): matrices built by adam()
def un(x):
    return x[0] if isinstance(x, list) and len(x) == 1 and not isinstance(x[0], list) else x


models = json.load(open(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "adam_mats.json")))
gapPhi = gapTheta = 0.0
nInv = nShortcutWrong = nFullRejectsInv = 0
nNonInv = nNonInvAccepted = nNonInvPlain = nNonInvPlainAccepted = 0
byKind = {}
for mo in models:
    kind = un(mo["kind"])
    lags_all = [int(x) for x in mo["lagsAll"]]
    F = np.array(mo["F"], float)
    k = len(lags_all)
    F = F.reshape(k, k) if F.ndim == 1 else F
    g, w = np.array(mo["g"], float).ravel(), np.array(mo["w"], float).ravel()
    lagsB = [int(un(x)) if isinstance(x, list) else int(x) for x in mo["lags"]]
    ar = [int(un(x)) for x in mo["ar"]]
    i_ = [int(un(x)) for x in mo["i"]]
    mq = [int(un(x)) for x in mo["ma"]]
    arc, mac = list(mo["arc"]), list(mo["mac"])
    arP, maP = [], []
    ia = im = 0
    for l, p in zip(lagsB, ar):
        arP.append((l, arc[ia:ia + p])); ia += p
    for l, q in zip(lagsB, mq):
        maP.append((l, mac[im:im + q])); im += q
    arP = [(l, c) for l, c in arP if len(c)]
    maP = [(l, c) for l, c in maP if len(c)]
    iP = [(l, d) for l, d in zip(lagsB, i_) if d]
    ari, maPoly = arima_polys(arP, iP, maP)
    extra = [1.0, -1.0] if kind != "plain" else [1.0]
    ariW, maW = pmul(ari, extra), pmul(maPoly, extra)
    D = F - np.outer(g, w)
    phi, theta = detpoly(F.tolist(), lags_all), detpoly(D.tolist(), lags_all)
    n = max(len(phi), len(ariW), len(theta), len(maW))
    pad = lambda p: np.pad(np.array(p, float), (0, n - len(p)))
    gapPhi = max(gapPhi, np.abs(pad(phi) - pad(ariW)).max())
    gapTheta = max(gapTheta, np.abs(pad(theta) - pad(maW)).max())
    # (d): the shortcut accepts when every MA root is outside the unit circle
    invertible = minroot(maPoly) > 1 + 1e-9
    xreg = kind == "xreg"
    mod = package_modulus(F, g, w, lags_all, xreg=xreg, nx=1 if xreg else 0, const=kind == "constant")
    full_ok = not (mod > 1 + 1e-50)
    byKind.setdefault(kind, [0, 0])
    byKind[kind][0] += 1
    if invertible:
        nInv += 1
        nFullRejectsInv += (not full_ok)
        byKind[kind][1] += 1
    else:
        nNonInv += 1
        nNonInvAccepted += full_ok
        if kind == "plain":
            nNonInvPlain += 1
            nNonInvPlainAccepted += full_ok

add("max abs difference between det(I - F L) of the adam() matrices and ari polynomial times (1 - B) per constant/regressor", gapPhi, models=len(models))
add("max abs difference between det(I - D L) of the adam() matrices and MA polynomial times (1 - B) per constant/regressor", gapTheta, models=len(models))
add("pure ARIMA models (plain, constant, static regressor) from adam() with all MA roots outside the unit circle", nInv, models=len(models))
add("of those, models the whole-model check smooth_eigens rejects (shortcut would be wrong)", nFullRejectsInv, models=nInv)
add("models from adam() with an MA root on or inside the unit circle (shortcut runs the whole check)", nNonInv, models=len(models))
add("of those, models the whole-model check accepts (root cancels)", nNonInvAccepted, models=nNonInv)
for kd, (a, b) in byKind.items():
    add("models of the kind", a, kind=kd, invertible=b)
results_out(res)
