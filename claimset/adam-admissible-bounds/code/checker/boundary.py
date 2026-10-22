"""AC3: the package's smooth_eigens() modulus when theta* has a root exactly on the unit circle that does not cancel."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
import numpy as np
from smooth.adam_general._eigenCalc import smooth_eigens

res = []
def case(name, ar, i, ma, lagsEts=None):
    ari, maP = arima_polys(ar, i, ma)
    F, g, w, lags = arima_matrices(ari, maP)
    if lagsEts:
        e = ([[1.0]], [0.3], [1.0], [1])
        F, g, w, lags = blockdiag([e, (F, g, w, lags)])
    mod = float(np.max(smooth_eigens(persistence=np.asfortranarray(np.array(g, float).reshape(-1, 1)),
        transition=np.asfortranarray(np.array(F, float)), measurement=np.asfortranarray(np.array([w], float)),
        lags_model_all=np.asarray(lags, dtype=np.int32), xreg_model=False, obs_in_sample=1, has_delta=False,
        xreg_number=0, constant_required=False)))
    res.append({"quantity": "package smooth_eigens modulus, theta* with an uncancelled root exactly on the unit circle: " + name,
                "value": mod, "conditions": {"admissible": bool(not mod > 1 + 1e-50)}})
case("MA(1) theta=1", [], [], [(1, [1.0])])
case("MA(1) theta=-1", [], [], [(1, [-1.0])])
case("MA(2) 1 - B + B^2", [], [], [(1, [-1.0, 1.0])])
case("MA(2) 1 + B^2", [], [], [(1, [0.0, 1.0])])
case("MA(2) 1 + B + B^2", [], [], [(1, [1.0, 1.0])])
case("AR(1) 0.5, MA(2) 1 - B + B^2", [(1, [0.5])], [], [(1, [-1.0, 1.0])])
case("I(1), MA(2) 1 - B + B^2", [], [(1, 1)], [(1, [-1.0, 1.0])])
case("I(1), MA(2) 0.6 factor times 1 - B + B^2, with 5 lags", [], [(1, 1)], [(1, [-0.4, 0.4, -0.6, 0.6])])
case("ETS(A,N,N)+I(1), MA(2) 1 - B + B^2", [], [(1, 1)], [(1, [-1.0, 1.0])], lagsEts=True)
case("I(2) with MA(2) 1 + 2B + B^2 (double unit root, cancels twice)", [], [(1, 2)], [(1, [2.0, 1.0])])
case("I(1) with MA(2) 1 + 2B + B^2 (double root at -1, one factor of Phi_2 left, no cancel with lag 1)", [], [(1, 1)], [(1, [2.0, 1.0])])
results_out(res)
