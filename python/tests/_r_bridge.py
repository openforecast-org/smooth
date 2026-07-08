"""Shared helpers for parity tests that compare Python output against R.

The R side is always loaded with ``devtools::load_all('.')`` so the tests
exercise the *local* R source (matching the development checkout), not a
potentially stale CRAN install. Outputs flow back over stdout as JSON
(via ``jsonlite::toJSON``).

Each test file that imports from here must also carry
``pytestmark = pytest.mark.r_parity`` so the suite stays opt-in
(``[tool.pytest.ini_options]`` deselects ``r_parity`` by default).
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Optional

import numpy as np

# Repo root (the directory containing the R package source). load_all() is
# called against this path so the R-side resolution matches the dev install.
REPO_ROOT = "/home/config/Misc/Python/Libraries/smooth"

_PRELUDE = (
    "suppressMessages(suppressWarnings({"
    "devtools::load_all('.', quiet=TRUE);"
    "library(jsonlite)"
    "}));"
)


def r_to_literal(value: Any) -> str:
    """Render a Python scalar / array as an R literal expression.

    Supports ``None`` (→ ``NULL``), bool / int / float scalars, strings
    (single-quoted), and 1-D numeric arrays / lists (→ R ``c(...)``).
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return repr(float(value))
    if isinstance(value, str):
        # R prefers single quotes; escape any inside.
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return "numeric(0)"
    return "c(" + ",".join(repr(float(v)) for v in arr) + ")"


def r_eval(expr: str, R_data: Optional[dict] = None) -> Any:  # noqa: N803
    """Evaluate an R expression and return the JSON-decoded result.

    Parameters
    ----------
    expr : str
        Any R expression. Must evaluate to something ``jsonlite::toJSON``
        can serialise — typically a numeric vector / matrix / list /
        named list.
    R_data : dict, optional
        Mapping of variable name → Python value, bound in R as
        ``<name> <- <r_to_literal(value)>`` before ``expr`` runs. Use for
        passing input series, model coefficients, etc.

    Returns
    -------
    Any
        Whatever ``json.loads`` produces — usually ``list``, ``dict``,
        ``float``, or nested combinations. Convert to ``np.ndarray`` /
        ``pd.DataFrame`` at the call site as needed.

    Notes
    -----
    Uses ``digits=15`` to preserve double-precision values across the
    JSON round-trip. ``--vanilla`` avoids picking up the user's
    ``.Rprofile`` so the test environment is reproducible.
    """
    bindings = ""
    if R_data:
        bindings = (
            ";".join(f"{k} <- {r_to_literal(v)}" for k, v in R_data.items()) + ";"
        )
    script = _PRELUDE + bindings + f"cat(jsonlite::toJSON({expr}, digits=15))"
    out = subprocess.check_output(
        ["Rscript", "--vanilla", "-e", script],
        text=True,
        cwd=REPO_ROOT,
    )
    return json.loads(out)


def r_array(expr: str, R_data: Optional[dict] = None) -> np.ndarray:  # noqa: N803
    """``r_eval`` wrapper that returns a NumPy array (any shape preserved)."""
    return np.asarray(r_eval(expr, R_data=R_data), dtype=float)


def r_dict(expr: str, R_data: Optional[dict] = None) -> dict:  # noqa: N803
    """``r_eval`` wrapper that asserts the result is a dict.

    ``jsonlite::toJSON`` on a named R list yields a dict. Useful for
    pulling several quantities out of a single R fit (one subprocess call
    instead of N) — e.g. ``list(coef=coef(m), vcov=vcov(m), …)``.
    """
    res = r_eval(expr, R_data=R_data)
    if not isinstance(res, dict):
        raise TypeError(f"Expected dict, got {type(res).__name__}: {res!r}")
    return res
