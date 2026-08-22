"""Parameter validation for :class:`smooth.CES`.

Kept out of ``test_ces.py`` because that module is marked ``r_parity`` as a
whole; these check the Python API's own guards and need no R installation.
"""

from pathlib import Path

import pandas as pd
import pytest

from smooth import CES

_DATA_DIR = Path(__file__).parent / "data"


class TestCESProvidedB:
    """The provided ``b`` must match the seasonality it belongs to."""

    y = pd.read_csv(_DATA_DIR / "ces_airpassengers.csv")["y"].values

    def test_full_rejects_a_real_b(self):
        with pytest.raises(ValueError, match="complex second smoothing parameter"):
            CES(seasonality="full", lags=[12], b=0.3).fit(self.y)

    def test_partial_rejects_a_complex_b(self):
        with pytest.raises(ValueError, match="real second smoothing parameter"):
            CES(seasonality="partial", lags=[12], b=complex(0.3, 0.1)).fit(self.y)

    @pytest.mark.parametrize("seasonality", ["none", "simple"])
    def test_b_is_dropped_where_there_is_none(self, seasonality):
        with pytest.warns(UserWarning, match="no second smoothing parameter"):
            m = CES(seasonality=seasonality, lags=[12], b=0.3)
        m.fit(self.y)
        assert m.b_ is None
        free = CES(seasonality=seasonality, lags=[12])
        free.fit(self.y)
        assert m.loglik == free.loglik

    def test_valid_b_is_honoured(self):
        free = CES(seasonality="full", lags=[12])
        free.fit(self.y)
        held = CES(seasonality="full", lags=[12], b=free.b_)
        held.fit(self.y)
        assert held.b_ == free.b_

        partial = CES(seasonality="partial", lags=[12], b=0.3)
        partial.fit(self.y)
        assert partial.b_ == 0.3
