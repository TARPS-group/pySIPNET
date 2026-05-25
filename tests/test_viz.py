"""Tests for pysipnet.viz.dashboard."""

from __future__ import annotations

import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")


def _make_result(include_litter: bool = False):
    """Return a minimal SIPNETResult with synthetic timeseries and climate."""
    from unittest.mock import MagicMock

    from pysipnet.result import SIPNETResult

    n = 10
    ts_cols = {
        "year": 2020,
        "day": range(1, n + 1),
        "time": 0.0,
        "nee": -1.0,
        "gpp": 3.0,
        "evapotranspiration": 0.2,
        "ra": 1.0,
        "rh": 1.0,
        "cum_nee": range(-n, 0),
        "plant_wood_c": 30000.0,
        "plant_leaf_c": 100.0,
        "soil_c": 10000.0,
        "soil_water": 6.0,
    }
    if include_litter:
        ts_cols["litter_c"] = 200.0

    ts = pd.DataFrame(ts_cols)

    clim_cols = {
        "year": 2020,
        "day": range(1, n + 1),
        "time": 0.0,
        "length": 1.0,
        "tair": 15.0,
        "tsoil": 10.0,
        "par": 8.0,
        "precip": 2.0,
        "vpd": 800.0,
        "vpd_soil": 400.0,
        "vpress": 1200.0,
        "wspd": 2.5,
    }

    from pysipnet.climate import ClimateDrivers

    climate = ClimateDrivers.from_dataframe(pd.DataFrame(clim_cols))

    result = MagicMock(spec=SIPNETResult)
    result.timeseries = ts
    result.climate = climate
    result.success = True
    result.returncode = 0
    return result


class TestDashboard:
    def test_returns_figure(self):
        from plotly.graph_objects import Figure

        from pysipnet.viz import dashboard

        fig = dashboard(_make_result())
        assert isinstance(fig, Figure)

    def test_has_four_row_layout(self):
        from pysipnet.viz import dashboard

        fig = dashboard(_make_result())
        # Multiple y-axes means traces are distributed across rows.
        y_axes = {trace.yaxis for trace in fig.data}
        assert len(y_axes) > 1

    def test_flux_traces_present(self):
        from pysipnet.viz import dashboard

        fig = dashboard(_make_result())
        names = {t.name for t in fig.data}
        assert "NEE" in names
        assert "GPP" in names
        assert "ET" in names

    def test_pool_traces_present(self):
        from pysipnet.viz import dashboard

        fig = dashboard(_make_result())
        names = {t.name for t in fig.data}
        assert "Plant Wood C" in names
        assert "Soil C" in names

    def test_missing_column_skipped(self):
        """litter_c absent when LITTER_POOL=0; dashboard should not error."""
        from pysipnet.viz import dashboard

        fig = dashboard(_make_result(include_litter=False))
        names = {t.name for t in fig.data}
        assert "Litter C" not in names

    def test_litter_shown_when_present(self):
        from pysipnet.viz import dashboard

        fig = dashboard(_make_result(include_litter=True))
        names = {t.name for t in fig.data}
        assert "Litter C" in names

    def test_cum_nee_hidden_by_default(self):
        from pysipnet.viz import dashboard

        fig = dashboard(_make_result())
        names = {t.name for t in fig.data}
        assert "Cumulative NEE" not in names

    def test_cum_nee_shown_when_requested(self):
        from pysipnet.viz import dashboard

        fig = dashboard(_make_result(), show_cum_nee=True)
        names = {t.name for t in fig.data}
        assert "Cumulative NEE" in names

    def test_empty_timeseries_raises(self):
        from unittest.mock import MagicMock

        from pysipnet.climate import ClimateDrivers
        from pysipnet.result import SIPNETResult
        from pysipnet.viz import dashboard

        result = MagicMock(spec=SIPNETResult)
        result.timeseries = pd.DataFrame()
        result.climate = ClimateDrivers.from_dataframe(
            pd.DataFrame(
                {
                    "year": [2020],
                    "day": [1],
                    "time": [0.0],
                    "length": [1.0],
                    "tair": [15.0],
                    "tsoil": [10.0],
                    "par": [8.0],
                    "precip": [2.0],
                    "vpd": [800.0],
                    "vpd_soil": [400.0],
                    "vpress": [1200.0],
                    "wspd": [2.5],
                }
            )
        )
        result.returncode = 1

        with pytest.raises(ValueError, match="empty"):
            dashboard(result)
