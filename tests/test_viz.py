"""Tests for pysipnet.viz.dashboard."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")


def _make_result(include_litter: bool = False):
    """Return a minimal SIPNETResult with synthetic outputs and climate."""
    from pysipnet.output import SIPNETOutput
    from pysipnet.result import RunProvenance, SIPNETResult

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

    provenance = MagicMock(spec=RunProvenance)
    provenance.returncode = 0

    result = MagicMock(spec=SIPNETResult)
    # Wrap the frame the way a real result does. Assigning a bare
    # DataFrame here is what let dashboard() ship broken: the mock
    # encoded the contract from before SIPNETOutput existed, so these
    # tests certified an API the code no longer had.
    result.outputs = SIPNETOutput.from_dataframe(ts)
    result.climate = climate
    result.provenance = provenance
    return result


class TestDashboard:
    def test_returns_figure(self):
        from plotly.graph_objects import Figure

        from pysipnet.viz import dashboard

        fig = dashboard(_make_result())
        assert isinstance(fig, Figure)

    def test_has_four_row_layout(self):
        import plotly.graph_objects as go

        from pysipnet.viz import dashboard

        fig = dashboard(_make_result())
        # Table traces don't live on cartesian axes; filter to Scatter only.
        scatter = [t for t in fig.data if isinstance(t, go.Scatter)]
        y_axes = {t.yaxis for t in scatter}
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
        assert "Wood C (stem)" in names
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

    def test_empty_outputs_raises(self):
        from pysipnet.climate import ClimateDrivers
        from pysipnet.output import SIPNETOutput
        from pysipnet.result import RunProvenance, SIPNETResult
        from pysipnet.viz import dashboard

        provenance = MagicMock(spec=RunProvenance)
        provenance.returncode = 1

        result = MagicMock(spec=SIPNETResult)
        result.outputs = SIPNETOutput.from_dataframe(pd.DataFrame())
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
        result.provenance = provenance

        with pytest.raises(ValueError, match="empty"):
            dashboard(result)


class TestDashboardOnARealResult:
    """Exercise dashboard() against an actual run, not a mock.

    Every other test here builds its result with MagicMock. That is what let
    dashboard() ship broken: the mock assigned a bare DataFrame to
    ``result.outputs``, encoding the contract from before SIPNETOutput existed,
    so the tests passed while every real result raised AttributeError. A mock
    can only ever assert the shape the author believed in.
    """

    def test_dashboard_accepts_a_real_result(self, minimal_params):
        import warnings
        from pathlib import Path

        from pysipnet.build import binary_path
        from pysipnet.climate import ClimateDrivers
        from pysipnet.io.clim_io import read_clim_file
        from pysipnet.parameters.model import ModelFlags
        from pysipnet.runner import SIPNETRunner
        from pysipnet.viz import dashboard

        if not binary_path().exists():
            pytest.skip("SIPNET binary not built; run 'make sipnet'")

        reference = Path(__file__).parent / "fixtures" / "niwot_reference" / "sipnet.clim"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = read_clim_file(reference, n_columns=14)
        climate = ClimateDrivers.from_dataframe(full.data.head(40).copy(), n_columns=14)

        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, climate)
        figure = dashboard(result)
        assert len(figure.data) > 0, "dashboard produced no traces"
