"""End-to-end integration tests: write inputs → run SIPNET → parse output.

These tests require the compiled SIPNET binary.  They are automatically skipped
when the binary is absent (e.g., in CI without a build step).

Build the binary with::

    make sipnet-standard
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysipnet.runner import ModelPreset, SIPNETRunner

_STANDARD_BINARY = SIPNETRunner(preset=ModelPreset.STANDARD).binary_path

pytestmark = pytest.mark.skipif(
    not _STANDARD_BINARY.exists(),
    reason=f"SIPNET binary not found at {_STANDARD_BINARY}; run 'make sipnet-standard'",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_climate(n_days: int = 30, year: int = 2010, start_doy: int = 150):
    """Return a ClimateDrivers with synthetic summer data."""
    from pysipnet.climate import ClimateDrivers

    rows = []
    for i in range(n_days):
        rows.append(
            {
                "year": year,
                "day": start_doy + i,
                "time": 0.0,
                "length": 1.0,
                "tair": 18.0 + 5.0 * np.sin(np.pi * i / n_days),
                "tsoil": 12.0 + 3.0 * np.sin(np.pi * i / n_days),
                "par": 15.0,
                "precip": 2.0,
                "vpd": 1200.0,
                "vpd_soil": 600.0,
                "vpress": 1500.0,
                "wspd": 2.0,
            }
        )
    df = pd.DataFrame(rows)
    return ClimateDrivers.from_dataframe(df, version="v1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_run_completes(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD, keep_workdir=False)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.success, (
            f"SIPNET exited with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_output_shape(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate(n_days=30)
        result = runner.run(minimal_params, climate)

        assert result.success
        assert len(result.timeseries) == 30
        assert result.timeseries.shape[1] > 10

    def test_key_columns_present(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.success
        for col in ("nee", "gpp", "npp", "evapotranspiration"):
            assert col in result.timeseries.columns, f"Missing column: {col}"

    def test_no_nans_in_output(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.success
        assert not result.timeseries.isnull().any().any(), "NaN values found in output"

    def test_convenience_accessors(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.success
        assert len(result.nee()) == 30
        assert len(result.gpp()) == 30
        assert len(result.et()) == 30

    def test_gpp_non_negative(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.success
        assert (result.gpp() >= 0).all(), "GPP should be non-negative"

    def test_carbon_balance_identity(self, minimal_params):
        """NEE ≈ Rtot − GPP at each timestep.

        SIPNET writes output with limited decimal precision (~2 dp), so we
        allow an absolute tolerance of 0.01 g C m⁻² rather than a relative one.
        """
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.success
        ts = result.timeseries
        computed_nee = ts["rtot"] - ts["gpp"]
        np.testing.assert_allclose(
            ts["nee"].values,
            computed_nee.values,
            atol=0.01,
            err_msg="NEE != Rtot - GPP (beyond output-precision tolerance)",
        )

    def test_clim_roundtrip(self, tmp_path):
        """Writing then reading a v1 climate file returns the same data."""
        from pysipnet.io.clim_io import read_clim_file, write_clim_file

        climate = _make_climate(n_days=10)
        clim_path = tmp_path / "test.clim"
        write_clim_file(climate, clim_path)
        climate2 = read_clim_file(clim_path, version="v1")

        pd.testing.assert_frame_equal(
            climate.data.reset_index(drop=True),
            climate2.data.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )
