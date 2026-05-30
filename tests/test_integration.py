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
# Core run tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_run_completes(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD, keep_workdir=False)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success, (
            f"SIPNET exited with code {result.provenance.returncode}\n"
            f"stdout: {result.provenance.stdout}\nstderr: {result.provenance.stderr}"
        )

    def test_output_shape(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate(n_days=30)
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert len(result.outputs.data) == 30
        assert result.outputs.data.shape[1] > 10

    def test_key_columns_present(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        for col in ("nee", "gpp", "npp", "evapotranspiration"):
            assert col in result.outputs.data.columns, f"Missing column: {col}"

    def test_no_nans_in_output(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert not result.outputs.data.isnull().any().any(), "NaN values found in output"

    def test_convenience_accessors(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert len(result.nee()) == 30
        assert len(result.gpp()) == 30
        assert len(result.et()) == 30

    def test_gpp_non_negative(self, minimal_params):
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert (result.gpp() >= 0).all(), "GPP should be non-negative"

    def test_carbon_balance_identity(self, minimal_params):
        """NEE ≈ Rtot − GPP at each timestep.

        SIPNET writes output with limited decimal precision (~2 dp), so we
        allow an absolute tolerance of 0.01 g C m⁻² rather than a relative one.
        """
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        ts = result.outputs.data
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


# ---------------------------------------------------------------------------
# Output I/O tests
# ---------------------------------------------------------------------------


class TestOutputIO:
    def test_eager_output_is_memory_backed(self, minimal_params):
        """Default run (no output_dir) returns a memory-backed SIPNETOutput."""
        from pysipnet.output import SIPNETOutput

        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        result = runner.run(minimal_params, _make_climate())

        assert isinstance(result.outputs, SIPNETOutput)
        assert result.outputs.source_path is None
        assert result.outputs._data is not None

    def test_output_dir_creates_file(self, minimal_params, tmp_path):
        """Runner-level output_dir copies sipnet.out before workdir cleanup."""
        output_dir = tmp_path / "outputs"
        runner = SIPNETRunner(
            preset=ModelPreset.STANDARD,
            output_dir=output_dir,
        )
        result = runner.run(minimal_params, _make_climate(), run_id="test_run")

        expected_file = output_dir / "sipnet_test_run.out"
        assert expected_file.exists(), f"Expected output file not found: {expected_file}"
        assert result.outputs.source_path == expected_file

    def test_lazy_output_not_loaded_until_accessed(self, minimal_params, tmp_path):
        """File-backed SIPNETOutput holds no DataFrame until .data is accessed."""
        runner = SIPNETRunner(
            preset=ModelPreset.STANDARD,
            output_dir=tmp_path / "outputs",
        )
        result = runner.run(minimal_params, _make_climate())

        assert result.outputs._data is None, "Data should not be loaded before first access"
        df = result.outputs.data
        assert df is not None
        assert len(df) == 30

    def test_per_call_output_dir_overrides_runner_default(self, minimal_params, tmp_path):
        """Per-call output_dir takes precedence over the runner-level default."""
        runner_dir = tmp_path / "runner_default"
        call_dir = tmp_path / "call_override"

        runner = SIPNETRunner(preset=ModelPreset.STANDARD, output_dir=runner_dir)
        runner.run(minimal_params, _make_climate(), run_id="override_run", output_dir=call_dir)

        assert (call_dir / "sipnet_override_run.out").exists()
        assert not runner_dir.exists(), "Runner-level dir should not be created when overridden"

    def test_per_call_none_suppresses_runner_output_dir(self, minimal_params, tmp_path):
        """Passing output_dir=None at call time suppresses the runner-level default."""
        runner_dir = tmp_path / "runner_default"
        runner = SIPNETRunner(preset=ModelPreset.STANDARD, output_dir=runner_dir)
        result = runner.run(minimal_params, _make_climate(), output_dir=None)

        assert result.outputs.source_path is None, "Should be in-memory when output_dir=None"
        assert not runner_dir.exists(), "Runner-level dir should not be created"

    def test_output_dir_inside_workdir_raises_before_run(self, minimal_params, tmp_path):
        """output_dir inside the workdir raises ValueError before the binary runs."""
        workdir_base = tmp_path / "workdirs"
        runner = SIPNETRunner(
            preset=ModelPreset.STANDARD,
            workdir_base=workdir_base,
        )
        # The workdir will be workdir_base/sipnet_myrun — so a subdir of that is invalid.
        bad_output_dir = workdir_base / "sipnet_myrun" / "outputs"

        with pytest.raises(ValueError, match="inside the run's working directory"):
            runner.run(minimal_params, _make_climate(), run_id="myrun", output_dir=bad_output_dir)

    def test_column_selection_returns_subset(self, minimal_params, tmp_path):
        """load(columns=...) returns only the requested columns plus time coords."""
        runner = SIPNETRunner(
            preset=ModelPreset.STANDARD,
            output_dir=tmp_path / "outputs",
        )
        result = runner.run(minimal_params, _make_climate())

        subset = result.outputs.load(columns=["nee", "gpp"])
        assert set(subset.columns) == {"year", "day", "time", "nee", "gpp"}
        assert len(subset) == 30

    def test_column_selection_memory_backed(self, minimal_params):
        """load(columns=...) works on memory-backed instances too."""
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        result = runner.run(minimal_params, _make_climate())

        subset = result.outputs.load(columns=["nee"])
        assert "nee" in subset.columns
        assert "year" in subset.columns
        assert "plant_wood_c" not in subset.columns

    def test_n_timesteps(self, minimal_params, tmp_path):
        """n_timesteps is correct for both memory-backed and file-backed outputs."""
        runner_mem = SIPNETRunner(preset=ModelPreset.STANDARD)
        runner_file = SIPNETRunner(preset=ModelPreset.STANDARD, output_dir=tmp_path / "outputs")
        result_mem = runner_mem.run(minimal_params, _make_climate(n_days=20))
        result_file = runner_file.run(minimal_params, _make_climate(n_days=20))

        assert result_mem.outputs.n_timesteps == 20
        assert result_file.outputs.n_timesteps == 20
