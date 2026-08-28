"""End-to-end integration tests: write inputs → run SIPNET → parse output.

These tests require the compiled SIPNET binary.  They are automatically skipped
when the binary is absent (e.g., in CI without a build step).

Build the binary with::

    make sipnet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import SIPNETRunner

_SIPNET_BINARY = SIPNETRunner(flags=ModelFlags.standard()).binary_path

pytestmark = pytest.mark.skipif(
    not _SIPNET_BINARY.exists(),
    reason=f"SIPNET binary not found at {_SIPNET_BINARY}; run 'make sipnet'",
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
    return ClimateDrivers.from_dataframe(df, n_columns=14)


# ---------------------------------------------------------------------------
# Core run tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_run_completes(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard(), keep_workdir=False)
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success, (
            f"SIPNET exited with code {result.provenance.returncode}\n"
            f"stdout: {result.provenance.stdout}\nstderr: {result.provenance.stderr}"
        )

    def test_output_shape(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate(n_days=30)
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert len(result.outputs.data) == 30
        assert result.outputs.data.shape[1] > 10

    def test_key_columns_present(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        for col in ("nee", "gpp", "npp", "evapotranspiration"):
            assert col in result.outputs.data.columns, f"Missing column: {col}"

    def test_no_nans_in_output(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert not result.outputs.data.isnull().any().any(), "NaN values found in output"

    def test_convenience_accessors(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert len(result.nee()) == 30
        assert len(result.gpp()) == 30
        assert len(result.et()) == 30

    def test_gpp_non_negative(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert (result.gpp() >= 0).all(), "GPP should be non-negative"

    def test_carbon_balance_identity(self, minimal_params):
        """NEE ≈ Rtot − GPP at each timestep.

        SIPNET writes output with limited decimal precision (~2 dp), so we
        allow an absolute tolerance of 0.01 g C m⁻² rather than a relative one.
        """
        runner = SIPNETRunner(flags=ModelFlags.standard())
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
        climate2 = read_clim_file(clim_path, n_columns=14)

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

        runner = SIPNETRunner(flags=ModelFlags.standard())
        result = runner.run(minimal_params, _make_climate())

        assert isinstance(result.outputs, SIPNETOutput)
        assert result.outputs.source_path is None
        assert result.outputs._data is not None

    def test_output_dir_creates_file(self, minimal_params, tmp_path):
        """Runner-level output_dir copies sipnet.out before workdir cleanup."""
        output_dir = tmp_path / "outputs"
        runner = SIPNETRunner(
            flags=ModelFlags.standard(),
            output_dir=output_dir,
        )
        result = runner.run(minimal_params, _make_climate(), run_id="test_run")

        expected_file = output_dir / "sipnet_test_run.out"
        assert expected_file.exists(), f"Expected output file not found: {expected_file}"
        assert result.outputs.source_path == expected_file

    def test_lazy_output_not_loaded_until_accessed(self, minimal_params, tmp_path):
        """File-backed SIPNETOutput holds no DataFrame until .data is accessed."""
        runner = SIPNETRunner(
            flags=ModelFlags.standard(),
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

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=runner_dir)
        runner.run(minimal_params, _make_climate(), run_id="override_run", output_dir=call_dir)

        assert (call_dir / "sipnet_override_run.out").exists()
        assert not runner_dir.exists(), "Runner-level dir should not be created when overridden"

    def test_per_call_none_suppresses_runner_output_dir(self, minimal_params, tmp_path):
        """Passing output_dir=None at call time suppresses the runner-level default."""
        runner_dir = tmp_path / "runner_default"
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=runner_dir)
        result = runner.run(minimal_params, _make_climate(), output_dir=None)

        assert result.outputs.source_path is None, "Should be in-memory when output_dir=None"
        assert not runner_dir.exists(), "Runner-level dir should not be created"

    def test_output_dir_inside_workdir_is_refused(self, minimal_params, tmp_path):
        """output_dir inside the workdir is refused before the binary runs.

        The working directory is deleted after the run, so writing output into
        it would delete the output too. The check is exercised directly rather
        than by predicting the path: each run now gets a freshly created
        directory with an unguessable name, which is what stops two concurrent
        runs sharing one.
        """
        runner = SIPNETRunner(flags=ModelFlags.standard(), workdir_base=tmp_path)
        workdir = tmp_path / "sipnet_somerun_abc123"
        workdir.mkdir()

        with pytest.raises(ValueError, match="inside the run's working directory"):
            runner._check_output_dir(workdir / "outputs", workdir)

    def test_output_dir_outside_workdir_is_allowed(self, minimal_params, tmp_path):
        runner = SIPNETRunner(flags=ModelFlags.standard(), workdir_base=tmp_path)
        workdir = tmp_path / "sipnet_somerun_abc123"
        workdir.mkdir()
        runner._check_output_dir(tmp_path / "elsewhere", workdir)  # must not raise

    def test_two_runs_sharing_a_run_id_get_separate_directories(self, minimal_params, tmp_path):
        """The collision that silently swapped results between concurrent runs.

        The working directory used to be named from the run id, so two runs
        with the same id under a shared temp directory wrote to the same path.
        Because a run succeeds by reading whatever sipnet.out it finds, the
        symptom was wrong numbers rather than an error — and the module
        docstring recommends exactly this pattern for ensembles.
        """
        runner = SIPNETRunner(flags=ModelFlags.standard(), workdir_base=tmp_path, keep_workdir=True)
        first = runner.run(minimal_params, _make_climate(), run_id="member")
        second = runner.run(minimal_params, _make_climate(), run_id="member")
        assert first.provenance.workdir != second.provenance.workdir
        assert first.provenance.workdir.exists()
        assert second.provenance.workdir.exists()

    def test_the_run_id_still_labels_the_directory(self, minimal_params, tmp_path):
        """Unpredictable, but still recognisable while debugging."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), workdir_base=tmp_path, keep_workdir=True)
        result = runner.run(minimal_params, _make_climate(), run_id="member42")
        assert result.provenance.workdir.name.startswith("sipnet_member42_")

    def test_a_run_id_that_escapes_the_workdir_base_is_refused(self, minimal_params, tmp_path):
        """A run id becomes part of a path that is deleted afterwards."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), workdir_base=tmp_path)
        with pytest.raises(ValueError, match="Invalid run_id"):
            runner.run(minimal_params, _make_climate(), run_id="../../escape")

    def test_column_selection_returns_subset(self, minimal_params, tmp_path):
        """load(columns=...) returns only the requested columns plus time coords."""
        runner = SIPNETRunner(
            flags=ModelFlags.standard(),
            output_dir=tmp_path / "outputs",
        )
        result = runner.run(minimal_params, _make_climate())

        subset = result.outputs.load(columns=["nee", "gpp"])
        assert set(subset.columns) == {"year", "day", "time", "nee", "gpp"}
        assert len(subset) == 30

    def test_column_selection_memory_backed(self, minimal_params):
        """load(columns=...) works on memory-backed instances too."""
        runner = SIPNETRunner(flags=ModelFlags.standard())
        result = runner.run(minimal_params, _make_climate())

        subset = result.outputs.load(columns=["nee"])
        assert "nee" in subset.columns
        assert "year" in subset.columns
        assert "plant_wood_c" not in subset.columns

    def test_n_timesteps(self, minimal_params, tmp_path):
        """n_timesteps is correct for both memory-backed and file-backed outputs."""
        runner_mem = SIPNETRunner(flags=ModelFlags.standard())
        runner_file = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result_mem = runner_mem.run(minimal_params, _make_climate(n_days=20))
        result_file = runner_file.run(minimal_params, _make_climate(n_days=20))

        assert result_mem.outputs.n_timesteps == 20
        assert result_file.outputs.n_timesteps == 20


# ---------------------------------------------------------------------------
# Litter pool
# ---------------------------------------------------------------------------


class TestLitterPool:
    """The litter pool must actually work when switched on.

    This configuration was unusable before the SIPNET v2.1.0 pin. The old
    pinned source did not compile with the litter pool enabled, and the commit
    that made it compile still left soil respiration unassigned on that code
    path, so ``rSoil`` stayed at zero: soil carbon accumulated without ever
    respiring, and both heterotrophic respiration and NEE were wrong.

    Nothing about that failure was loud. The run succeeded and the output
    looked plausible, which is exactly why it is worth asserting on directly.
    """

    @pytest.fixture
    def litter_params(self, minimal_params):
        """Parameters with the two values the litter pool requires."""
        data = minimal_params.model_dump()
        data["respiration"]["litter_breakdown_rate"] = 0.5
        data["respiration"]["frac_litter_respired"] = 0.5
        return type(minimal_params).model_validate(data)

    def test_run_succeeds(self, litter_params):
        result = SIPNETRunner(flags=ModelFlags.forest()).run(litter_params, _make_climate())
        assert result.provenance.success, result.provenance.stderr

    def test_soil_respiration_is_not_zero(self, litter_params):
        """The specific regression: rSoil must be computed, not left at zero."""
        result = SIPNETRunner(flags=ModelFlags.forest()).run(litter_params, _make_climate())
        r_soil = result.outputs.data["r_soil"]
        assert (r_soil > 0).any(), (
            "soil respiration is zero for every timestep with the litter pool on, "
            "which is the pre-v2.1.0 defect this test exists to catch"
        )

    def test_litter_pool_holds_carbon(self, litter_params):
        """With the pool on, litter carbon should be tracked rather than left at zero."""
        result = SIPNETRunner(flags=ModelFlags.forest()).run(litter_params, _make_climate())
        assert (result.outputs.data["litter_c"] > 0).any()

    def test_litter_pool_stays_empty_when_switched_off(self, minimal_params):
        """The complement: SIPNET writes the column but leaves it at zero."""
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, _make_climate())
        assert (result.outputs.data["litter_c"] == 0).all()

    def test_enabling_the_litter_pool_changes_the_answer(self, litter_params):
        """A flag that reaches SIPNET must visibly affect the model.

        If the flag were silently dropped from sipnet.in, the two runs would
        agree and every other test here would still pass.
        """
        climate = _make_climate()
        with_pool = SIPNETRunner(flags=ModelFlags.forest()).run(litter_params, climate)
        without = SIPNETRunner(flags=ModelFlags.standard()).run(litter_params, climate)
        assert not np.allclose(
            with_pool.outputs.data["nee"].to_numpy(),
            without.outputs.data["nee"].to_numpy(),
        ), "turning the litter pool on made no difference to NEE"


# ---------------------------------------------------------------------------
# Mass balance
# ---------------------------------------------------------------------------


class TestMassBalance:
    """SIPNET checks its own carbon and nitrogen closure; read its verdict.

    At v2.1.0 the closure errors were output columns (``bcdeltaC`` /
    ``bcdeltaN``) and this asserted they stayed near zero. The pinned version
    reports them as log warnings from ``checkBalance()`` in ``balance.c``
    instead, so the test reads stderr.

    Reading SIPNET's verdict is better than re-deriving the balance ourselves:
    it is the model's own accounting, and a change to how the model accounts
    for carbon shows up here rather than needing our arithmetic updated to
    match.
    """

    def test_carbon_balance_does_not_warn(self, minimal_params):
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, _make_climate())
        log = (result.provenance.stdout or "") + (result.provenance.stderr or "")
        failures = [ln for ln in log.splitlines() if "Carbon balance check failed" in ln]
        assert not failures, "SIPNET reported a carbon imbalance:\n" + "\n".join(failures[:5])

    def test_nitrogen_balance_does_not_warn(self, minimal_params):
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, _make_climate())
        log = (result.provenance.stdout or "") + (result.provenance.stderr or "")
        failures = [ln for ln in log.splitlines() if "Nitrogen balance check failed" in ln]
        assert not failures, "SIPNET reported a nitrogen imbalance:\n" + "\n".join(failures[:5])

    def test_the_balance_check_actually_runs(self, minimal_params):
        """Guard against the check being silently compiled out or skipped.

        These two tests pass trivially if SIPNET never checks anything, so
        confirm the machinery is present in the pinned source.
        """
        from pathlib import Path as _Path

        balance_c = _Path(__file__).parent.parent / "sipnet" / "src" / "sipnet" / "balance.c"
        if not balance_c.exists():
            pytest.skip("SIPNET submodule not populated")
        text = balance_c.read_text()
        assert "Carbon balance check failed" in text
        assert "Nitrogen balance check failed" in text

    def test_the_old_output_columns_are_really_gone(self, minimal_params):
        """Documents why this class reads the log rather than the output."""
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, _make_climate())
        columns = set(result.outputs.data.columns)
        assert "balance_delta_c" not in columns
        assert "balance_delta_n" not in columns


# ---------------------------------------------------------------------------
# Events through the runner
# ---------------------------------------------------------------------------


class TestRunnerAppliesEvents:
    """The runner's own event plumbing, which nothing else exercises.

    `test_events_contract.py` writes `events.in` by hand and invokes the binary
    directly, so it proves the file format but never that `SIPNETRunner.run`
    writes that file, or that it sets `EVENTS` correctly. Removing the runner's
    entire event block leaves the rest of the suite green.
    """

    @pytest.fixture
    def tillage(self):
        from pysipnet.events import EventSequence, TillageEvent

        # Day 160 falls inside the synthetic climate built by _make_climate.
        return EventSequence(events=[TillageEvent(year=2010, day=160, tillage_effect=0.4)])

    def test_events_reach_sipnet(self, minimal_params, tillage):
        """SIPNET echoes what it applied; the value must be the one we passed."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), keep_workdir=True)
        result = runner.run(minimal_params, _make_climate(), events=tillage)
        assert result.provenance.success, result.provenance.stderr

        events_out = Path(result.provenance.workdir) / "events.out"
        assert events_out.exists(), "the runner did not produce events.out"
        text = events_out.read_text()
        assert "till" in text, f"SIPNET applied no tillage event:\n{text}"
        assert "0.4" in text, f"SIPNET applied a different value:\n{text}"

    def test_events_file_is_written_into_the_workdir(self, minimal_params, tillage):
        runner = SIPNETRunner(flags=ModelFlags.standard(), keep_workdir=True)
        result = runner.run(minimal_params, _make_climate(), events=tillage)
        written = (Path(result.provenance.workdir) / "events.in").read_text()
        assert written.split() == ["2010", "160", "till", "0.4"]

    def test_events_change_the_result(self, minimal_params, tillage):
        """If the runner silently dropped events, this is what would notice."""
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        with_events = runner.run(minimal_params, climate, events=tillage)
        without = runner.run(minimal_params, climate)
        assert not np.allclose(
            with_events.outputs.data["nee"].to_numpy(),
            without.outputs.data["nee"].to_numpy(),
        ), "applying a tillage event made no difference to NEE"

    def test_no_events_means_events_are_switched_off(self, minimal_params):
        """A stale events.in in the working directory must not be picked up."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), keep_workdir=True)
        result = runner.run(minimal_params, _make_climate())
        config = (Path(result.provenance.workdir) / "sipnet.in").read_text()
        assert "EVENTS = 0" in config
        assert not (Path(result.provenance.workdir) / "events.in").exists()


class TestFailedRunsRaise:
    """A failed run must not come back looking like a successful empty one.

    Returning an empty DataFrame deferred the failure to whatever the caller
    did next — usually ``result.nee()``, which raises KeyError a long way from
    the cause, while SIPNET's own explanation sat unread on the provenance
    object. In an ensemble the empties were collected silently.
    """

    @pytest.fixture
    def broken_climate(self):
        """A climate frame SIPNET refuses: no rows to read."""
        from pysipnet.climate import ClimateDrivers

        return ClimateDrivers.from_dataframe(_make_climate().data.head(0).copy())

    def test_a_failed_run_raises(self, minimal_params, broken_climate):
        from pysipnet.runner import SIPNETRunError

        with pytest.raises(SIPNETRunError):
            SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, broken_climate)

    def test_the_error_carries_sipnets_own_explanation(self, minimal_params, broken_climate):
        """The reason lives in SIPNET's output; the exception must surface it."""
        from pysipnet.runner import SIPNETRunError

        with pytest.raises(SIPNETRunError) as exc:
            SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, broken_climate)
        assert exc.value.returncode != 0
        combined = exc.value.stdout + exc.value.stderr
        assert combined.strip(), "the error carried neither stdout nor stderr"
        assert "climate" in str(exc.value).lower()

    def test_check_false_returns_a_result_instead(self, minimal_params, broken_climate):
        """Opt-out for ensembles, where one failure should not stop the rest."""
        result = SIPNETRunner(flags=ModelFlags.standard()).run(
            minimal_params, broken_climate, check=False
        )
        assert result.provenance.success is False
        assert result.outputs.data.empty

    def test_a_successful_run_is_unaffected(self, minimal_params):
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, _make_climate())
        assert result.provenance.success
        assert not result.outputs.data.empty
