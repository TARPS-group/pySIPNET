"""End-to-end integration tests: write inputs → run SIPNET → parse output.

These tests require the compiled SIPNET binary.  They are automatically skipped
when the binary is absent (e.g., in CI without a build step).

Build the binary with::

    make sipnet
"""

from __future__ import annotations

import subprocess
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
                "day_of_year": start_doy + i,
                "hour_of_day": 0.0,
                "time_step_length": 1.0,
                "air_temperature": 18.0 + 5.0 * np.sin(np.pi * i / n_days),
                "soil_temperature": 12.0 + 3.0 * np.sin(np.pi * i / n_days),
                "photosynthetically_active_radiation": 15.0,
                "precipitation": 2.0,
                "vapor_pressure_deficit": 1200.0,
                "soil_vapor_pressure_deficit": 600.0,
                "vapor_pressure": 1500.0,
                "wind_speed": 2.0,
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
        assert len(result.outputs.pandas) == 30
        assert result.outputs.pandas.shape[1] > 10

    def test_key_columns_present(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        for col in (
            "net_ecosystem_exchange",
            "gross_primary_production",
            "net_primary_production",
            "evapotranspiration",
        ):
            assert col in result.outputs.pandas.columns, f"Missing column: {col}"

    def test_no_nans_in_output(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert not result.outputs.pandas.isnull().any().any(), "NaN values found in output"

    def test_variable_lookup_by_alias(self, minimal_params):
        """``outputs["nee"]`` and the full name are the same array, with metadata."""
        runner = SIPNETRunner(flags=ModelFlags.standard())
        result = runner.run(minimal_params, _make_climate())

        by_alias = result.outputs["nee"]
        by_name = result.outputs["net_ecosystem_exchange"]
        assert by_alias.name == "net_ecosystem_exchange"
        assert by_alias.attrs["units"] == "g m-2"
        assert by_alias.attrs["constituent"] == "C"
        np.testing.assert_array_equal(by_alias.values, by_name.values)
        assert result.outputs["gpp"].sizes["time"] == 30

    def test_gpp_non_negative(self, minimal_params):
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        assert bool((result.outputs["gpp"] >= 0).all()), "GPP should be non-negative"

    def test_dataset_has_one_time_dimension_with_step_bounds(self, minimal_params):
        """The xarray view is 1-D in time, dense, and knows when each step ends."""
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        ds = result.outputs.xarray
        assert dict(ds.sizes) == {"time": 30, "bounds": 2}
        assert ds["net_ecosystem_exchange"].dims == ("time",)
        assert not ds["net_ecosystem_exchange"].isnull().any()
        assert ds["time"].attrs["long_name"] == "End of timestep"
        expected_end = (
            ds["time_step_start"].values
            + pd.to_timedelta(climate.pandas["time_step_length"].to_numpy(), unit="D").to_numpy()
        )
        np.testing.assert_array_equal(ds["time"].values, expected_end)
        assert ds["wood_carbon"].attrs["time_reference"] == "value at the end of the timestep"
        assert ds["net_ecosystem_exchange"].attrs["time_reference"] == "total over the timestep"

    def test_dataset_records_which_run_produced_it(self, minimal_params):
        """A prediction saved to disk should say which run and which flags made it."""
        import json

        runner = SIPNETRunner(flags=ModelFlags.standard())
        result = runner.run(minimal_params, _make_climate(), run_id="calibration_042")

        attrs = result.outputs.xarray.attrs
        assert attrs["run_id"] == "calibration_042" == result.provenance.run_id
        assert json.loads(attrs["model_flags"]) == result.flags.model_dump()
        assert attrs["time_step_length_source"] == "climate drivers"
        assert attrs["time_zone"].startswith("naive")
        assert result.outputs.xarray["time"].attrs["time_zone"] == attrs["time_zone"]

    def test_a_file_backed_dataset_records_its_run_too(self, minimal_params, tmp_path):
        """The provenance has to survive the other construction path as well."""
        import json

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate(), run_id="member_7")

        assert result.outputs.source_path is not None
        attrs = result.outputs[["nee"]].attrs
        assert attrs["run_id"] == "member_7"
        assert json.loads(attrs["model_flags"])["litter_pool"] is False

    def test_a_file_backed_output_refuses_a_flag_zeroed_variable_without_reading(
        self, minimal_params, tmp_path
    ):
        """The flags alone settle it; the file should not be touched to find out."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate())

        with pytest.raises(ValueError, match="litter_pool"):
            result.outputs.select(["litter_carbon"], format="pandas")
        with pytest.raises(ValueError, match="litter_pool"):
            result.outputs[["litter_carbon"]]
        assert result.outputs._frame is None, "refusing a variable must not read the file"

    def test_selecting_a_time_coordinate_is_not_a_duplicate_column(self, minimal_params, tmp_path):
        """'time' and 'day' are documented aliases of the time coordinates."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate())

        frame = result.outputs.select(["year"], format="pandas")
        assert list(frame.columns) == ["year", "day_of_year", "hour_of_day"]
        assert result.outputs["time"].sizes["time"] == 30
        assert set(result.outputs[["nee", "day"]].data_vars) == {"net_ecosystem_exchange"}

    def test_an_empty_selection_reads_only_the_time_coordinates(self, minimal_params, tmp_path):
        """A programmatically built variable list can legitimately be empty."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate())

        assert list(result.outputs.select([], format="pandas").columns) == [
            "year",
            "day_of_year",
            "hour_of_day",
        ]
        assert result.outputs[[]].sizes["time"] == 30

    def test_a_variable_the_file_does_not_contain_is_reported_once(
        self, minimal_params, tmp_path, monkeypatch
    ):
        """A registry name absent from this file must not re-read the file forever."""
        import pysipnet.io.output_reader as output_reader

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate())

        reads = []
        real_read = output_reader.read_output_file

        def counting_read(path, variables=None):
            reads.append(variables)
            return real_read(path, variables=variables)

        monkeypatch.setattr(output_reader, "read_output_file", counting_read)

        # bcdeltaC is a v2.1.0 column the registry still maps; this file has none.
        with pytest.raises(KeyError, match="are not in"):
            result.outputs.select(["bcdeltaC"], format="pandas")
        assert len(reads) == 1

    def test_carbon_balance_identity(self, minimal_params):
        """NEE ≈ Rtot − GPP at each timestep.

        SIPNET writes output with limited decimal precision (~2 dp), so we
        allow an absolute tolerance of 0.01 g C m⁻² rather than a relative one.
        """
        runner = SIPNETRunner(flags=ModelFlags.standard())
        climate = _make_climate()
        result = runner.run(minimal_params, climate)

        assert result.provenance.success
        ts = result.outputs.pandas
        computed_nee = ts["ecosystem_respiration"] - ts["gross_primary_production"]
        np.testing.assert_allclose(
            ts["net_ecosystem_exchange"].values,
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
            climate.pandas.reset_index(drop=True),
            climate2.pandas.reset_index(drop=True),
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
        assert result.outputs._frame is not None

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
        """File-backed SIPNETOutput holds no DataFrame until .pandas is accessed."""
        runner = SIPNETRunner(
            flags=ModelFlags.standard(),
            output_dir=tmp_path / "outputs",
        )
        result = runner.run(minimal_params, _make_climate())

        assert result.outputs._frame is None, "Data should not be loaded before first access"
        df = result.outputs.pandas
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

    def test_a_second_run_will_not_overwrite_the_first_ones_output(self, minimal_params, tmp_path):
        """The same collision one level out, where a random suffix is not available.

        The output file is named from the run id because users are told to
        predict that name. A second run with the same id would replace it, and
        because a file-backed output is read lazily, the first result would then
        answer with the second run's numbers while still reporting success.
        """
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        first = runner.run(minimal_params, _make_climate(n_days=30), run_id="member")

        with pytest.raises(FileExistsError, match="distinct run_id"):
            runner.run(minimal_params, _make_climate(n_days=20), run_id="member")

        # The refusal must leave the first run's output exactly as it was.
        assert first.outputs.n_timesteps == 30

    def test_a_refused_run_does_not_execute_sipnet_or_leave_a_workdir(
        self, minimal_params, tmp_path, monkeypatch
    ):
        """Refusing before the binary runs is the point: a wasted run is not free."""
        runner = SIPNETRunner(
            flags=ModelFlags.standard(),
            output_dir=tmp_path / "outputs",
            workdir_base=tmp_path / "work",
        )
        runner.run(minimal_params, _make_climate(), run_id="member")
        before = sorted((tmp_path / "work").iterdir())

        executed = []
        real_run = subprocess.run
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: (executed.append(a), real_run(*a, **k))[1]
        )

        with pytest.raises(FileExistsError):
            runner.run(minimal_params, _make_climate(), run_id="member")
        assert executed == [], "SIPNET ran for a result that was never going to be kept"
        assert sorted((tmp_path / "work").iterdir()) == before

    def test_overwrite_allows_rerunning_one_id(self, minimal_params, tmp_path):
        """A loop that reruns one member under a fixed id is a legitimate thing to do."""
        runner = SIPNETRunner(
            flags=ModelFlags.standard(), output_dir=tmp_path / "outputs", overwrite=True
        )
        runner.run(minimal_params, _make_climate(n_days=30), run_id="member")
        second = runner.run(minimal_params, _make_climate(n_days=20), run_id="member")

        assert second.outputs.n_timesteps == 20
        assert len(list((tmp_path / "outputs").iterdir())) == 1

    def test_overwrite_can_be_decided_per_call(self, minimal_params, tmp_path):
        """Both directions: the call decides, whatever the runner's default is."""
        permissive = SIPNETRunner(
            flags=ModelFlags.standard(), output_dir=tmp_path / "outputs", overwrite=True
        )
        strict = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")

        strict.run(minimal_params, _make_climate(), run_id="member")
        assert (
            strict.run(
                minimal_params, _make_climate(n_days=20), run_id="member", overwrite=True
            ).outputs.n_timesteps
            == 20
        )
        with pytest.raises(FileExistsError):
            permissive.run(minimal_params, _make_climate(), run_id="member", overwrite=False)

    def test_a_failed_run_does_not_block_the_retry(self, minimal_params, tmp_path):
        """Nothing is written for a run that failed, so its id is still free."""
        from pysipnet.climate import ClimateDrivers

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        empty = ClimateDrivers.from_dataframe(_make_climate().pandas.head(0).copy())

        failed = runner.run(minimal_params, empty, run_id="member", check=False)
        assert failed.provenance.success is False
        assert list((tmp_path / "outputs").iterdir()) == []

        retried = runner.run(minimal_params, _make_climate(), run_id="member")
        assert retried.provenance.success

    def test_two_concurrent_runs_sharing_an_id_cannot_swap_results(self, minimal_params, tmp_path):
        """The pre-run check cannot see a run that has not finished yet.

        Both runs pass it, then both copy. Claiming the name and creating the
        file have to be one operation, or the loser's result reads the winner's
        numbers while reporting success.
        """
        import threading

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        finished: dict[str, object] = {}
        refused: list[str] = []
        lengths = {"long": 30, "short": 12}

        def go(tag: str) -> None:
            try:
                finished[tag] = runner.run(
                    minimal_params, _make_climate(n_days=lengths[tag]), run_id="member"
                )
            except FileExistsError:
                refused.append(tag)

        threads = [threading.Thread(target=go, args=(tag,)) for tag in lengths]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(refused) == 1, "one of the two runs had to be turned away"
        for tag, result in finished.items():
            assert result.outputs.n_timesteps == lengths[tag], (
                f"the {tag} run's result read the other run's numbers"
            )

    def test_a_dangling_symlink_does_not_let_a_run_write_outside_output_dir(
        self, minimal_params, tmp_path
    ):
        """Path.exists() follows links, so a broken one looks like free space."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        elsewhere = tmp_path / "elsewhere.out"
        (output_dir / "sipnet_member.out").symlink_to(elsewhere)

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=output_dir)
        with pytest.raises(FileExistsError):
            runner.run(minimal_params, _make_climate(), run_id="member")
        assert not elsewhere.exists()

    def test_a_stored_output_path_survives_a_change_of_directory(
        self, minimal_params, tmp_path, monkeypatch
    ):
        """An ensemble scheduler may chdir per task; a relative path would then move."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        monkeypatch.chdir(tmp_path / "a")

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir="outputs")
        result = runner.run(minimal_params, _make_climate(), run_id="member")
        assert result.outputs.source_path.is_absolute()

        monkeypatch.chdir(tmp_path / "b")
        with pytest.raises(FileExistsError):
            runner.run(minimal_params, _make_climate(n_days=20), run_id="member")
        assert result.outputs.n_timesteps == 30

    def test_distinct_run_ids_get_distinct_files(self, minimal_params, tmp_path):
        """Including the default, which is a fresh UUID every time.

        Ids that differ only in case are a special case: on a case-insensitive
        filesystem they name one file, and the run is refused rather than
        silently sharing it.
        """
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        results = [runner.run(minimal_params, _make_climate()) for _ in range(3)]

        paths = {r.outputs.source_path for r in results}
        assert len(paths) == 3

    def test_the_run_id_still_labels_the_directory(self, minimal_params, tmp_path):
        """Unpredictable, but still recognizable while debugging."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), workdir_base=tmp_path, keep_workdir=True)
        result = runner.run(minimal_params, _make_climate(), run_id="member42")
        assert result.provenance.workdir.name.startswith("sipnet_member42_")

    def test_a_run_id_that_escapes_the_workdir_base_is_refused(self, minimal_params, tmp_path):
        """A run id becomes part of a path that is deleted afterwards."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), workdir_base=tmp_path)
        with pytest.raises(ValueError, match="Invalid run_id"):
            runner.run(minimal_params, _make_climate(), run_id="../../escape")

    def test_column_selection_returns_subset(self, minimal_params, tmp_path):
        """dataframe(...) returns only the requested variables plus time coords."""
        runner = SIPNETRunner(
            flags=ModelFlags.standard(),
            output_dir=tmp_path / "outputs",
        )
        result = runner.run(minimal_params, _make_climate())

        subset = result.outputs.select(["nee", "gpp"], format="pandas")
        assert list(subset.columns) == [
            "year",
            "day_of_year",
            "hour_of_day",
            "net_ecosystem_exchange",
            "gross_primary_production",
        ]
        assert len(subset) == 30

    def test_column_selection_memory_backed(self, minimal_params):
        """dataframe(...) works on memory-backed instances too."""
        runner = SIPNETRunner(flags=ModelFlags.standard())
        result = runner.run(minimal_params, _make_climate())

        subset = result.outputs.select(["nee"], format="pandas")
        assert "net_ecosystem_exchange" in subset.columns
        assert "year" in subset.columns
        assert "wood_carbon" not in subset.columns

    def test_variable_selection_as_xarray(self, minimal_params, tmp_path):
        """dataset(...) reads only the requested variables into a Dataset."""
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate())

        ds = result.outputs[["nee"]]
        assert set(ds.data_vars) == {"net_ecosystem_exchange"}
        assert "time_step_start" in ds.coords

    def test_getitem_with_a_list_gives_a_dataset(self, minimal_params, tmp_path):
        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate())

        ds = result.outputs[["nee", "gpp"]]
        assert set(ds.data_vars) == {"net_ecosystem_exchange", "gross_primary_production"}
        np.testing.assert_array_equal(
            ds["net_ecosystem_exchange"].to_numpy(), result.outputs["nee"].to_numpy()
        )

    def test_every_column_is_read_at_most_once(self, minimal_params, tmp_path, monkeypatch):
        """A likelihood over several variables must not re-read the file per variable.

        This is the guarantee that matters for calibration loops, and it cannot
        be observed from the returned values — only by counting reads.
        """
        import pysipnet.io.output_reader as output_reader

        runner = SIPNETRunner(flags=ModelFlags.standard(), output_dir=tmp_path / "outputs")
        result = runner.run(minimal_params, _make_climate())

        requested: list[list[str] | None] = []
        real_read = output_reader.read_output_file

        def counting_read(path, variables=None):
            requested.append(variables)
            return real_read(path, variables=variables)

        monkeypatch.setattr(output_reader, "read_output_file", counting_read)

        out = result.outputs
        assert out["nee"].sizes["time"] == 30
        assert out["nee"].sizes["time"] == 30
        assert set(out[["nee", "gpp"]].data_vars) == {
            "net_ecosystem_exchange",
            "gross_primary_production",
        }
        assert not out.select(["soil_respiration", "gpp"], format="pandas").empty

        read_columns = [c for call in requested for c in (call or [])]
        assert len(read_columns) == len(set(read_columns)), (
            f"a column was read more than once: {requested}"
        )
        assert set(read_columns) == {
            "year",
            "day_of_year",
            "hour_of_day",
            "net_ecosystem_exchange",
            "gross_primary_production",
            "soil_respiration",
        }

        # The documented exception: a whole-file view re-reads, because only the
        # file states the order its columns belong in. Selections after it do not.
        assert len(out.pandas.columns) > 30
        assert requested[-1] is None
        before = len(requested)
        assert out["nee"].sizes["time"] == 30
        assert not out.select(["gpp", "soil_respiration"], format="pandas").empty
        assert len(requested) == before, f"a cached column was re-read: {requested[before:]}"

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
        data["respiration"]["litter_respired_fraction"] = 0.5
        return type(minimal_params).model_validate(data)

    def test_run_succeeds(self, litter_params):
        result = SIPNETRunner(flags=ModelFlags(litter_pool=True)).run(
            litter_params, _make_climate()
        )
        assert result.provenance.success, result.provenance.stderr

    def test_soil_respiration_is_not_zero(self, litter_params):
        """The specific regression: rSoil must be computed, not left at zero."""
        result = SIPNETRunner(flags=ModelFlags(litter_pool=True)).run(
            litter_params, _make_climate()
        )
        r_soil = result.outputs.pandas["soil_respiration"]
        assert (r_soil > 0).any(), (
            "soil respiration is zero for every timestep with the litter pool on, "
            "which is the pre-v2.1.0 defect this test exists to catch"
        )

    def test_litter_pool_holds_carbon(self, litter_params):
        """With the pool on, litter carbon should be tracked rather than left at zero."""
        result = SIPNETRunner(flags=ModelFlags(litter_pool=True)).run(
            litter_params, _make_climate()
        )
        assert (result.outputs.pandas["litter_carbon"] > 0).any()

    def test_litter_pool_stays_empty_when_switched_off(self, minimal_params):
        """The complement: SIPNET writes the column but leaves it at zero."""
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, _make_climate())
        assert (result.outputs.pandas["litter_carbon"] == 0).all()

    def test_enabling_the_litter_pool_changes_the_answer(self, litter_params):
        """A flag that reaches SIPNET must visibly affect the model.

        If the flag were silently dropped from sipnet.in, the two runs would
        agree and every other test here would still pass.
        """
        climate = _make_climate()
        with_pool = SIPNETRunner(flags=ModelFlags(litter_pool=True)).run(litter_params, climate)
        without = SIPNETRunner(flags=ModelFlags.standard()).run(litter_params, climate)
        assert not np.allclose(
            with_pool.outputs["nee"].to_numpy(),
            without.outputs["nee"].to_numpy(),
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
        columns = set(result.outputs.pandas.columns)
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
            with_events.outputs["nee"].to_numpy(),
            without.outputs["nee"].to_numpy(),
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
    did next — usually a column lookup, which raises KeyError a long way from
    the cause, while SIPNET's own explanation sat unread on the provenance
    object. In an ensemble the empties were collected silently.
    """

    @pytest.fixture
    def broken_climate(self):
        """A climate frame SIPNET refuses: no rows to read."""
        from tests.helpers import empty_climate

        return empty_climate()

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

    def test_the_error_crosses_a_process_boundary(self, minimal_params):
        """An ensemble worker's failure reaches the driver as the real error."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        from pysipnet.runner import SIPNETRunError
        from tests.helpers import WORKER_TIMEOUT_SECONDS, run_with_no_climate_rows

        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
            future = pool.submit(run_with_no_climate_rows, minimal_params)
            err = future.exception(timeout=WORKER_TIMEOUT_SECONDS)

        assert isinstance(err, SIPNETRunError)
        assert err.returncode != 0
        assert (err.stdout + err.stderr).strip()
        assert isinstance(err.workdir, Path)
        assert "climate" in str(err).lower()

    def test_check_false_returns_a_result_instead(self, minimal_params, broken_climate):
        """Opt-out for ensembles, where one failure should not stop the rest."""
        result = SIPNETRunner(flags=ModelFlags.standard()).run(
            minimal_params, broken_climate, check=False
        )
        assert result.provenance.success is False
        assert result.outputs.pandas.empty

    def test_selecting_from_a_failed_run_points_at_the_failure(
        self, minimal_params, broken_climate
    ):
        """The message a caller sees first should name the cause, not the symptom."""
        result = SIPNETRunner(flags=ModelFlags.standard()).run(
            minimal_params, broken_climate, check=False
        )
        for select in (
            lambda: result.outputs["nee"],
            lambda: result.outputs.select(["nee"], format="pandas"),
        ):
            with pytest.raises(KeyError, match="provenance.stderr"):
                select()

    def test_a_successful_run_is_unaffected(self, minimal_params):
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, _make_climate())
        assert result.provenance.success
        assert not result.outputs.pandas.empty


# ---------------------------------------------------------------------------
# Snow flag
# ---------------------------------------------------------------------------


class TestSnowFlag:
    """At the pinned SIPNET the snow flag does not switch the snowpack off.

    SIPNET's documentation says ``SNOW = 0`` treats all precipitation as
    liquid. The source disagrees: nothing in the flux code reads ``ctx.snow``,
    so snow falls and accumulates either way and the flag only decides whether
    ``snowMelt`` is required. The registry describes the column accordingly.
    If this test starts failing, upstream has fixed it, and the descriptions
    of ``ModelFlags.snow`` and ``snow_water_equivalent`` need revisiting.
    """

    @staticmethod
    def _freezing_climate(n_days: int = 10):
        from pysipnet.climate import ClimateDrivers

        rows = [
            {
                "year": 2010,
                "day_of_year": 10 + i,
                "hour_of_day": 0.0,
                "time_step_length": 1.0,
                "air_temperature": -5.0,
                "soil_temperature": -2.0,
                "photosynthetically_active_radiation": 5.0,
                "precipitation": 10.0,
                "vapor_pressure_deficit": 300.0,
                "soil_vapor_pressure_deficit": 200.0,
                "vapor_pressure": 300.0,
                "wind_speed": 2.0,
            }
            for i in range(n_days)
        ]
        return ClimateDrivers.from_dataframe(pd.DataFrame(rows))

    def test_file_backed_climate_is_not_read_to_build_the_result(self, minimal_params, tmp_path):
        """The step lengths for the Dataset are fetched lazily, so from_path stays lazy."""
        from pysipnet.climate import ClimateDrivers

        path = tmp_path / "site.clim"
        self._freezing_climate().to_file(path)
        climate = ClimateDrivers.from_path(path)
        result = SIPNETRunner(flags=ModelFlags.standard()).run(minimal_params, climate)
        assert result.provenance.success
        assert climate._data is None, "building the result must not read the climate file"
        assert "time_step_start" in result.outputs.xarray.coords
        assert climate._data is not None, "the Dataset needs the step lengths"

    def test_snow_melts_identically_with_the_flag_off_when_the_rate_is_supplied(
        self, minimal_params
    ):
        """With snow_melt_rate written to the file, the flag changes nothing at all."""
        climate = self._freezing_climate(n_days=10)
        df = climate.pandas.copy()
        df.loc[5:, "air_temperature"] = 10.0  # five freezing days, then a thaw
        from pysipnet.climate import ClimateDrivers

        thaw = ClimateDrivers.from_dataframe(df)
        on = SIPNETRunner(flags=ModelFlags(snow=True)).run(minimal_params, thaw)
        off = SIPNETRunner(flags=ModelFlags(snow=False)).run(minimal_params, thaw)
        swe = off.outputs["snow_water_equivalent"]
        assert swe.max() > swe[-1], "the thaw should melt some snow with the flag off"
        np.testing.assert_array_equal(
            on.outputs["snow_water_equivalent"].to_numpy(), swe.to_numpy()
        )

    def test_snowpack_accumulates_with_the_flag_off(self, minimal_params):
        climate = self._freezing_climate()
        on = SIPNETRunner(flags=ModelFlags(snow=True)).run(minimal_params, climate)
        off = SIPNETRunner(flags=ModelFlags(snow=False)).run(minimal_params, climate)

        swe_off = off.outputs["snow_water_equivalent"]
        assert swe_off[-1] > swe_off[0] > 0, "snow should accumulate below 0 °C"
        np.testing.assert_array_equal(
            on.outputs["snow_water_equivalent"].to_numpy(), swe_off.to_numpy()
        )
