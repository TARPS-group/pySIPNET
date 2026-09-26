"""Unit tests for RunConfig."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pysipnet.climate import ClimateDrivers
from pysipnet.config import RunConfig
from pysipnet.parameters.model import ModelFlags

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CLIM_ROWS = 10


@pytest.fixture
def in_memory_climate() -> ClimateDrivers:
    """Minimal in-memory ClimateDrivers (no source file)."""
    return ClimateDrivers.from_dataframe(
        pd.DataFrame(
            {
                "year": [2020] * _CLIM_ROWS,
                "day_of_year": list(range(1, _CLIM_ROWS + 1)),
                "hour_of_day": [0.0] * _CLIM_ROWS,
                "timestep_length": [1.0] * _CLIM_ROWS,
                "air_temperature": [15.0] * _CLIM_ROWS,
                "soil_temperature": [12.0] * _CLIM_ROWS,
                "photosynthetically_active_radiation": [8.0] * _CLIM_ROWS,
                "precipitation": [2.0] * _CLIM_ROWS,
                "vapor_pressure_deficit": [100.0] * _CLIM_ROWS,
                "soil_vapor_pressure_deficit": [80.0] * _CLIM_ROWS,
                "vapor_pressure": [900.0] * _CLIM_ROWS,
                "wind_speed": [2.0] * _CLIM_ROWS,
            }
        )
    )


@pytest.fixture
def file_backed_climate(tmp_path, in_memory_climate) -> ClimateDrivers:
    """File-backed ClimateDrivers backed by a temp file."""
    clim_file = tmp_path / "source.clim"
    in_memory_climate.to_file(clim_file)
    return ClimateDrivers.from_path(clim_file)


# ---------------------------------------------------------------------------
# Default mode (reference_only=False): climate data copied into directory
# ---------------------------------------------------------------------------


class TestCopyMode:
    def test_creates_config_json_and_clim(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        assert (tmp_path / "run" / "config.json").exists()
        assert (tmp_path / "run" / "sipnet.clim").exists()

    def test_no_events_file_by_default(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        assert not (tmp_path / "run" / "events.in").exists()

    def test_save_returns_resolved_path(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        returned = config.save(tmp_path / "run")
        assert returned == (tmp_path / "run").resolve()

    def test_config_json_climate_mode_is_copy(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        data = json.loads((tmp_path / "run" / "config.json").read_text())
        assert data["climate"]["mode"] == "copy"

    def test_metadata_fields_present(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        data = json.loads((tmp_path / "run" / "config.json").read_text())
        assert "sipnet_commit" in data
        assert "pysipnet_version" in data
        assert "created_at" in data

    def test_roundtrip_flags(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.flags == ModelFlags.standard()

    def test_roundtrip_params(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.params.model_dump() == minimal_params.model_dump()

    def test_roundtrip_climate_timesteps(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.climate.n_timesteps == _CLIM_ROWS

    def test_loaded_climate_is_file_backed(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.climate.source_path is not None

    def test_roundtrip_no_events(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.events is None

    def test_events_roundtrip(self, tmp_path, minimal_params, in_memory_climate):
        from pysipnet.events import EventSequence, IrrigationEvent, IrrigationMethod

        events = EventSequence(
            events=[
                IrrigationEvent(year=2020, day=5, amount=3.0, method=IrrigationMethod.SOIL),
            ]
        )
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
            events=events,
        )
        config.save(tmp_path / "run")
        assert (tmp_path / "run" / "events.in").exists()
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.events is not None
        assert len(loaded.events) == 1
        ev = loaded.events.events[0]
        assert ev.year == 2020
        assert ev.day == 5

    def test_litter_pool_flags_roundtrip(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags(litter_pool=True),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.flags == ModelFlags(litter_pool=True)

    def test_overwrites_existing_directory(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        config.save(tmp_path / "run")
        # Save again — should not raise
        config.save(tmp_path / "run")
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.flags == ModelFlags.standard()


# ---------------------------------------------------------------------------
# reference_only=True: path + hash stored, no climate file written
# ---------------------------------------------------------------------------


class TestReferenceOnly:
    def test_no_clim_file_written(self, tmp_path, minimal_params, file_backed_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=file_backed_climate,
        )
        config.save(tmp_path / "run", reference_only=True)
        assert not (tmp_path / "run" / "sipnet.clim").exists()

    def test_config_json_records_path_and_hash(self, tmp_path, minimal_params, file_backed_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=file_backed_climate,
        )
        config.save(tmp_path / "run", reference_only=True)
        data = json.loads((tmp_path / "run" / "config.json").read_text())
        assert data["climate"]["mode"] == "reference"
        assert "path" in data["climate"]
        assert "sha256" in data["climate"]
        assert len(data["climate"]["sha256"]) == 64  # hex SHA-256

    def test_roundtrip_flags_and_params(self, tmp_path, minimal_params, file_backed_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=file_backed_climate,
        )
        config.save(tmp_path / "run", reference_only=True)
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.flags == ModelFlags.standard()
        assert loaded.params.model_dump() == minimal_params.model_dump()

    def test_roundtrip_climate_timesteps(self, tmp_path, minimal_params, file_backed_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=file_backed_climate,
        )
        config.save(tmp_path / "run", reference_only=True)
        loaded = RunConfig.load(tmp_path / "run")
        assert loaded.climate.n_timesteps == _CLIM_ROWS

    def test_raises_for_in_memory_climate(self, tmp_path, minimal_params, in_memory_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=in_memory_climate,
        )
        with pytest.raises(ValueError, match="reference_only=True requires a file-backed"):
            config.save(tmp_path / "run", reference_only=True)

    def test_load_raises_if_file_missing(self, tmp_path, minimal_params, file_backed_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=file_backed_climate,
        )
        config.save(tmp_path / "run", reference_only=True)

        cfg_path = tmp_path / "run" / "config.json"
        data = json.loads(cfg_path.read_text())
        data["climate"]["path"] = "/nonexistent/does_not_exist.clim"
        cfg_path.write_text(json.dumps(data))

        with pytest.raises(FileNotFoundError, match="no longer exists"):
            RunConfig.load(tmp_path / "run")

    def test_hash_mismatch_warns(self, tmp_path, minimal_params, file_backed_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=file_backed_climate,
        )
        config.save(tmp_path / "run", reference_only=True)

        cfg_path = tmp_path / "run" / "config.json"
        data = json.loads(cfg_path.read_text())
        data["climate"]["sha256"] = "a" * 64  # deliberately wrong
        cfg_path.write_text(json.dumps(data))

        with pytest.warns(UserWarning, match="changed since"):
            RunConfig.load(tmp_path / "run")

    def test_matching_hash_no_warning(self, tmp_path, minimal_params, file_backed_climate):
        config = RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=file_backed_climate,
        )
        config.save(tmp_path / "run", reference_only=True)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RunConfig.load(tmp_path / "run")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestLoadErrors:
    def test_missing_config_json(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.json"):
            RunConfig.load(tmp_path / "nonexistent_dir")

    def test_empty_directory(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="config.json"):
            RunConfig.load(tmp_path / "empty")


# ---------------------------------------------------------------------------
# from_result
# ---------------------------------------------------------------------------


class TestFromResult:
    def test_from_result_copies_flags_params_climate(self, minimal_params, in_memory_climate):
        from pysipnet.parameters.model import ModelFlags
        from pysipnet.result import RunProvenance, SIPNETResult

        provenance = RunProvenance(
            flags=ModelFlags.standard(),
            binary_path=Path("/fake/sipnet"),
            run_id="test-abc",
            workdir=Path("/fake/workdir"),
            returncode=0,
            success=True,
            stdout="",
            stderr="",
        )
        result = SIPNETResult(
            outputs=pd.DataFrame(),
            parameters=minimal_params,
            climate=in_memory_climate,
            flags=ModelFlags.standard(),
            provenance=provenance,
            events=None,
        )
        config = RunConfig.from_result(result)
        assert config.flags == ModelFlags.standard()
        assert config.params.model_dump() == minimal_params.model_dump()
        assert config.events is None

    def test_from_result_preserves_events(self, minimal_params, in_memory_climate):
        from pysipnet.events import EventSequence, IrrigationEvent, IrrigationMethod
        from pysipnet.parameters.model import ModelFlags
        from pysipnet.result import RunProvenance, SIPNETResult

        events = EventSequence(
            events=[IrrigationEvent(year=2020, day=10, amount=5.0, method=IrrigationMethod.CANOPY)]
        )
        provenance = RunProvenance(
            flags=ModelFlags(litter_pool=True),
            binary_path=Path("/fake/sipnet"),
            run_id="test-xyz",
            workdir=Path("/fake/workdir"),
            returncode=0,
            success=True,
            stdout="",
            stderr="",
        )
        result = SIPNETResult(
            outputs=pd.DataFrame(),
            parameters=minimal_params,
            climate=in_memory_climate,
            flags=ModelFlags(litter_pool=True),
            provenance=provenance,
            events=events,
        )
        config = RunConfig.from_result(result)
        assert config.flags == ModelFlags(litter_pool=True)
        assert config.events is not None
        assert len(config.events) == 1


# ---------------------------------------------------------------------------
# The climate file's layout survives a save and load
# ---------------------------------------------------------------------------


class TestClimateLayout:
    """The layout is read back from the file, so neither layout is lost in a config.

    A config used to reload every climate as 14 columns, so one saved with
    12-column drivers loaded without complaint and failed on first read.
    """

    @pytest.mark.parametrize("n_columns", [12, 14])
    def test_copy_mode(self, tmp_path, minimal_params, in_memory_climate, n_columns):
        climate = ClimateDrivers.from_dataframe(in_memory_climate.pandas, n_columns=n_columns)
        RunConfig(flags=ModelFlags.standard(), params=minimal_params, climate=climate).save(
            tmp_path / "run"
        )
        loaded = RunConfig.load(tmp_path / "run").climate
        assert loaded.n_columns == n_columns
        pd.testing.assert_frame_equal(loaded.pandas, climate.pandas, check_exact=False)

    @pytest.mark.parametrize("n_columns", [12, 14])
    def test_reference_mode(self, tmp_path, minimal_params, in_memory_climate, n_columns):
        source = tmp_path / "source.clim"
        ClimateDrivers.from_dataframe(in_memory_climate.pandas, n_columns=n_columns).to_file(source)
        RunConfig(
            flags=ModelFlags.standard(),
            params=minimal_params,
            climate=ClimateDrivers.from_path(source),
        ).save(tmp_path / "run", reference_only=True)
        loaded = RunConfig.load(tmp_path / "run").climate
        assert loaded.n_columns == n_columns
        assert len(loaded.pandas) == _CLIM_ROWS
