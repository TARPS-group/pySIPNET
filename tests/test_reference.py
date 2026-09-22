"""The bundled Niwot reference data: reachable as installed, and still what it claims to be.

Two things can go wrong with data that ships inside a package, and neither
shows up in a test that reads the source tree:

1. **It does not ship.** ``pyproject.toml`` lists the package; whether the
   data files under it end up in the wheel is the build backend's decision.
   :func:`test_wheel_ships_the_reference_files` builds the wheel and looks,
   which is the only honest check — the defect this module guards against is
   precisely that the source tree has files the wheel does not.
2. **It drifts.** The inputs are valuable because SIPNET's developers wrote
   them, not pySIPNET. ``README.md`` beside them says they are byte-identical
   to the submodule's copy; the tests here make that claim executable, so a
   re-save, a reformat or a changed line ending fails loudly.

The loaders themselves are tested through :mod:`importlib.resources`, never a
path relative to this file, so they prove the same thing an installed consumer
would rely on.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysipnet.climate import ClimateDrivers
from pysipnet.io.reference import (
    NIWOT_CLIM_FILE,
    NIWOT_OUTPUT_FILE,
    NIWOT_OUTPUT_RUN_ID,
    NIWOT_PARAM_FILE,
    NIWOT_README_FILE,
    NiwotReferenceFiles,
    niwot_reference_climate,
    niwot_reference_files,
    niwot_reference_output,
)
from pysipnet.output import SIPNETOutput
from pysipnet.parameters.model import ModelFlags
from pysipnet.variables import OUTPUT_VARIABLES

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_NIWOT = REPO_ROOT / "sipnet" / "tests" / "smoke" / "niwot"

#: Where the files sit inside the wheel, as ``RECORD`` and ``zipfile`` spell them.
WHEEL_DATA_DIR = "pysipnet/data/niwot"
SHIPPED_FILES = (NIWOT_PARAM_FILE, NIWOT_CLIM_FILE, NIWOT_OUTPUT_FILE, NIWOT_README_FILE)

#: How many rows of the upstream climate the bundled copy keeps.
N_CLIM_ROWS = 800


# ---------------------------------------------------------------------------
# Location: through importlib.resources, as a consumer would find them
# ---------------------------------------------------------------------------


def test_files_are_found_through_importlib_resources():
    paths = niwot_reference_files()
    assert isinstance(paths, NiwotReferenceFiles)
    expected_dir = files("pysipnet") / "data" / "niwot"
    assert paths.directory == Path(str(expected_dir))
    for path in (paths.param, paths.clim, paths.output, paths.readme):
        assert path.is_file(), path
        assert path.parent == paths.directory


def test_file_names_are_the_documented_ones():
    paths = niwot_reference_files()
    assert paths.param.name == NIWOT_PARAM_FILE == "sipnet.param"
    assert paths.clim.name == NIWOT_CLIM_FILE == "sipnet.clim"
    assert paths.output.name == NIWOT_OUTPUT_FILE == "niwot_standard.out.csv"
    assert paths.readme.name == NIWOT_README_FILE == "README.md"


# ---------------------------------------------------------------------------
# Provenance: the README's claims, made executable
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def upstream_niwot() -> Path:
    if not (UPSTREAM_NIWOT / "sipnet.param").is_file():
        pytest.skip("SIPNET submodule not populated; run 'git submodule update --init sipnet'")
    return UPSTREAM_NIWOT


def test_param_file_is_byte_identical_to_upstream(upstream_niwot: Path):
    ours = niwot_reference_files().param.read_bytes()
    theirs = (upstream_niwot / "sipnet.param").read_bytes()
    assert ours == theirs, "sipnet.param has drifted from the submodule's copy"


def test_clim_file_is_the_first_800_upstream_rows_byte_for_byte(upstream_niwot: Path):
    ours = niwot_reference_files().clim.read_bytes()
    with (upstream_niwot / "sipnet.clim").open("rb") as handle:
        theirs = b"".join(handle.readline() for _ in range(N_CLIM_ROWS))
    assert ours == theirs, "sipnet.clim has drifted from the first 800 rows of the submodule's copy"


def test_readme_records_the_pinned_commit():
    from pysipnet.version import SIPNET_PINNED_COMMIT

    text = niwot_reference_files().readme.read_text()
    assert SIPNET_PINNED_COMMIT in text, "README.md must name the commit the files were copied at"


# ---------------------------------------------------------------------------
# Loaders: typed, consistent with each other, and quiet
# ---------------------------------------------------------------------------


def test_climate_loads_without_warnings(recwarn: pytest.WarningsRecorder):
    climate = niwot_reference_climate()
    assert isinstance(climate, ClimateDrivers)
    assert climate.n_columns == 14
    assert len(climate.pandas) == N_CLIM_ROWS
    assert not recwarn.list, [str(w.message) for w in recwarn.list]


def test_climate_step_lengths_are_not_uniform():
    """The record's whole point for weighting tests: day and night steps differ."""
    lengths = niwot_reference_climate().pandas["time_step_length"]
    assert lengths.min() < 0.3 < 0.6 < lengths.max()


def test_output_is_the_golden_with_the_climates_step_lengths():
    output = niwot_reference_output()
    assert isinstance(output, SIPNETOutput)
    golden = pd.read_csv(niwot_reference_files().output)
    assert output.n_timesteps == len(golden) == 60
    pd.testing.assert_frame_equal(output.pandas, golden)

    climate = niwot_reference_climate().pandas.head(len(golden))
    assert output.time_step_length is not None
    np.testing.assert_array_equal(output.time_step_length, climate["time_step_length"])
    ds = output.xarray
    assert ds.attrs["time_step_length_source"] == "climate drivers"
    assert ds.attrs["run_id"] == NIWOT_OUTPUT_RUN_ID


def test_output_rows_line_up_with_the_first_climate_rows():
    """The golden is the model run on the climate's first rows; each row must say so."""
    golden = pd.read_csv(niwot_reference_files().output)
    climate = niwot_reference_climate().pandas.head(len(golden))
    for column in ("year", "day_of_year", "hour_of_day"):
        np.testing.assert_array_equal(golden[column].to_numpy(), climate[column].to_numpy())


def test_output_header_is_the_registry():
    output = niwot_reference_output()
    assert list(output.pandas.columns) == [v.name for v in OUTPUT_VARIABLES]


def test_output_carries_standard_flags_and_refuses_zero_columns():
    output = niwot_reference_output()
    assert output.flags == ModelFlags.standard()
    with pytest.raises(ValueError, match="litter_pool"):
        output["litter_carbon"]
    assert output["nee"].shape == (60,)


# ---------------------------------------------------------------------------
# Packaging: the files are in the wheel, unchanged
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_wheel_ships_the_reference_files(tmp_path: Path):
    """Build the wheel the way pip would and check the data is in it, byte for byte.

    Reading the source tree proves nothing here: the defect this guards is a
    source tree with files the wheel lacks. hatchling is in the dev group so
    the build needs no network.
    """
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = list(tmp_path.glob("pysipnet-*.whl"))
    assert len(wheels) == 1, wheels

    paths = niwot_reference_files()
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())
        for name in SHIPPED_FILES:
            member = f"{WHEEL_DATA_DIR}/{name}"
            assert member in members, f"{member} is not in the wheel; members: {sorted(members)}"
            assert wheel.read(member) == (paths.directory / name).read_bytes(), name
        stray = [m for m in members if m.startswith("tests/") or "fixtures" in m]
        assert not stray, stray
