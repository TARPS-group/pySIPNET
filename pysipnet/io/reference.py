"""The bundled Niwot Ridge reference data: real SIPNET inputs and real SIPNET output.

pySIPNET ships three data files under ``pysipnet/data/niwot/`` so that anything
that installs the package — a downstream project's test suite most of all — has
genuine SIPNET data to work with, without a source checkout and without a
compiled binary:

- ``sipnet.param`` and ``sipnet.clim`` are SIPNET's own Niwot Ridge smoke-test
  inputs, authored by the SIPNET developers rather than by pySIPNET.  The
  parameter file is byte-identical to the pinned submodule's copy; the climate
  file is its first 800 rows, about one year of sub-daily steps starting in
  November 1998.  Their value is that pySIPNET did not write them, so they are
  never re-saved or reformatted.
- ``niwot_standard.out.csv`` is pySIPNET's golden baseline: what the pinned
  SIPNET wrote when run under :meth:`~pysipnet.parameters.model.ModelFlags.standard`
  on those parameters and the first 60 rows of that climate.  It is real model
  output, but a narrow slice of it — one dormant-season month in which
  photosynthesis is almost entirely inactive and ten of the 35 columns are
  identically zero.  It changes whenever the baseline is deliberately
  regenerated (a SIPNET pin bump, an intended wrapper change), so it is a fixed
  sample of the model's output, not a reference solution.

``README.md`` beside the files records the provenance in full.

The functions here locate the files through :mod:`importlib.resources`, so they
work from an installed wheel as well as from a source tree, and turn them into
the package's own types::

    from pysipnet import niwot_reference_output

    output = niwot_reference_output()      # a SIPNETOutput, 60 steps, no binary needed
    nee = output["nee"]                    # with the climate's own step lengths

:func:`niwot_reference_files` gives the paths themselves, for anything the typed
loaders do not cover — reading ``sipnet.param`` with
:func:`~pysipnet.io.param_io.read_param_file`, or running the binary on the
original files by hand.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pysipnet.io.clim_io import read_clim_file

if TYPE_CHECKING:
    from pysipnet.climate import ClimateDrivers
    from pysipnet.output import SIPNETOutput

_DATA_DIR_PARTS = ("data", "niwot")

#: File names inside the bundled directory.
NIWOT_PARAM_FILE = "sipnet.param"
NIWOT_CLIM_FILE = "sipnet.clim"
NIWOT_OUTPUT_FILE = "niwot_standard.out.csv"
NIWOT_README_FILE = "README.md"

#: Identifier recorded on the golden output's Dataset.
NIWOT_OUTPUT_RUN_ID = "niwot_standard"

# The upstream climate record has three rows with vapor pressure deficit ≤ 0,
# which ClimateDrivers reports on load. SIPNET clamps them and the data is
# upstream's, not ours to correct, so the warning tells a caller nothing they
# can act on. Only this one message is silenced, and only for this file.
_KNOWN_UPSTREAM_DATA_WARNING = r"\d+ timestep\(s\) have vapor_pressure_deficit ≤ 0 Pa"


@dataclass(frozen=True)
class NiwotReferenceFiles:
    """Filesystem paths to the bundled Niwot Ridge reference files.

    Attributes
    ----------
    param:
        SIPNET's own ``sipnet.param`` for Niwot Ridge, byte-identical to the
        pinned submodule's ``tests/smoke/niwot/sipnet.param``.
    clim:
        The first 800 rows of the matching ``sipnet.clim``, 14-column layout.
    output:
        pySIPNET's golden baseline: standard-flag output on the first 60 rows,
        as a CSV with the registry's column names.
    readme:
        The provenance note that travels with the files.
    """

    param: Path
    clim: Path
    output: Path
    readme: Path

    @property
    def directory(self) -> Path:
        """The directory holding all four files."""
        return self.param.parent


def niwot_reference_files() -> NiwotReferenceFiles:
    """Locate the bundled reference files, from a wheel or a source tree alike.

    Raises
    ------
    FileNotFoundError
        If the package's data directory is not a real directory on disk. That
        is the case when pySIPNET is imported from an archive, which none of
        its compiled dependencies support either, so it is reported rather
        than worked around.
    """
    root = files("pysipnet")
    for part in _DATA_DIR_PARTS:
        root = root / part
    if not isinstance(root, Path) or not root.is_dir():
        raise FileNotFoundError(
            f"pySIPNET's reference data directory is not on disk at {root!s}; "
            "the package must be installed as a directory, not an archive."
        )
    paths = NiwotReferenceFiles(
        param=root / NIWOT_PARAM_FILE,
        clim=root / NIWOT_CLIM_FILE,
        output=root / NIWOT_OUTPUT_FILE,
        readme=root / NIWOT_README_FILE,
    )
    missing = [
        p.name for p in (paths.param, paths.clim, paths.output, paths.readme) if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"pySIPNET's reference data is incomplete under {root!s}: missing {missing}. "
            "The installation is broken; reinstall the package."
        )
    return paths


def niwot_reference_climate() -> ClimateDrivers:
    """The bundled Niwot Ridge climate record, as :class:`~pysipnet.climate.ClimateDrivers`.

    800 sub-daily steps from day 305 of 1998, with day and night steps of
    different lengths (0.29 to 0.63 days), which is what makes it a useful
    record for anything that must weight by step length.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_KNOWN_UPSTREAM_DATA_WARNING)
        return read_clim_file(niwot_reference_files().clim, n_columns=14)


def niwot_reference_output() -> SIPNETOutput:
    """The golden baseline as a memory-backed :class:`~pysipnet.output.SIPNETOutput`.

    The step lengths come from the bundled climate, so the Dataset's time axis
    is the one a real run would carry rather than one inferred from
    timestamps, and the flags are :meth:`~pysipnet.parameters.model.ModelFlags.standard`,
    the flags the baseline was produced under, so selecting a variable SIPNET
    wrote as constant zero is refused as it would be for a live run.

    See the module docstring for what the baseline does and does not cover.
    """
    from pysipnet.output import SIPNETOutput
    from pysipnet.parameters.model import ModelFlags

    paths = niwot_reference_files()
    frame = pd.read_csv(paths.output)
    climate = niwot_reference_climate().pandas.head(len(frame))

    time_columns = ["year", "day_of_year", "hour_of_day"]
    if not frame[time_columns].to_numpy().tolist() == climate[time_columns].to_numpy().tolist():
        raise RuntimeError(
            f"{paths.output.name} does not start where {paths.clim.name} starts, so its step "
            "lengths cannot be taken from the climate. The bundled data is inconsistent; "
            "regenerate the baseline with 'python -m tests.test_golden'."
        )

    return SIPNETOutput.from_dataframe(
        frame,
        time_step_length=climate["time_step_length"].to_numpy(),
        flags=ModelFlags.standard(),
        run_id=NIWOT_OUTPUT_RUN_ID,
    )
