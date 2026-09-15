"""SIPNET model output container.

:class:`SIPNETOutput` holds the parsed ``sipnet.out`` file either in memory
(eager) or as a reference to a file on disk (lazy), and exposes it two ways:

- :attr:`SIPNETOutput.data` — a flat :class:`pandas.DataFrame`, one row per
  timestep, columns named as in :mod:`pysipnet.variables`.
- :attr:`SIPNETOutput.dataset` — an :class:`xarray.Dataset` with a single
  ``time`` dimension, a datetime coordinate, and every variable carrying its
  units, description and time reference as attributes. This is the
  representation to stack across ensemble members or write to netCDF.

The two construction modes are:

- :meth:`SIPNETOutput.from_dataframe` — memory-backed; data immediately available.
- :meth:`SIPNETOutput.from_path` — file-backed; the file is not read until
  :attr:`data` or :attr:`dataset` is first accessed. The file is verified to
  exist at construction time so that a missing file is caught immediately.

The file-backed mode is the natural choice when :class:`~pysipnet.runner.SIPNETRunner`
is configured with an ``output_dir``: the runner copies ``sipnet.out`` there
before deleting the temporary working directory. In a 1 000-member ensemble
this means nothing is held in memory until the caller asks for it.

Time convention
---------------
SIPNET labels each row with the **start** of its timestep. Pools are reported
at the **end** of the step, fluxes are totals **over** the step. The Dataset
makes this explicit: the ``time`` coordinate is the step start and says so in
its attributes, ``time_step_end`` and ``time_step_length`` are coordinates when
the step lengths are known, and every data variable has a ``time_reference``
attribute in words.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pysipnet.variables import (
    OUTPUT_VARIABLES_BY_NAME,
    TIME_COORDINATE_NAMES,
    VariableSpec,
    resolve_output_variable,
    resolve_output_variable_names,
)

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


class SIPNETOutput:
    """Parsed SIPNET ``.out`` output, memory-backed or file-backed.

    Prefer the factory methods :meth:`from_path` and :meth:`from_dataframe`;
    they are the public construction interface.

    Parameters
    ----------
    data:
        In-memory DataFrame. Mutually exclusive with *source_path*.
    source_path:
        Path to a persistent ``.out`` file. Mutually exclusive with *data*.
        The file is checked for existence at construction time.
    time_step_length:
        Length of each timestep in days, one value per row, taken from the
        climate drivers. Optional; when given, :attr:`dataset` carries
        ``time_step_end`` and ``time_step_length`` coordinates.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | None = None,
        source_path: Path | None = None,
        time_step_length: np.ndarray | None = None,
    ) -> None:
        if (data is None) == (source_path is None):
            raise ValueError(
                "Exactly one of 'data' or 'source_path' must be provided, not both or neither."
            )
        self._data: pd.DataFrame | None = data
        self._dataset: xr.Dataset | None = None
        self.source_path: Path | None = source_path
        self.time_step_length: np.ndarray | None = (
            None if time_step_length is None else np.asarray(time_step_length, dtype=float)
        )

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_path(
        cls, path: str | Path, *, time_step_length: np.ndarray | None = None
    ) -> SIPNETOutput:
        """Create a file-backed instance without reading the output into memory.

        The file is not parsed until :attr:`data` or :attr:`dataset` is
        accessed, but its existence is verified immediately so that a missing
        or prematurely deleted file is detected at construction.

        Parameters
        ----------
        path:
            Path to a ``sipnet.out`` file that will persist for the lifetime of
            this object. Never point this at a file inside a temporary working
            directory that will be deleted; use
            :attr:`~pysipnet.runner.SIPNETRunner.output_dir` to have the file
            copied to a stable location first.
        time_step_length:
            Timestep lengths in days, one per row; see the class docstring.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"SIPNET output file not found: {path}\n"
                "Ensure the file is in a stable location outside the run's working "
                "directory, which is deleted after each run."
            )
        return cls(source_path=path, time_step_length=time_step_length)

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, *, time_step_length: np.ndarray | None = None
    ) -> SIPNETOutput:
        """Create a memory-backed instance from an already-parsed DataFrame.

        Parameters
        ----------
        df:
            DataFrame returned by :func:`~pysipnet.io.output_reader.read_output_file`.
        time_step_length:
            Timestep lengths in days, one per row; see the class docstring.
        """
        return cls(data=df, time_step_length=time_step_length)

    # ── Data access ────────────────────────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        """The full output as a DataFrame, one row per timestep.

        For file-backed instances the first access reads and caches the file;
        later accesses are free. Column names are the registry names in
        :mod:`pysipnet.variables` (``net_ecosystem_exchange``, ``wood_carbon``,
        ...). Use :meth:`variable` or ``output["nee"]`` to look one up by alias.
        """
        if self._data is None:
            from pysipnet.io.output_reader import read_output_file

            self._data = read_output_file(self._require_source())
        return self._data

    @property
    def dataset(self) -> xr.Dataset:
        """The full output as an :class:`xarray.Dataset` with a ``time`` dimension.

        Built from :attr:`data` on first access and cached. See
        :func:`output_dataframe_to_dataset` for the layout.
        """
        if self._dataset is None:
            self._dataset = output_dataframe_to_dataset(self.data, self.time_step_length)
        return self._dataset

    def __getitem__(self, name: str) -> xr.DataArray:
        """One variable as a :class:`xarray.DataArray`, by name or alias.

        ``output["nee"]``, ``output["NEE"]`` and
        ``output["net_ecosystem_exchange"]`` all return the same array, with
        units, description and time reference in ``.attrs``.
        """
        return self.dataset[resolve_output_variable(name).name]

    def variable(self, name: str) -> pd.Series:
        """One variable as a :class:`pandas.Series`, by name or alias."""
        return self.data[resolve_output_variable(name).name]

    @property
    def variables(self) -> tuple[VariableSpec, ...]:
        """Registry specs for the columns present, in column order.

        Columns the registry does not know (from a newer SIPNET) are omitted.
        """
        return tuple(
            OUTPUT_VARIABLES_BY_NAME[c] for c in self.data.columns if c in OUTPUT_VARIABLES_BY_NAME
        )

    def _require_source(self) -> Path:
        """Return the backing file path, or explain why there isn't one.

        An instance holds either in-memory data or a path to read from. If
        neither is set the object was built past its constructor, and reading
        would otherwise fail somewhere deeper with a less obvious message.
        """
        if self.source_path is None:
            raise ValueError("This SIPNETOutput has neither loaded data nor a file to read from.")
        return self.source_path

    def load(
        self, variables: list[str] | None = None, *, as_xarray: bool = False
    ) -> pd.DataFrame | xr.Dataset:
        """Explicitly load output, optionally restricting to a subset of variables.

        Unlike :attr:`data`, this method does **not** cache its result when
        *variables* is given: each call reads the file afresh so that only the
        requested columns are held in memory. This is the pattern for
        memory-constrained ensemble post-processing::

            nee_frames = [r.outputs.load(variables=["nee"]) for r in results]

        The time coordinates (``year``, ``day_of_year``, ``hour_of_day``) are
        always included.

        Parameters
        ----------
        variables:
            Variable names or aliases to return. ``None`` returns everything and
            behaves like :attr:`data` (or :attr:`dataset`).
        as_xarray:
            Return an :class:`xarray.Dataset` instead of a DataFrame.
        """
        if variables is None:
            return self.dataset if as_xarray else self.data

        requested = resolve_output_variable_names(variables)

        if self._data is not None:
            keep = [c for c in self._data.columns if c in TIME_COORDINATE_NAMES or c in requested]
            frame = self._data[keep]
        else:
            from pysipnet.io.output_reader import read_output_file

            frame = read_output_file(self._require_source(), variables=requested)

        if as_xarray:
            return output_dataframe_to_dataset(frame, self.time_step_length)
        return frame

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def n_timesteps(self) -> int:
        """Number of output timesteps.

        For file-backed instances this triggers a full data load if not already
        cached.
        """
        return len(self.data)

    def __repr__(self) -> str:
        if self.source_path is not None:
            loaded = "loaded" if self._data is not None else "not yet loaded"
            return f"SIPNETOutput(source_path={str(self.source_path)!r}, {loaded})"
        return f"SIPNETOutput(in_memory, timesteps={len(self.data)})"


# ── DataFrame → Dataset ────────────────────────────────────────────────────────


def output_dataframe_to_dataset(
    df: pd.DataFrame, time_step_length: np.ndarray | None = None
) -> xr.Dataset:
    """Turn a parsed output DataFrame into a self-describing :class:`xarray.Dataset`.

    Layout (see :func:`pysipnet.dataset.dataframe_to_dataset`):

    - one dimension, ``time``, whose coordinate is the **start** of each
      timestep as ``datetime64``, built from ``year``, ``day_of_year`` and
      ``hour_of_day``;
    - those three columns kept as auxiliary coordinates on ``time``;
    - when *time_step_length* is given, ``time_step_length`` (as
      ``timedelta64``) and ``time_step_end`` coordinates as well;
    - one data variable per remaining column, with the attributes from
      :meth:`~pysipnet.variables.VariableSpec.xarray_attributes`.

    Columns the registry does not know become variables with no attributes.
    """
    from pysipnet.dataset import dataframe_to_dataset

    return dataframe_to_dataset(
        df,
        attributes_for=_attributes_for,
        time_step_length=time_step_length,
        source="SIPNET output, via pySIPNET",
    )


def _attributes_for(name: str) -> dict[str, Any]:
    spec = OUTPUT_VARIABLES_BY_NAME.get(name)
    return {} if spec is None else spec.xarray_attributes()
