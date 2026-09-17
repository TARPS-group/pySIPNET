"""SIPNET model output container.

:class:`SIPNETOutput` holds the parsed ``sipnet.out`` file either in memory
(eager) or as a reference to a file on disk (lazy), and exposes it as an
:class:`xarray.Dataset`, as a :class:`pandas.DataFrame`, or one variable at a
time:

- ``output["nee"]`` — one variable as a :class:`xarray.DataArray`, carrying its
  units, its time coordinate and the interval each value covers.
- ``output[["nee", "gpp"]]`` — several variables as a Dataset.
- :meth:`SIPNETOutput.dataset` / :meth:`SIPNETOutput.dataframe` — the same
  selection, spelled out, in either library.
- :attr:`SIPNETOutput.xarray` / :attr:`SIPNETOutput.pandas` — everything.

Every one of these accepts aliases (``"nee"``, ``"NEE"``,
``"net_ecosystem_exchange"``), and a selection never re-reads a column already
in memory: asking for a variable twice costs one read, and asking for several
at once is a single read.  Defining a likelihood over three output variables
therefore does not read the file three times, no matter how the caller spells
the request.

The two construction modes are:

- :meth:`SIPNETOutput.from_dataframe` — memory-backed; data immediately available.
- :meth:`SIPNETOutput.from_path` — file-backed; the file is not read until a
  variable is asked for.  The file is verified to exist at construction time so
  that a missing file is caught immediately.

The file-backed mode is the natural choice when :class:`~pysipnet.runner.SIPNETRunner`
is configured with an ``output_dir``: the runner copies ``sipnet.out`` there
before deleting the temporary working directory.  In a 1 000-member ensemble
this means nothing is held in memory until the caller asks for it, and asking
for one variable holds one variable.

Time convention
---------------
SIPNET labels each row with the **start** of its timestep.  Pools are reported
at the **end** of the step, fluxes are totals **over** the step.  The Dataset
makes this explicit: the ``time`` coordinate is the step start and says so in
its attributes, ``time_step_end``, ``time_step_length`` and a CF
``time_bounds`` variable give the interval each row covers, and every data
variable has a ``time_reference`` attribute in words.

Flag-dependent variables
------------------------
SIPNET writes some columns as constant zero unless the matching model flag is
on — ``litter_carbon`` without ``litter_pool``, the nitrogen group without
``nitrogen_cycle``.  When the flags are known, selecting such a variable by
name raises rather than handing back a column of zeros that a likelihood would
consume without complaint.  The full views (:attr:`pandas`, :attr:`xarray`)
still contain the column, because they are a faithful view of the file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from pysipnet.dataset import TimeAxis, build_time_axis, dataset_from_frame
from pysipnet.variables import (
    OUTPUT_VARIABLES_BY_NAME,
    TIME_COORDINATE_NAMES,
    VariableSpec,
    resolve_output_variable_names,
)

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from pysipnet.parameters.model import ModelFlags


TimeStepLengths: TypeAlias = "np.ndarray | Callable[[], np.ndarray]"

_SOURCE = "SIPNET output, via pySIPNET"


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
        climate drivers, or a zero-argument callable returning that array so a
        file-backed climate is not read until the Dataset needs it.  Optional;
        when omitted the lengths are inferred from consecutive timestamps and
        the Dataset's ``time_step_length_source`` attribute says so.
    flags:
        The :class:`~pysipnet.parameters.model.ModelFlags` the run used, so that
        selecting a variable SIPNET wrote as constant zero can be refused.
    run_id:
        Identifier of the run, recorded in the Dataset's attributes.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | None = None,
        source_path: Path | None = None,
        time_step_length: TimeStepLengths | None = None,
        flags: ModelFlags | None = None,
        run_id: str | None = None,
    ) -> None:
        if (data is None) == (source_path is None):
            raise ValueError(
                "Exactly one of 'data' or 'source_path' must be provided, not both or neither."
            )
        self.source_path: Path | None = source_path
        self.flags = flags
        self.run_id = run_id

        # Holds whichever columns have been read so far; _complete says whether
        # that is all of them.
        self._frame: pd.DataFrame | None = data
        self._complete: bool = data is not None
        self._dataset: xr.Dataset | None = None
        self._axis: TimeAxis | None = None
        self._time_step_length: TimeStepLengths | None = time_step_length

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        time_step_length: TimeStepLengths | None = None,
        flags: ModelFlags | None = None,
        run_id: str | None = None,
    ) -> SIPNETOutput:
        """Create a file-backed instance without reading the output into memory.

        The file is not parsed until a variable is asked for, but its existence
        is verified immediately so that a missing or prematurely deleted file is
        detected at construction.

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
        flags:
            Model flags the run used; see the class docstring.
        run_id:
            Identifier of the run, recorded in the Dataset's attributes.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"SIPNET output file not found: {path}\n"
                "Ensure the file is in a stable location outside the run's working "
                "directory, which is deleted after each run."
            )
        return cls(source_path=path, time_step_length=time_step_length, flags=flags, run_id=run_id)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        time_step_length: TimeStepLengths | None = None,
        flags: ModelFlags | None = None,
        run_id: str | None = None,
    ) -> SIPNETOutput:
        """Create a memory-backed instance from an already-parsed DataFrame.

        Parameters
        ----------
        df:
            DataFrame returned by :func:`~pysipnet.io.output_reader.read_output_file`.
        time_step_length:
            Timestep lengths in days, one per row; see the class docstring.
        flags:
            Model flags the run used; see the class docstring.
        run_id:
            Identifier of the run, recorded in the Dataset's attributes.
        """
        return cls(data=df, time_step_length=time_step_length, flags=flags, run_id=run_id)

    # ── Data access ────────────────────────────────────────────────────────────

    @property
    def pandas(self) -> pd.DataFrame:
        """The full output as a :class:`pandas.DataFrame`, one row per timestep.

        For file-backed instances the first access reads and caches the whole
        file; later accesses are free. A whole-file read after some variables
        have been selected individually re-reads those columns, because only
        the file itself states the order they belong in — the one case where a
        column is read twice. Column names are the registry names in
        :mod:`pysipnet.variables` (``net_ecosystem_exchange``, ``wood_carbon``,
        ...), in the order SIPNET wrote them. Use ``output["nee"]`` or
        :meth:`dataframe` to look columns up by alias.
        """
        if not self._complete:
            from pysipnet.io.output_reader import read_output_file

            # A full read replaces whatever subset was accumulated, which keeps
            # the columns in the order SIPNET wrote them.
            self._frame = read_output_file(self._require_source())
            self._complete = True
        assert self._frame is not None
        return self._frame

    @property
    def xarray(self) -> xr.Dataset:
        """The full output as an :class:`xarray.Dataset` with a ``time`` dimension.

        Built from :attr:`pandas` on first access and cached. See
        :meth:`dataset` for the layout.
        """
        if self._dataset is None:
            self._dataset = self._build_dataset(self.pandas)
        return self._dataset

    def dataframe(self, variables: Sequence[str] | None = None) -> pd.DataFrame:
        """The output, or the named variables of it, as a DataFrame.

        The time coordinates (``year``, ``day_of_year``, ``hour_of_day``) are
        always included, because they identify each row.

        Parameters
        ----------
        variables:
            Variable names or aliases. ``None`` returns everything, exactly as
            :attr:`pandas` does.
        """
        if variables is None:
            return self.pandas
        names = self._resolve(variables)
        self._ensure_columns(names)
        assert self._frame is not None
        return self._frame[[*TIME_COORDINATE_NAMES, *names]]

    def dataset(self, variables: Sequence[str] | None = None) -> xr.Dataset:
        """The output, or the named variables of it, as a self-describing Dataset.

        Layout:

        - one dimension, ``time``, whose coordinate is the **start** of each
          timestep as ``datetime64``;
        - ``year``, ``day_of_year`` and ``hour_of_day`` kept as auxiliary
          coordinates on ``time``;
        - ``time_step_length``, ``time_step_end`` and a CF ``time_bounds``
          variable describing the interval each row covers;
        - one data variable per selected column, with the attributes from
          :meth:`~pysipnet.variables.VariableSpec.xarray_attributes`.

        Columns the registry does not know become variables with no attributes.

        Parameters
        ----------
        variables:
            Variable names or aliases. ``None`` returns everything, exactly as
            :attr:`xarray` does.
        """
        if variables is None:
            return self.xarray
        return self._build_dataset(self.dataframe(variables))

    def __getitem__(self, key: str | Sequence[str]) -> xr.DataArray | xr.Dataset:
        """One variable as a DataArray, or several as a Dataset, by name or alias.

        ``output["nee"]``, ``output["NEE"]`` and
        ``output["net_ecosystem_exchange"]`` all return the same array, with
        units, description and time reference in ``.attrs``.
        ``output[["nee", "gpp"]]`` returns both in one Dataset, read in one go.
        """
        if isinstance(key, str):
            name = self._resolve([key])[0]
            if self._dataset is not None:
                return self._dataset[name]
            return self.dataset([key])[name]
        return self.dataset(key)

    @property
    def variables(self) -> tuple[VariableSpec, ...]:
        """Registry specs for the columns present, in column order.

        Columns the registry does not know (from a newer SIPNET) are omitted.
        """
        return tuple(
            OUTPUT_VARIABLES_BY_NAME[c]
            for c in self.pandas.columns
            if c in OUTPUT_VARIABLES_BY_NAME
        )

    @property
    def time_step_length(self) -> np.ndarray | None:
        """Timestep lengths in days, one per row, or ``None`` when not supplied.

        Resolved on first access when a callable was given, which is what lets
        a file-backed climate stay unread until the Dataset is built. ``None``
        means the Dataset will infer the lengths from the timestamps instead.
        """
        if callable(self._time_step_length):
            self._time_step_length = np.asarray(self._time_step_length(), dtype=float)
        elif self._time_step_length is not None:
            self._time_step_length = np.asarray(self._time_step_length, dtype=float)
        return self._time_step_length

    @property
    def n_timesteps(self) -> int:
        """Number of output timesteps.

        For file-backed instances this triggers a full data load if not already
        cached.
        """
        return len(self.pandas)

    def __repr__(self) -> str:
        if self.source_path is not None:
            loaded = "loaded" if self._complete else "not yet loaded"
            return f"SIPNETOutput(source_path={str(self.source_path)!r}, {loaded})"
        return f"SIPNETOutput(in_memory, timesteps={len(self.pandas)})"

    # ── Internals ──────────────────────────────────────────────────────────────

    def _resolve(self, variables: Sequence[str]) -> list[str]:
        """Canonical names for a selection, refusing variables the flags zero out."""
        if isinstance(variables, str):
            raise TypeError(
                "Pass a sequence of variable names, or a single name to output[...]; "
                f"got the string {variables!r}."
            )
        names = resolve_output_variable_names(list(variables))
        if self.flags is not None:
            for name in names:
                self._check_flag(name)
        return names

    def _check_flag(self, name: str) -> None:
        spec = OUTPUT_VARIABLES_BY_NAME.get(name)
        if spec is None or spec.requires_flag is None:
            return
        assert self.flags is not None
        if not getattr(self.flags, spec.requires_flag):
            raise ValueError(
                f"{name!r} is constant zero in this run: SIPNET only fills it when the "
                f"{spec.requires_flag!r} flag is on, and this run had it off. Re-run with "
                f"ModelFlags(..., {spec.requires_flag}=True), or read the raw column from "
                "output.pandas if the zeros are genuinely what you want."
            )

    def _ensure_columns(self, names: Sequence[str]) -> None:
        """Read whichever of *names* is not in memory yet, in a single pass."""
        have = set(self._frame.columns) if self._frame is not None else set()
        missing = [name for name in names if name not in have]
        if not missing:
            return

        if self.source_path is None:
            raise KeyError(
                f"{missing} are not in this in-memory output, and there is no file to read "
                "them from. It was built from a DataFrame that does not contain them."
            )

        import pandas as pd

        from pysipnet.io.output_reader import read_output_file

        new = read_output_file(self._require_source(), variables=missing)
        if self._frame is None:
            self._frame = new
        else:
            added = [c for c in new.columns if c not in have]
            self._frame = pd.concat([self._frame, new[added]], axis=1)

    def _time_axis(self, df: pd.DataFrame) -> TimeAxis:
        """The shared time coordinates, built once per output.

        Every view of a run has the same rows, so the axis — a third of the cost
        of reading the file — is built on the first Dataset and reused by the
        rest.
        """
        if self._axis is None:
            self._axis = build_time_axis(
                df, attributes_for=_attributes_for, time_step_length=self.time_step_length
            )
        return self._axis

    def _build_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        import xarray as xr

        if df.empty:
            return xr.Dataset()
        return dataset_from_frame(
            df,
            self._time_axis(df),
            attributes_for=_attributes_for,
            source=_SOURCE,
            extra_attrs=self._provenance_attrs(),
        )

    def _provenance_attrs(self) -> dict[str, Any]:
        """Run identity, as netCDF-safe scalars, so a saved prediction says where it came from."""
        attrs: dict[str, Any] = {}
        if self.run_id is not None:
            attrs["run_id"] = self.run_id
        if self.flags is not None:
            attrs["model_flags"] = json.dumps(self.flags.model_dump(), sort_keys=True)
        return attrs

    def _require_source(self) -> Path:
        """Return the backing file path, or explain why there isn't one.

        An instance holds either in-memory data or a path to read from. If
        neither is set the object was built past its constructor, and reading
        would otherwise fail somewhere deeper with a less obvious message.
        """
        if self.source_path is None:
            raise ValueError("This SIPNETOutput has neither loaded data nor a file to read from.")
        return self.source_path


# ── DataFrame → Dataset ────────────────────────────────────────────────────────


def output_dataframe_to_dataset(
    df: pd.DataFrame, time_step_length: np.ndarray | None = None
) -> xr.Dataset:
    """Turn a parsed output DataFrame into a self-describing :class:`xarray.Dataset`.

    The free-function form of :meth:`SIPNETOutput.dataset`, for a frame that did
    not come from a :class:`SIPNETOutput`. See :func:`pysipnet.dataset.dataframe_to_dataset`
    for the layout.
    """
    from pysipnet.dataset import dataframe_to_dataset

    return dataframe_to_dataset(
        df,
        attributes_for=_attributes_for,
        time_step_length=time_step_length,
        source=_SOURCE,
    )


def _attributes_for(name: str) -> dict[str, Any]:
    spec = OUTPUT_VARIABLES_BY_NAME.get(name)
    return {} if spec is None else spec.xarray_attributes()
