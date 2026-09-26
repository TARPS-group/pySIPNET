"""SIPNET model output container.

:class:`SIPNETOutput` holds the parsed ``sipnet.out`` file either in memory
(eager) or as a reference to a file on disk (lazy), and exposes it as an
:class:`xarray.Dataset`, as a :class:`pandas.DataFrame`, or one variable at a
time:

- ``output["nee"]`` — one variable as a :class:`xarray.DataArray`, carrying its
  units, its time coordinate and the interval each value covers.
- ``output[["nee", "gpp"]]`` — several variables as a Dataset.
- :meth:`SIPNETOutput.select` — the same selection, in either library:
  ``select(["nee"], format="pandas")``.
- :attr:`SIPNETOutput.xarray` / :attr:`SIPNETOutput.pandas` — everything.

Every one of these accepts aliases (``"nee"``, ``"NEE"``,
``"net_ecosystem_exchange"``), and a selection never re-reads a column already
in memory: asking for a variable twice costs one read, and asking for several
at once is a single read.  Defining a likelihood over three output variables
therefore does not read the file three times, no matter how the caller spells
the request.

:meth:`SIPNETOutput.select` is the memory-efficient route: it reads and keeps
only the columns named, where :attr:`pandas` and :attr:`xarray` read and cache
the whole file.

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
therefore puts its ``time`` coordinate at the step **end**, the one instant at
which every variable's CF ``cell_methods`` is literally true; ``timestep_start``,
``timestep_length`` and a CF ``time_bounds`` variable give the interval each
row covers, and every data variable has ``kind`` and ``time_reference``
attributes in words.  See :mod:`pysipnet.dataset` for the reasoning, and
:func:`pysipnet.resample.resample` for combining steps into coarser ones.

An output that knows the :class:`~pysipnet.climate.ClimateDrivers` it was run
on — every output a :class:`~pysipnet.runner.SIPNETRunner` returns — takes its
time axis from them, row for row: their start times, their step lengths and
their declared clock.  SIPNET's own labels are rounded to 0.01 h, so the
drivers' are the exact version of the same thing.  Built from a file or frame
alone, the output falls back to the labels SIPNET printed; the Dataset's
``time_axis_source`` attribute says which.  SIPNET has no time zone, so
neither is on any clock but the drivers'.

Flag-dependent variables
------------------------
SIPNET writes some columns as constant zero unless the matching model flag is
on — ``litter_carbon`` without ``litter_pool``, the nitrogen group without
``nitrogen_cycle``.  When the flags are known, selecting such a variable by
name raises rather than handing back a column of zeros that a likelihood would
consume without complaint.  The full views (:attr:`pandas`, :attr:`xarray`)
still contain the column, because they are a faithful view of the file.
:func:`~pysipnet.variables.check_variable_is_written` is the same check on its
own, for refusing such a variable before any run exists.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, overload

import numpy as np

from pysipnet.dataset import (
    TimeAxis,
    build_time_axis,
    dataset_from_dataframe,
    without_absent_bounds,
)
from pysipnet.variables import (
    OUTPUT_VARIABLES_BY_NAME,
    TIME_COORDINATE_NAMES,
    VariableSpec,
    check_variable_is_written,
    resolve_output_variable_names,
)

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from pysipnet.climate import ClimateDrivers
    from pysipnet.parameters.model import ModelFlags


OutputFormat: TypeAlias = Literal["xarray", "pandas"]

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
    climate:
        The :class:`~pysipnet.climate.ClimateDrivers` the run was driven by.
        The Dataset's time axis is then theirs: their start times, their step
        lengths and their ``time_zone``.  A file-backed climate is not read
        until a Dataset is built.  The output must have exactly one row per
        climate row, with labels SIPNET could have printed from them, or
        building the Dataset raises.  Without it — an output file re-opened
        on its own — the axis is rebuilt from the labels SIPNET printed and
        the step lengths are inferred from them; the Dataset's
        ``time_axis_source`` and ``timestep_length_source`` attributes say
        so.
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
        climate: ClimateDrivers | None = None,
        flags: ModelFlags | None = None,
        run_id: str | None = None,
    ) -> None:
        if (data is None) == (source_path is None):
            raise ValueError(
                "Exactly one of 'data' or 'source_path' must be provided, not both or neither."
            )
        self.climate = climate
        self.source_path: Path | None = source_path
        self.flags = flags
        self.run_id = run_id

        # Holds whichever columns have been read so far; _complete says whether
        # that is all of them.
        self._frame: pd.DataFrame | None = data
        self._complete: bool = data is not None
        self._dataset: xr.Dataset | None = None
        self._axis: TimeAxis | None = None

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        climate: ClimateDrivers | None = None,
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
        climate:
            The climate drivers the run used; see the class docstring.
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
        return cls(
            source_path=path,
            climate=climate,
            flags=flags,
            run_id=run_id,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        climate: ClimateDrivers | None = None,
        flags: ModelFlags | None = None,
        run_id: str | None = None,
    ) -> SIPNETOutput:
        """Create a memory-backed instance from an already-parsed DataFrame.

        Parameters
        ----------
        df:
            DataFrame returned by :func:`~pysipnet.io.output_reader.read_output_file`.
        climate:
            The climate drivers the run used; see the class docstring.
        flags:
            Model flags the run used; see the class docstring.
        run_id:
            Identifier of the run, recorded in the Dataset's attributes.
        """
        return cls(data=df, climate=climate, flags=flags, run_id=run_id)

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
        ``output[[...]]`` for the layout.
        """
        if self._dataset is None:
            self._dataset = self._build_dataset(self.pandas)
        return self._dataset

    @overload
    def select(
        self, variables: Sequence[str], *, format: Literal["xarray"] = ...
    ) -> xr.Dataset: ...

    @overload
    def select(self, variables: Sequence[str], *, format: Literal["pandas"]) -> pd.DataFrame: ...

    def select(
        self, variables: Sequence[str], *, format: OutputFormat = "xarray"
    ) -> xr.Dataset | pd.DataFrame:
        """The named variables, reading only those from the file.

        This is the memory-efficient way to work with an output: on a
        file-backed instance it reads the named columns and holds only those,
        where :attr:`pandas` and :attr:`xarray` read and cache all 35. Across an
        ensemble that is the difference between keeping one column per member
        and keeping every member's full output. It is not a speed optimization —
        the parser still scans every field of every line, so reading a few
        columns costs about four fifths of reading them all.

        A column already in memory is never read again, so selecting variables
        one at a time costs the same as selecting them together.

        The time coordinates (``year``, ``day_of_year``, ``hour_of_day``) are
        always included, because they identify each row.

        Parameters
        ----------
        variables:
            Variable names or aliases (``"nee"``, ``"NEE"``,
            ``"net_ecosystem_exchange"`` are the same variable).
        format:
            ``"xarray"`` (default) for a Dataset carrying units, the time
            coordinate and the interval each value covers; ``"pandas"`` for a
            plain DataFrame. ``output[[...]]`` is shorthand for the default.

        Returns
        -------
        xarray.Dataset or pandas.DataFrame
            The Dataset has one dimension, ``time``, whose coordinate is the
            **end** of each timestep as ``datetime64``; ``timestep_start``,
            ``timestep_length`` and a CF ``time_bounds`` variable describing
            the interval each row covers; ``year``, ``day_of_year`` and
            ``hour_of_day`` for the start of the step, from the climate
            drivers when this output has them; and one
            data variable per selected column, carrying the attributes from
            :meth:`~pysipnet.variables.VariableSpec.xarray_attributes`. Columns
            the registry does not know become variables with no attributes.
        """
        if format not in ("xarray", "pandas"):
            raise ValueError(f"format must be 'xarray' or 'pandas', not {format!r}.")
        frame = self._select_frame(variables)
        return frame if format == "pandas" else self._build_dataset(frame)

    def __getitem__(self, key: str | Sequence[str]) -> xr.DataArray | xr.Dataset:
        """One variable as a DataArray, or several as a Dataset, by name or alias.

        ``output["nee"]``, ``output["NEE"]`` and
        ``output["net_ecosystem_exchange"]`` all return the same array, with
        units, description and time reference in ``.attrs``.
        ``output[["nee", "gpp"]]`` is shorthand for :meth:`select` and returns
        both in one Dataset, read in one go.

        A single array carries ``timestep_start`` and ``timestep_length``
        but not the two-dimensional ``time_bounds``, so its ``time`` has no
        ``bounds`` attribute; the Dataset from ``output[[...]]`` has both.
        """
        if isinstance(key, str):
            name = self._resolve([key])[0]
            if self._dataset is not None and name in self._dataset:
                return without_absent_bounds(self._dataset[name])
            return without_absent_bounds(self._build_dataset(self._select_frame([key]))[name])
        return self.select(key)

    @property
    def variables(self) -> tuple[VariableSpec, ...]:
        """Registry specs for the columns present, in column order.

        Columns the registry does not know (from a newer SIPNET) are omitted.
        Only the file itself lists its columns, so this reads all of them on a
        file-backed output that has so far read only a selection.
        """
        return tuple(
            OUTPUT_VARIABLES_BY_NAME[c]
            for c in self.pandas.columns
            if c in OUTPUT_VARIABLES_BY_NAME
        )

    @property
    def timestep_length(self) -> np.ndarray | None:
        """Timestep lengths in days, one per row, from the climate drivers.

        ``None`` when this output has no drivers; its Dataset then infers the
        lengths from the timestamps. Reads a file-backed climate.
        """
        if self.climate is None:
            return None
        return self.climate.pandas["timestep_length"].to_numpy(dtype=float)

    @property
    def n_timesteps(self) -> int:
        """Number of output timesteps.

        Any column already in memory answers this, so it does not pull the rest
        of the file in behind it. On a file-backed output that has read nothing
        yet, it reads the file.
        """
        if self._frame is not None:
            return len(self._frame)
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
                check_variable_is_written(name, self.flags)
        return names

    def _ensure_columns(self, names: Sequence[str]) -> None:
        """Read whichever of *names* is not in memory yet, in a single pass.

        The time coordinates count as needed whatever was asked for: they
        identify the rows, so an empty selection still has to read them.
        """
        needed = [*TIME_COORDINATE_NAMES, *names]
        have = set(self._frame.columns) if self._frame is not None else set()
        missing = [name for name in needed if name not in have]
        if not missing:
            return

        if self.source_path is None:
            self._report_absent(missing, have)

        import pandas as pd

        from pysipnet.io.output_reader import read_output_file

        new = read_output_file(self._require_source(), variables=missing)
        if self._frame is None:
            self._frame = new
        else:
            added = [c for c in new.columns if c not in have]
            self._frame = pd.concat([self._frame, new[added]], axis=1)

        # read_output_file returns what the file has, not what was asked for. A
        # name the registry knows but this file lacks would otherwise leave the
        # frame short, so every later call would read the file again and fail
        # somewhere inside pandas.
        still_missing = [name for name in needed if name not in self._frame.columns]
        if still_missing:
            self._report_absent(still_missing, set(self._frame.columns))

    def _report_absent(self, missing: Sequence[str], have: set[Any]) -> None:
        """Explain why a requested column cannot be produced, and stop.

        The time coordinates are needed for every selection but were not what
        the caller asked for, so they are named only when nothing else is
        missing.
        """
        asked_for = [name for name in missing if name not in TIME_COORDINATE_NAMES]
        missing = asked_for or missing
        if self.source_path is None:
            if not have:
                raise KeyError(
                    f"{list(missing)} cannot be read: this output is empty. SIPNET wrote no "
                    "rows, which usually means the run failed — check provenance.success and "
                    "provenance.stderr."
                )
            raise KeyError(
                f"{list(missing)} are not in this in-memory output, and there is no file to "
                "read them from. It was built from a DataFrame that does not contain them."
            )
        if not have:
            raise KeyError(
                f"{list(missing)} cannot be read: {self.source_path} is empty. SIPNET wrote no "
                "rows, which usually means the run failed — check provenance.success and "
                "provenance.stderr."
            )
        if any(not isinstance(column, str) for column in have):
            raise KeyError(
                f"{list(missing)} cannot be selected by name: {self.source_path} has no header "
                "row, so its columns are only known by position. Read it with .pandas and name "
                "the columns yourself."
            )
        raise KeyError(
            f"{list(missing)} are not in {self.source_path}. The variable registry knows the "
            "name, but this file does not contain it — a different SIPNET version, or a run "
            "whose flags left the column out."
        )

    def _select_frame(self, variables: Sequence[str]) -> pd.DataFrame:
        """The named columns plus the time coordinates, read if not already held."""
        names = self._resolve(variables)
        self._ensure_columns(names)
        assert self._frame is not None
        # Asking for a time coordinate by name must not list it twice; they are
        # always present.
        selected = [name for name in names if name not in TIME_COORDINATE_NAMES]
        return self._frame[[*TIME_COORDINATE_NAMES, *selected]]

    def _time_axis(self, df: pd.DataFrame) -> TimeAxis:
        """The shared time coordinates, built once per output.

        Every view of a run has the same rows, so the axis does not depend on
        which variables were selected: it is built on the first Dataset and
        reused by the rest.
        """
        if self._axis is None:
            self._axis = build_time_axis(df, attributes_for=_attributes_for, drivers=self.climate)
        return self._axis

    def _build_dataset(self, df: pd.DataFrame) -> xr.Dataset:
        import xarray as xr

        if df.empty:
            return xr.Dataset()
        return dataset_from_dataframe(
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


def build_output_dataset(df: pd.DataFrame, climate: ClimateDrivers | None = None) -> xr.Dataset:
    """Turn a parsed output DataFrame into a self-describing :class:`xarray.Dataset`.

    :func:`~pysipnet.dataset.build_xarray_dataset` with the output variable
    registry already supplied, for a frame that did not come from a
    :class:`SIPNETOutput` — one that did would give you the same thing through
    ``output[[...]]``. Pass the *climate* the rows were produced from to put
    them on its time axis; see :func:`~pysipnet.dataset.build_time_axis` for
    what happens without it.
    """
    from pysipnet.dataset import build_xarray_dataset

    return build_xarray_dataset(df, attributes_for=_attributes_for, drivers=climate, source=_SOURCE)


def _attributes_for(name: str) -> dict[str, Any]:
    spec = OUTPUT_VARIABLES_BY_NAME.get(name)
    return {} if spec is None else spec.xarray_attributes()
