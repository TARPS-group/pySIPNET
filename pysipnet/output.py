"""SIPNET model output container.

:class:`SIPNETOutput` holds the parsed ``sipnet.out`` file either in memory
(eager) or as a reference to a file on disk (lazy).  The two modes are created
via factory methods:

- :meth:`SIPNETOutput.from_dataframe` — memory-backed; data immediately available.
- :meth:`SIPNETOutput.from_path` — file-backed; the file is not read until
  :attr:`data` is first accessed.  The file is verified to exist at construction
  time so that a missing file is caught immediately rather than when the caller
  later tries to access the data.

The file-backed mode is the natural choice when :class:`~pysipnet.runner.SIPNETRunner`
is configured with an ``output_dir``: the runner copies ``sipnet.out`` there
before deleting the temporary working directory, then returns an
``SIPNETOutput.from_path(...)`` pointing at the persistent copy.  In a
1 000-member ensemble this means zero DataFrames are held in memory until the
caller explicitly requests them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Time-coordinate columns always included when columns= is specified.
_TIME_COLS: frozenset[str] = frozenset({"year", "day", "time"})


class SIPNETOutput:
    """Parsed SIPNET ``.out`` output, memory-backed or file-backed.

    Parameters
    ----------
    data:
        In-memory DataFrame.  Mutually exclusive with *source_path*.
    source_path:
        Path to a persistent ``.out`` file.  Mutually exclusive with *data*.
        The file is checked for existence at construction time.

    See the factory methods :meth:`from_path` and :meth:`from_dataframe` for
    the public construction interface.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | None = None,
        source_path: Path | None = None,
    ) -> None:
        if (data is None) == (source_path is None):
            raise ValueError(
                "Exactly one of 'data' or 'source_path' must be provided, not both or neither."
            )
        self._data: pd.DataFrame | None = data
        self.source_path: Path | None = source_path

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_path(cls, path: str | Path) -> SIPNETOutput:
        """Create a file-backed instance without reading the output into memory.

        The file is not parsed until :attr:`data` is accessed.  However, the
        file's existence is verified immediately so that a missing or
        prematurely deleted file is detected at the point of construction rather
        than later at access time.

        Parameters
        ----------
        path:
            Path to a ``sipnet.out`` file that will persist for the lifetime of
            this object.  Never point this at a file inside a temporary working
            directory that will be deleted — use
            :attr:`~pysipnet.runner.SIPNETRunner.output_dir` to ensure the file
            is copied to a stable location first.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"SIPNET output file not found: {path}\n"
                "Ensure the file is in a stable location outside the run's working "
                "directory, which is deleted after each run."
            )
        return cls(source_path=path)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SIPNETOutput:
        """Create a memory-backed instance from an already-parsed DataFrame.

        Parameters
        ----------
        df:
            DataFrame returned by :func:`~pysipnet.io.output_reader.read_output_file`.
        """
        return cls(data=df)

    # ── Data access ────────────────────────────────────────────────────────────

    @property
    def data(self) -> pd.DataFrame:
        """The full output timeseries as a DataFrame.

        For file-backed instances, the first access reads and caches the output
        file from :attr:`source_path`.  Subsequent accesses return the cached
        copy at no cost.

        Column names use ``snake_case`` translations of SIPNET's ``camelCase``
        output headers (e.g. ``plant_wood_c``, ``nee``, ``gpp``).
        """
        if self._data is None:
            from pysipnet.io.output_reader import read_output_file

            self._data = read_output_file(self.source_path)
        return self._data

    def load(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Explicitly load output data, optionally restricting to a column subset.

        Unlike :attr:`data`, this method does **not** cache its result when
        *columns* is specified — each call reads the file afresh so that only
        the requested columns are held in memory.  This is the preferred
        pattern for memory-constrained ensemble post-processing:

        .. code-block:: python

            nee_frames = [r.outputs.load(columns=["nee"]) for r in results]

        The time-coordinate columns (``year``, ``day``, ``time``) are always
        included in the returned DataFrame regardless of what *columns*
        specifies.

        Parameters
        ----------
        columns:
            Column names to return, using ``snake_case`` names (e.g.
            ``"nee"``, ``"gpp"``, ``"plant_wood_c"``).  ``None`` returns all
            columns and has the same behaviour as accessing :attr:`data`.

        Returns
        -------
        pandas.DataFrame
            Requested columns plus ``year``, ``day``, ``time``.
        """
        if columns is None:
            return self.data

        requested = set(columns)
        all_cols = list(_TIME_COLS | requested)

        if self._data is not None:
            available = [c for c in all_cols if c in self._data.columns]
            return self._data[available]

        from pysipnet.io.output_reader import read_output_file

        return read_output_file(self.source_path, columns=list(requested))

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
        return f"SIPNETOutput(in_memory, timesteps={len(self._data)})"
