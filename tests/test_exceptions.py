"""pySIPNET's exceptions survive pickling, so they cross process boundaries.

Ensemble runners execute runs in worker processes and send failures back to
the driver through pickle. ``BaseException`` rebuilds an exception from
``cls(*self.args)``, so a class whose ``__init__`` takes more than the message
pickles without complaint and then fails to unpickle — and in a
``ProcessPoolExecutor`` that failure breaks the pool for every later run.
"""

from __future__ import annotations

import copy
import importlib
import multiprocessing
import pickle
import pkgutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

import pysipnet
from pysipnet.build import BinaryVersionError, BuildError, DownloadError
from pysipnet.io.output_reader import UnknownOutputColumnWarning
from pysipnet.runner import SIPNETRunError

_RUN_ERROR_FIELDS = ("returncode", "stdout", "stderr", "workdir")

EXAMPLE_EXCEPTIONS: dict[type[BaseException], Callable[[], BaseException]] = {
    SIPNETRunError: lambda: SIPNETRunError(
        "SIPNET exited with code 1",
        returncode=1,
        stdout="reading climate\n",
        stderr="Error: no climate data\n",
        workdir=Path("/scratch/run-7"),
    ),
    BinaryVersionError: lambda: BinaryVersionError("wrong tag"),
    DownloadError: lambda: DownloadError("checksum mismatch"),
    BuildError: lambda: BuildError("make failed"),
    UnknownOutputColumnWarning: lambda: UnknownOutputColumnWarning("column 'foo'"),
}

# Modules whose import needs a dependency pySIPNET does not require.
OPTIONAL_DEPENDENCY_MODULES = {"pysipnet.ensemble"}


class _SubclassedRunError(SIPNETRunError):
    pass


def _make_run_error(cls: type[SIPNETRunError] = SIPNETRunError) -> SIPNETRunError:
    return cls(
        "SIPNET exited with code 3",
        returncode=3,
        stdout="out",
        stderr="err",
        workdir=Path("/scratch/run-3"),
    )


def _raise_run_error(returncode: int) -> None:
    raise SIPNETRunError(
        f"SIPNET exited with code {returncode}",
        returncode=returncode,
        stdout="out",
        stderr="err",
        workdir=Path(f"/scratch/run-{returncode}"),
    )


def _return_value(value: int) -> int:
    return value


def _assert_same_run_error(restored: BaseException, original: SIPNETRunError) -> None:
    assert type(restored) is type(original)
    assert isinstance(restored, SIPNETRunError)
    assert restored.args == original.args
    assert str(restored) == str(original)
    for name in _RUN_ERROR_FIELDS:
        assert getattr(restored, name) == getattr(original, name), name
    assert isinstance(restored.workdir, Path)


def _pysipnet_exception_classes() -> set[type[BaseException]]:
    found: set[type[BaseException]] = set()
    for info in pkgutil.walk_packages(pysipnet.__path__, prefix="pysipnet."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            if info.name in OPTIONAL_DEPENDENCY_MODULES:
                continue
            raise
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, BaseException)
                and value.__module__ == module.__name__
            ):
                found.add(value)
    return found


class TestRunErrorRoundTrip:
    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_keeps_type_message_and_attributes(self, protocol):
        err = _make_run_error()
        _assert_same_run_error(pickle.loads(pickle.dumps(err, protocol=protocol)), err)

    @pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
    def test_copy_keeps_type_message_and_attributes(self, copier):
        err = _make_run_error()
        _assert_same_run_error(copier(err), err)

    def test_notes_and_extra_attributes_survive(self):
        err = _make_run_error()
        err.add_note("member 12 of the ensemble")
        err.member = 12  # type: ignore[attr-defined]
        restored = pickle.loads(pickle.dumps(err))
        assert restored.__notes__ == ["member 12 of the ensemble"]
        assert restored.member == 12

    def test_a_subclass_comes_back_as_itself(self):
        err = _make_run_error(_SubclassedRunError)
        _assert_same_run_error(pickle.loads(pickle.dumps(err)), err)


class TestAcrossAProcessBoundary:
    """What an ensemble runner actually does with a failed run."""

    def test_error_raised_in_a_worker_arrives_intact(self):
        spawn = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
            failed = pool.submit(_raise_run_error, 4).exception()
            later = pool.submit(_return_value, 5).result()

        assert isinstance(failed, SIPNETRunError)
        assert failed.returncode == 4
        assert failed.stdout == "out"
        assert failed.stderr == "err"
        assert failed.workdir == Path("/scratch/run-4")
        assert str(failed) == "SIPNET exited with code 4"
        assert later == 5, "the failed run broke the pool for the next one"


class TestEveryExceptionPickles:
    def test_every_exception_class_has_an_example(self):
        """A new exception class must be added to EXAMPLE_EXCEPTIONS.

        That is the point: a class with a custom ``__init__`` can reintroduce
        this failure, so each one has to be round-tripped here.
        """
        missing = _pysipnet_exception_classes() - EXAMPLE_EXCEPTIONS.keys()
        assert not missing, f"add examples for {sorted(c.__qualname__ for c in missing)}"

    @pytest.mark.parametrize("cls", list(EXAMPLE_EXCEPTIONS), ids=lambda c: c.__qualname__)
    def test_round_trips(self, cls):
        err = EXAMPLE_EXCEPTIONS[cls]()
        err.add_note("note")
        restored = pickle.loads(pickle.dumps(err))
        assert type(restored) is cls
        assert restored.args == err.args
        assert str(restored) == str(err)
        assert vars(restored) == vars(err)
