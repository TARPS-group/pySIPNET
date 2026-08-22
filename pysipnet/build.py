"""Compile the SIPNET binary and check that it is present.

The ``Makefile`` at the repository root is the main way to build SIPNET. This
module wraps it so that scripts and tests can build and locate the binary
without shelling out to ``make`` themselves.

Typical usage::

    from pysipnet.build import build_sipnet, ensure_binary

    build_sipnet()     # compile, or do nothing if the binary already exists
    ensure_binary()    # return the path, or raise if it has not been built

There is a single binary. Every model option that pySIPNET can set — snow,
litter pool, nitrogen cycle, and the rest — is chosen at run time through the
``sipnet.in`` config file that :class:`pysipnet.runner.SIPNETRunner` writes, so
no compile-time configuration is involved.

The SIPNET source lives in the ``sipnet/`` git submodule. On a fresh clone the
submodule directory is empty; :func:`init_submodule` fills it in, and
:func:`build_sipnet` calls that for you.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_SIPNET_DIR = _REPO_ROOT / "sipnet"
_CACHE_DIR = _REPO_ROOT / ".sipnet_cache"

BINARY_NAME = "sipnet"
"""Filename of the compiled binary inside the cache directory."""


def binary_path() -> Path:
    """Return where the compiled SIPNET binary is expected to live.

    This is a pure path computation: it does not check whether the file
    exists. Use :func:`ensure_binary` when you need that guarantee.
    """
    return _CACHE_DIR / BINARY_NAME


def init_submodule() -> None:
    """Populate the ``sipnet/`` submodule if it is empty.

    A fresh ``git clone`` of pySIPNET leaves submodule directories empty until
    they are explicitly initialised. Missing ``sipnet/Makefile`` is the signal
    that this has not happened yet.
    """
    if not (_SIPNET_DIR / "Makefile").exists():
        subprocess.run(
            ["git", "submodule", "update", "--init", "sipnet"],
            cwd=_REPO_ROOT,
            check=True,
        )


def build_sipnet(*, force: bool = False) -> Path:
    """Compile SIPNET and return the path to the binary.

    Parameters
    ----------
    force:
        Compile even when a binary is already present. Use this after
        changing the pinned SIPNET version, since the existing binary was
        built from the previous source and this function otherwise trusts it.

    Returns
    -------
    Path
        Location of the compiled binary.
    """
    target = binary_path()
    if target.exists() and not force:
        return target

    init_submodule()
    subprocess.run(["make", "sipnet"], cwd=_REPO_ROOT, check=True)
    return target


def ensure_binary() -> Path:
    """Return the path to the compiled binary, raising if it is missing.

    This never compiles anything, which makes it the right check to run at the
    start of a batch of model runs: it fails immediately and with a useful
    message rather than part-way through.

    Raises
    ------
    FileNotFoundError
        If the binary has not been built yet.
    """
    target = binary_path()
    if not target.exists():
        raise FileNotFoundError(
            f"SIPNET binary not found at {target}. "
            "Build it by running 'make sipnet' from the repository root, "
            "or by calling pysipnet.build.build_sipnet()."
        )
    return target


def binary_sha256() -> str:
    """Return the SHA-256 digest of the compiled binary.

    Recording this alongside model output pins down exactly which build
    produced a given result, which matters when comparing runs made weeks
    apart or on different machines.
    """
    return hashlib.sha256(ensure_binary().read_bytes()).hexdigest()


def sipnet_version() -> str:
    """Return the version string reported by the compiled binary.

    Reads the version from the binary itself rather than from the submodule
    checkout, so the answer describes what will actually run.
    """
    result = subprocess.run(
        [str(ensure_binary()), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    # SIPNET prints e.g. "SIPNET version 2.1.0 (v2.1.0)"; keep everything
    # after the "version" keyword so the git tag suffix is preserved.
    _, _, version = result.stdout.strip().partition("version ")
    return version or result.stdout.strip()
