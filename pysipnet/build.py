"""SIPNET binary compilation and cache management.

Programmatic interface to the build system.  The ``Makefile`` at the repo root
is the primary build entry point; this module provides a Python API for
scripted builds and for verifying the binary cache.

Typical usage::

    from pysipnet.build import build_preset, ensure_binary
    from pysipnet.runner import ModelPreset

    build_preset(ModelPreset.FOREST)          # compile (or skip if up to date)
    ensure_binary(ModelPreset.STANDARD)       # raise if binary missing

Submodule initialisation
------------------------
The SIPNET source lives in the ``sipnet/`` git submodule.  If the submodule
has not been initialised (e.g., after a fresh ``git clone``), call
:func:`init_submodule` before building.

Patch application
-----------------
The build process applies a source patch before compiling that wraps SIPNET's
compile-time flag ``#define`` statements with ``#ifndef`` guards.  This
enables flag overrides via compiler ``-D`` arguments.  The patch is
idempotent and is applied automatically by :func:`build_preset`.

See ``patches/apply_flags_patch.py`` for details.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysipnet.runner import ModelPreset

_REPO_ROOT = Path(__file__).parent.parent
_SIPNET_DIR = _REPO_ROOT / "sipnet"
_CACHE_DIR = _REPO_ROOT / ".sipnet_cache"
_PATCH_SCRIPT = _REPO_ROOT / "patches" / "apply_flags_patch.py"


def init_submodule() -> None:
    """Initialise and update the SIPNET git submodule if needed."""
    if not (_SIPNET_DIR / "Makefile").exists():
        subprocess.run(
            ["git", "submodule", "update", "--init", "sipnet"],
            cwd=_REPO_ROOT,
            check=True,
        )


def apply_patch() -> None:
    """Apply the ``#ifndef`` flag patch to the SIPNET source (idempotent)."""
    subprocess.run(
        [sys.executable, str(_PATCH_SCRIPT), str(_SIPNET_DIR)],
        check=True,
    )


def build_preset(preset: ModelPreset, *, force: bool = False) -> Path:
    """Compile a SIPNET binary for the given *preset*.

    Parameters
    ----------
    preset:
        The :class:`~pysipnet.runner.ModelPreset` to build.
    force:
        If ``True``, rebuild even if the binary already exists.

    Returns
    -------
    Path
        Path to the compiled binary.
    """

    target = _CACHE_DIR / preset.binary_name
    if target.exists() and not force:
        return target

    init_submodule()
    apply_patch()

    make_target = f"sipnet-{preset.value}"
    subprocess.run(
        ["make", make_target],
        cwd=_REPO_ROOT,
        check=True,
    )
    return target


def ensure_binary(preset: ModelPreset) -> Path:
    """Return the path to the binary for *preset*, raising if it does not exist.

    Unlike :func:`build_preset`, this does not attempt to compile.
    """
    target = _CACHE_DIR / preset.binary_name
    if not target.exists():
        raise FileNotFoundError(
            f"SIPNET binary not found: {target}. "
            f"Run 'make sipnet-{preset.value}' or call build_preset({preset!r})."
        )
    return target


def binary_sha256(preset: ModelPreset) -> str:
    """Return the SHA-256 hex digest of the compiled binary for *preset*."""
    path = ensure_binary(preset)
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()
