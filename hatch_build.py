"""Hatch build hook: bundle a SIPNET binary into a platform-specific wheel.

Off by default, so ``uv build`` produces the ordinary pure-Python wheel. When
``$PYSIPNET_BUNDLE_SIPNET`` names a platform key from
``pysipnet.version.SIPNET_WHEEL_PLATFORM_TAGS`` (``darwin-arm64``,
``linux-x86_64``), the binary that ``pysipnet stage-bundle <key>`` put at
``pysipnet/bin/<key>/sipnet`` is included in the wheel as
``pysipnet/bin/sipnet`` and the wheel is tagged for that platform, so pip
installs it only where the binary can run and takes the pure wheel elsewhere.
The staging directory is named by the same key as the tag, so a wheel cannot
carry the other platform's binary.

The hook does not download anything itself: that would mean re-implementing the
digest and archive checks in :mod:`pysipnet.build` without that module's
dependencies, since the build environment has only hatchling. Staging is a
separate, verified step, and this hook only decides what the wheel says about
itself. ``pysipnet/version.py`` is loaded by path for the same reason: it has
no imports, where importing the package would pull in pandas.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

BUNDLE_ENV_VAR = "PYSIPNET_BUNDLE_SIPNET"
BUNDLED_BINARY = Path("pysipnet") / "bin" / "sipnet"


def staged_binary(key: str) -> Path:
    """Where ``pysipnet stage-bundle <key>`` puts the binary, relative to the repo root."""
    return Path("pysipnet") / "bin" / key / "sipnet"


def _load_version_module(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_pysipnet_version", root / "pysipnet/version.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BundleSipnetHook(BuildHookInterface):  # type: ignore[type-arg]
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return
        key = os.environ.get(BUNDLE_ENV_VAR)
        if not key:
            return

        root = Path(self.root)
        tags = _load_version_module(root).SIPNET_WHEEL_PLATFORM_TAGS
        if key not in tags:
            raise RuntimeError(
                f"{BUNDLE_ENV_VAR}={key!r} is not a platform pySIPNET knows a binary for; "
                f"choose one of {sorted(tags)}."
            )
        binary = root / staged_binary(key)
        if not binary.is_file():
            raise RuntimeError(
                f"{BUNDLE_ENV_VAR} is set but there is no binary at {binary}. "
                f"Stage one first: pysipnet stage-bundle {key}"
            )

        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{tags[key]}"
        build_data["force_include"][str(binary)] = BUNDLED_BINARY.as_posix()
