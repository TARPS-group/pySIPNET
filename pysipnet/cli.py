"""The ``pysipnet`` command: install the SIPNET binary, and say where things are.

Installing the Python package does not install SIPNET, so this is the one
step a user has to take after ``pip install``::

    pysipnet install-sipnet              # download if published for this machine, else compile
    pysipnet install-sipnet --method compile
    pysipnet info                        # what pySIPNET pins, where it looked, what it found

Everything here is a thin layer over :mod:`pysipnet.build`; the command exists
so that a non-developer never has to open Python to get set up, and so the
same verb works on a laptop, a cluster login node and in CI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pysipnet.build import (
    _BUNDLED_DIR,
    BINARY_ENV_VAR,
    BINARY_NAME,
    CACHE_DIR_ENV_VAR,
    BinaryVersionError,
    BuildError,
    DownloadError,
    describe_binary_search,
    fetch_release_binary,
    find_binary,
    install_sipnet,
    install_target,
    platform_key,
    prebuilt_unavailable_reason,
    verify_binary_matches_pin,
)
from pysipnet.version import (
    PYSIPNET_VERSION,
    SIPNET_PINNED_COMMIT,
    SIPNET_PINNED_TAG,
    SIPNET_RELEASE_ASSETS,
)


def staged_bundle_path(key: str) -> Path:
    """Where ``stage-bundle`` puts the binary for platform *key*: ``pysipnet/bin/<key>/sipnet``.

    One directory per platform, so the wheel build can only bundle the binary
    whose directory matches the platform it is tagging the wheel for.
    """
    return _BUNDLED_DIR / key / BINARY_NAME


def _cmd_install(args: argparse.Namespace) -> int:
    try:
        path = install_sipnet(method=args.method, force=args.force)
        version = verify_binary_matches_pin(path)
    except (DownloadError, BuildError, BinaryVersionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"SIPNET {version} at {path}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"pySIPNET {PYSIPNET_VERSION}")
    print(f"pins SIPNET {SIPNET_PINNED_TAG} ({SIPNET_PINNED_COMMIT})")
    print(f"platform {platform_key()}")
    reason = prebuilt_unavailable_reason()
    if reason is None:
        print("prebuilt binary: available for this machine")
    else:
        print(f"prebuilt binary: not usable here — {reason}")
    print(f"install target: {install_target()}")
    print(f"environment: ${BINARY_ENV_VAR} names a binary, ${CACHE_DIR_ENV_VAR} moves the cache")
    print("searched:")
    print(describe_binary_search())
    found = find_binary()
    if found is None:
        print("no binary found; run 'pysipnet install-sipnet'")
        return 1
    try:
        version = verify_binary_matches_pin(found.path)
    except BinaryVersionError as exc:
        print(f"binary: {exc}")
        return 1
    print(f"using: {found.path} ({found.source}), SIPNET {version}")
    return 0


def _cmd_stage_bundle(args: argparse.Namespace) -> int:
    """Put the published binary for a platform at ``pysipnet/bin/<platform>/sipnet``.

    The step before building a platform wheel: ``hatch_build.py`` bundles the
    binary in the directory named by ``$PYSIPNET_BUNDLE_SIPNET``. The binary is
    verified by digest; it is run to confirm its version only when this
    machine can execute it.
    """
    key: str = args.platform
    try:
        path = fetch_release_binary(key, staged_bundle_path(key))
    except (DownloadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"staged {key} binary at {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pysipnet", description="Set up the SIPNET binary that pySIPNET drives."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser(
        "install-sipnet",
        help="download the published SIPNET binary, or compile the pinned source",
    )
    install.add_argument(
        "--method",
        choices=("auto", "download", "compile"),
        default="auto",
        help="auto (default): download when a usable binary is published for this machine, "
        "else compile; download or compile: insist on that route",
    )
    install.add_argument(
        "--force", action="store_true", help="replace a binary already at the install target"
    )
    install.set_defaults(func=_cmd_install)

    info = sub.add_parser("info", help="show the pin, the search order, and the binary found")
    info.set_defaults(func=_cmd_info)

    stage = sub.add_parser(
        "stage-bundle",
        help="stage a published binary at pysipnet/bin/<platform>/ for building a platform wheel",
    )
    stage.add_argument("platform", choices=sorted(SIPNET_RELEASE_ASSETS))
    stage.set_defaults(func=_cmd_stage_bundle)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
