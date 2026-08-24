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
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from pysipnet.version import (
    SIPNET_RELEASE_ASSETS,
    SIPNET_RELEASE_REPO,
    SIPNET_RELEASE_TAG,
    SIPNET_TARGET_VERSION,
)

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


# ── Downloading a prebuilt binary ─────────────────────────────────────────────
#
# The SIPNET project publishes compiled binaries with each release, so a user
# without a C toolchain can still run the model. Compiling from source remains
# the default and works everywhere; this is a convenience, and it is never
# automatic.
#
# What arrives over the network is executed, so it is checked first. The
# SHA-256 of each archive is pinned in pysipnet.version and verified before
# anything is unpacked, and the archive's members are inspected before
# extraction so a hostile path cannot write outside the destination.


class DownloadError(RuntimeError):
    """A prebuilt binary could not be fetched, verified, or unpacked."""


def platform_key() -> str:
    """Return the key identifying this machine in :data:`SIPNET_RELEASE_ASSETS`.

    Normalises what :mod:`platform` reports, since the same architecture goes
    by more than one name: ``arm64`` and ``aarch64`` are the same thing, as are
    ``x86_64`` and ``AMD64``.

    Returns a key even for platforms upstream does not build for; use
    :func:`release_asset` to find out whether an archive actually exists.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    architectures = {
        "arm64": "arm64",
        "aarch64": "arm64",
        "x86_64": "x86_64",
        "amd64": "x86_64",
    }
    return f"{system}-{architectures.get(machine, machine)}"


def release_asset(key: str | None = None) -> tuple[str, str]:
    """Return the ``(filename, sha256)`` published for a platform.

    Parameters
    ----------
    key:
        A platform key as returned by :func:`platform_key`. Defaults to this
        machine's.

    Raises
    ------
    DownloadError
        If upstream publishes no binary for that platform. Compiling from
        source is always available, so the message says so.
    """
    key = key or platform_key()
    if key not in SIPNET_RELEASE_ASSETS:
        supported = ", ".join(sorted(SIPNET_RELEASE_ASSETS))
        raise DownloadError(
            f"No prebuilt SIPNET binary is published for {key}. "
            f"Prebuilt binaries exist for: {supported}. "
            "Build from source instead with 'make sipnet', which works on any "
            "platform with a C compiler."
        )
    return SIPNET_RELEASE_ASSETS[key]


def release_url(filename: str) -> str:
    """Return the download URL for a release archive."""
    return (
        f"https://github.com/{SIPNET_RELEASE_REPO}/releases/download/"
        f"{SIPNET_RELEASE_TAG}/{filename}"
    )


def _sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks so size does not matter."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_archive_members(archive: tarfile.TarFile) -> None:
    """Reject an archive that would write outside the directory it unpacks into.

    An archive member may name any path it likes, including an absolute one or
    one climbing out with ``..``, and a symlink member can redirect a later
    write anywhere on disk. Extracting without checking is the "tar slip"
    vulnerability.

    The pinned checksum already makes a hostile archive very unlikely, but this
    costs nothing and does not depend on the checksum being right.
    """
    for member in archive.getmembers():
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise DownloadError(
                f"Refusing to extract archive: member {member.name!r} points outside "
                "the destination directory."
            )
        if member.issym() or member.islnk():
            raise DownloadError(
                f"Refusing to extract archive: member {member.name!r} is a link, "
                "which could redirect a write outside the destination directory."
            )
        if not (member.isfile() or member.isdir()):
            raise DownloadError(
                f"Refusing to extract archive: member {member.name!r} is neither a "
                "regular file nor a directory."
            )


def _find_binary(root: Path) -> Path:
    """Return the ``sipnet`` executable somewhere beneath *root*.

    The archive layout is upstream's to change, so this searches rather than
    assuming a path. A layout change should not break the download.
    """
    candidates = [p for p in root.rglob(BINARY_NAME) if p.is_file()]
    if not candidates:
        contents = ", ".join(sorted(str(p.relative_to(root)) for p in root.rglob("*"))) or "(empty)"
        raise DownloadError(
            f"No file named {BINARY_NAME!r} in the downloaded archive. Contents: {contents}"
        )
    # Prefer the shallowest match, so a stray copy in a subdirectory cannot win.
    return min(candidates, key=lambda p: len(p.relative_to(root).parts))


def download_sipnet(*, force: bool = False, timeout: float = 120.0) -> Path:
    """Fetch the prebuilt SIPNET binary for this platform and install it.

    An alternative to :func:`build_sipnet` for machines without a C compiler.
    Compiling gives the same result and works on every platform, so prefer it
    when a toolchain is available.

    The archive is verified against the SHA-256 pinned in
    :data:`pysipnet.version.SIPNET_RELEASE_ASSETS` before anything is unpacked,
    and the installed binary is then asked for its version to confirm it is the
    release this pySIPNET targets.

    Parameters
    ----------
    force:
        Replace an existing binary. Without this, an existing binary is kept
        and returned unchanged.
    timeout:
        Seconds to wait for the download.

    Returns
    -------
    Path
        Location of the installed binary.

    Raises
    ------
    DownloadError
        If no binary is published for this platform, the download fails, the
        checksum does not match, the archive looks unsafe, or the installed
        binary reports an unexpected version.
    """
    target = binary_path()
    if target.exists() and not force:
        return target

    filename, expected_sha256 = release_asset()
    url = release_url(filename)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / filename

        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                archive.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DownloadError(
                f"Could not download {url}: {exc}. Build from source instead with 'make sipnet'."
            ) from exc

        actual_sha256 = _sha256_of(archive)
        if actual_sha256 != expected_sha256:
            raise DownloadError(
                f"Checksum mismatch for {filename}.\n"
                f"  expected {expected_sha256}\n"
                f"  actual   {actual_sha256}\n"
                "The download was not installed. Either the file was corrupted in "
                "transit, or the pinned checksum in pysipnet.version is stale — "
                "check it against the release before changing it."
            )

        unpacked = tmp_path / "unpacked"
        unpacked.mkdir()
        try:
            with tarfile.open(archive, "r:gz") as tar:
                _check_archive_members(tar)
                # Python's own "data" filter rejects the same things
                # _check_archive_members does, and a few more besides. Use it
                # where available as a second line of defence; it also settles
                # the extraction behaviour, which otherwise differs by version
                # and warns on 3.12 and 3.13. Absent only on Python 3.11.0-3.11.3,
                # where the explicit check above still applies.
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(unpacked, filter="data")  # noqa: S202
                else:
                    tar.extractall(unpacked)  # noqa: S202 - members checked above
        except tarfile.TarError as exc:
            raise DownloadError(f"Could not unpack {filename}: {exc}") from exc

        source = _find_binary(unpacked)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o755)

    # The binary is on disk now, so ask it what it is rather than trusting the
    # filename. A mismatch means the pinned asset and the pinned version have
    # drifted apart.
    try:
        reported = sipnet_version()
    except (subprocess.SubprocessError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise DownloadError(
            f"The downloaded binary would not run: {exc}. It has been removed."
        ) from exc

    if not reported.startswith(SIPNET_TARGET_VERSION.lstrip("v")):
        target.unlink(missing_ok=True)
        raise DownloadError(
            f"The downloaded binary reports version {reported!r}, but this "
            f"pySIPNET targets {SIPNET_TARGET_VERSION!r}. It has been removed. "
            "The pinned asset in pysipnet.version is probably out of step with "
            "SIPNET_TARGET_VERSION."
        )

    return target
