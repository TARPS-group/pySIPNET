"""Get a SIPNET binary, and check that the one you have is the right one.

The ``Makefile`` at the repository root is the main way to build SIPNET. This
module wraps it so that scripts and tests can build and locate the binary
without shelling out to ``make`` themselves.

Two ways to get one::

    from pysipnet.build import build_sipnet, download_sipnet, ensure_binary

    build_sipnet()     # compile from the submodule source; works anywhere
    download_sipnet()  # fetch a prebuilt binary; no C toolchain needed
    ensure_binary()    # return the path, or raise if there is no binary yet

Compiling is the default. Downloading is available for the platforms upstream
publishes binaries for, verifies a pinned checksum before unpacking anything,
and never happens unless asked.

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
import http.client
import os
import platform
import re
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
            "Build it by running 'make sipnet' from the repository root, or "
            "fetch a prebuilt one with 'make sipnet-download'. The equivalent "
            "Python calls are build_sipnet() and download_sipnet()."
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


# The published archives are around 1 MB. These caps are generous enough never
# to bite in practice, and they bound the damage if the URL ever serves
# something else: the checksum cannot help, because it can only be computed
# after the bytes have already been read.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_UNPACKED_BYTES = 256 * 1024 * 1024

# An archive filename becomes a path and part of a URL, so it is restricted to
# a plain name. Without this, an absolute path in SIPNET_RELEASE_ASSETS would
# silently escape the temporary directory, because Path("/tmp") / "/etc/x" is
# "/etc/x" — the left operand is discarded.
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_filename(filename: str) -> str:
    """Return *filename* if it is a plain name, else raise.

    Guards the pinned data in :mod:`pysipnet.version` rather than the network:
    an edit there looks like a routine pin update, so it should not be able to
    turn into a write outside the download directory.
    """
    if not _SAFE_FILENAME.match(filename) or filename in {".", ".."}:
        raise DownloadError(
            f"Refusing to use archive filename {filename!r}. Filenames must be plain "
            "names without path separators. Check SIPNET_RELEASE_ASSETS in "
            "pysipnet/version.py."
        )
    return filename


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
    filename, digest = SIPNET_RELEASE_ASSETS[key]
    return _check_filename(filename), digest


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only while they stay on HTTPS.

    GitHub release downloads redirect to a storage host, so redirects are on
    the normal path and cannot simply be disabled. The stdlib handler accepts
    http and ftp targets as well, which would silently drop the connection to
    plaintext and expose the fetch to anyone on the network path.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise DownloadError(f"Refusing to follow a redirect to a non-HTTPS address: {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str, timeout: float):
    """Open *url* over HTTPS, refusing to be redirected off it.

    The one place this module touches the network. Kept separate so tests have
    a single, stable thing to intercept: patching a stdlib internal instead
    would quietly stop working the moment the implementation changed, and the
    tests would start making real requests without failing.
    """
    opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler)
    return opener.open(url, timeout=timeout)  # noqa: S310


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
    """Reject an archive that would write outside its directory or blow up on disk.

    An archive member may name any path it likes, including an absolute one or
    one climbing out with ``..``, and a symlink member can redirect a later
    write anywhere on disk. Extracting without checking is the "tar slip"
    vulnerability.

    The pinned checksum already makes a hostile archive unlikely. These checks
    cost nothing and do not depend on the checksum being right, which is the
    point of having them.
    """
    seen_lowercase: dict[str, str] = {}
    total_bytes = 0

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
        if member.mode is not None and member.mode & (0o4000 | 0o2000 | 0o0002):
            raise DownloadError(
                f"Refusing to extract archive: member {member.name!r} requests unsafe "
                f"permissions ({member.mode:#o}) — setuid, setgid or world-writable."
            )

        # Two members differing only in case collide into one file on a
        # case-insensitive filesystem such as macOS's default. An archive
        # reviewed on Linux, where both survive, would then install something
        # different here.
        lowered = member.name.lower()
        if lowered in seen_lowercase and seen_lowercase[lowered] != member.name:
            raise DownloadError(
                f"Refusing to extract archive: members {seen_lowercase[lowered]!r} and "
                f"{member.name!r} differ only in case and would collide on a "
                "case-insensitive filesystem."
            )
        seen_lowercase[lowered] = member.name

        total_bytes += max(member.size, 0)
        if member.size > MAX_UNPACKED_BYTES or total_bytes > MAX_UNPACKED_BYTES:
            raise DownloadError(
                f"Refusing to extract archive: it expands to at least "
                f"{total_bytes / 1e6:.0f} MB, over the {MAX_UNPACKED_BYTES / 1e6:.0f} MB "
                "limit. The real SIPNET archives are a few megabytes."
            )


def _find_binary(root: Path) -> Path:
    """Return the ``sipnet`` executable somewhere beneath *root*.

    The archive layout is upstream's to change, so this searches rather than
    assuming a path.

    Matching is on the exact name, not a glob, because ``rglob`` is
    case-insensitive on macOS and would happily return a file called
    ``SIPNET``. Ties are broken by path so the result does not depend on
    directory order, which varies by filesystem.
    """
    candidates = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name == BINARY_NAME),
        # Shallowest first, then alphabetical: a stray copy deeper in the tree
        # cannot displace one at the root, and equal depths resolve the same
        # way everywhere.
        key=lambda path: (len(path.relative_to(root).parts), path.parts),
    )
    if not candidates:
        contents = ", ".join(sorted(repr(str(q.relative_to(root))) for q in root.rglob("*")))
        raise DownloadError(
            f"No file named {BINARY_NAME!r} in the downloaded archive. "
            f"Contents: {contents or '(empty)'}"
        )
    return candidates[0]


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
    # Re-check here rather than trusting release_asset alone. The filename
    # becomes a path below, and Path("/tmp") / "/etc/x" is "/etc/x" — the left
    # operand is simply discarded — so an unchecked name is an arbitrary write,
    # and the write happens before the checksum can say anything about it.
    filename = _check_filename(filename)
    url = release_url(filename)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / filename

        try:
            with _open_url(url, timeout) as response:
                # Stream rather than read() in one go: the checksum can only be
                # computed after the bytes arrive, so an oversized body has to
                # be stopped while it is still arriving.
                downloaded = 0
                with archive.open("wb") as handle:
                    while chunk := response.read(65536):
                        downloaded += len(chunk)
                        if downloaded > MAX_ARCHIVE_BYTES:
                            raise DownloadError(
                                f"Refusing to download more than "
                                f"{MAX_ARCHIVE_BYTES / 1e6:.0f} MB from {url}. The real "
                                "SIPNET archives are a few megabytes."
                            )
                        handle.write(chunk)
        except DownloadError:
            raise
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
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

        # Install in two steps. The new binary is staged beside the target and
        # checked there, then moved into place with os.replace, which is
        # atomic. Overwriting the target directly would destroy a working
        # binary whenever the new one turns out to be unusable, and would let
        # a concurrent run observe a half-written executable.
        staged = target.with_name(target.name + ".incoming")
        try:
            shutil.copy2(source, staged)
            staged.chmod(0o755)
            _check_staged_binary(staged)
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)

    return target


def _check_staged_binary(path: Path) -> None:
    """Confirm a freshly unpacked binary is the release we expect.

    Asks the binary what it is rather than trusting the filename it arrived
    under. A mismatch means the pinned asset and ``SIPNET_TARGET_VERSION`` have
    drifted apart, which is what a half-finished pin bump looks like.
    """
    try:
        result = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, check=True
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise DownloadError(f"The downloaded binary would not run: {exc}.") from exc

    _, _, reported = result.stdout.strip().partition("version ")
    reported = reported or result.stdout.strip()

    expected = SIPNET_TARGET_VERSION.removeprefix("v")
    # Match on the full version token so "2.1.05" and "2.1.0-rc1" do not pass
    # as "2.1.0".
    if not re.match(rf"{re.escape(expected)}(?![\w.])", reported):
        raise DownloadError(
            f"The downloaded binary reports version {reported!r}, but this "
            f"pySIPNET targets {SIPNET_TARGET_VERSION!r}. It was not installed. "
            "The pinned asset in pysipnet.version is probably out of step with "
            "SIPNET_TARGET_VERSION."
        )
