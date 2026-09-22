"""Find, fetch or compile the SIPNET binary, and check it is the pinned one.

Installing the Python package does not install SIPNET. Getting a binary is an
explicit step, and this module is where it happens::

    pysipnet install-sipnet          # from the shell; downloads or compiles
    pysipnet info                    # where pySIPNET looked and what it found

    from pysipnet.build import install_sipnet, ensure_binary
    install_sipnet()                 # the same, from Python
    ensure_binary()                  # the path, or raise if there is none yet

Where the binary is looked for
------------------------------
:func:`binary_candidates` lists the places, in order; the first that exists
wins:

1. ``$PYSIPNET_BINARY`` — an explicit path, for a cluster module or a shared
   build.
2. ``pysipnet/bin/sipnet`` inside the installed package — present only in a
   platform-specific wheel that bundles the binary.
3. ``.sipnet_cache/sipnet`` at the repository root — when pySIPNET is running
   from a source checkout, which is where ``make sipnet`` puts it.
4. A per-user cache directory named by the pinned SIPNET commit, under
   ``platformdirs.user_cache_dir("pysipnet")`` or ``$PYSIPNET_CACHE_DIR``.
   This is where :func:`install_sipnet` puts a binary outside a checkout.

Naming the user-cache directory by commit means a pin bump can never pick up a
stale binary from there. The checkout cache is flat, so it can go stale; the
runner asks every binary for its version before the first run and refuses one
built from a different tag (:func:`verify_binary_matches_pin`), which is what
catches that.

Getting one
-----------
:func:`download_sipnet` fetches the archive upstream publishes for this
platform, checks its SHA-256 against the digest pinned in
:mod:`pysipnet.version` before unpacking, and asks the installed binary for its
version. :func:`build_sipnet` compiles: from the ``sipnet/`` submodule with
``make sipnet`` in a checkout, and outside one by fetching the pinned commit
with git and running SIPNET's own Makefile. :func:`install_sipnet` picks between
them. None of them run unless asked, and none of them run inside
:class:`~pysipnet.runner.SIPNETRunner`: a model run that reached for the
network would be a surprise inside an ensemble and a failure on a compute node.

There is a single binary. Every model option pySIPNET can set is chosen at run
time through the ``sipnet.in`` file the runner writes, so no compile-time
configuration is involved.
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
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any, Literal

import platformdirs

from pysipnet.version import (
    SIPNET_PINNED_COMMIT,
    SIPNET_PINNED_TAG,
    SIPNET_PREBUILT_REQUIREMENTS,
    SIPNET_RELEASE_ASSETS,
    SIPNET_RELEASE_REPO,
    SIPNET_RELEASE_TAG,
    SIPNET_SOURCE_REPO_URL,
)

_REPO_ROOT = Path(__file__).parent.parent
_SIPNET_DIR = _REPO_ROOT / "sipnet"
_CACHE_DIR = _REPO_ROOT / ".sipnet_cache"
_BUNDLED_DIR = Path(__file__).parent / "bin"

BINARY_NAME = "sipnet"
"""Filename of the compiled binary inside any of the candidate directories."""

BINARY_ENV_VAR = "PYSIPNET_BINARY"
"""Environment variable naming an explicit binary; checked before anything else."""

CACHE_DIR_ENV_VAR = "PYSIPNET_CACHE_DIR"
"""Environment variable replacing the per-user cache root, e.g. for a shared filesystem."""

BinarySource = Literal["argument", "environment", "bundled", "source tree", "user cache"]


class BinaryVersionError(RuntimeError):
    """The SIPNET binary found was built from a different source than this pySIPNET pins."""


@dataclass(frozen=True)
class BinaryCandidate:
    """One place a SIPNET binary may be, and whether it is there."""

    source: BinarySource
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.is_file()


def in_source_tree() -> bool:
    """Whether pySIPNET is running from a checkout rather than an installed wheel.

    A checkout has the repository Makefile beside ``pyproject.toml``; a wheel
    installed into ``site-packages`` has neither. Editable installs count as a
    checkout, which is what a developer wants.
    """
    return (_REPO_ROOT / "Makefile").is_file() and (_REPO_ROOT / "pyproject.toml").is_file()


def user_cache_dir() -> Path:
    """The per-user directory a binary is installed into outside a checkout.

    Named by the pinned commit, so a pin bump looks in a new, empty directory
    rather than finding the previous version's binary. ``$PYSIPNET_CACHE_DIR``
    replaces the platform default root, for a cluster where home directories
    are small or a scratch filesystem is shared between nodes.
    """
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    root = (
        Path(override).expanduser() if override else Path(platformdirs.user_cache_dir("pysipnet"))
    )
    return root / "sipnet" / SIPNET_PINNED_COMMIT[:12]


def binary_candidates() -> list[BinaryCandidate]:
    """Every place a binary is looked for, in the order they are tried.

    The environment variable comes first because it is the only one a user
    sets deliberately. A bundled binary is guaranteed to match the pin, so it
    outranks the caches. The checkout cache comes before the user cache so a
    developer's ``make sipnet`` is what their tests run.
    """
    candidates: list[BinaryCandidate] = []
    explicit = os.environ.get(BINARY_ENV_VAR)
    if explicit:
        candidates.append(BinaryCandidate("environment", Path(explicit).expanduser()))
    candidates.append(BinaryCandidate("bundled", _BUNDLED_DIR / BINARY_NAME))
    if in_source_tree():
        candidates.append(BinaryCandidate("source tree", _CACHE_DIR / BINARY_NAME))
    candidates.append(BinaryCandidate("user cache", user_cache_dir() / BINARY_NAME))
    return candidates


def find_binary() -> BinaryCandidate | None:
    """The first candidate that exists, or ``None`` when there is no binary anywhere."""
    return next((c for c in binary_candidates() if c.exists), None)


def install_target() -> Path:
    """Where :func:`install_sipnet` puts a binary on this machine.

    The checkout cache in a source tree, because that is where ``make sipnet``
    and the tests expect it; the per-user cache everywhere else.
    """
    if in_source_tree():
        return _CACHE_DIR / BINARY_NAME
    return user_cache_dir() / BINARY_NAME


def binary_path() -> Path:
    """The binary pySIPNET will run: the first existing candidate, else the install target.

    The second half of that makes the return value meaningful before anything
    is installed — it is where a binary *would* go — so callers can print it in
    a message. Use :func:`ensure_binary` when the file has to exist.
    """
    found = find_binary()
    return found.path if found is not None else install_target()


def describe_binary_search() -> str:
    """One line per candidate saying whether it was there.

    Used in error messages and by ``pysipnet info``.
    """
    lines = []
    for candidate in binary_candidates():
        state = "found" if candidate.exists else "not found"
        lines.append(f"  {candidate.source:<12} {candidate.path}  ({state})")
    return "\n".join(lines)


def init_submodule() -> None:
    """Populate the ``sipnet/`` submodule if it is empty.

    A fresh ``git clone`` of pySIPNET leaves submodule directories empty until
    they are explicitly initialized. Missing ``sipnet/Makefile`` is the signal
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

    In a source checkout this runs ``make sipnet``, which compiles the
    ``sipnet/`` submodule into ``.sipnet_cache/``. Anywhere else it fetches
    the pinned commit from upstream with git into the user cache and runs
    SIPNET's own Makefile there, so compiling works from an installed package
    too. Both need ``make`` and a C compiler; the second also needs ``git``.

    Parameters
    ----------
    force:
        Compile even when a binary is already present at the install target.
        Use this after changing the pinned SIPNET version in a checkout, since
        the checkout cache is not named by commit and would otherwise be
        trusted.

    Returns
    -------
    Path
        Location of the compiled binary.
    """
    target = install_target()
    if target.exists() and not force:
        return target

    if in_source_tree():
        init_submodule()
        subprocess.run(["make", "sipnet"], cwd=_REPO_ROOT, check=True)
        return target

    _compile_pinned_source(target)
    return target


def _require_tool(name: str, purpose: str) -> None:
    if shutil.which(name) is None:
        raise BuildError(
            f"'{name}' was not found on PATH, and it is needed to {purpose}. "
            "Install it, or fetch a prebuilt binary with 'pysipnet install-sipnet "
            "--method download' if one is published for this platform."
        )


class BuildError(RuntimeError):
    """SIPNET could not be compiled from source."""


def _compile_pinned_source(target: Path) -> None:
    """Fetch the pinned SIPNET commit with git and compile it beside *target*.

    The source is fetched by commit and tag rather than as a GitHub archive
    tarball: a tarball's bytes are generated on request and have changed
    before, so a pinned digest of one is brittle, whereas a commit hash is
    exactly the identity we want and git verifies it for us. Fetching the tag
    as well lets SIPNET's Makefile stamp ``git describe`` into the binary the
    way a full checkout would; the tag is also passed explicitly so the stamp
    does not depend on how the fetch went.
    """
    for tool, purpose in (
        ("git", "fetch the pinned SIPNET source"),
        ("make", "run SIPNET's Makefile"),
        ("cc", "compile SIPNET"),
    ):
        _require_tool(tool, purpose)

    source_dir = target.parent / "src"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=source_dir, check=True, capture_output=True, text=True)

    try:
        git("init", "-q")
        git("remote", "add", "origin", SIPNET_SOURCE_REPO_URL)
        git("fetch", "-q", "--depth", "1", "origin", SIPNET_PINNED_COMMIT)
        git("fetch", "-q", "--depth", "1", "origin", "tag", SIPNET_PINNED_TAG, "--no-tags")
        git("checkout", "-q", SIPNET_PINNED_COMMIT)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != SIPNET_PINNED_COMMIT:
            raise BuildError(
                f"Fetched SIPNET source is at {head}, not the pinned {SIPNET_PINNED_COMMIT}."
            )
        subprocess.run(
            ["make", "sipnet", f"GIT_HASH={SIPNET_PINNED_TAG}"],
            cwd=source_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise BuildError(
            f"Compiling SIPNET failed while running {exc.cmd!r} (exit {exc.returncode}).\n"
            f"{exc.stderr or exc.stdout or ''}".rstrip()
        ) from exc

    built = source_dir / BINARY_NAME
    if not built.is_file():
        raise BuildError(f"SIPNET's Makefile finished but produced no {built}.")
    _install_binary(built, target)
    shutil.rmtree(source_dir, ignore_errors=True)


def _install_binary(source: Path, target: Path) -> None:
    """Copy *source* to *target* atomically, checking its version first.

    The new binary is staged beside the target and checked there, then moved
    into place with :func:`os.replace`. Overwriting the target directly would
    destroy a working binary whenever the new one turns out to be unusable,
    and would let a concurrent run observe a half-written executable.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".incoming")
    try:
        shutil.copy2(source, staged)
        staged.chmod(0o755)
        _check_staged_binary(staged)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


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
    found = find_binary()
    if found is None:
        raise FileNotFoundError(missing_binary_message())
    return found.path


def missing_binary_message() -> str:
    """The error text for "there is no SIPNET binary", listing where it was looked for."""
    return (
        "No SIPNET binary was found. Looked in:\n"
        f"{describe_binary_search()}\n"
        "Install one with 'pysipnet install-sipnet' (Python: pysipnet.build.install_sipnet()), "
        f"or point ${BINARY_ENV_VAR} at an existing binary."
    )


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
    # SIPNET prints e.g. "SIPNET version 2.1.0 (v2.2.0-alpha.1)"; keep
    # everything after the "version" keyword so the tag suffix is preserved.
    return _version_of(ensure_binary())


def sipnet_build_tag(version_string: str | None = None) -> str:
    """Return the ``git describe`` tag a binary was built from.

    SIPNET's Makefile injects ``git describe --tags`` into the binary, and
    ``--version`` prints it in parentheses after the numeric version::

        SIPNET version 2.1.0 (v2.2.0-alpha.1)
                              ^^^^^^^^^^^^^^

    That parenthesized part is the only thing in the output that identifies
    which source a binary came from. The numeric version cannot do the job: it
    comes from ``version.h``, which lags behind pre-release tags — at
    ``v2.2.0-alpha.1`` it still reads ``2.1.0``. Checking the numeric version
    at a pre-release pin would accept a binary built from a different release
    and report success.

    Parameters
    ----------
    version_string:
        Output of :func:`sipnet_version`. Read from the installed binary when
        omitted.

    Returns
    -------
    str
        The tag, e.g. ``"v2.2.0-alpha.1"``. Empty if the binary carries no tag,
        which happens when it was compiled outside a git checkout so the
        Makefile had no ``git describe`` to inject.
    """
    text = sipnet_version() if version_string is None else version_string
    match = re.search(r"\(([^)]*)\)", text)
    return match.group(1).strip() if match else ""


def _version_of(path: Path) -> str:
    """Ask the binary at *path* what it is; the tail of ``--version`` after ``version``."""
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=True)
    _, _, reported = result.stdout.strip().partition("version ")
    return reported or result.stdout.strip()


_verified: dict[tuple[Path, int, int], str] = {}


def verify_binary_matches_pin(path: Path) -> str:
    """Check that the binary at *path* was built from the pinned tag, and return its version.

    Called by :class:`~pysipnet.runner.SIPNETRunner` before its first run, so a
    binary from another release — a stale checkout cache after a pin bump, or
    a cluster module of a different version named by ``$PYSIPNET_BINARY`` — is
    refused instead of quietly producing a different model's output. The
    result is cached on the file's path, size and modification time, so an
    ensemble pays one ``--version`` call per process, not per run.

    Raises
    ------
    BinaryVersionError
        If the binary will not run, carries no build tag, or carries a tag
        other than :data:`~pysipnet.version.SIPNET_PINNED_TAG`.
    """
    stat = path.stat()
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key in _verified:
        return _verified[key]

    try:
        version = _version_of(path)
    except (subprocess.SubprocessError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise BinaryVersionError(
            f"The SIPNET binary at {path} would not run: {detail.strip()}"
        ) from exc

    tag = sipnet_build_tag(version)
    if tag != SIPNET_PINNED_TAG:
        raise BinaryVersionError(
            f"The SIPNET binary at {path} was built from "
            f"{tag or 'an untagged commit'!r}, but this pySIPNET pins {SIPNET_PINNED_TAG!r} "
            f"(full version string: {version!r}). Rebuild or reinstall it with "
            "'pysipnet install-sipnet --force', or pass verify_binary=False to "
            "SIPNETRunner to run it anyway."
        )
    _verified[key] = version
    return version


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

    Normalizes what :mod:`platform` reports, since the same architecture goes
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
            "Compile from source instead with 'pysipnet install-sipnet --method compile', "
            "which works on any platform with git, make and a C compiler."
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

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not newurl.lower().startswith("https://"):
            raise DownloadError(f"Refusing to follow a redirect to a non-HTTPS address: {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str, timeout: float) -> Any:
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


def prebuilt_unavailable_reason(key: str | None = None) -> str | None:
    """Why the published binary for *key* cannot be used here, or ``None`` if it can.

    Upstream builds on recent systems, and what it produces runs only on
    systems at least as recent: the macOS binary is linked for macOS 26 and the
    Linux binary needs glibc 2.34. This checks the running machine against the
    requirement recorded in :data:`~pysipnet.version.SIPNET_PREBUILT_REQUIREMENTS`
    so :func:`install_sipnet` can compile instead of downloading something that
    the loader would refuse.
    """
    key = key or platform_key()
    if key not in SIPNET_RELEASE_ASSETS:
        supported = ", ".join(sorted(SIPNET_RELEASE_ASSETS))
        return f"no prebuilt binary is published for {key} (published: {supported})"
    if key != platform_key():
        return None  # not this machine; nothing to compare against

    requirement = SIPNET_PREBUILT_REQUIREMENTS[key]
    if key.startswith("darwin"):
        release = platform.mac_ver()[0]
        have = _version_tuple(release)
        need = _version_tuple(requirement.minimum)
        if have and have < need:
            return (
                f"the published macOS binary needs macOS {requirement.minimum}+, this is {release}"
            )
    elif key.startswith("linux"):
        libc, libc_version = platform.libc_ver()
        if libc == "glibc" and libc_version:
            if _version_tuple(libc_version) < _version_tuple(requirement.minimum):
                return (
                    f"the published Linux binary needs glibc {requirement.minimum}+, "
                    f"this system has {libc_version}"
                )
        elif libc:
            return f"the published Linux binary needs glibc, this system reports {libc}"
    return None


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def download_sipnet(*, force: bool = False, timeout: float = 120.0) -> Path:
    """Fetch the prebuilt SIPNET binary for this platform and install it.

    An alternative to :func:`build_sipnet` for machines without a C compiler.
    The archive is verified against the SHA-256 pinned in
    :data:`pysipnet.version.SIPNET_RELEASE_ASSETS` before anything is unpacked,
    and the installed binary is then asked for its version to confirm it is the
    release this pySIPNET targets. It goes to :func:`install_target`.

    Parameters
    ----------
    force:
        Replace an existing binary at the install target. Without this, an
        existing binary is kept and returned unchanged.
    timeout:
        Seconds to wait for the download.

    Raises
    ------
    DownloadError
        If no binary is published for this platform, the download fails, the
        checksum does not match, the archive looks unsafe, or the installed
        binary reports an unexpected version.
    """
    target = install_target()
    if target.exists() and not force:
        return target
    fetch_release_binary(platform_key(), target, timeout=timeout)
    return target


def fetch_release_binary(
    key: str, target: Path, *, timeout: float = 120.0, verify_runs: bool | None = None
) -> Path:
    """Download the archive published for platform *key* and install its binary at *target*.

    The building block behind :func:`download_sipnet`, and what the wheel
    build uses to stage a binary for another platform. Verification of the
    archive is the same in both cases. Whether the installed binary is *run*
    to confirm its version defaults to "only when *key* is this machine",
    since a binary for another platform cannot be executed here.
    """
    if verify_runs is None:
        verify_runs = key == platform_key()

    filename, expected_sha256 = release_asset(key)
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
                f"Could not download {url}: {exc}. Compile from source instead with "
                "'pysipnet install-sipnet --method compile'."
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
                # where available as a second line of defense; it also settles
                # the extraction behavior, which otherwise differs by version
                # and warns on 3.12 and 3.13. Absent only on Python 3.11.0-3.11.3,
                # where the explicit check above still applies.
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(unpacked, filter="data")  # noqa: S202
                else:
                    tar.extractall(unpacked)  # noqa: S202 - members checked above
        except tarfile.TarError as exc:
            raise DownloadError(f"Could not unpack {filename}: {exc}") from exc

        source = _find_binary(unpacked)
        if verify_runs:
            _install_binary(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(target.name + ".incoming")
            shutil.copy2(source, staged)
            staged.chmod(0o755)
            os.replace(staged, target)

    return target


InstallMethod = Literal["auto", "download", "compile"]


def install_sipnet(*, method: InstallMethod = "auto", force: bool = False) -> Path:
    """Get a SIPNET binary onto this machine, and return its path.

    What ``pysipnet install-sipnet`` runs. With ``method="auto"``: a binary
    that already exists anywhere in the search order is returned as is, unless
    *force*; otherwise the published binary is downloaded when one exists for
    this platform and can run here, and SIPNET is compiled from the pinned
    source when not. ``"download"`` and ``"compile"`` insist on one route.

    Parameters
    ----------
    method:
        ``"auto"``, ``"download"`` or ``"compile"``.
    force:
        Install even if a binary already exists at the install target,
        replacing it.
    """
    if method == "auto" and not force:
        found = find_binary()
        if found is not None:
            return found.path

    if method == "download":
        return download_sipnet(force=force)
    if method == "compile":
        return build_sipnet(force=force)

    reason = prebuilt_unavailable_reason()
    if reason is None:
        return download_sipnet(force=force)
    return build_sipnet(force=force)


def _check_staged_binary(path: Path) -> None:
    """Confirm a freshly unpacked or compiled binary is the release we expect.

    Asks the binary what it is rather than trusting the filename it arrived
    under. A mismatch means the pinned asset and :data:`SIPNET_PINNED_TAG` have
    drifted apart, which is what a half-finished pin bump looks like.

    Checks the ``git describe`` tag rather than the numeric version, because
    the numeric version lags pre-release tags and would accept a binary from
    the wrong release.
    """
    try:
        reported = _version_of(path)
    except (subprocess.SubprocessError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise DownloadError(
            f"The new binary would not run: {detail.strip()}. It was not installed. "
            f"{prebuilt_unavailable_reason() or ''}".rstrip()
        ) from exc

    tag = sipnet_build_tag(reported)
    if tag != SIPNET_PINNED_TAG:
        raise DownloadError(
            f"The new binary was built from {tag or 'an untagged commit'!r}, "
            f"but this pySIPNET pins {SIPNET_PINNED_TAG!r}. It was not installed. "
            f"(Full version string: {reported!r}.) The pinned asset in "
            "pysipnet.version is probably out of step with SIPNET_PINNED_TAG."
        )
