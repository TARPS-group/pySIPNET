"""Which version of SIPNET this release of pySIPNET drives.

pySIPNET is pinned to one exact SIPNET commit, recorded here and in the
``sipnet/`` git submodule. Pinning by commit rather than by branch means that
anyone who clones this repository compiles the same model source, so results
are reproducible and a version change is an explicit, reviewable edit.

Anything that differs between SIPNET versions — the climate file layout, which
parameters exist, the output columns — is keyed off the constants here, so the
rest of the package does not need to know the version at all.

Three constants describe the pin, and they are not interchangeable:

:data:`SIPNET_PINNED_COMMIT`
    The authoritative answer to "which source". A commit cannot move.
:data:`SIPNET_PINNED_TAG`
    The human name for that commit, and what ``git describe --tags`` reports.
    This is what a compiled binary can be checked against, and what names the
    release assets. A tag *can* be moved by whoever owns it, which is worth
    knowing when pinning to a pre-release.
:data:`SIPNET_NUMERIC_VERSION`
    What SIPNET's own ``version.h`` says. Useful for reporting, useless for
    identifying the pin: it lags behind pre-release tags. At
    ``v2.2.0-alpha.1`` it still reads ``2.1.0``.
"""

PYSIPNET_VERSION: str = "0.1.0.dev0"

# ── SIPNET source pinning ─────────────────────────────────────────────────────
# The sipnet/ submodule is checked out to this commit, which is the
# v2.2.0-alpha.1 pre-release.
#
# Why a v2 release rather than something older: from v2.0.0 onward SIPNET
# chooses its model options at run time, so pySIPNET builds one binary and
# writes the options into sipnet.in. Before that, options were compile-time
# #define switches, which forced one binary per combination of options and a
# patch to the SIPNET source to make those switches overridable at all.
SIPNET_PINNED_COMMIT: str = "41fa853e7131f542c52fcc0f4e3ea76892b52eda"

# What `git describe --tags` reports at that commit. SIPNET's own Makefile
# injects this into the binary as GIT_HASH, so `sipnet --version` prints it —
# which makes it the one thing a compiled binary can be checked against.
SIPNET_PINNED_TAG: str = "v2.2.0-alpha.1"

# What SIPNET's version.h declares. Note this is *not* 2.2.0: the header was
# not bumped for the alpha, so it lags the tag. Recorded for reporting, and
# deliberately not used to verify the pin — see tests/test_build.py.
SIPNET_NUMERIC_VERSION: str = "2.1.0"

# ── Climate file column counts ────────────────────────────────────────────────
# SIPNET accepts the climate file in two layouts, and detects which one it has
# been given by counting the columns on the first line.
#
# 12 columns (current):
#     year day time length tair tsoil par precip vpd vpdSoil vPress wspd
#
# 14 columns (older files, still accepted): the same 12 values wrapped in two
# columns SIPNET no longer uses — a leading site identifier and a trailing soil
# wetness value. SIPNET reads both, ignores both, and logs that it did so.
CLIM_COLS_12: int = 12
CLIM_COLS_14: int = 14

# ── Prebuilt binaries ─────────────────────────────────────────────────────────
# From v2.0.0 onward the SIPNET project publishes compiled binaries with each
# release, so pySIPNET can fetch one instead of compiling. Downloading is never
# automatic: call pysipnet.build.download_sipnet() or run `make sipnet-download`.
#
# The SHA-256 of each archive is pinned here, exactly as the pinned commit is,
# and is checked before anything is extracted. The digests come from GitHub's
# own release metadata (`gh release view v2.1.0 --repo PecanProject/sipnet
# --json assets`). Pinning them means the identity of a downloaded binary is a
# reviewable part of this file rather than whatever the network returned.
#
# When bumping the pin, refresh these together with SIPNET_PINNED_COMMIT: a
# stale digest fails the download loudly, which is the intended behaviour.

SIPNET_RELEASE_REPO: str = "PecanProject/sipnet"
SIPNET_RELEASE_TAG: str = SIPNET_PINNED_TAG

SIPNET_RELEASE_ASSETS: dict[str, tuple[str, str]] = {
    # platform key -> (archive filename, SHA-256 of the archive)
    "darwin-arm64": (
        "sipnet-macos-arm64-v2.2.0-alpha.1.tar.gz",
        "05dab157a0969ba18e18a4c05769c4d8f9297b32fa94f6c308cb2555de84653c",
    ),
    "linux-x86_64": (
        "sipnet-linux-x86_64-v2.2.0-alpha.1.tar.gz",
        "f97a33bed82ba2a15479679712446eae53f8f2e91c05e5d73f8bdf6fce33b76f",
    ),
}
"""Archives published for the pinned release, keyed by platform.

Only the platforms upstream builds for appear here. Anywhere else — Intel
macOS, Windows, ARM Linux — has to compile from source, which always works.
"""
