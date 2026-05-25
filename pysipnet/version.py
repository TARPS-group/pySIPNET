"""SIPNET version constants and the pySIPNET–SIPNET version contract.

pySIPNET is pinned to a specific SIPNET commit.  All version-specific behaviour
(file format differences, available parameters, output column layout) is
gated on :data:`SIPNET_TARGET_VERSION`.

When SIPNET v2 support is added, introduce a new constant and route
version-specific logic through the existing adapters in ``pysipnet/io/`` and
``pysipnet/parameters/``.
"""

PYSIPNET_VERSION: str = "0.1.0.dev0"

# ── SIPNET source pinning ──────────────────────────────────────────────────────
# The SIPNET submodule in ``sipnet/`` is checked out to this exact commit.
# It is the last commit before PR #114 ("SIP78 Convert switches to run time
# options part 3") migrated all compile-time flags to runtime CLI options,
# defining the boundary between SIPNET v1 (compile-time flags, 14-column
# climate file) and v2 (runtime flags, 12-column climate file).
SIPNET_PINNED_COMMIT: str = "e4abf14f2445133c785b756025a2e39e60c7760f"
SIPNET_TARGET_VERSION: str = "v1"

# ── Climate file column counts per version ─────────────────────────────────────
# v1: loc | year day time length tair tsoil par precip vpd vpdSoil vPress wspd | soilWetness
#     col 1 (loc) and col 14 (soilWetness) are present but ignored by SIPNET.
# v2: year day time length tair tsoil par precip vpd vpdSoil vPress wspd  (12 cols)
CLIM_COLS_V1: int = 14
CLIM_COLS_V2: int = 12
