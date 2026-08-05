# Niwot Ridge fidelity reference fixture

These files are an **independently-authored** SIPNET v1 input set, used by
`tests/test_fidelity.py` to verify that the pySIPNET wrapper reproduces the
output of the SIPNET binary run by hand on the same inputs.

## Provenance

Copied verbatim from the SIPNET submodule's own smoke-test suite:

    sipnet/tests/smoke/niwot/{sipnet.param, sipnet.clim}

- `sipnet.param` — full file, copied unchanged.
- `sipnet.clim` — the **first 800 rows** of the upstream file (≈ one simulation
  year at Niwot's sub-daily cadence), trimmed only to keep the fixture small and
  CI fast. Values are otherwise unchanged.

Pinned SIPNET submodule commit at time of copy:

    e4abf14f2445133c785b756025a2e39e60c7760f  (v1.3.0-82-ge4abf14)

The point of using an upstream, wrapper-independent file is that a bug in the
pySIPNET `.param` writer (a wrong name, unit, or dropped field) would make the
wrapper's output diverge from the binary's, which this fixture would catch.
If a future submodule bump reorganizes or changes the upstream smoke fixtures,
re-copy these files and update the commit hash above.
