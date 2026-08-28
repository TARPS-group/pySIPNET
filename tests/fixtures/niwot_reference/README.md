# Niwot Ridge fidelity reference fixture

These files are an **independently-authored** SIPNET input set, used by
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

    41fa853e7131f542c52fcc0f4e3ea76892b52eda  (v2.2.0-alpha.1)

`sipnet.param` is byte-identical to the upstream file at that commit, and
`sipnet.clim` matches its first 800 rows exactly. Keep it that way: the whole
value of this fixture is that pySIPNET did not write it, so any drift weakens
what the fidelity test proves.

Note that the upstream file still lists parameters SIPNET removed at v2.0.0,
so a run against it logs unknown-parameter warnings. Those come from upstream's
own fixture, not from pySIPNET's writer; `tests/test_param_file_contract.py`
checks our writer separately and requires it to produce none.

The point of using an upstream, wrapper-independent file is that a bug in the
pySIPNET `.param` writer (a wrong name, unit, or dropped field) would make the
wrapper's output diverge from the binary's, which this fixture would catch.
If a future submodule bump reorganizes or changes the upstream smoke fixtures,
re-copy these files and update the commit hash above.
