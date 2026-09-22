# Niwot Ridge reference data

These files ship inside the pySIPNET package so that anything that installs
it — a downstream project's test suite most of all — has real SIPNET inputs
and real SIPNET output to work with, without a source checkout and without a
compiled binary. Reach them through `pysipnet.io.reference`:
`niwot_reference_files()` for the paths, `niwot_reference_climate()` and
`niwot_reference_output()` for the typed objects.

They are also the fixtures pySIPNET's own tests use. There is exactly one copy.

## The inputs: `sipnet.param` and `sipnet.clim`

An **independently-authored** SIPNET input set, used by `tests/test_fidelity.py`
to verify that the pySIPNET wrapper reproduces the output of the SIPNET binary
run by hand on the same inputs.

### Provenance

Copied verbatim from the SIPNET submodule's own smoke-test suite:

    sipnet/tests/smoke/niwot/{sipnet.param, sipnet.clim}

- `sipnet.param` — full file, copied unchanged.
- `sipnet.clim` — the **first 800 rows** of the upstream file (≈ one simulation
  year at Niwot's sub-daily cadence, from day 305 of 1998), trimmed only to
  keep the fixture small and CI fast. Values are otherwise unchanged. 14-column
  layout.

Pinned SIPNET submodule commit at time of copy:

    41fa853e7131f542c52fcc0f4e3ea76892b52eda  (v2.2.0-alpha.1)

`sipnet.param` is byte-identical to the upstream file at that commit, and
`sipnet.clim` matches its first 800 rows exactly. Keep it that way: the whole
value of this fixture is that pySIPNET did not write it, so any drift weakens
what the fidelity test proves. `tests/test_reference.py` compares both files to
the submodule byte for byte, so a re-save, reformat or changed line ending
fails the suite. Never open these in an editor that normalizes whitespace or
line endings.

Note that the upstream file still lists parameters SIPNET removed at v2.0.0,
so a run against it logs unknown-parameter warnings. Those come from upstream's
own fixture, not from pySIPNET's writer; `tests/test_param_file_contract.py`
checks our writer separately and requires it to produce none.

The upstream climate record has three rows with vapor pressure deficit ≤ 0,
which `ClimateDrivers` warns about on load. `niwot_reference_climate()`
silences that one warning, because SIPNET clamps the values and the data is
upstream's, not ours to correct.

The point of using an upstream, wrapper-independent file is that a bug in the
pySIPNET `.param` writer (a wrong name, unit, or dropped field) would make the
wrapper's output diverge from the binary's, which this fixture would catch.
If a future submodule bump reorganizes or changes the upstream smoke fixtures,
re-copy these files and update the commit hash above.

## The output: `niwot_standard.out.csv`

pySIPNET's **golden baseline**: what the pinned SIPNET wrote when run through
the wrapper under `ModelFlags.standard()` on `sipnet.param` and the first 60
rows of `sipnet.clim`, saved with the output registry's column names. It is
pySIPNET's frozen numeric regression baseline (`tests/test_golden.py`) and, for
a consumer, real model output that needs no binary.

Know what it is before building on it:

- **60 timesteps, days 305–334 of 1998** — one dormant-season month. GPP is
  zero in most rows, and ten of the 35 columns (`litter_carbon`, the nitrogen
  group, `methane`) are identically zero because their flags are off. See
  issue #31 for what that leaves unconstrained.
- **It changes when the baseline is deliberately regenerated** — a SIPNET pin
  bump, or an intended wrapper change — with

      python -m tests.test_golden        # from the repo root

  Review the diff before committing, and record the before-and-after values
  in the commit. Regeneration is now a visible change for consumers, so treat
  it as one.
- **It was written by pandas**, unlike the two inputs, so it does not share
  their byte-identity guarantee; its contract is that it equals what the
  current pin produces, which `test_golden` checks whenever a binary is
  present.
