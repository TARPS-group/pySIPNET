# SIPNET Version Contract

## Pinned release

pySIPNET targets one exact SIPNET commit, recorded in `pysipnet/version.py` and
checked out in the `sipnet/` submodule:

```
41fa853e7131f542c52fcc0f4e3ea76892b52eda   # v2.2.0-alpha.1
```

This is the **v2.2.0-alpha.1** pre-release, created by the SIPNET developers so
that everyone's analyses pin to the same version.

Pinning to a commit rather than a branch means everyone who clones pySIPNET
compiles the same model source, so results are reproducible and a version
change is a deliberate, reviewable edit rather than something that happens on
its own.

Two tests keep the pin honest, both in `tests/test_build.py`: one asserts the
submodule is checked out at `SIPNET_PINNED_COMMIT`, the other that the compiled
binary was built from `SIPNET_PINNED_TAG`. A stale binary left over from a
previous pin has no other symptom, so without these it would quietly produce
output from the wrong model.

The second check reads the `git describe` tag, not the numeric version. SIPNET's
`version.h` still declares `2.1.0` at this tag, so `sipnet --version` prints
`2.1.0 (v2.2.0-alpha.1)` and the number alone cannot tell one release from
another. Checking it would produce a test that passes against the wrong binary.

Note also that a tag, unlike a commit, can be moved by whoever owns it. If
upstream re-tags this pre-release the commit assertion fails, which is the
behaviour we want.

## What this pin gives you

**Model options are chosen when you run, not when you compile.** From SIPNET
v2.0.0 onward, every feature switch lives in the `sipnet.in` config file. There
is one binary, built by `make sipnet` with no compiler flags, and pySIPNET
expresses the configuration as a
[`ModelFlags`](api/index.md) object written into that file for each run.

Earlier SIPNET versions chose these options at compile time, which meant one
binary per combination of options and a patch to the SIPNET source to make the
switches overridable at all. Both are gone.

**The litter pool works.** Before v2.0.0 the litter pool did not compile, and
the commit that fixed compilation still left soil respiration unassigned on
that code path, so soil carbon accumulated without ever respiring.

**Wood carbon is counted once.** Up to v2.0.0, the wood pool update added
photosynthesis and wood creation together. v2.1.0 separates the wood pool from
a storage-lag term. This changes results, which is why parameter values fitted
against an older SIPNET do not carry over.

## File formats at this pin

### Climate file

SIPNET decides which layout it has been given by counting the columns on the
first line.

| Columns | Notes |
|:--------|:------|
| 12      | The current layout: `year day time length tair tsoil par precip vpd vpdSoil vPress wspd`. |
| 14      | An older layout carrying the same 12 values, plus a leading site identifier and a trailing soil wetness value. SIPNET parses those two extra fields and then ignores them. |

Anything else is a hard error, including 13 columns, which an earlier SIPNET
accepted.

pySIPNET writes 14 columns. SIPNET accepts them, noting in its log that it took
the older layout. Both layouts can be read, selected with `n_columns=12` or
`n_columns=14`. Note the discriminator is
the column count, not a version: one SIPNET version reads both.

### Parameter file

Two columns, `name  value`, with `!` for comments. Files may carry extra
columns from older tooling; SIPNET warns and ignores them.

An unrecognised parameter name is only a warning, so a renamed parameter would
silently stop having any effect. `tests/test_param_file_contract.py` guards
against that by running the binary and failing on any unknown-parameter line.

### Output file

35 columns with a header row. Every column is always present: a switched-off
process writes zeros rather than omitting its column.

Older SIPNET wrote a `Notes:` line above the header, which v2.1.0 removed. The
output reader detects the header by content rather than expecting that line, so
files from older versions still read. Columns are matched by name, never by
position, so a column set that changes between versions costs only a mapping
entry.

## Moving to a newer SIPNET

Bump `SIPNET_PINNED_COMMIT`, `SIPNET_PINNED_TAG` and `SIPNET_NUMERIC_VERSION`
together, refresh `SIPNET_RELEASE_ASSETS`, update the
submodule, and rebuild with `make sipnet` — `build_sipnet()` trusts an existing
binary unless you pass `force=True`.

Expect to revisit:

1. **Required parameters.** Which parameters SIPNET insists on, and the
   conditions attached to them, are mirrored in
   `SIPNETParameters.validate_for_flags`.
2. **Model flags.** Any new feature switch belongs in `ModelFlags`, along with
   the restrictions SIPNET enforces on combining it with others.
3. **Output columns.** New columns need entries in
   `SIPNET_TO_PYTHON_OUTPUT`; unmapped ones keep their SIPNET spelling rather
   than being dropped.
4. **Golden fixtures.** Regenerate with `python -m tests.test_golden` and
   review the diff, recording the before-and-after values in the commit.

The public API — `SIPNETRunner.run(params, climate)` — should not need to
change.

## Not yet used from this pin

Two capabilities exist in the pinned SIPNET but are not wired up in pySIPNET
yet.

**Prescribed phenology.** `leafon` and `leafoff` events in the events file set
leaf-out and leaf-fall timing from observed dates instead of from a fitted
parameter, which removes a dimension from the calibration problem. They are
mutually exclusive with the calculated triggers: SIPNET refuses to start if
both are configured. Three event types are unmodelled in total — `leafon`,
`leafoff` and `plantdeath` — and `tests/test_events_contract.py` fails if a
fourth appears upstream. Tracked in issue #25.

**Restart checkpoints.** `RESTART_IN` and `RESTART_OUT` save and reload model
state, which is the efficient route to sequential data assimilation. Tracked in
issue #24.
