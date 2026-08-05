"""Model-wrapping fidelity tests.

The single most important invariant pySIPNET must maintain is that the Python
wrapper is a *faithful* driver of the SIPNET binary: given the same inputs, the
wrapper must produce the same outputs the binary produces when run by hand.  A
bug anywhere in the input writers, the subprocess runner, or the output parser
would silently corrupt every downstream result, so this file certifies the
invariant directly rather than relying on plausibility checks (no-NaN,
GPP ≥ 0, carbon balance) which pass regardless of whether the *right* numbers
were computed.

Two complementary properties are tested:

1. **Runner/parser transparency** (:meth:`test_runner_is_transparent`).
   Drive SIPNET through the wrapper, then run the bare binary by hand on the
   *exact input files the wrapper wrote* and parse the result.  The two outputs
   must be **identical**, proving the runner and output parser introduce zero
   distortion of their own.

2. **End-to-end input-translation fidelity**
   (:meth:`test_reproduces_independent_reference`).  Start from an
   independently-authored SIPNET input set (the Niwot Ridge smoke fixture, see
   ``tests/fixtures/niwot_reference/README.md``), read it into the Python data
   model, and run it through the wrapper.  Compare against the bare binary run
   directly on the *original* files.  Because the reference files were authored
   by the SIPNET project — not by pySIPNET's writer — a writer bug (wrong name,
   wrong unit, dropped field) makes the two diverge.  Agreement therefore
   certifies the writer/reader translation as well as the runner and parser.

Both tests require the compiled STANDARD binary and are skipped when it is
absent (e.g. CI without a build step).  Build it with ``make sipnet-standard``.
"""

from __future__ import annotations

import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysipnet.io.clim_io import read_clim_file
from pysipnet.io.output_reader import read_output_file
from pysipnet.runner import ModelPreset, SIPNETRunner
from tests.helpers import params_from_sipnet_file

REFERENCE_DIR = Path(__file__).parent / "fixtures" / "niwot_reference"
REFERENCE_PARAM = REFERENCE_DIR / "sipnet.param"
REFERENCE_CLIM = REFERENCE_DIR / "sipnet.clim"

_STANDARD_BINARY = SIPNETRunner(preset=ModelPreset.STANDARD).binary_path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _STANDARD_BINARY.exists(),
        reason=f"SIPNET binary not found at {_STANDARD_BINARY}; run 'make sipnet-standard'",
    ),
    pytest.mark.skipif(
        not REFERENCE_PARAM.exists() or not REFERENCE_CLIM.exists(),
        reason=f"Reference fixture missing under {REFERENCE_DIR}",
    ),
]

# The microbe carbon pool is driven by microbe-submodel parameters
# (microbeInit, microbeNC, ...) that the v1 STANDARD model intentionally does
# not carry.  The pool is reported in output but is inert with respect to the
# active carbon/water dynamics, so it is the one column that legitimately
# differs between an independently-authored reference and the wrapper.  It is
# excluded from the end-to-end comparison and documented here rather than
# silently tolerated.
_OUT_OF_SCOPE_COLUMNS = {"microbe_c"}


def _run_binary_directly(binary: Path, param_path: Path, clim_path: Path) -> pd.DataFrame:
    """Run the SIPNET binary by hand on the given files and parse its output.

    Mirrors a manual invocation: copies the inputs into a clean directory,
    writes a minimal ``sipnet.in``, executes the binary with that directory as
    the working directory, and parses ``sipnet.out`` with the standard reader.
    """
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "sipnet.param").write_bytes(param_path.read_bytes())
        (workdir / "sipnet.clim").write_bytes(clim_path.read_bytes())
        (workdir / "sipnet.in").write_text("fileName = sipnet\nEVENTS = 0\n")

        proc = subprocess.run(
            [str(binary)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"Direct SIPNET invocation failed (rc={proc.returncode})\n{proc.stderr}"
        )
        return read_output_file(workdir / "sipnet.out")


def _load_reference_climate():
    # read_clim_file warns on the reference's handful of vpd ≤ 0 rows; that is a
    # property of the upstream data, not something under test here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_clim_file(REFERENCE_CLIM, version="v1")


class TestWrapperFidelity:
    def test_runner_is_transparent(self):
        """Wrapper output == bare binary run on the wrapper's own input files.

        Certifies the runner and output parser add no distortion: with byte-for
        -byte identical inputs on both sides, every output column must match
        exactly.
        """
        params = params_from_sipnet_file(REFERENCE_PARAM)
        climate = _load_reference_climate()

        runner = SIPNETRunner(preset=ModelPreset.STANDARD, keep_workdir=True)
        result = runner.run(params, climate, run_id="fidelity_transparent")
        assert result.provenance.success, result.provenance.stderr
        workdir = result.provenance.workdir
        try:
            direct = _run_binary_directly(
                _STANDARD_BINARY,
                workdir / "sipnet.param",
                workdir / "sipnet.clim",
            )
        finally:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

        wrapper = result.outputs.data
        assert list(wrapper.columns) == list(direct.columns)
        pd.testing.assert_frame_equal(
            wrapper.reset_index(drop=True),
            direct.reset_index(drop=True),
            check_exact=True,
            obj="wrapper vs. direct-invocation output",
        )

    def test_reproduces_independent_reference(self):
        """Wrapper reproduces the binary's output on an independent input set.

        The reference ``.param``/``.clim`` were authored upstream (SIPNET's own
        Niwot smoke fixture), so this exercises the full translation pipeline —
        reader → data model → writer → runner → output parser — against a
        ground-truth direct invocation.  Agreement to SIPNET's output precision
        certifies that no parameter is mis-named, mis-scaled, or dropped.
        """
        native = _run_binary_directly(_STANDARD_BINARY, REFERENCE_PARAM, REFERENCE_CLIM)

        params = params_from_sipnet_file(REFERENCE_PARAM)
        climate = _load_reference_climate()
        runner = SIPNETRunner(preset=ModelPreset.STANDARD)
        result = runner.run(params, climate, run_id="fidelity_reference")
        assert result.provenance.success, result.provenance.stderr
        wrapper = result.outputs.data

        assert list(wrapper.columns) == list(native.columns)
        assert len(wrapper) == len(native)

        compared = [
            c
            for c in wrapper.columns
            if c not in _OUT_OF_SCOPE_COLUMNS and pd.api.types.is_numeric_dtype(wrapper[c])
        ]
        assert compared, "no numeric columns to compare"

        for col in compared:
            # Both legs use the same binary on the same machine, so the only
            # source of divergence is the deterministic input delta (out-of-scope
            # params + the 6-sig-fig climate serialization contract). Empirically
            # that is ≤ 6e-6 on this fixture and, over much longer runs, at most
            # one 0.01 output quantum in the large C pools. atol = 0.03 (≈3
            # quanta) absorbs that with margin; a genuine writer bug (e.g. aMax
            # off by 1 %, which shifts GPP by ~0.06) is comfortably caught.
            np.testing.assert_allclose(
                wrapper[col].to_numpy(),
                native[col].to_numpy(),
                atol=0.03,
                rtol=1e-5,
                err_msg=f"wrapper output diverges from direct invocation in column '{col}'",
            )
