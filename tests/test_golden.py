"""Golden-output regression test.

Where :mod:`tests.test_fidelity` proves the wrapper reproduces the binary *at
the current moment*, this test pins the actual output *numbers* to a checked-in
baseline.  It is a canary: if a submodule bump changes the model, a compile flag
is altered by accident, the ``.param`` writer regresses, or the output parser's
column contract shifts, the frozen values (or column set) stop matching and the
test fails — even though the wrapper and binary would still agree with each
other (so :mod:`tests.test_fidelity` alone would not notice).

The baseline is a fixed, in-repo input: the Niwot reference parameters and the
first :data:`_N_TIMESTEPS` rows of its climate.  Regenerate the golden after an
*intended* change with::

    python -m tests.test_golden        # from the repo root

and review the resulting diff before committing it.

Requires the compiled STANDARD binary; skipped when absent.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from pysipnet.climate import ClimateDrivers
from pysipnet.io.clim_io import read_clim_file
from pysipnet.runner import SIPNETRunner
from pysipnet.parameters.model import ModelFlags
from tests.helpers import params_from_sipnet_file

REFERENCE_DIR = Path(__file__).parent / "fixtures" / "niwot_reference"
REFERENCE_PARAM = REFERENCE_DIR / "sipnet.param"
REFERENCE_CLIM = REFERENCE_DIR / "sipnet.clim"
GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "niwot_standard.out.csv"

_N_TIMESTEPS = 60  # ~3–4 weeks at Niwot's sub-daily cadence; keeps the golden compact

_STANDARD_BINARY = SIPNETRunner(flags=ModelFlags.standard()).binary_path

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


def _run_baseline() -> pd.DataFrame:
    """Run the frozen baseline input through the wrapper and return its output."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # upstream data has a few vpd ≤ 0 rows
        full = read_clim_file(REFERENCE_CLIM, n_columns=14)
    climate = ClimateDrivers.from_dataframe(full.data.head(_N_TIMESTEPS).copy(), n_columns=14)
    params = params_from_sipnet_file(REFERENCE_PARAM)
    result = SIPNETRunner(flags=ModelFlags.standard()).run(params, climate, run_id="golden")
    assert result.provenance.success, result.provenance.stderr
    return result.outputs.data


def test_output_matches_golden():
    if not GOLDEN.exists():
        pytest.fail(
            f"Golden baseline missing at {GOLDEN}. Generate it with "
            "'python -m tests.test_golden' and commit the result."
        )

    produced = _run_baseline().reset_index(drop=True)
    golden = pd.read_csv(GOLDEN)

    assert list(produced.columns) == list(golden.columns), (
        "Output column contract changed.\n"
        f"  produced: {list(produced.columns)}\n"
        f"  golden:   {list(golden.columns)}"
    )
    assert len(produced) == len(golden)

    # atol = 0.02 absorbs cross-platform last-digit rounding in SIPNET's
    # 2-decimal text output (the golden may be generated on a different OS /
    # compiler than CI's); any real drift — a flag change, submodule numeric
    # change, or writer regression — moves values far beyond this.
    pd.testing.assert_frame_equal(
        produced,
        golden,
        check_dtype=False,
        rtol=1e-4,
        atol=0.02,
        obj="produced vs. golden output",
    )


def _write_golden() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    _run_baseline().reset_index(drop=True).to_csv(GOLDEN, index=False)
    print(f"Wrote golden baseline: {GOLDEN} ({_N_TIMESTEPS} timesteps)")


if __name__ == "__main__":
    _write_golden()
