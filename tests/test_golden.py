"""Golden-output regression test.

Where :mod:`tests.test_fidelity` proves the wrapper reproduces the binary *at
the current moment*, this test pins the actual output *numbers* to a checked-in
baseline.  It is a canary: if a submodule bump changes the model, a compile flag
is altered by accident, the ``.param`` writer regresses, or the output parser's
column contract shifts, the frozen values (or column set) stop matching and the
test fails — even though the wrapper and binary would still agree with each
other (so :mod:`tests.test_fidelity` alone would not notice).

The baseline is a fixed input that ships with the package: the Niwot reference
parameters and the first :data:`_N_TIMESTEPS` rows of its climate, under
``pysipnet/data/niwot/`` (see :mod:`pysipnet.io.reference`).  The golden lives
beside them and is shipped too, so regenerating it is a visible change for
anyone who builds tests on it.  Regenerate after an *intended* change with::

    python -m tests.test_golden        # from the repo root

and review the resulting diff before committing it.

Requires the compiled SIPNET binary; skipped when absent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pysipnet.climate import ClimateDrivers
from pysipnet.io.reference import niwot_reference_climate, niwot_reference_files
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import SIPNETRunner
from tests.helpers import params_from_sipnet_file

_REFERENCE = niwot_reference_files()
REFERENCE_PARAM = _REFERENCE.param
GOLDEN = _REFERENCE.output

_N_TIMESTEPS = 60  # ~3–4 weeks at Niwot's sub-daily cadence; keeps the golden compact

_SIPNET_BINARY = SIPNETRunner(flags=ModelFlags.standard()).binary_path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _SIPNET_BINARY.exists(),
        reason=f"SIPNET binary not found at {_SIPNET_BINARY}; run 'make sipnet'",
    ),
]


def _run_baseline() -> pd.DataFrame:
    """Run the frozen baseline input through the wrapper and return its output."""
    full = niwot_reference_climate()
    climate = ClimateDrivers.from_dataframe(full.pandas.head(_N_TIMESTEPS).copy(), n_columns=14)
    params = params_from_sipnet_file(REFERENCE_PARAM)
    result = SIPNETRunner(flags=ModelFlags.standard()).run(params, climate, run_id="golden")
    assert result.provenance.success, result.provenance.stderr
    return result.outputs.pandas


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
