"""The ``.clim`` layout contract: pySIPNET reads exactly the layouts SIPNET reads.

SIPNET decides a climate file's layout by counting the fields on its first
line (``readClimData`` in ``src/sipnet/sipnet.c``), accepts 12 or 14, and exits
on anything else. pySIPNET detects the layout the same way rather than taking
the caller's word for it, which is only safe if the two agree. These tests run
the binary on the same files pySIPNET reads and assert that each accepts
exactly what the other accepts, and that the two accepted layouts are the same
data to the model.

They build every variant from the upstream-authored Niwot record, which is in
the legacy 14-column layout, so no pySIPNET writer is involved in making them.
"""

from __future__ import annotations

import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from pysipnet.io.clim_io import detect_clim_layout, read_clim_file
from pysipnet.io.output_reader import read_output_file
from pysipnet.io.reference import niwot_reference_files
from pysipnet.parameters.model import ModelFlags
from pysipnet.runner import SIPNETRunner

_SIPNET_BINARY = SIPNETRunner(flags=ModelFlags.standard()).binary_path

pytestmark = pytest.mark.skipif(
    not _SIPNET_BINARY.exists(),
    reason=f"SIPNET binary not found at {_SIPNET_BINARY}; run 'make sipnet'",
)

_N_ROWS = 60


def _niwot_rows() -> list[list[str]]:
    lines = niwot_reference_files().clim.read_text().splitlines()[:_N_ROWS]
    return [line.split() for line in lines]


def _write(path: Path, rows: list[list[str]]) -> Path:
    path.write_text("\n".join(" ".join(row) for row in rows) + "\n")
    return path


def _variants(tmp_path: Path) -> dict[str, Path]:
    rows = _niwot_rows()
    assert {len(row) for row in rows} == {14}
    two_sites = [list(row) for row in rows]
    two_sites[5][0] = str(int(two_sites[5][0]) + 1)
    return {
        "14": _write(tmp_path / "legacy.clim", rows),
        "12": _write(tmp_path / "standard.clim", [row[1:13] for row in rows]),
        "13": _write(tmp_path / "no_loc.clim", [row[1:] for row in rows]),
        "11": _write(tmp_path / "short.clim", [row[1:12] for row in rows]),
        "two sites": _write(tmp_path / "two_sites.clim", two_sites),
    }


@dataclass(frozen=True)
class _Run:
    accepted: bool
    log: str
    output: pd.DataFrame | None


def _run_sipnet(clim: Path) -> _Run:
    """Run the bare binary on *clim*; SIPNET logs its errors to stdout."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "sipnet.param").write_bytes(niwot_reference_files().param.read_bytes())
        (workdir / "sipnet.clim").write_bytes(clim.read_bytes())
        (workdir / "sipnet.in").write_text("fileName = sipnet\nEVENTS = 0\n")
        proc = subprocess.run(
            [str(_SIPNET_BINARY)], cwd=workdir, capture_output=True, text=True, timeout=120
        )
        accepted = proc.returncode == 0
        output = read_output_file(workdir / "sipnet.out") if accepted else None
        return _Run(accepted=accepted, log=proc.stdout + proc.stderr, output=output)


def _pysipnet_reads(clim: Path) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            read_clim_file(clim)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("variant", ["12", "13", "14", "11", "two sites"])
def test_pysipnet_accepts_exactly_what_sipnet_accepts(tmp_path, variant):
    clim = _variants(tmp_path)[variant]
    sipnet_accepts = _run_sipnet(clim).accepted
    assert _pysipnet_reads(clim) == sipnet_accepts
    assert sipnet_accepts == (variant in {"12", "14"})


def test_sipnet_names_the_same_reasons(tmp_path):
    """Not just the verdicts: SIPNET refuses the refused files for the reasons pySIPNET gives."""
    variants = _variants(tmp_path)
    assert "format unrecognized" in _run_sipnet(variants["13"]).log
    assert "multiple locations" in _run_sipnet(variants["two sites"]).log


def test_the_two_layouts_are_the_same_data(tmp_path):
    """Same model output from SIPNET, same frame from pySIPNET, so the layout is only framing."""
    variants = _variants(tmp_path)
    assert detect_clim_layout(variants["12"]) == 12
    assert detect_clim_layout(variants["14"]) == 14

    standard, legacy = _run_sipnet(variants["12"]), _run_sipnet(variants["14"])
    assert standard.output is not None and legacy.output is not None
    assert len(standard.output) == _N_ROWS
    pd.testing.assert_frame_equal(standard.output, legacy.output)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_standard = read_clim_file(variants["12"])
        from_legacy = read_clim_file(variants["14"])
    assert (from_standard.n_columns, from_legacy.n_columns) == (12, 14)
    pd.testing.assert_frame_equal(from_standard.pandas, from_legacy.pandas)
