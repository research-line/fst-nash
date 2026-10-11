import subprocess
import sys
from pathlib import Path

import pytest

from scripts.xcl1_fold_switching_calibrated import DDG_SALT_150MM, salt_ddG


def test_validation_evidence_ledger_runs():
    root = Path(__file__).parent.parent
    script = root / "scripts" / "validation_evidence_ledger.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "full_validation_pass_count" in res.stdout


def test_xcl1_150mm_anchor_and_paper_formula_stay_synchronized():
    root = Path(__file__).parent.parent
    assert salt_ddG(150.0) == pytest.approx(DDG_SALT_150MM)

    pub_dir = root / "publications" / "v1.4"
    for paper_name in ("fst_nash_v1.4_en.tex", "fst_nash_v1.4_ger.tex"):
        tex = (pub_dir / paper_name).read_text(encoding="utf-8")
        assert r"\Delta\Delta G_{150}" in tex
        assert r"\frac{f([\text{NaCl}])}{f(150\,\text{mM})}" in tex
