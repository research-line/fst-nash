import subprocess
import sys
from pathlib import Path


def test_validation_evidence_ledger_runs():
    root = Path(__file__).parent.parent
    script = root / "scripts" / "validation_evidence_ledger.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "full_validation_pass_count" in res.stdout
