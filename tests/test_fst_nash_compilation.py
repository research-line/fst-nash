import compileall
from pathlib import Path


def test_python_code_compilation():
    root = Path(__file__).parent.parent
    assert compileall.compile_dir(root / "code", quiet=True)
    assert compileall.compile_dir(root / "scripts", quiet=True)
