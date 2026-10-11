from pathlib import Path
import re


def test_pyproject_metadata():
    root = Path(__file__).parent.parent
    pyproject_path = root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist"

    txt = pyproject_path.read_text(encoding="utf-8")
    assert re.search(r'name\s*=\s*"fst-nash"', txt)
    assert re.search(r'version\s*=\s*"1\.4\.0"', txt)
    assert re.search(r'description\s*=\s*"Construction-Conditioned Potential-Game Diagnostics for Chaperone Models"', txt)
    assert re.search(r'"Preprint DOI"\s*=\s*"https://doi\.org/10\.5281/zenodo\.23082682"', txt)
    assert re.search(r'"Concept DOI"\s*=\s*"https://doi\.org/10\.5281/zenodo\.20402751"', txt)


def test_citation_cff():
    root = Path(__file__).parent.parent
    cff_path = root / "CITATION.cff"
    assert cff_path.exists(), "CITATION.cff must exist"

    txt = cff_path.read_text(encoding="utf-8")
    assert re.search(r'version:\s*"1\.4"', txt)
    assert re.search(r'doi:\s*"10\.5281/zenodo\.23082682"', txt)
    assert re.search(r'title:\s*"Construction-Conditioned Potential-Game Diagnostics for Chaperone Models"', txt)


def test_publication_v1_4_artifacts():
    root = Path(__file__).parent.parent
    pub_dir = root / "publications" / "v1.4"
    assert pub_dir.exists(), "publications/v1.4 must exist"

    en_pdf = pub_dir / "fst_nash_v1.4_en.pdf"
    ger_pdf = pub_dir / "fst_nash_v1.4_ger.pdf"
    en_tex = pub_dir / "fst_nash_v1.4_en.tex"
    ger_tex = pub_dir / "fst_nash_v1.4_ger.tex"
    supp_zip = pub_dir / "fst_nash_supplementary_proof_notes.zip"

    assert en_pdf.is_file() and en_pdf.stat().st_size > 100_000
    assert ger_pdf.is_file() and ger_pdf.stat().st_size > 100_000
    assert en_tex.is_file() and en_tex.stat().st_size > 10_000
    assert ger_tex.is_file() and ger_tex.stat().st_size > 10_000
    assert supp_zip.is_file() and supp_zip.stat().st_size > 1_000


def test_supplementary_proof_notes():
    root = Path(__file__).parent.parent
    supp_dir = root / "supplementary" / "proof-notes"
    assert supp_dir.exists(), "supplementary/proof-notes must exist"

    readme = supp_dir / "README.md"
    note = supp_dir / "C2_tautology_proof.md"

    assert readme.is_file() and "Errata" in readme.read_text(encoding="utf-8")
    assert note.is_file()
