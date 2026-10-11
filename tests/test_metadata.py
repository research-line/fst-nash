from pathlib import Path
import tomllib
import yaml


def test_pyproject_metadata():
    root = Path(__file__).parent.parent
    pyproject_path = root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    assert project.get("name") == "fst-nash"
    assert project.get("version") == "1.4.0"
    assert project.get("description") == "Construction-Conditioned Potential-Game Diagnostics for Chaperone Models"

    urls = project.get("urls", {})
    assert urls.get("Preprint DOI") == "https://doi.org/10.5281/zenodo.23082682"
    assert urls.get("Concept DOI") == "https://doi.org/10.5281/zenodo.20402751"


def test_citation_cff():
    root = Path(__file__).parent.parent
    cff_path = root / "CITATION.cff"
    assert cff_path.exists(), "CITATION.cff must exist"

    with open(cff_path, "r", encoding="utf-8") as f:
        cff = yaml.safe_load(f)

    assert cff.get("version") == "1.4"
    assert cff.get("doi") == "10.5281/zenodo.23082682"
    assert cff.get("title") == "Construction-Conditioned Potential-Game Diagnostics for Chaperone Models"


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
