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


def test_publication_v1_4_pdf_textlayer():
    root = Path(__file__).parent.parent
    pub_dir = root / "publications" / "v1.4"
    try:
        import fitz
    except ImportError:
        return

    for pdf_name in ("fst_nash_v1.4_en.pdf", "fst_nash_v1.4_ger.pdf"):
        pdf_path = pub_dir / pdf_name
        assert pdf_path.exists(), f"{pdf_name} must exist"
        doc = fitz.open(pdf_path)
        total_chars = sum(len(page.get_text()) for page in doc)
        replacement_chars = sum(page.get_text().count("\ufffd") for page in doc)
        doc.close()
        assert total_chars > 25000, f"Expected >25000 chars, got {total_chars}"
        assert replacement_chars == 0, f"Found replacement chars in {pdf_name}"


def test_four_cycle_exact_potential_game_contract():
    """
    Validates the Monderer & Shapley (1996) 4-cycle condition for exact potential games.
    This mathematical diagnostic core was defended in the FST-Nash review (BEWEISNOTIZ_2.md):
    A finite 2-player game with payoffs u1, u2 is an exact potential game iff
    for every 4-cycle of unilateral deviations, the sum of payoff differences vanishes.
    """
    import numpy as np

    # 1. Exact Potential Game Case
    P = np.array([[2.0, 1.0], [0.5, 3.0]])
    c1 = np.array([0.2, 0.7])
    c2 = np.array([0.1, -0.4])
    u1 = P + c1
    u2 = P + c2.reshape(-1, 1)

    # 4-cycle: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
    d1 = u1[1, 0] - u1[0, 0]
    d2 = u2[1, 1] - u2[1, 0]
    d3 = u1[0, 1] - u1[1, 1]
    d4 = u2[0, 0] - u2[0, 1]
    cycle_sum = d1 + d2 + d3 + d4
    assert abs(cycle_sum) < 1e-12, f"Potential game cycle sum must vanish, got {cycle_sum}"

    # 2. Strict Non-Potential Game Negative Control (Matching Pennies)
    u1_mp = np.array([[1.0, -1.0], [-1.0, 1.0]])
    u2_mp = np.array([[-1.0, 1.0], [1.0, -1.0]])
    d1_mp = u1_mp[1, 0] - u1_mp[0, 0]
    d2_mp = u2_mp[1, 1] - u2_mp[1, 0]
    d3_mp = u1_mp[0, 1] - u1_mp[1, 1]
    d4_mp = u2_mp[0, 0] - u2_mp[0, 1]
    cycle_sum_mp = d1_mp + d2_mp + d3_mp + d4_mp
    assert abs(cycle_sum_mp) > 1.0, "Non-potential game must exhibit non-vanishing cycle sum"

