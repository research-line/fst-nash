#!/usr/bin/env python3
"""
Download and validate the 28-protein benchmark dataset for FST-Nash.

Downloads PDB files from RCSB, validates chain extraction, residue counts,
and standard amino acid content. Outputs a validation report.

Usage:
    PYTHONIOENCODING=utf-8 python download_pdb_dataset.py [--data-dir ../data]
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

try:
    from Bio.PDB import PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1_extended as map3to1
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
CONFIG_PATH = SCRIPT_DIR / "dataset_config.json"


def download_pdb(pdb_id: str, out_path: Path) -> bool:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        urllib.request.urlretrieve(url, str(out_path))
        return True
    except urllib.error.HTTPError as e:
        print(f"  ERROR downloading {pdb_id}: HTTP {e.code}")
        return False
    except Exception as e:
        print(f"  ERROR downloading {pdb_id}: {e}")
        return False


def validate_pdb(pdb_path: Path, chain_id: str, expected_res: int) -> dict:
    result = {
        "file_exists": pdb_path.exists(),
        "file_size_kb": round(pdb_path.stat().st_size / 1024, 1) if pdb_path.exists() else 0,
        "chain_found": False,
        "residue_count": 0,
        "nonstandard_aa": [],
        "sequence": "",
        "valid": False,
    }

    if not pdb_path.exists() or not HAS_BIOPYTHON:
        return result

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("X", str(pdb_path))
    except Exception as e:
        result["error"] = str(e)
        return result

    model = next(structure.get_models())

    if chain_id not in [c.id for c in model.get_chains()]:
        available = [c.id for c in model.get_chains()]
        result["error"] = f"Chain {chain_id} not found; available: {available}"
        return result

    result["chain_found"] = True
    chain = model[chain_id]

    residues = [r for r in chain.get_residues() if r.id[0] == " "]
    residues = [r for r in residues if "N" in r and "CA" in r and "C" in r]
    result["residue_count"] = len(residues)

    seq = []
    nonstandard = []
    for r in residues:
        name3 = r.get_resname().strip().upper()
        aa1 = map3to1.get(name3.capitalize(), "?")
        if aa1 not in AA20:
            nonstandard.append(f"{name3}({r.id[1]})")
            aa1 = "X"
        seq.append(aa1)

    result["sequence"] = "".join(seq)
    result["nonstandard_aa"] = nonstandard

    res_diff = abs(result["residue_count"] - expected_res)
    result["valid"] = (
        result["chain_found"]
        and result["residue_count"] >= 20
        and res_diff <= 15
        and len(nonstandard) == 0
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Download FST-Nash PDB dataset")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    all_proteins = []
    for fold_class, proteins in config["proteins"].items():
        for p in proteins:
            p["fold_class"] = fold_class
            all_proteins.append(p)

    print(f"FST-Nash PDB Dataset — {len(all_proteins)} Proteine")
    print(f"Zielverzeichnis: {data_dir}")
    print(f"BioPython verfügbar: {HAS_BIOPYTHON}")
    print("=" * 70)

    results = {}
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    n_valid = 0

    for p in all_proteins:
        pdb_id = p["pdb"]
        chain = p["chain"]
        pdb_file = data_dir / f"{pdb_id}.pdb"

        print(f"\n[{p['fold_class']:16s}] {pdb_id} chain {chain} — {p['name']}")

        if pdb_file.exists() and args.skip_existing and not args.validate_only:
            print(f"  Bereits vorhanden ({pdb_file.stat().st_size / 1024:.1f} KB)")
            n_skipped += 1
        elif not args.validate_only:
            print(f"  Downloading von RCSB...")
            ok = download_pdb(pdb_id, pdb_file)
            if ok:
                print(f"  OK ({pdb_file.stat().st_size / 1024:.1f} KB)")
                n_downloaded += 1
            else:
                n_failed += 1
                results[pdb_id] = {"valid": False, "error": "Download failed"}
                continue

        if HAS_BIOPYTHON:
            v = validate_pdb(pdb_file, chain, p["res"])
            results[pdb_id] = v

            status = "VALID" if v["valid"] else "ISSUE"
            print(f"  Validierung: {status} — {v['residue_count']} Residuen "
                  f"(erwartet ~{p['res']}), Split: {p['split']}")

            if v.get("error"):
                print(f"  WARNUNG: {v['error']}")
            if v["nonstandard_aa"]:
                print(f"  Non-standard AAs: {v['nonstandard_aa']}")
            if v["valid"]:
                n_valid += 1
        else:
            print(f"  (BioPython fehlt — keine Validierung)")

    print("\n" + "=" * 70)
    print(f"ZUSAMMENFASSUNG:")
    print(f"  Neu heruntergeladen: {n_downloaded}")
    print(f"  Bereits vorhanden:   {n_skipped}")
    print(f"  Download-Fehler:     {n_failed}")
    if HAS_BIOPYTHON:
        print(f"  Validiert (OK):      {n_valid}/{len(all_proteins)}")

        issues = [(pid, r) for pid, r in results.items() if not r.get("valid", False)]
        if issues:
            print(f"\n  PROBLEME ({len(issues)}):")
            for pid, r in issues:
                err = r.get("error", "")
                ns = r.get("nonstandard_aa", [])
                rc = r.get("residue_count", 0)
                print(f"    {pid}: residues={rc}, nonstandard={ns}, error={err}")

    manifest_path = data_dir / "dataset_manifest.json"
    manifest = {
        "total": len(all_proteins),
        "valid": n_valid,
        "proteins": {}
    }
    for p in all_proteins:
        pid = p["pdb"]
        v = results.get(pid, {})
        manifest["proteins"][pid] = {
            "name": p["name"],
            "chain": p["chain"],
            "fold_class": p["fold_class"],
            "split": p["split"],
            "expected_residues": p["res"],
            "actual_residues": v.get("residue_count", None),
            "resolution": p["resolution"],
            "valid": v.get("valid", None),
            "sequence": v.get("sequence", None),
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest geschrieben: {manifest_path}")


if __name__ == "__main__":
    main()
