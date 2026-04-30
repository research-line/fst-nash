#!/usr/bin/env python3
"""
fetch_clinvar_variants.py
=========================
Download and filter ClinVar variants for a specific gene.

Uses NCBI E-utilities to fetch variants, filters for:
- Missense only
- High confidence (≥2 stars review status)
- Clear pathogenic/benign classification

Output: CSV ready for nash_mutation_score.py

Usage:
    python fetch_clinvar_variants.py --gene TP53 --out tp53_variants.csv
"""

import argparse
import csv
import re
import time
from typing import List, Dict, Optional
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


def fetch_clinvar_ids(gene: str, retmax: int = 5000) -> List[str]:
    """
    Search ClinVar for variant IDs associated with a gene.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    # Simpler query - just gene name
    query = f"{gene}[gene]"

    params = {
        "db": "clinvar",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    print(f"Searching ClinVar for {gene} variants...")

    with urllib.request.urlopen(url, timeout=30) as response:
        import json
        data = json.loads(response.read().decode('utf-8'))

    ids = data.get("esearchresult", {}).get("idlist", [])
    count = data.get("esearchresult", {}).get("count", 0)
    print(f"  Found {count} total, fetching {len(ids)} variant IDs")
    return ids


def fetch_variant_details(variant_ids: List[str], batch_size: int = 100) -> List[Dict]:
    """
    Fetch detailed information for each variant.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    variants = []

    for i in range(0, len(variant_ids), batch_size):
        batch = variant_ids[i:i+batch_size]
        print(f"  Fetching details {i+1}-{min(i+batch_size, len(variant_ids))}...")

        params = {
            "db": "clinvar",
            "id": ",".join(batch),
            "rettype": "vcv",
            "retmode": "xml",
        }

        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                xml_data = response.read().decode('utf-8')

            # Parse XML
            root = ET.fromstring(xml_data)

            for record in root.findall(".//VariationArchive"):
                variant = parse_clinvar_record(record)
                if variant:
                    variants.append(variant)

        except Exception as e:
            print(f"    Error fetching batch: {e}")

        # Be nice to NCBI
        time.sleep(0.5)

    return variants


def parse_clinvar_record(record: ET.Element) -> Optional[Dict]:
    """
    Parse a ClinVar VariationArchive record.
    """
    try:
        # Get variation ID
        var_id = record.get("VariationID", "")

        # Get clinical significance
        interp = record.find(".//ClinicalAssertion//Interpretation")
        if interp is None:
            interp = record.find(".//Interpretations/Interpretation")

        if interp is None:
            return None

        significance = interp.get("Description", "").lower()

        # Map to binary label
        if "pathogenic" in significance and "conflicting" not in significance:
            label = "pathogenic"
        elif "benign" in significance and "conflicting" not in significance:
            label = "benign"
        else:
            return None  # Skip uncertain/conflicting

        # Get review status (stars)
        review = record.find(".//ReviewStatus")
        review_status = review.text if review is not None else ""

        # Filter for quality
        high_quality_reviews = [
            "criteria provided, multiple submitters, no conflicts",
            "reviewed by expert panel",
            "practice guideline",
            "criteria provided, single submitter"
        ]

        if not any(hq in review_status.lower() for hq in high_quality_reviews):
            return None

        # Get protein change (HGVS)
        hgvs = None
        for name in record.findall(".//Name"):
            text = name.text or ""
            # Look for protein notation like p.Arg175His
            match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', text)
            if match:
                aa_map = {
                    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D',
                    'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G',
                    'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
                    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S',
                    'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
                }
                wt = aa_map.get(match.group(1), '')
                pos = match.group(2)
                mut = aa_map.get(match.group(3), '')
                if wt and mut:
                    hgvs = f"{wt}{pos}{mut}"
                    break

        if not hgvs:
            return None

        return {
            "variant_id": var_id,
            "mutation": hgvs,
            "label": label,
            "significance": significance,
            "review_status": review_status,
        }

    except Exception as e:
        return None


def save_variants(variants: List[Dict], output_path: str):
    """
    Save variants to CSV.
    """
    # Remove duplicates by mutation
    seen = set()
    unique = []
    for v in variants:
        if v["mutation"] not in seen:
            seen.add(v["mutation"])
            unique.append(v)

    # Sort by position
    def get_pos(v):
        match = re.search(r'\d+', v["mutation"])
        return int(match.group()) if match else 0

    unique.sort(key=get_pos)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["mutation", "label", "significance", "review_status", "variant_id"])
        writer.writeheader()
        writer.writerows(unique)

    print(f"\nWrote {len(unique)} unique variants to {output_path}")

    # Summary
    pathogenic = sum(1 for v in unique if v["label"] == "pathogenic")
    benign = sum(1 for v in unique if v["label"] == "benign")
    print(f"  Pathogenic: {pathogenic}")
    print(f"  Benign: {benign}")


def main():
    ap = argparse.ArgumentParser(
        description="Fetch ClinVar missense variants for a gene"
    )
    ap.add_argument("--gene", required=True, help="Gene symbol (e.g., TP53)")
    ap.add_argument("--out", required=True, help="Output CSV file")
    ap.add_argument("--retmax", type=int, default=5000, help="Max variants to fetch")
    args = ap.parse_args()

    # Fetch variant IDs
    ids = fetch_clinvar_ids(args.gene, args.retmax)

    if not ids:
        print("No variants found!")
        return

    # Fetch details
    variants = fetch_variant_details(ids)

    if not variants:
        print("No valid variants after filtering!")
        return

    # Save
    save_variants(variants, args.out)


if __name__ == "__main__":
    main()
