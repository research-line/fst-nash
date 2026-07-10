"""Build the FST-Nash pre-review validation-evidence ledger.

The ledger separates five evidence axes that were previously mixed in prose:
construction conditioning, experimental dynamics, an AlphaFold baseline, an
energetic-frustration baseline, and an independent chaperone-cycle anchor.

It deliberately does not turn the presence of a result file into scientific
validation.  Each row is checked against the existing machine-readable result
that supports its status, and a strong-validation pass requires independent,
quantitative evidence on every applicable external axis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN_DATE = "2026-07-10"


def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_inputs() -> dict[str, dict[str, Any]]:
    data = {
        "xcl1": load_json("results/xcl1_fold_switching_calibrated_results.json"),
        "groel": load_json("results/groel_kinetic_consistency_results.json"),
        "hsp90": load_json("results/hsp90_calibrated_results.json"),
        "dnaj": load_json("scripts/results/dnaj_holdout_results.json"),
        "hsp26": load_json("scripts/results/hsp26_holdout_results.json"),
        "hsp33": load_json("scripts/results/hsp33_holdout_results.json"),
        "hsc70": load_json("scripts/results/hsc70_holdout_results.json"),
        "thermosome": load_json("scripts/results/thermosome_holdout_results.json"),
    }

    salt_150 = next(
        row for row in data["xcl1"]["salt_vorhersage"] if row["NaCl_mM"] == 150
    )
    require(data["xcl1"]["pg_test"]["is_PG"] is True, "XCL1 PG check drifted")
    require(abs(salt_150["f_Ltn40"] - 0.237) < 1e-12, "XCL1 150 mM value drifted")

    groel_summary = data["groel"]["summary"]
    require(groel_summary["checks_passed"] == 2, "GroEL passed-check count drifted")
    require(groel_summary["total_checks"] == 3, "GroEL total-check count drifted")
    require(
        not any(
            row["in_experimental_range"]
            for row in data["groel"]["check_2_cycle_time"]["conditions"]
        ),
        "GroEL cycle-time status drifted",
    )

    require(
        data["hsp90"]["kalibrierungsstatus"]["delta_S"]["status"] == "ANGENOMMEN",
        "Hsp90 delta_S is no longer assumption-bound",
    )

    for key in ("dnaj", "hsp26", "hsp33", "hsc70"):
        require(
            data[key]["vorhersage_vergleich"]["overall_hit"] is True,
            f"{key} hold-out convention-consistency result drifted",
        )

    require(
        data["thermosome"]["modell_A"]["pg_test"]["is_PG"] is True
        and data["thermosome"]["modell_B"]["pg_test"]["is_PG"] is False,
        "Thermosome dual-model contrast drifted",
    )
    return data


def build_rows() -> list[dict[str, Any]]:
    common = {
        "construction_conditioned": True,
        "alphafold_baseline": "not_computed",
        "energetic_frustration_baseline": "not_computed",
        "claim_level": "diagnostic_only",
    }

    rows = [
        {
            **common,
            "case": "XCL1 salt shift",
            "experimental_dynamic_evidence": "quantitative_but_partially_circular",
            "independent_chaperone_cycle_anchor": "not_applicable_no_chaperone_cycle",
            "provenance": "results/xcl1_fold_switching_calibrated_results.json",
            "gate_reason": (
                "The 150 mM population reuses Tyler inputs; the 70/75 mM point is "
                "qualitative, and neither requested baseline is computed."
            ),
        },
        {
            **common,
            "case": "GroEL/GroES cycle",
            "experimental_dynamic_evidence": "qualitative_partial_2_of_3",
            "independent_chaperone_cycle_anchor": "partial_qualitative",
            "provenance": "results/groel_kinetic_consistency_results.json",
            "gate_reason": (
                "The ATP-pattern/timescale check passes 2/3 categories, but all modeled "
                "cycle times miss the stated 15--20 s experimental range."
            ),
        },
        {
            **common,
            "case": "Hsp90/Cdc37",
            "experimental_dynamic_evidence": "parameter_partial_delta_s_assumed",
            "independent_chaperone_cycle_anchor": "source_parameter_only",
            "provenance": "results/hsp90_calibrated_results.json",
            "gate_reason": (
                "The non-PG result follows from asymmetric payoffs and delta_S remains "
                "assumed rather than independently measured."
            ),
        },
        {
            **common,
            "case": "DnaJ hold-out",
            "experimental_dynamic_evidence": "convention_consistency_only",
            "independent_chaperone_cycle_anchor": "not_present",
            "provenance": "scripts/results/dnaj_holdout_results.json",
            "gate_reason": "The hit follows after adopting the shared-delta convention.",
        },
        {
            **common,
            "case": "Hsp26 hold-out",
            "experimental_dynamic_evidence": "convention_consistency_only",
            "independent_chaperone_cycle_anchor": "not_present",
            "provenance": "scripts/results/hsp26_holdout_results.json",
            "gate_reason": "The hit follows after adopting the shared-delta convention.",
        },
        {
            **common,
            "case": "Hsp33 hold-out",
            "experimental_dynamic_evidence": "convention_consistency_only",
            "independent_chaperone_cycle_anchor": "not_present",
            "provenance": "scripts/results/hsp33_holdout_results.json",
            "gate_reason": "The hit follows after adopting the shared-delta convention.",
        },
        {
            **common,
            "case": "Hsc70 hold-out",
            "experimental_dynamic_evidence": "convention_consistency_only",
            "independent_chaperone_cycle_anchor": "not_present",
            "provenance": "scripts/results/hsc70_holdout_results.json",
            "gate_reason": "The R1 row-2 hit is a shared-delta convention-consistency check.",
        },
        {
            **common,
            "case": "Thermosome dual model",
            "experimental_dynamic_evidence": "model_choice_demonstration",
            "independent_chaperone_cycle_anchor": "not_independent",
            "provenance": "scripts/results/thermosome_holdout_results.json",
            "gate_reason": (
                "Opposite PG outcomes are intentionally generated by two defensible game "
                "constructions; this is a guardrail, not independent validation."
            ),
        },
    ]
    for row in rows:
        row["strong_validation_pass"] = (
            row["experimental_dynamic_evidence"] == "quantitative_independent"
            and row["alphafold_baseline"] == "computed"
            and row["energetic_frustration_baseline"] == "computed"
            and row["independent_chaperone_cycle_anchor"]
            in {"quantitative_independent", "not_applicable_no_chaperone_cycle"}
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_validation_pass_count = sum(row["strong_validation_pass"] for row in rows)
    return {
        "row_count": len(rows),
        "construction_guardrail_count": sum(row["construction_conditioned"] for row in rows),
        "quantitative_but_partially_circular_count": sum(
            row["experimental_dynamic_evidence"] == "quantitative_but_partially_circular"
            for row in rows
        ),
        "partial_dynamic_anchor_count": sum(
            row["independent_chaperone_cycle_anchor"] == "partial_qualitative"
            for row in rows
        ),
        "holdout_convention_consistency_count": sum(
            row["experimental_dynamic_evidence"] == "convention_consistency_only"
            for row in rows
        ),
        "alphafold_computed_count": sum(
            row["alphafold_baseline"] == "computed" for row in rows
        ),
        "energetic_frustration_computed_count": sum(
            row["energetic_frustration_baseline"] == "computed" for row in rows
        ),
        "independent_cycle_quantitative_count": sum(
            row["independent_chaperone_cycle_anchor"] == "quantitative_independent"
            for row in rows
        ),
        "full_validation_pass_count": full_validation_pass_count,
        "claim_upgrade_allowed": full_validation_pass_count > 0,
        "paper_waterline": (
            "Diagnostic classification only; one partially circular quantitative XCL1 "
            "check and one partial GroEL cycle anchor do not establish a validated "
            "mechanistic model."
        ),
    }


def write_json(rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    path = RESULTS / "validation_evidence_ledger.json"
    payload = {
        "schema": "fst-nash-validation-evidence-v1",
        "run_date": RUN_DATE,
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(rows: list[dict[str, Any]]) -> Path:
    path = RESULTS / "validation_evidence_ledger.csv"
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    path = RESULTS / "VALIDATION_EVIDENCE_LEDGER.md"
    lines = [
        "# FST-Nash Validation-Evidence Ledger",
        "",
        f"**Run date:** {RUN_DATE}",
        "**Decision:** diagnostic classification only; no claim upgrade.",
        "",
        "This ledger records evidence status rather than treating an existing result file as validation.",
        "A strong pass requires quantitative, independent evidence on all applicable external axes.",
        "",
        "| Case | Experimental dynamics | AlphaFold baseline | Energetic-frustration baseline | Independent cycle anchor | Strong pass |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {experimental_dynamic_evidence} | {alphafold_baseline} | "
            "{energetic_frustration_baseline} | {independent_chaperone_cycle_anchor} | "
            "{strong_validation_pass} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Rows: {summary['row_count']}",
            f"- Construction-conditioned rows: {summary['construction_guardrail_count']}",
            f"- Computed AlphaFold baselines: {summary['alphafold_computed_count']}",
            "- Computed energetic-frustration baselines: "
            f"{summary['energetic_frustration_computed_count']}",
            "- Quantitative independent cycle anchors: "
            f"{summary['independent_cycle_quantitative_count']}",
            f"- Full validation passes: {summary['full_validation_pass_count']}",
            f"- Claim upgrade allowed: {str(summary['claim_upgrade_allowed']).lower()}",
            "",
            "## Waterline",
            "",
            summary["paper_waterline"],
            "",
            "The next valid progress unit is a preregistered case with a computed AlphaFold "
            "conformational baseline, a matched energetic-frustration baseline, and an "
            "independent dynamic/cycle observable not reused to set the game payoffs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    validate_inputs()
    rows = build_rows()
    summary = summarize(rows)
    require(len(rows) == 8, "Unexpected ledger row count")
    require(summary["full_validation_pass_count"] == 0, "Strong-pass waterline drifted")
    require(summary["claim_upgrade_allowed"] is False, "Claim-upgrade guardrail drifted")

    outputs = [write_json(rows, summary), write_csv(rows), write_markdown(rows, summary)]
    print(
        json.dumps(
            {
                "rows": summary["row_count"],
                "full_validation_pass_count": summary["full_validation_pass_count"],
                "claim_upgrade_allowed": summary["claim_upgrade_allowed"],
                "outputs": [str(path.relative_to(ROOT)) for path in outputs],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
