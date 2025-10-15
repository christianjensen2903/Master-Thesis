from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import]


def build_celex_to_type(json_path: Path) -> dict[str, str]:
    celex_to_type: dict[str, str] = {}
    with json_path.open("rb") as f:
        for celex, obj in ijson.kvitems(f, ""):
            if not isinstance(obj, dict):
                continue
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
            t: str | None = None
            if meta and isinstance(meta.get("document_type"), str):
                t = meta["document_type"].strip()
            if t is None and isinstance(obj.get("document_type"), str):
                t = obj["document_type"].strip()
            celex_to_type[celex] = t if t else "<unknown>"
    return celex_to_type


def celex_family(celex: str) -> str:
    # Example: 61954CJ0002 -> family CJ (positions 6-7, 0-based index 5:7)
    if len(celex) >= 7:
        return celex[5:7]
    return "??"


def family_type_distribution(celex_to_type: dict[str, str]) -> dict[str, Counter[str]]:
    dist: dict[str, Counter[str]] = {}
    for celex, doc_type in celex_to_type.items():
        fam = celex_family(celex)
        if fam not in dist:
            dist[fam] = Counter()
        dist[fam][doc_type] += 1
    return dist


def collect_unknowns(
    json_path: Path, celex_to_type: dict[str, str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with json_path.open("rb") as f:
        for src_celex, obj in ijson.kvitems(f, ""):
            if not isinstance(obj, dict):
                continue
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
            src_type = "<unknown>"
            if meta and isinstance(meta.get("document_type"), str):
                src_type = meta["document_type"].strip() or "<unknown>"
            elif isinstance(obj.get("document_type"), str):
                src_type = obj["document_type"].strip() or "<unknown>"

            refs: Any = obj.get("references")
            if not isinstance(refs, list) and isinstance(meta, dict):
                refs = meta.get("references")
            if not isinstance(refs, list):
                continue

            seen_targets: set[str] = set()
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                dst_celex = ref.get("target")
                if (
                    not isinstance(dst_celex, str)
                    or not dst_celex
                    or dst_celex in seen_targets
                ):
                    continue
                seen_targets.add(dst_celex)

                dst_present = dst_celex in celex_to_type
                dst_type = celex_to_type.get(dst_celex, "<missing>")
                if not dst_present:
                    reason = "missing_celex"
                elif dst_type == "<unknown>":
                    reason = "missing_document_type"
                else:
                    # Not unknown; skip
                    continue

                rows.append(
                    {
                        "src_celex": src_celex,
                        "src_type": src_type,
                        "dst_celex": dst_celex,
                        "dst_present": "yes" if dst_present else "no",
                        "dst_type": dst_type,
                        "reason": reason,
                        "dst_family": celex_family(dst_celex),
                    }
                )
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    json_path = project_root / "data" / "par-to-par.json"
    out_csv = project_root / "artifacts" / "unknown_citations_samples.csv"

    celex_to_type = build_celex_to_type(json_path)
    rows = collect_unknowns(json_path, celex_to_type)
    fam_dist = family_type_distribution(celex_to_type)

    # Summaries
    total = len(rows)
    by_reason: Counter[str] = Counter(r["reason"] for r in rows)
    by_src_type: Counter[str] = Counter(r["src_type"] for r in rows)
    by_dst_family: Counter[str] = Counter(r["dst_family"] for r in rows)

    print("Unknown target analysis")
    print("Total unknown target citations:", total)
    print("By reason:")
    for k, v in by_reason.most_common():
        print(f"  {k}: {v}")
    print("By source document_type:")
    for k, v in by_src_type.most_common():
        print(f"  {k}: {v}")
    print("By target CELEX family (code at positions 6-7):")
    for k, v in by_dst_family.most_common():
        print(f"  {k}: {v}")

    # Heuristic mapping of family codes to likely document types based on present CELEXs
    print()
    print(
        "Family code -> top present document_types (to infer what unknowns represent):"
    )
    for fam, _ in by_dst_family.most_common(20):
        type_counts = fam_dist.get(fam)
        if not type_counts:
            continue
        top = ", ".join([f"{t} ({c})" for t, c in type_counts.most_common(3)])
        print(f"  {fam}: {top}")

    # Export up to 20k samples to CSV (to keep file manageable)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "src_celex",
                "src_type",
                "dst_celex",
                "dst_present",
                "dst_type",
                "reason",
                "dst_family",
            ],
        )
        writer.writeheader()
        for r in rows[:20000]:
            writer.writerow(r)

    print()
    print("Sample CSV:", out_csv)


if __name__ == "__main__":
    main()
