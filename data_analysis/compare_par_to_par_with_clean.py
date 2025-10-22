from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import ijson  # type: ignore
import orjson
import re
from datetime import datetime


@dataclass(frozen=True)
class Citation:
    celex_from: str
    paragraph_from: int
    celex_to: str
    paragraph_to: int | None


def load_clean_citations(clean_csv_path: str) -> dict[str, list[Citation]]:
    """Return mapping: CELEX_FROM -> set of (CELEX_TO, NUMBER_TO) for CJ cases only.

    Only include rows where NUMBER_TO is present (paragraph) and CELEX_FROM starts with '619' and contains 'CJ'.
    """
    mapping: dict[str, list[Citation]] = defaultdict(list)
    with open(clean_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            celex_from = row.get("CELEX_FROM")
            par_from_str = row.get("NUMBER_FROM")
            celex_to = row.get("CELEX_TO")
            par_to_str = row.get("NUMBER_TO")
            date_from = row.get("DATE_FROM")
            if date_from:
                try:
                    date_obj = datetime.strptime(date_from, "%Y-%m-%d")
                    if date_obj.year > 2021 or (
                        date_obj.year == 2021 and date_obj.month > 9
                    ):
                        continue
                except ValueError:
                    # Skip if date parsing fails
                    continue

            # Skip rows with missing required fields
            if not celex_from or not par_from_str or not celex_to or not par_to_str:
                continue

            par_from = int(par_from_str)
            par_to = int(par_to_str)

            # Filter: CJ cases only (source) and paragraph target available
            if "CJ" not in celex_from:
                continue

            mapping[celex_from].append(
                Citation(
                    celex_from=celex_from,
                    celex_to=celex_to,
                    paragraph_to=par_to,
                    paragraph_from=par_from,
                )
            )
    return mapping


def load_par_to_par_citations(par_to_par_path: str) -> dict[str, list[Citation]]:
    """Load entire par-to-par.json into memory and yield paragraph-level citations with target 'to'."""
    with open(par_to_par_path, "rb") as f:
        data = orjson.loads(f.read())

    mapping: dict[str, list[Citation]] = defaultdict(list)
    for celex_from, case in data.items():
        if "CJ" not in celex_from:
            continue
        refs = case.get("meta", {}).get("references") or case.get("references") or []
        for ref in refs:
            target = ref.get("target")
            unit = ref.get("unit")
            cited_unit = ref.get("cited_unit")
            from_pars = ref.get("from") or []
            to_pars = ref.get("to") or []

            if unit != "paragraphs" or (cited_unit and cited_unit != "paragraphs"):
                continue

            if "CJ" not in target:
                continue

            date_from = case.get("meta", {}).get("date")
            if date_from:
                try:
                    date_obj = datetime.strptime(date_from, "%Y-%m-%d")
                    if date_obj.year > 2021 or (
                        date_obj.year == 2021 and date_obj.month > 9
                    ):
                        continue
                except ValueError:
                    # Skip if date parsing fails
                    continue
            digit_pattern = re.compile(r"\d+")

            for from_par in from_pars:

                digit_match = digit_pattern.search(from_par)
                if not digit_match:
                    print(
                        f"Skipping from_par {from_par} because it doesn't match the pattern."
                    )
                    continue
                from_par = int(digit_match.group())

                if not to_pars:
                    mapping[celex_from].append(
                        Citation(
                            celex_from=celex_from,
                            paragraph_from=int(from_par),
                            celex_to=target,
                            paragraph_to=None,
                        )
                    )
                    continue

                for to_par in to_pars:
                    digit_match = digit_pattern.search(to_par)
                    if not digit_match:
                        print(
                            f"Skipping to_par {to_par} because it doesn't match the pattern."
                        )
                        continue
                    to_par = int(digit_match.group())

                    mapping[celex_from].append(
                        Citation(
                            celex_from=celex_from,
                            paragraph_from=int(from_par),
                            celex_to=target,
                            paragraph_to=int(to_par),
                        )
                    )

    return mapping


def compare_citations(
    clean_map: dict[str, list[Citation]],
    par_to_par_map: dict[str, list[Citation]],
    citations_not_found: dict[str, list[dict[str, str | int | None]]],
) -> dict:
    """Compare clean citations with par-to-par citations bidirectionally and calculate agreement rates."""
    # Direction 1: Clean -> Par-to-par
    clean_to_par_total = 0
    clean_to_par_exact_matches = 0
    clean_to_par_partial_matches = 0
    clean_to_par_no_matches = 0

    # Direction 2: Par-to-par -> Clean
    par_to_clean_total = 0
    par_to_clean_exact_matches = 0
    par_to_clean_partial_matches = 0
    par_to_clean_no_matches = 0

    disagreements = []

    # Get all CELEX IDs that appear in either dataset
    all_celex_ids = set(clean_map.keys()) | set(par_to_par_map.keys())

    for celex_id in all_celex_ids:
        clean_citations = clean_map.get(celex_id, [])
        par_citations = par_to_par_map.get(celex_id, [])

        if not clean_citations and not par_citations:
            continue

        # Convert to sets for easier comparison
        clean_set = set()
        par_set = set()

        # Build clean citations set (celex_to, paragraph_from, paragraph_to)
        for citation in clean_citations:
            clean_set.add(
                (citation.celex_to, citation.paragraph_from, citation.paragraph_to)
            )

        # Build par-to-par citations set
        for citation in par_citations:
            par_set.add(
                (citation.celex_to, citation.paragraph_from, citation.paragraph_to)
            )

        # Direction 1: Check each clean citation against par-to-par
        for clean_citation in clean_citations:
            clean_to_par_total += 1
            found_match = False

            # Look for exact match (celex_to and paragraph_to both match)
            if (
                clean_citation.celex_to,
                clean_citation.paragraph_from,
                clean_citation.paragraph_to,
            ) in par_set:
                clean_to_par_exact_matches += 1
                continue

            # Look for partial match (celex_to and paragraph_from match, but paragraph_to is None in par-to-par)
            for par_citation in par_citations:
                if (
                    par_citation.celex_to == clean_citation.celex_to
                    and par_citation.paragraph_from == clean_citation.paragraph_from
                    and par_citation.paragraph_to is None
                ):
                    clean_to_par_partial_matches += 1
                    found_match = True
                    break

            if not found_match:
                clean_to_par_no_matches += 1
                # Add citation to not found list
                citations_not_found["only_in_clean"].append(
                    {
                        "celex_from": clean_citation.celex_from,
                        "paragraph_from": clean_citation.paragraph_from,
                        "celex_to": clean_citation.celex_to,
                        "paragraph_to": clean_citation.paragraph_to,
                    }
                )

        # Direction 2: Check each par-to-par citation against clean
        for par_citation in par_citations:

            if par_citation.paragraph_to is None:
                continue

            par_to_clean_total += 1
            found_match = False

            # Look for exact match (celex_to and paragraph_to both match)
            if (
                par_citation.celex_to,
                par_citation.paragraph_from,
                par_citation.paragraph_to,
            ) in clean_set:
                par_to_clean_exact_matches += 1
                continue

            # Look for partial match (celex_to and paragraph_from match, but paragraph_to is None in par-to-par)
            if par_citation.paragraph_to is None:
                for clean_citation in clean_citations:
                    if (
                        clean_citation.celex_to == par_citation.celex_to
                        and clean_citation.paragraph_from == par_citation.paragraph_from
                    ):
                        par_to_clean_partial_matches += 1
                        found_match = True
                        break

            if not found_match:
                par_to_clean_no_matches += 1
                # Add citation to not found list
                citations_not_found["only_in_par"].append(
                    {
                        "celex_from": par_citation.celex_from,
                        "paragraph_from": par_citation.paragraph_from,
                        "celex_to": par_citation.celex_to,
                        "paragraph_to": par_citation.paragraph_to,
                    }
                )

        # Record disagreements for detailed analysis
        if clean_citations or par_citations:
            clean_celexes = set(c.celex_to for c in clean_citations)
            par_celexes = set(c.celex_to for c in par_citations)

            only_in_clean = clean_celexes - par_celexes
            only_in_par = par_celexes - clean_celexes
            common = clean_celexes & par_celexes

            if only_in_clean or only_in_par:
                disagreements.append(
                    {
                        "celex": celex_id,
                        "clean_citations": list(clean_celexes),
                        "par_references": list(par_celexes),
                        "only_in_clean": list(only_in_clean),
                        "only_in_par": list(only_in_par),
                        "common": list(common),
                    }
                )

    return {
        # Direction 1: Clean -> Par-to-par
        "clean_to_par": {
            "total_comparisons": clean_to_par_total,
            "exact_matches": clean_to_par_exact_matches,
            "partial_matches": clean_to_par_partial_matches,
            "no_matches": clean_to_par_no_matches,
            "exact_match_rate": (
                clean_to_par_exact_matches / clean_to_par_total
                if clean_to_par_total > 0
                else 0
            ),
            "partial_match_rate": (
                clean_to_par_partial_matches / clean_to_par_total
                if clean_to_par_total > 0
                else 0
            ),
            "no_match_rate": (
                clean_to_par_no_matches / clean_to_par_total
                if clean_to_par_total > 0
                else 0
            ),
        },
        # Direction 2: Par-to-par -> Clean
        "par_to_clean": {
            "total_comparisons": par_to_clean_total,
            "exact_matches": par_to_clean_exact_matches,
            "partial_matches": par_to_clean_partial_matches,
            "no_matches": par_to_clean_no_matches,
            "exact_match_rate": (
                par_to_clean_exact_matches / par_to_clean_total
                if par_to_clean_total > 0
                else 0
            ),
            "partial_match_rate": (
                par_to_clean_partial_matches / par_to_clean_total
                if par_to_clean_total > 0
                else 0
            ),
            "no_match_rate": (
                par_to_clean_no_matches / par_to_clean_total
                if par_to_clean_total > 0
                else 0
            ),
        },
        "disagreements": disagreements,
    }


def main() -> None:
    clean_path = "data/clean_data.csv"
    par_to_par_path = "data/par-to-par.json"

    print("Loading clean citations...")
    clean_map = load_clean_citations(clean_path)
    print(
        f"Loaded {sum(len(citations) for citations in clean_map.values())} clean citations from {len(clean_map)} cases"
    )

    print("Loading par-to-par citations...")
    par_to_par_map = load_par_to_par_citations(par_to_par_path)
    print(
        f"Loaded {sum(len(citations) for citations in par_to_par_map.values())} par-to-par citations from {len(par_to_par_map)} cases"
    )

    # Initialize citations not found list
    citations_not_found: dict[str, list[dict[str, str | int | None]]] = {
        "only_in_clean": [],
        "only_in_par": [],
    }

    print("Comparing citations...")
    results = compare_citations(clean_map, par_to_par_map, citations_not_found)

    print("\n=== BIDIRECTIONAL AGREEMENT ANALYSIS ===")

    # Direction 1: Clean -> Par-to-par
    clean_to_par = results["clean_to_par"]
    print(f"\n--- CLEAN -> PAR-TO-PAR ---")
    print(f"Total comparisons: {clean_to_par['total_comparisons']}")
    print(
        f"Exact matches: {clean_to_par['exact_matches']} ({clean_to_par['exact_match_rate']:.2%})"
    )
    print(
        f"Partial matches: {clean_to_par['partial_matches']} ({clean_to_par['partial_match_rate']:.2%})"
    )
    print(
        f"No matches: {clean_to_par['no_matches']} ({clean_to_par['no_match_rate']:.2%})"
    )
    print(
        f"Overall agreement rate: {(clean_to_par['exact_matches'] + clean_to_par['partial_matches']) / clean_to_par['total_comparisons']:.2%}"
    )

    # Direction 2: Par-to-par -> Clean
    par_to_clean = results["par_to_clean"]
    print(f"\n--- PAR-TO-PAR -> CLEAN ---")
    print(f"Total comparisons: {par_to_clean['total_comparisons']}")
    print(
        f"Exact matches: {par_to_clean['exact_matches']} ({par_to_clean['exact_match_rate']:.2%})"
    )
    print(
        f"Partial matches: {par_to_clean['partial_matches']} ({par_to_clean['partial_match_rate']:.2%})"
    )
    print(
        f"No matches: {par_to_clean['no_matches']} ({par_to_clean['no_match_rate']:.2%})"
    )
    print(
        f"Overall agreement rate: {(par_to_clean['exact_matches'] + par_to_clean['partial_matches']) / par_to_clean['total_comparisons']:.2%}"
    )

    # Save detailed citations not found in each dataset

    with open("artifacts/citations_not_found.json", "w") as f:
        f.write(orjson.dumps(citations_not_found, option=orjson.OPT_INDENT_2).decode())
    print(f"Detailed citations not found saved to citations_not_found.json")
    print(f"Citations only in clean data: {len(citations_not_found['only_in_clean'])}")
    print(
        f"Citations only in par-to-par data: {len(citations_not_found['only_in_par'])}"
    )


if __name__ == "__main__":
    main()
