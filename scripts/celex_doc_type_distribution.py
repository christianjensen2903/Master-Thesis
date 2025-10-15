from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import]


def load_unique_celex_codes(csv_path: Path) -> set[str]:
    unique_celex: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            celex_from = row.get("CELEX_FROM", "").strip()
            celex_to = row.get("CELEX_TO", "").strip()
            if celex_from:
                unique_celex.add(celex_from)
            if celex_to:
                unique_celex.add(celex_to)
    return unique_celex


def stream_celex_to_doc_type(json_path: Path, celex_filter: set[str]) -> dict[str, str]:
    celex_to_doc_type: dict[str, str] = {}
    # The JSON is a large object; stream keys at the top level
    with json_path.open("rb") as f:
        # Each top-level key is a CELEX; we stream pairs (key, value)
        for celex, obj in ijson.kvitems(f, ""):
            if celex not in celex_filter:
                continue
            # document_type likely under meta; handle common locations defensively
            doc_type: str | None = None
            if isinstance(obj, dict):
                meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
                if meta and isinstance(meta.get("document_type"), str):
                    doc_type = meta["document_type"].strip()
                # Some entries might store document_type at root
                if doc_type is None and isinstance(obj.get("document_type"), str):
                    doc_type = obj["document_type"].strip()
            if doc_type:
                celex_to_doc_type[celex] = doc_type
            else:
                # Mark as unknown if missing
                celex_to_doc_type[celex] = "<unknown>"
            # Early stop if we've mapped all CELEX codes
            if len(celex_to_doc_type) == len(celex_filter):
                break
    return celex_to_doc_type


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "clean_data.csv"
    json_path = project_root / "data" / "par-to-par.json"

    unique_celex = load_unique_celex_codes(csv_path)
    celex_to_doc = stream_celex_to_doc_type(json_path, unique_celex)

    # Count distribution using only CELEXs that were found
    counter: Counter[str] = Counter(
        celex_to_doc[c] for c in unique_celex if c in celex_to_doc
    )

    # Report missing CELEXs (not present in JSON)
    missing = [c for c in unique_celex if c not in celex_to_doc]

    print("Total unique CELEX in CSV:", len(unique_celex))
    print("Found in JSON:", len(unique_celex) - len(missing))
    print("Missing in JSON:", len(missing))
    print()
    print("Document type distribution (unique CELEX across FROM/TO):")
    for doc_type, count in counter.most_common():
        print(f"{doc_type}\t{count}")


if __name__ == "__main__":
    main()
