from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import]


def stream_type_edges(json_path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    celex_to_type: dict[str, str] = {}
    edges: list[tuple[str, str]] = []

    with json_path.open("rb") as f:
        for celex, obj in ijson.kvitems(f, ""):
            if not isinstance(obj, dict):
                continue

            # Get source type
            src_type: str | None = None
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
            if meta and isinstance(meta.get("document_type"), str):
                src_type = meta["document_type"].strip()
            if src_type is None and isinstance(obj.get("document_type"), str):
                src_type = obj["document_type"].strip()
            if not src_type:
                src_type = "<unknown>"
            celex_to_type[celex] = src_type

            # Collect references (can be under root or under meta)
            refs = obj.get("references")
            if not isinstance(refs, list) and isinstance(meta, dict):
                refs = meta.get("references")
            if isinstance(refs, list):
                # Deduplicate targets per source to avoid double counting the same target
                seen_targets: set[str] = set()
                for ref in refs:
                    if not isinstance(ref, dict):
                        continue
                    target = ref.get("target")
                    if isinstance(target, str) and target:
                        if target in seen_targets:
                            continue
                        seen_targets.add(target)
                        edges.append((celex, target))
    return celex_to_type, edges


def build_type_to_type_counts(
    celex_to_type: dict[str, str], edges: list[tuple[str, str]]
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for src, dst in edges:
        src_type = celex_to_type.get(src, "<unknown>")
        dst_type = celex_to_type.get(dst, "<unknown>")
        counts[(src_type, dst_type)] += 1
    return counts


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    json_path = project_root / "data" / "par-to-par.json"

    celex_to_type, edges = stream_type_edges(json_path)
    counts = build_type_to_type_counts(celex_to_type, edges)

    # Summaries
    total_edges = sum(counts.values())
    print(
        f"Total unique citations (source->target, deduped per source-target): {total_edges}"
    )
    print()
    # Present as sorted by count desc
    for (src_t, dst_t), c in counts.most_common():
        print(f"{src_t} -> {dst_t}\t{c}")


if __name__ == "__main__":
    main()
