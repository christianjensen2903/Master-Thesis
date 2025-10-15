from __future__ import annotations

from pathlib import Path
from typing import Any

from collections import Counter

import ijson  # type: ignore[import]
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import]


RAW_TO_CANONICAL: dict[str, str] = {
    "Judgment": "Judgment",
    "Arrêt": "Judgment",
    "Opinion of the Advocate General": "Opinion of the Advocate General",
    "Conclusions de l’avocat général": "Opinion of the Advocate General",
    "Conclusions de l'avocat général": "Opinion of the Advocate General",
    "View of the Advocate General": "Opinion of the Advocate General",
    "Reports of cases": "Reports of cases",
    "Recueil de la jurisprudence": "Reports of cases",
    "Third-party proceedings": "Third-party proceedings",
    "Case": "Case",
    "<unknown>": "<unknown>",
}

CANONICAL_TO_FR: dict[str, str] = {
    "Judgment": "Arrêt",
    "Opinion of the Advocate General": "Conclusions de l’avocat général",
    "Reports of cases": "Recueil de la jurisprudence",
    "Third-party proceedings": "Third-party proceedings",
    "Case": "Case",
    "<unknown>": "<unknown>",
}


def canonicalize(doc_type: str) -> str:
    value = doc_type.strip() if doc_type else "<unknown>"
    return RAW_TO_CANONICAL.get(value, value)


def stream_counts(json_path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    # Pass 1: CELEX -> canonical document type
    celex_to_type: dict[str, str] = {}
    with json_path.open("rb") as f:
        for celex, obj in ijson.kvitems(f, ""):
            if not isinstance(obj, dict):
                continue
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
            src_type_val: str | None = None
            if meta and isinstance(meta.get("document_type"), str):
                src_type_val = meta["document_type"]
            if src_type_val is None and isinstance(obj.get("document_type"), str):
                src_type_val = obj["document_type"]
            celex_to_type[celex] = canonicalize(src_type_val or "<unknown>")

    # Pass 2: accumulate counts by mapping targets through celex_to_type
    with json_path.open("rb") as f:
        for celex, obj in ijson.kvitems(f, ""):
            if not isinstance(obj, dict):
                continue
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
            src_type_val2: str | None = None
            if meta and isinstance(meta.get("document_type"), str):
                src_type_val2 = meta["document_type"]
            if src_type_val2 is None and isinstance(obj.get("document_type"), str):
                src_type_val2 = obj["document_type"]
            src_canon = canonicalize(src_type_val2 or "<unknown>")

            refs: Any = obj.get("references")
            if not isinstance(refs, list) and isinstance(meta, dict):
                refs = meta.get("references")
            if not isinstance(refs, list):
                continue

            seen_targets2: set[str] = set()
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                target = ref.get("target")
                if isinstance(target, str) and target and target not in seen_targets2:
                    seen_targets2.add(target)
                    dst_canon = celex_to_type.get(target, "<unknown>")
                    counts[(src_canon, dst_canon)] += 1

    return counts


def draw_graph(
    counts: Counter[tuple[str, str]], label_map: dict[str, str], out_path: Path
) -> None:
    G = nx.DiGraph()

    # Nodes from counts
    nodes: set[str] = set()
    for (src, dst), c in counts.items():
        if c <= 0:
            continue
        nodes.add(src)
        nodes.add(dst)
    for n in nodes:
        G.add_node(n)

    # Edges with weights
    for (src, dst), c in counts.items():
        if c <= 0:
            continue
        G.add_edge(src, dst, weight=c)

    # Layout and drawing
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 10))
    # Node labels
    labels = {n: label_map.get(n, n) for n in G.nodes()}

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="#e0ecf4",
        edgecolors="#1f78b4",
        linewidths=1.5,
        node_size=1800,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # Edge widths scaled
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        min_w, max_w = min(weights), max(weights)
    else:
        min_w, max_w = 1, 1

    def scale_width(w: int) -> float:
        if max_w == min_w:
            return 2.0
        # scale to [0.5, 6.0]
        return 0.5 + 5.5 * ((w - min_w) / (max_w - min_w))

    widths = [scale_width(G[u][v]["weight"]) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, width=widths, edge_color="#636363", arrows=True, arrowsize=20
    )

    # Edge labels as counts
    edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5
    )

    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    json_path = project_root / "data" / "par-to-par.json"
    counts = stream_counts(json_path)

    # English labels: identity
    en_labels = {k: k for k in RAW_TO_CANONICAL.values()}
    # Also ensure keys for any canonical seen
    for s, d in list(counts.keys()):
        en_labels.setdefault(s, s)
        en_labels.setdefault(d, d)

    # French labels
    fr_labels = {k: CANONICAL_TO_FR.get(k, k) for k in en_labels.keys()}

    out_dir = project_root / "artifacts"
    draw_graph(counts, en_labels, out_dir / "type_citations_en.png")
    draw_graph(counts, fr_labels, out_dir / "type_citations_fr.png")


if __name__ == "__main__":
    main()
