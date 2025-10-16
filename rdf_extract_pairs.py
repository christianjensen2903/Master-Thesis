from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Iterable
from urllib.parse import unquote
from tqdm import tqdm  # type: ignore


CDM_NS = "http://publications.europa.eu/ontology/cdm#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS = "http://www.w3.org/2002/07/owl#"


_FRAGMENT_CITING = f"{{{CDM_NS}}}fragment_citing_source".replace(
    "ontology/cdm#fragment_citing_source", "ontology/annotation#fragment_citing_source"
)
_FRAGMENT_CITED = f"{{{CDM_NS}}}fragment_cited_target".replace(
    "ontology/cdm#fragment_cited_target", "ontology/annotation#fragment_cited_target"
)

# The annotation terms live under the annotation namespace, but the files alias it
# as j.2. ElementTree strips prefixes, so we match by localname as fallback.


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _is_owl_axiom(elem: ET.Element) -> bool:
    return any(
        _local(child.tag) == "type"
        and child.get(f"{{{RDF_NS}}}resource", "").endswith("#Axiom")
        for child in elem
    )


def _text(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    txt = elem.text if elem.text is not None else None
    return txt.strip() if txt else None


def _extract_resource(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    value = elem.get(f"{{{RDF_NS}}}resource")
    return value.strip() if value else None


def _iter_files(rdf_dir: Path) -> Iterable[Path]:
    for p in sorted(rdf_dir.glob("*.rdf")):
        if p.is_file():
            yield p


def _get_number_of_files(rdf_dir: Path) -> int:
    return len(list(rdf_dir.glob("*.rdf"))) if rdf_dir.is_dir() else 0


def _parse_date_map(xml_root: ET.Element) -> dict[str, str]:
    celex_to_date: dict[str, str] = {}

    # Collect date_document and work_date_document attached to a celex resource
    for desc in xml_root.findall(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    ):
        about = desc.get(f"{{{RDF_NS}}}about")
        if not about or "/celex/" not in about:
            continue
        celex_id = unquote(about.rsplit("/", 1)[-1])

        # prefer work_date_document, fallback to date_document
        date_val = None
        for child in desc:
            lname = _local(child.tag)
            if lname == "work_date_document" or lname == "date_document":
                candidate = _text(child)
                if candidate:
                    date_val = candidate
                    # keep looking to prefer work_date_document encountered later
        if date_val:
            celex_to_date[celex_id] = date_val

    return celex_to_date


@dataclass
class TargetLocation:
    article: int | None
    paragraph: int | None
    point: int | None
    line: int | None
    page: int | None
    column: int | None
    raw: str


@dataclass
class Citation:
    celex_from: str
    # Source location broken down into pages/paragraphs/columns
    # Pages come from tokens like "P 1212"; paragraphs from "N 25"; columns from "C 3".
    source_pages: list[int]
    source_paragraphs: list[int]
    source_columns: list[int]
    celex_to: str
    target_location: TargetLocation
    fragment_source: str
    fragment_target: str


def _parse_source_fragment(fragment: str) -> tuple[list[int], list[int], list[int]]:
    """Parse a citing source fragment extracting pages (P), paragraphs (N), columns (C).

    Examples:
    - "P 1455 1456" -> pages=[1455, 1456]
    - "N 25 30" -> paragraphs=[25, 30]
    - "P 757 758 N 17" -> pages=[757, 758], paragraphs=[17]
    - "C 3" -> columns=[3]
    """
    pages: list[int] = []
    paragraphs: list[int] = []
    columns: list[int] = []

    # Capture groups starting with a token followed by one or more integers separated by spaces
    for token, nums in re.findall(r"\b(P|N|C)\s+((?:\d+\s*)+)", fragment):
        values = [int(x) for x in re.findall(r"\d+", nums)]
        if token == "P":
            pages.extend(values)
        elif token == "N":
            paragraphs.extend(values)
        elif token == "C":
            columns.extend(values)

    # Also handle compact single tokens like "C5" that may appear without spaces
    # Avoid double counting if they were already captured above
    for token, num in re.findall(r"\b(P|N|C)(\d+)\b", fragment):
        val = int(num)
        if token == "P" and val not in pages:
            pages.append(val)
        elif token == "N" and val not in paragraphs:
            paragraphs.append(val)
        elif token == "C" and val not in columns:
            columns.append(val)

    return pages, paragraphs, columns


def _parse_target_fragment(fragment: str) -> TargetLocation:
    """Parse a target fragment like "A18P1PT9", "A03", or "A03P1L1".

    Recognized tokens (case-sensitive):
    - A<d+>: article
    - P<d+>: paragraph when part of an A...P... block, otherwise page
    - PT<d+>: point
    - L<d+>: line
    - N<d+>: paragraph (standalone notation)
    - C<d+>: column

    Any unrecognized tokens are ignored; the raw string is preserved.
    """
    article: int | None = None
    paragraph: int | None = None
    point: int | None = None
    line: int | None = None
    page: int | None = None
    column: int | None = None

    # Prefer the structured A...P...PT...L... pattern where P means paragraph
    m = re.search(r"A(\d+)(?:P(\d+))?(?:PT(\d+))?(?:L(\d+))?", fragment)
    if m:
        a, p, pt, l = m.groups()
        article = int(a)
        if p is not None:
            paragraph = int(p)
        if pt is not None:
            point = int(pt)
        if l is not None:
            line = int(l)

    # Standalone N denotes paragraph when no paragraph parsed yet
    n_match = re.search(r"\bN\s*(\d+)\b|\bN(\d+)\b", fragment)
    if n_match and paragraph is None:
        paragraph = int(next(g for g in n_match.groups() if g is not None))

    # Columns: C<number> possibly embedded like "22-09C5"
    c_match = re.search(r"\bC\s*(\d+)\b|C(\d+)", fragment)
    if c_match:
        column = int(next(g for g in c_match.groups() if g is not None))

    # Pages: P<number> when not part of an A... block (i.e., not already used for paragraph)
    if paragraph is None:
        p_page_match = re.search(r"\bP\s*(\d+)\b|\bP(\d+)\b", fragment)
        if p_page_match:
            page = int(next(g for g in p_page_match.groups() if g is not None))

    return TargetLocation(
        article=article,
        paragraph=paragraph,
        point=point,
        line=line,
        page=page,
        column=column,
        raw=fragment,
    )


def _parse_pairs_from_axioms(xml_root: ET.Element, celex: str) -> list[Citation]:
    pairs: list[Citation] = []

    # OWL axiom nodes carry annotatedSource/annotatedTarget and fragment_* data.
    for desc in xml_root.findall(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    ):
        if not _is_owl_axiom(desc):
            continue

        annotated_source = None
        annotated_target = None
        fragment_source = None
        fragment_target = None
        annotated_property = None

        for child in desc:
            lname = _local(child.tag)
            if lname == "annotatedSource":
                annotated_source = _extract_resource(child)
            elif lname == "annotatedTarget":
                annotated_target = _extract_resource(child)
            elif lname == "annotatedProperty":
                annotated_property = _extract_resource(child)
            elif lname == "fragment_citing_source":
                fragment_source = _text(child)
            elif lname == "fragment_cited_target":
                fragment_target = _text(child)

        # Only consider citations where the annotated property is work_cites_work
        if (
            annotated_property != f"{CDM_NS}work_cites_work"
            and annotated_property != f"{CDM_NS}cites"
        ):
            continue

        # Require CELEX URIs on both ends; fragments are optional
        if not annotated_source or not annotated_target:
            continue
        if "/celex/" not in annotated_source or "/celex/" not in annotated_target:
            continue

        from_celex = unquote(annotated_source.rsplit("/", 1)[-1])
        to_celex = unquote(annotated_target.rsplit("/", 1)[-1])

        if from_celex != celex:
            continue

        # Parse available fragments; default missing ones to empty structured values
        if fragment_source:
            source_pages, source_pars, source_cols = _parse_source_fragment(
                fragment_source
            )
        else:
            source_pages, source_pars, source_cols = [], [], []

        if fragment_target:
            target_loc = _parse_target_fragment(fragment_target)
        else:
            target_loc = TargetLocation(
                article=None,
                paragraph=None,
                point=None,
                line=None,
                page=None,
                column=None,
                raw="",
            )
        if not (source_pars or source_pages or source_cols) and (
            target_loc.article is None
            and target_loc.paragraph is None
            and target_loc.point is None
            and target_loc.line is None
            and target_loc.page is None
            and target_loc.column is None
        ):
            # No usable location info on either side; skip
            continue

        pairs.append(
            Citation(
                celex_from=from_celex,
                source_pages=source_pages,
                source_paragraphs=source_pars,
                source_columns=source_cols,
                celex_to=to_celex,
                target_location=target_loc,
                fragment_source=fragment_source or "",
                fragment_target=fragment_target or "",
            )
        )

    return pairs


def extract_pairs_from_file(path: Path) -> tuple[list[Citation], dict[str, str]]:

    tree = ET.parse(path)
    root = tree.getroot()
    date_map = _parse_date_map(root)
    # Only keep pairs where the annotated source CELEX matches the current file's CELEX
    file_celex = path.stem
    all_pairs = _parse_pairs_from_axioms(root, file_celex)
    pairs = [p for p in all_pairs if p.celex_from == file_celex]
    return pairs, date_map


def build_celex_date_index(rdf_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for rdf_file in tqdm(
        _iter_files(rdf_dir),
        desc="Building celex date index",
        total=_get_number_of_files(rdf_dir),
    ):
        try:
            pairs, date_map = extract_pairs_from_file(rdf_file)
        except ET.ParseError:
            continue
        # merge dates
        for k, v in date_map.items():
            index.setdefault(k, v)
    return index


def _gather_same_as_mappings(xml_root: ET.Element) -> list[set[str]]:
    """Return a list of sets of sameAs resources (URIs) that co-occur in a description."""
    bundles: list[set[str]] = []
    for desc in xml_root.findall(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    ):
        uris: set[str] = set()
        for child in desc:
            if _local(child.tag) == "sameAs":
                uri = _extract_resource(child)
                if uri:
                    uris.add(uri)
        if uris:
            bundles.append(uris)
    return bundles


def _extract_meta_for_sources(
    xml_root: ET.Element, sources: set[str]
) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {s: {} for s in sources}

    # Map ECLI by walking sameAs bundles that contain celex/<id>
    for uris in _gather_same_as_mappings(xml_root):
        celex_ids = {
            unquote(uri.rsplit("/", 1)[-1]) for uri in uris if "/celex/" in uri
        }
        ecli_vals = [unquote(uri.rsplit("/", 1)[-1]) for uri in uris if "/ecli/" in uri]
        if not ecli_vals:
            continue
        ecli_val = ecli_vals[0]
        for cid in celex_ids:
            if cid in meta:
                meta[cid]["ECLI"] = ecli_val

    # For each source, try to find caseno by inspecting its own description
    for desc in xml_root.findall(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    ):
        about = desc.get(f"{{{RDF_NS}}}about")
        if not about or "/celex/" not in about:
            continue
        celex_id = unquote(about.rsplit("/", 1)[-1])
        if celex_id not in sources:
            continue
        case_uri = None
        for child in desc:
            lname = _local(child.tag)
            if lname in ("work_part_of_dossier", "part_of"):
                uri = _extract_resource(child)
                if uri and "/case/" in uri:
                    case_uri = uri
        if case_uri:
            meta[celex_id]["caseno"] = unquote(case_uri.rsplit("/", 1)[-1])

    return meta


def _group_pairs_by_source(pairs: Iterable[Citation]) -> dict[str, list[Citation]]:
    grouped: dict[str, list[Citation]] = {}
    for p in pairs:
        grouped.setdefault(p.celex_from, []).append(p)
    return grouped


def build_par_to_par_json(
    *,
    all_pairs: Iterable[Citation],
    date_index: dict[str, str],
    xml_roots_by_file: dict[Path, ET.Element],
) -> dict[str, dict[str, object]]:
    # Group pairs by their source CELEX
    grouped = _group_pairs_by_source(all_pairs)

    result: dict[str, dict[str, object]] = {}

    # Build meta per source by scanning xml roots from files that contributed those pairs
    # First map source -> set(files) that included it
    source_files: dict[str, set[Path]] = {}
    for file_path, _root in xml_roots_by_file.items():
        # Identify pairs from this file
        for source_id in grouped.keys():
            # Heuristic: if the filename starts with the source CELEX, assume it's represented here
            if file_path.name.startswith(source_id):
                source_files.setdefault(source_id, set()).add(file_path)

    for source_celex, pairs in grouped.items():
        # Assemble references array
        refs = []
        for pair in pairs:
            refs.append(
                {
                    # Compatibility: keep target as CELEX for downstream scripts
                    "target": pair.celex_to,
                    # New structured locations
                    "source": {
                        "pages": pair.source_pages,
                        "paragraphs": pair.source_paragraphs,
                        "columns": pair.source_columns,
                        "raw": pair.fragment_source,
                    },
                    "target_location": {
                        "article": pair.target_location.article,
                        "paragraph": pair.target_location.paragraph,
                        "point": pair.target_location.point,
                        "line": pair.target_location.line,
                        "page": pair.target_location.page,
                        "column": pair.target_location.column,
                        "raw": pair.target_location.raw,
                    },
                }
            )

        # Build meta
        meta: dict[str, object] = {}
        if source_celex in date_index:
            meta["date"] = date_index[source_celex]

        # Try to enrich with ECLI and caseno using any xml root we have for this source
        ecli_caseno: dict[str, str] = {}
        for fp in sorted(source_files.get(source_celex, set())):
            root_for_meta = xml_roots_by_file.get(fp)
            if not root_for_meta:
                continue
            md = _extract_meta_for_sources(root_for_meta, {source_celex})
            ecli_caseno.update(md.get(source_celex, {}))
        meta.update(ecli_caseno)

        result[source_celex] = {
            "meta": meta,
            "references": refs,
        }

    return result


def _build_entry_from_root(
    xml_root: ET.Element, file_celex: str
) -> tuple[str, dict[str, object]] | None:
    pairs = _parse_pairs_from_axioms(xml_root, file_celex)
    if not pairs:
        return None

    refs: list[dict[str, object]] = []
    for pair in pairs:
        refs.append(
            {
                "target": pair.celex_to,
                "source": {
                    "pages": pair.source_pages,
                    "paragraphs": pair.source_paragraphs,
                    "columns": pair.source_columns,
                    "raw": pair.fragment_source,
                },
                "target_location": {
                    "article": pair.target_location.article,
                    "paragraph": pair.target_location.paragraph,
                    "point": pair.target_location.point,
                    "line": pair.target_location.line,
                    "page": pair.target_location.page,
                    "column": pair.target_location.column,
                    "raw": pair.target_location.raw,
                },
            }
        )

    meta: dict[str, object] = {}
    date_map = _parse_date_map(xml_root)
    if file_celex in date_map:
        meta["date"] = date_map[file_celex]

    enrich = _extract_meta_for_sources(xml_root, {file_celex}).get(file_celex, {})
    meta.update(enrich)

    return file_celex, {"meta": meta, "references": refs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract paragraph-to-paragraph metadata pairs from EUR-Lex RDF files"
    )
    parser.add_argument(
        "--rdf-dir",
        type=Path,
        default=Path("eurlex_rdf"),
        help="Directory containing .rdf files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/parsed_par_pairs.json"),
        help="Output JSON or JSONL file path (auto by extension)",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=100,
        help="Number of CELEX entries between file flushes (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rdf_dir: Path = args.rdf_dir
    out_path: Path = args.output

    # Stream to disk to avoid high RAM usage
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_jsonl = out_path.suffix.lower() == ".jsonl"

    entries_written: int = 0
    flush_interval: int = max(1, int(args.flush_interval))  # periodic flushes

    if is_jsonl:
        with out_path.open("w", encoding="utf-8") as f:
            for rdf_file in tqdm(
                _iter_files(rdf_dir),
                desc="Extracting pairs",
                total=_get_number_of_files(rdf_dir),
            ):
                try:
                    root = ET.parse(rdf_file).getroot()
                except ET.ParseError:
                    continue
                file_celex = rdf_file.stem
                built = _build_entry_from_root(root, file_celex)
                if not built:
                    continue
                celex_key, entry_obj = built
                line_obj = {
                    "celex": celex_key,
                    "meta": entry_obj["meta"],
                    "references": entry_obj["references"],
                }
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                entries_written += 1
                if entries_written % flush_interval == 0:
                    f.flush()
        print(f"Wrote {entries_written} CELEX entries to {out_path} (JSONL)")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            f.write("{\n")
            first: bool = True
            for rdf_file in tqdm(
                _iter_files(rdf_dir),
                desc="Extracting pairs",
                total=_get_number_of_files(rdf_dir),
            ):
                try:
                    root = ET.parse(rdf_file).getroot()
                except ET.ParseError:
                    continue
                file_celex = rdf_file.stem
                built = _build_entry_from_root(root, file_celex)
                if not built:
                    continue
                celex_key, entry_obj = built
                if not first:
                    f.write(",\n")
                f.write(json.dumps(celex_key))
                f.write(": ")
                f.write(json.dumps(entry_obj, ensure_ascii=False))
                first = False
                entries_written += 1
                if entries_written % flush_interval == 0:
                    f.flush()
            f.write("\n}\n")
        print(f"Wrote {entries_written} CELEX entries to {out_path}")


if __name__ == "__main__":
    main()

    # print all pairs for 61970CJ0003
    # rdf_dir = Path("eurlex_rdf")
    # # rdf_file = rdf_dir / "61970CJ0003.rdf"
    # rdf_file = rdf_dir / "62020CJ0692.rdf"
    # pairs, _ = extract_pairs_from_file(rdf_file)
    # print(pairs)
