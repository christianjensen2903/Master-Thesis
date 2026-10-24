"""Collect the CELEX ids of CJ cases cited (at paragraph level) by the newly
scraped 2021+ judgments, so they can be scraped with parallel_scraping.py.

We need the *full text* of every cited case to populate TEXT_TO in the
par-to-par dataset. The old xlsx is not a reliable text source (it only stores
paragraphs that themselves participated in old citations, missing ~39% of the
specific cited paragraphs), so we scrape the cited cases directly.

Output: data_processing/cited_cases_to_scrape.txt (one CELEX per line) — only
cited CJ cases that do not already have a folder under data_processing/judgments.
"""

from pathlib import Path

from meta_parser import EcjXmlParser

HERE = Path(__file__).parent
JUDGMENTS_DIR = HERE / "judgments"
CITING_LISTS = [HERE / "cj_cases_2021.txt", HERE / "cj_cases_2022-2026.txt"]
OUTPUT = HERE / "cited_cases_to_scrape.txt"


def read_citing_ids() -> list[str]:
    ids: list[str] = []
    for f in CITING_LISTS:
        if f.exists():
            ids += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted(set(ids))


def metadata_path(celex: str) -> Path | None:
    for lang in ("eng", "fra"):
        p = JUDGMENTS_DIR / celex / f"{lang}_metadata.xml"
        if p.exists():
            return p
    return None


def collect_cited_cj(citing_ids: list[str]) -> set[str]:
    """Return all unique CJ targets cited at paragraph level by the citing cases."""
    cited: set[str] = set()
    no_meta = 0
    for celex in citing_ids:
        mp = metadata_path(celex)
        if mp is None:
            no_meta += 1
            continue
        try:
            parser = EcjXmlParser(str(mp))
            parser.read_celex()
            citations = parser.read_citations_and_citing_paragraphs()
        except Exception:
            continue
        for c in citations:
            target = c.get("target", "")
            # Only CJ judgments, and only true paragraph-to-paragraph citations.
            if "CJ" not in target:
                continue
            if not c.get("from") or not c.get("to"):
                continue
            cited.add(target)
    print(f"Citing cases without metadata: {no_meta}")
    return cited


def main() -> None:
    citing_ids = read_citing_ids()
    print(f"Citing cases (2021+): {len(citing_ids)}")

    cited = collect_cited_cj(citing_ids)
    print(f"Unique cited CJ targets: {len(cited)}")

    on_disk = {p.name for p in JUDGMENTS_DIR.iterdir() if p.is_dir()}
    already = cited & on_disk
    to_scrape = sorted(cited - on_disk)
    print(f"  already on disk: {len(already)}")
    print(f"  to scrape:       {len(to_scrape)}")

    OUTPUT.write_text("\n".join(to_scrape) + "\n", encoding="utf-8")
    print(f"Wrote {len(to_scrape)} CELEX ids to {OUTPUT}")


if __name__ == "__main__":
    main()
