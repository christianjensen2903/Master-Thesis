"""Build an updated paragraph-to-paragraph citation dataset from the newly
scraped 2021+ ECJ judgments and merge it with the existing dataset.

Pipeline
--------
1. For every case we need (citing 2021+ cases and the cases they cite), parse
   the judgment into numbered paragraphs, choosing the language version that
   actually yields content: English > French > native (most paragraphs wins).
   Also read the case title and decision date from the XML metadata.
2. For every citing case, read the paragraph-level citations from the metadata
   (WORK_CITES_WORK) and expand each into (from_paragraph, to_paragraph) pairs
   restricted to CJ -> CJ paragraph citations.
3. Join paragraph texts to produce rows matching the old schema:
   CELEX_FROM, TITLE_FROM, NUMBER_FROM, DATE_FROM, TEXT_FROM,
   CELEX_TO,   TITLE_TO,   NUMBER_TO,   DATE_TO,   TEXT_TO.
   Rows whose FROM or TO text cannot be resolved are dropped and counted.
4. Merge with data/par-to-par-20220126.xlsx, drop self-citations and temporal
   violations, dedup, and write the unified dataset.

Run from anywhere:  python data_processing/build_par_to_par.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Import the package first (REPO on path) so `judgment_parser` resolves to the
# judgment_parser/ package and not data_processing/judgment_parser.py, then make
# meta_parser importable from data_processing.
sys.path.insert(0, str(REPO))
from judgment_parser import JudgementParser  # noqa: E402

sys.path.append(str(REPO / "data_processing"))
from meta_parser import EcjXmlParser  # noqa: E402

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

JUDGMENTS_DIR = REPO / "data_processing" / "judgments"
CITING_LISTS = [
    REPO / "data_processing" / "cj_cases_2021.txt",
    REPO / "data_processing" / "cj_cases_2022-2026.txt",
]
OLD_XLSX = REPO / "data" / "par-to-par-20220126.xlsx"
OUT_NEW = REPO / "data" / "par-to-par-new.csv"
OUT_MERGED = REPO / "data" / "par-to-par-updated.csv"

LANG_PRIORITY = ["eng", "fra"]

_jp = JudgementParser()


# --------------------------------------------------------------------------- #
# Per-case parsing (cached)
# --------------------------------------------------------------------------- #
_par_cache: dict[str, dict[int, str]] = {}
_meta_cache: dict[str, dict] = {}


def _judgment_files(celex: str) -> list[Path]:
    d = JUDGMENTS_DIR / celex
    if not d.is_dir():
        return []
    return [d / f for f in __import__("os").listdir(d) if f.endswith("_judgment.html")]


def parse_paragraphs(celex: str) -> dict[int, str]:
    """Parse paragraphs, picking the best language version (eng>fra>native).

    A version is acceptable only if it yields paragraphs; otherwise we fall
    back to the next language. This skips the EUR-Lex PDF-only shell pages that
    parse to zero paragraphs.
    """
    if celex in _par_cache:
        return _par_cache[celex]

    files = _judgment_files(celex)
    ordered: list[Path] = []
    for lang in LANG_PRIORITY:
        for f in files:
            if f.name == f"{lang}_judgment.html":
                ordered.append(f)
    ordered += [f for f in files if f.name.split("_")[0] not in LANG_PRIORITY]

    result: dict[int, str] = {}
    for f in ordered:
        try:
            pars = _jp.extract_paragraphs(str(f))
        except Exception:
            pars = {}
        if pars:
            result = pars
            break

    _par_cache[celex] = result
    return result


def read_meta(celex: str) -> dict:
    """Return {'title': str, 'date': str|None} from the XML metadata."""
    if celex in _meta_cache:
        return _meta_cache[celex]

    info = {"title": "", "date": None}
    for lang in ("eng", "fra"):
        p = JUDGMENTS_DIR / celex / f"{lang}_metadata.xml"
        if not p.exists():
            continue
        try:
            parser = EcjXmlParser(str(p))
            parser.read_celex()
            info["title"] = parser.read_full_title() or ""
            dates = parser.read_document_date()
            info["date"] = dates[0] if dates else None
        except Exception:
            continue
        break
    _meta_cache[celex] = info
    return info


def read_citations(celex: str) -> list[dict]:
    for lang in ("eng", "fra"):
        p = JUDGMENTS_DIR / celex / f"{lang}_metadata.xml"
        if not p.exists():
            continue
        try:
            parser = EcjXmlParser(str(p))
            parser.read_celex()
            return parser.read_citations_and_citing_paragraphs()
        except Exception:
            return []
    return []


def fmt_text(num: int, text: str) -> str:
    """Match the old dataset format: the paragraph number prefixes the text.

    Some parsers (older judgment formats) keep the leading paragraph number in
    the extracted text while modern ones strip it. Normalise by removing a
    leading number equal to ``num`` before prepending it once.
    """
    t = text.strip()
    m = re.match(r"^(\d+)", t)
    if m and m.group(1) == str(num):
        t = t[m.end():].lstrip(". \t\r\n")
    return f"{num} {t}".strip()


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def read_citing_ids() -> list[str]:
    ids: list[str] = []
    for f in CITING_LISTS:
        if f.exists():
            ids += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted(set(ids))


def build_new_rows() -> tuple[pd.DataFrame, Counter]:
    citing_ids = read_citing_ids()
    rows = []
    stats = Counter()

    for celex_from in tqdm(citing_ids, desc="Building rows"):
        pars_from = parse_paragraphs(celex_from)
        if not pars_from:
            stats["citing_unparseable"] += 1
            continue
        meta_from = read_meta(celex_from)

        for cit in read_citations(celex_from):
            target = cit.get("target", "")
            if "CJ" not in target:
                continue
            if not cit.get("from") or not cit.get("to"):
                continue
            # paragraph-to-paragraph only
            if cit.get("unit") != "paragraphs" or cit.get("cited_unit") != "paragraphs":
                stats["non_paragraph_unit"] += 1
                continue

            pars_to = parse_paragraphs(target)
            meta_to = read_meta(target)

            for fn in cit["from"]:
                for tn in cit["to"]:
                    try:
                        fn_i, tn_i = int(fn), int(tn)
                    except ValueError:
                        continue
                    stats["pairs_considered"] += 1

                    if target == celex_from:
                        stats["dropped_self"] += 1
                        continue
                    if fn_i not in pars_from:
                        stats["dropped_from_par_missing"] += 1
                        continue
                    if not pars_to:
                        stats["dropped_to_case_unavailable"] += 1
                        continue
                    if tn_i not in pars_to:
                        stats["dropped_to_par_missing"] += 1
                        continue

                    rows.append(
                        {
                            "CELEX_FROM": celex_from,
                            "TITLE_FROM": meta_from["title"],
                            "NUMBER_FROM": fn_i,
                            "DATE_FROM": meta_from["date"],
                            "TEXT_FROM": fmt_text(fn_i, pars_from[fn_i]),
                            "CELEX_TO": target,
                            "TITLE_TO": meta_to["title"],
                            "NUMBER_TO": tn_i,
                            "TEXT_TO": fmt_text(tn_i, pars_to[tn_i]),
                            "DATE_TO": meta_to["date"],
                        }
                    )
                    stats["rows_built"] += 1

    df = pd.DataFrame(rows)
    return df, stats


def main() -> None:
    df_new, stats = build_new_rows()

    print("\n=== Build stats ===")
    for k in [
        "pairs_considered",
        "rows_built",
        "dropped_from_par_missing",
        "dropped_to_case_unavailable",
        "dropped_to_par_missing",
        "dropped_self",
        "non_paragraph_unit",
        "citing_unparseable",
    ]:
        print(f"  {k:32s}: {stats.get(k, 0)}")

    if df_new.empty:
        print("No new rows built; aborting.")
        return

    # Normalise dates and apply the same filters as the old cleaning notebook.
    for col in ("DATE_FROM", "DATE_TO"):
        df_new[col] = pd.to_datetime(df_new[col], errors="coerce")
    before = len(df_new)
    df_new = df_new.dropna(subset=["DATE_FROM", "DATE_TO"])
    df_new = df_new[df_new["DATE_FROM"] >= df_new["DATE_TO"]]
    df_new = df_new.drop_duplicates(
        subset=["CELEX_FROM", "NUMBER_FROM", "CELEX_TO", "NUMBER_TO"]
    )
    print(f"  after date/temporal/dedup filters: {len(df_new)} (removed {before - len(df_new)})")
    print(f"  unique citing cases: {df_new['CELEX_FROM'].nunique()}")

    df_new.to_csv(OUT_NEW, index=False)
    print(f"Wrote new rows -> {OUT_NEW}")

    # --- merge with old ---
    df_old = pd.read_excel(OLD_XLSX)
    for col in ("DATE_FROM", "DATE_TO"):
        df_old[col] = pd.to_datetime(df_old[col], errors="coerce")
    df_old = df_old.dropna()
    df_old = df_old[df_old["CELEX_FROM"] != df_old["CELEX_TO"]]
    df_old = df_old[df_old["DATE_FROM"] >= df_old["DATE_TO"]]

    merged = pd.concat([df_old, df_new], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(
        subset=["CELEX_FROM", "NUMBER_FROM", "CELEX_TO", "NUMBER_TO"]
    )
    print("\n=== Merge ===")
    print(f"  old rows: {len(df_old)}  new rows: {len(df_new)}")
    print(f"  merged: {len(merged)} (removed {before - len(merged)} overlap/dup)")
    merged.to_csv(OUT_MERGED, index=False)
    print(f"Wrote merged dataset -> {OUT_MERGED}")


if __name__ == "__main__":
    main()
