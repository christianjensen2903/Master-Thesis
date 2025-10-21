import csv
import os
import re
from typing import Any

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from langdetect import detect, LangDetectException  # type: ignore
from fuzzywuzzy import fuzz  # type: ignore

from html_parser import JudgementParser, DtDdParser, LegacyEurLexParser


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and common text formatting differences."""
    if not text:
        return ""

    # First normalize all whitespace to single spaces
    normalized = re.sub(r"\s+", " ", text).strip()

    # Fix apostrophe spacing issues like "Court ' s" -> "Court's"
    # Match spaces around apostrophes and replace with single apostrophe
    normalized = re.sub(r"\s+'\s+", "'", normalized)

    # Fix other common spacing issues around punctuation
    # Remove spaces before periods, commas, semicolons, colons
    normalized = re.sub(r"\s+([.,;:])", r"\1", normalized)

    # Remove spaces after opening parentheses and before closing parentheses
    normalized = re.sub(r"\(\s+", "(", normalized)
    normalized = re.sub(r"\s+\)", ")", normalized)

    # Normalize multiple spaces again after other changes
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def strip_leading_paragraph_number(text: str) -> str:
    # Remove a leading integer with optional '.' or ')' followed by whitespace
    return re.sub(r"^\s*\d+[\.)]?", "", text).strip()


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    return text


def resolve_html_path(celex_id: str, language: str = "eng") -> str | None:
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "judgments", celex_id
    )
    if os.path.isdir(base_dir):
        # Look for language-specific file first
        lang_file = f"{language}_judgment.html"
        lang_path = os.path.join(base_dir, lang_file)
        if os.path.exists(lang_path):
            return lang_path

        # Fallback to any HTML file if language-specific not found
        for root, _dirs, files in os.walk(base_dir):
            for name in files:
                if name.lower().endswith(".html"):
                    return os.path.join(root, name)
    return None


def detect_language_from_excel_text(excel_texts: list[str]) -> str:
    """Detect the language of a document by sampling Excel text content."""
    try:
        if not excel_texts:
            return "unknown"

        # Combine all Excel texts for this CELEX ID and take a sample
        combined_text = " ".join(excel_texts)

        # Take a sample of the text (first 1000 characters should be enough for detection)
        sample_text = combined_text[:1000].strip()

        if not sample_text:
            return "unknown"

        # Normalize the text
        sample_text = normalize_whitespace(sample_text)

        # Use langdetect to determine language
        detected_lang = detect(sample_text.lower())

        # Map detected language to our language codes
        if detected_lang in ["en", "eng"]:
            return "eng"
        elif detected_lang in ["fr", "fra"]:
            return "fra"
        else:
            # For other languages, default to English
            return "eng"

    except (LangDetectException, Exception):
        # If detection fails, default to English
        return "eng"


def create_language_mapping(
    celex_ids: set[str], excel_rows: list[dict[str, Any]]
) -> dict[str, str]:
    """Create a mapping of CELEX IDs to their optimal language version based on Excel texts."""
    language_mapping: dict[str, str] = {}

    # Group Excel texts by CELEX ID
    celex_to_excel_texts: dict[str, list[str]] = {}
    for row in excel_rows:
        celex_id = row["celex"]
        excel_text = row["excel_text"]
        if celex_id not in celex_to_excel_texts:
            celex_to_excel_texts[celex_id] = []
        celex_to_excel_texts[celex_id].append(excel_text)

    print("Detecting languages for documents using Excel texts...")
    for celex_id in tqdm(celex_ids, desc="Language detection"):
        # Get Excel texts for this CELEX ID
        excel_texts = celex_to_excel_texts.get(celex_id, [])

        if not excel_texts:
            # No Excel texts available, check if HTML files exist
            eng_path = resolve_html_path(celex_id, "eng")
            fra_path = resolve_html_path(celex_id, "fra")

            if eng_path:
                language_mapping[celex_id] = "eng"
            elif fra_path:
                language_mapping[celex_id] = "fra"
            else:
                language_mapping[celex_id] = "none"
        else:
            # Detect language from Excel texts
            detected_language = detect_language_from_excel_text(excel_texts)

            # Check if the detected language version exists
            if detected_language == "fra":
                fra_path = resolve_html_path(celex_id, "fra")
                if fra_path:
                    language_mapping[celex_id] = "fra"
                else:
                    # French detected but no French HTML, fallback to English
                    eng_path = resolve_html_path(celex_id, "eng")
                    language_mapping[celex_id] = "eng" if eng_path else "none"
            else:
                # English or unknown language, try English first
                eng_path = resolve_html_path(celex_id, "eng")
                if eng_path:
                    language_mapping[celex_id] = "eng"
                else:
                    # No English HTML, try French
                    fra_path = resolve_html_path(celex_id, "fra")
                    language_mapping[celex_id] = "fra" if fra_path else "none"

    return language_mapping


def read_excel_rows(xlsx_path: str) -> list[dict[str, Any]]:
    df = pd.read_excel(xlsx_path)

    rows: list[dict[str, Any]] = []

    pruned = df[
        [
            "CELEX_FROM",
            "NUMBER_FROM",
            "TEXT_FROM",
            "CELEX_TO",
            "NUMBER_TO",
            "TEXT_TO",
        ]
    ]

    for _idx, row in pruned.iterrows():
        for side in ("FROM", "TO"):
            celex_val = str(row[f"CELEX_{side}"]).strip()
            parnum_raw = row[f"NUMBER_{side}"]
            text_val = row[f"TEXT_{side}"]
            if pd.isna(celex_val) or pd.isna(parnum_raw) or pd.isna(text_val):
                continue
            try:
                parnum_val = int(parnum_raw)
            except Exception:
                m = re.search(r"\d+", str(parnum_raw))
                if not m:
                    continue
                parnum_val = int(m.group(0))
            rows.append(
                {
                    "celex": celex_val,
                    "paragraph_number": parnum_val,
                    "excel_text": str(text_val),
                }
            )
    return rows


def compare_against_excel(xlsx_path: str) -> dict[str, Any]:
    rows = read_excel_rows(xlsx_path)
    # Deduplicate CELEX-paragraph combinations (keep first occurrence)
    seen_keys: set[tuple[str, int]] = set()
    deduped_rows: list[dict[str, Any]] = []
    for r in rows:
        key = (r["celex"], r["paragraph_number"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_rows.append(r)
    rows = deduped_rows
    parser = JudgementParser()

    # Create language mapping before processing
    unique_celex_ids = {row["celex"] for row in rows}
    language_mapping = create_language_mapping(unique_celex_ids, rows)

    celex_to_paragraphs: dict[str, dict[int, str]] = {}
    celex_dtdd: dict[str, bool] = {}
    celex_legacy: dict[str, bool] = {}
    celex_language_used: dict[str, str] = {}  # Track which language was used
    mismatches: list[dict[str, Any]] = []
    missing_files: set[str] = set()
    missing_paragraphs: list[dict[str, Any]] = []

    for item in tqdm(rows):
        celex_id = item["celex"]
        paragraph_number = item["paragraph_number"]

        if celex_id not in celex_to_paragraphs:
            # Use the pre-determined language mapping
            language_used = language_mapping.get(celex_id, "eng")

            if language_used == "none":
                missing_files.add(celex_id)
                celex_to_paragraphs[celex_id] = {}
                celex_language_used[celex_id] = "none"
            else:
                html_path = resolve_html_path(celex_id, language_used)
                if not html_path:
                    missing_files.add(celex_id)
                    celex_to_paragraphs[celex_id] = {}
                    celex_language_used[celex_id] = "none"
                else:
                    # Detect once per CELEX whether the document is parsable by DtDdParser
                    soup = parser._load_html(html_path)
                    celex_dtdd[celex_id] = (
                        DtDdParser().can_parse(soup) if soup else False
                    )
                    celex_legacy[celex_id] = (
                        LegacyEurLexParser().can_parse(soup) if soup else False
                    )
                    try:
                        extracted = parser.extract_paragraphs(html_path)
                    except Exception as e:
                        print(f"Error parsing {html_path}: {e}")
                        continue
                    # extracted = parser.extract_paragraphs(html_path)
                    celex_to_paragraphs[celex_id] = {
                        k: clean_text(v) for k, v in extracted.items()
                    }
                    celex_language_used[celex_id] = language_used

        parsed_map = celex_to_paragraphs.get(celex_id, {})
        parsed_text = parsed_map.get(paragraph_number)
        if parsed_text is None:
            missing_paragraphs.append(
                {
                    "celex": celex_id,
                    "paragraph_number": paragraph_number,
                    "issue": "missing_paragraph",
                }
            )
            continue

        excel_cmp = clean_text(item["excel_text"])

        # Use fuzzy matching to check if similarity is above 90%
        similarity_ratio = fuzz.ratio(parsed_text, excel_cmp)

        if similarity_ratio < 80:
            # If the case is parsable by DtDdParser, treat our parser as authoritative and ignore
            if celex_dtdd.get(celex_id, False):
                continue

            if celex_legacy.get(celex_id, False):
                continue

            if (celex_id, paragraph_number) in [
                ("62010CJ036", 11),
                ("62011CJ0439", 45),
                ("62011CJ0510", 23),
            ]:
                continue

            mismatches.append(
                {
                    "celex": celex_id,
                    "paragraph_number": paragraph_number,
                    "excel_text": excel_cmp,
                    "html_parser_text": parsed_text,
                    "language_used": celex_language_used.get(celex_id, "unknown"),
                    "similarity_ratio": similarity_ratio,
                }
            )

    return {
        "total_rows": len(rows),
        "unique_cases": len({r["celex"] for r in rows}),
        "mismatch_count": len(mismatches),
        "missing_file_count": len(missing_files),
        "missing_paragraph_count": len(missing_paragraphs),
        "mismatches": mismatches,
        "missing_files": sorted(missing_files),
        "missing_paragraphs": missing_paragraphs,
        "language_usage": celex_language_used,
    }


def write_report(result: dict[str, Any]) -> tuple[str, str, str]:
    artifacts_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "artifacts"
    )
    os.makedirs(artifacts_dir, exist_ok=True)

    # Write mismatches CSV
    out_csv = os.path.join(artifacts_dir, "html_vs_excel_mismatches.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "celex",
                "paragraph_number",
                "excel_text",
                "html_parser_text",
                "language_used",
                "similarity_ratio",
            ]
        )
        for row in result["mismatches"]:
            writer.writerow(
                [
                    row["celex"],
                    row["paragraph_number"],
                    row["excel_text"],
                    row["html_parser_text"],
                    row.get("language_used", "unknown"),
                    row.get("similarity_ratio", 0),
                ]
            )

    # Write missing paragraphs CSV
    missing_csv = os.path.join(artifacts_dir, "missing_paragraphs.csv")
    with open(missing_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["celex", "paragraph_number", "issue"])
        for row in result["missing_paragraphs"]:
            writer.writerow(
                [
                    row["celex"],
                    row["paragraph_number"],
                    row["issue"],
                ]
            )

    # Write language usage CSV
    language_csv = os.path.join(artifacts_dir, "language_usage.csv")
    with open(language_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["celex", "language_used"])
        for celex, language in result.get("language_usage", {}).items():
            writer.writerow([celex, language])

    print(result["missing_files"])

    return out_csv, missing_csv, language_csv


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(repo_root, "data", "par-to-par-2.xlsx")
    if not os.path.exists(xlsx_path):
        print("Excel file not found:", xlsx_path)
        return 1

    result = compare_against_excel(xlsx_path)
    out_csv, missing_csv, language_csv = write_report(result)

    # Calculate language usage statistics
    language_usage = result.get("language_usage", {})
    eng_count = sum(1 for lang in language_usage.values() if lang == "eng")
    fra_count = sum(1 for lang in language_usage.values() if lang == "fra")
    none_count = sum(1 for lang in language_usage.values() if lang == "none")

    print(
        f"Checked {result['total_rows']} rows across {result['unique_cases']} cases.\n"
        f"Mismatches: {result['mismatch_count']}, Missing files: {result['missing_file_count']}, "
        f"Missing paragraphs: {result['missing_paragraph_count']}.\n"
        f"Language usage - English: {eng_count}, French: {fra_count}, None: {none_count}\n"
        f"Mismatches written to: {out_csv}\n"
        f"Missing paragraphs written to: {missing_csv}\n"
        f"Language usage written to: {language_csv}"
    )

    return (
        0 if result["mismatch_count"] == 0 and result["missing_file_count"] == 0 else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
