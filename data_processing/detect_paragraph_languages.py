import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

from langdetect import detect, LangDetectException  # type: ignore
from tqdm import tqdm  # type: ignore


def normalize_text(text: str) -> str:
    """Normalize text for better language detection."""
    # Remove special markers like <DATE>, <CASE>, <PARAGRAPH>
    text = re.sub(r"<[A-Z]+>", "", text)
    # Convert to lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove extra punctuation
    text = text.strip()
    return text


def detect_language_safe(text: str) -> str | None:
    """Safely detect language with normalization."""
    try:
        normalized = normalize_text(text)
        if len(normalized) < 20:  # Need enough text for reliable detection
            return None
        return detect(normalized)
    except LangDetectException:
        return None


def extract_celex_from_par_to_par(csv_path: str) -> dict[str, list[str]]:
    """Extract all paragraph texts for each CELEX from par-to-par CSV."""
    celex_paragraphs = defaultdict(list)

    print(f"Reading paragraph data from {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Extracting paragraphs"):
            celex_from = row.get("CELEX_FROM", "")
            celex_to = row.get("CELEX_TO", "")
            text_from = row.get("TEXT_FROM", "")
            text_to = row.get("TEXT_TO", "")

            if celex_from and text_from:
                celex_paragraphs[celex_from].append(text_from)
            if celex_to and text_to:
                celex_paragraphs[celex_to].append(text_to)

    return dict(celex_paragraphs)


def detect_celex_language(paragraphs: list[str]) -> tuple[str | None, dict[str, int]]:
    """Detect the primary language of a CELEX by majority vote on paragraphs."""
    language_counts: Counter[str] = Counter()

    for para in paragraphs:
        lang = detect_language_safe(para)
        if lang:
            language_counts[lang] += 1

    if not language_counts:
        return None, {}

    # Get majority language
    majority_lang = language_counts.most_common(1)[0][0]
    return majority_lang, dict(language_counts)


def map_langdetect_to_judgment_lang(detected_lang: str) -> str | None:
    """Map langdetect language codes to judgment file prefixes."""
    mapping = {
        "en": "eng",
        "fr": "fra",
        "de": "deu",
        "nl": "nld",
        "it": "ita",
        "es": "spa",
        "pt": "por",
        "da": "dan",
        "sv": "swe",
        "fi": "fin",
        "el": "ell",
        "cs": "ces",
        "et": "est",
        "hu": "hun",
        "lt": "lit",
        "lv": "lav",
        "mt": "mlt",
        "pl": "pol",
        "sk": "slk",
        "sl": "slv",
        "bg": "bul",
        "ro": "ron",
        "hr": "hrv",
    }
    return mapping.get(detected_lang)


def create_language_mapping(par_to_par_csv: str, output_json: str) -> None:
    """Create a mapping of CELEX to detected language."""
    celex_paragraphs = extract_celex_from_par_to_par(par_to_par_csv)

    language_mapping: dict[str, Any] = {}
    language_distribution: Counter[str] = Counter()
    stats = {
        "total": len(celex_paragraphs),
        "detected": 0,
        "undetected": 0,
    }

    print("\nDetecting languages for each CELEX...")
    for celex, paragraphs in tqdm(celex_paragraphs.items(), desc="Language detection"):
        detected_lang, lang_counts = detect_celex_language(paragraphs)

        if detected_lang:
            judgment_lang = map_langdetect_to_judgment_lang(detected_lang)
            language_mapping[celex] = {
                "detected_language": detected_lang,
                "judgment_language": judgment_lang,
                "language_counts": lang_counts,
                "total_paragraphs": len(paragraphs),
                "confidence": (
                    lang_counts.get(detected_lang, 0) / len(paragraphs)
                    if paragraphs
                    else 0
                ),
            }
            stats["detected"] += 1
            language_distribution[detected_lang] += 1
        else:
            language_mapping[celex] = {
                "detected_language": None,
                "judgment_language": None,
                "language_counts": {},
                "total_paragraphs": len(paragraphs),
                "confidence": 0,
            }
            stats["undetected"] += 1

    # Save the mapping
    print(f"\nSaving language mapping to {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mapping": language_mapping,
                "statistics": {
                    "total_celexes": stats["total"],
                    "detected": stats["detected"],
                    "undetected": stats["undetected"],
                    "language_distribution": dict(language_distribution),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Print statistics
    print("\n" + "=" * 60)
    print("LANGUAGE DETECTION STATISTICS")
    print("=" * 60)
    print(f"Total CELEXes: {stats['total']}")
    print(
        f"Detected: {stats['detected']} ({stats['detected']/stats['total']*100:.1f}%)"
    )
    print(
        f"Undetected: {stats['undetected']} ({stats['undetected']/stats['total']*100:.1f}%)"
    )
    print("\nLanguage Distribution:")
    for lang, count in language_distribution.most_common():
        judgment_lang = map_langdetect_to_judgment_lang(lang)
        print(
            f"  {lang} ({judgment_lang}): {count} ({count/stats['detected']*100:.1f}%)"
        )
    print("=" * 60)


def main() -> None:
    base_dir = Path(__file__).parent.parent
    par_to_par_csv = base_dir / "data" / "par-to-par-og.csv"
    output_json = base_dir / "data" / "celex_language_mapping.json"

    create_language_mapping(str(par_to_par_csv), str(output_json))


if __name__ == "__main__":
    main()
