import json
import re
from pathlib import Path
from typing import Any, Callable
from tqdm import tqdm  # type: ignore
from bs4 import BeautifulSoup

from judgment_parser import ECJProcessor, ECJProcessor15, ECJTextProcessor


def parse_old_format_judgment(html_path: str) -> dict[int, str]:
    """
    Parse old-format ECJ judgments where all paragraphs are in a single <em> block.

    Old HTML files (pre-1990s) have all text in one <em> tag without individual <p> tags.
    Paragraphs are separated by ". " (period + spaces) or just multiple spaces.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    paragraphs = {}

    # Parse sections: Grounds (MO), Decision on costs (CO), Operative part (DI)
    for section_name in ["MO", "CO", "DI"]:
        section_anchor = soup.find("a", attrs={"name": section_name})
        if not section_anchor:
            continue

        h2 = section_anchor.find_next("h2")
        em = h2.find_next("em") if h2 else None
        if not em:
            continue

        text = em.get_text()

        # Split by punctuation/spaces followed by digit + space + uppercase letter
        parts = re.split(r"(?:[.?!]\s+|\s{2,})(?=\d+\s+[A-Z])", text.strip())

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Extract paragraph number from start of text
            match = re.match(r"^(\d+)\s+", part)
            if match:
                paragraphs[int(match.group(1))] = part

    return paragraphs


def try_parser(parser_callable: Callable[[], dict[int, str]]) -> dict[int, str]:
    """Try a parser and return results, or empty dict on failure."""
    try:
        return parser_callable()
    except Exception:
        return {}


def find_cj_judgments(judgments_dir: str) -> list[str]:
    """Find all CJ judgment directories."""
    judgments_path = Path(judgments_dir)
    cj_judgments = []

    for item in judgments_path.iterdir():
        if item.is_dir() and "CJ" in item.name:
            cj_judgments.append(item.name)

    return sorted(cj_judgments)


def get_available_languages(judgment_dir: Path) -> dict[str, Path]:
    """Get available language versions for a judgment."""
    languages = {}

    # Check for English version
    eng_path = judgment_dir / "eng_judgment.html"
    if eng_path.exists():
        languages["eng"] = eng_path

    # Check for French version
    fra_path = judgment_dir / "fra_judgment.html"
    if fra_path.exists():
        languages["fra"] = fra_path

    return languages


def try_all_parsers(path_str: str) -> dict[int, str]:
    """Try all parsers on a file and return the best result."""
    results = [
        try_parser(lambda: ECJProcessor(path_str).read_paragraphs()),
        try_parser(lambda: ECJProcessor15(path_str).read_paragraphs()),
        try_parser(lambda: parse_old_format_judgment(path_str)),
    ]
    best_result = max(results, key=len)
    return best_result


def process_paragraphs(
    celex: str, judgment_dir: Path, language_mapping: dict[str, Any]
) -> dict[int, str]:
    """Process a single judgment, using detected language first, then fallback."""
    languages = get_available_languages(judgment_dir)

    if not languages:
        print(f"No language versions found for {celex}")
        return {}

    # Get the detected language for this CELEX
    celex_info = language_mapping.get(celex, {})
    detected_lang = celex_info.get("judgment_language")  # e.g., 'eng' or 'fra'

    # Try detected language first if available
    if detected_lang and detected_lang in languages:
        path_str = str(languages[detected_lang])
        result = try_all_parsers(path_str)
        if result:
            return result

    # Fallback: try all available languages and pick the best result
    parsed_versions = {}
    for lang, path in languages.items():
        path_str = str(path)
        parsed_versions[lang] = try_all_parsers(path_str)

    # Choose the language version with most paragraphs
    if not parsed_versions:
        return {}

    return max(parsed_versions.values(), key=len)


def load_metadata(par_to_par_path: str) -> dict[str, Any]:
    """Load metadata from par-to-par.json."""
    print(f"Loading metadata from {par_to_par_path}")

    try:
        with open(par_to_par_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {len(metadata)} cases")
        return metadata
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return {}


def load_language_mapping(mapping_path: str) -> dict[str, Any]:
    """Load CELEX language mapping from JSON."""
    print(f"Loading language mapping from {mapping_path}")
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = data.get("mapping", {})
        stats = data.get("statistics", {})
        print(f"Loaded language mapping for {len(mapping)} CELEXes")
        print(f"Language distribution: {stats.get('language_distribution', {})}")
        return mapping
    except Exception as e:
        print(f"Failed to load language mapping: {e}")
        return {}


def construct_judgments_json(
    judgments_dir: str,
    par_to_par_path: str,
    output_path: str,
    language_mapping_path: str,
) -> None:
    cj_judgments = find_cj_judgments(judgments_dir)
    metadata = load_metadata(par_to_par_path)
    language_mapping = load_language_mapping(language_mapping_path)
    judgments_data = {}
    failed_count = 0

    for celex in tqdm(
        cj_judgments, desc=f"Processing judgments (Failed: {failed_count})"
    ):
        judgment_dir = Path(judgments_dir) / celex

        celex_meta = metadata.get(celex, {}).get("meta", {})

        paragraphs = process_paragraphs(celex, judgment_dir, language_mapping)

        judgments_data[celex] = {
            "paragraphs": paragraphs,
            "meta": celex_meta,
        }

        if not paragraphs:
            failed_count += 1

    # Save the results
    print(f"Saving results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judgments_data, f, ensure_ascii=False, indent=2)


def main():
    base_dir = Path(__file__).parent.parent
    judgments_dir = base_dir / "judgments"
    par_to_par_path = base_dir / "data" / "par-to-par.json"
    output_path = base_dir / "data" / "judgments.json"
    language_mapping_path = base_dir / "data" / "celex_language_mapping.json"
    construct_judgments_json(
        str(judgments_dir),
        str(par_to_par_path),
        str(output_path),
        str(language_mapping_path),
    )


if __name__ == "__main__":
    main()
