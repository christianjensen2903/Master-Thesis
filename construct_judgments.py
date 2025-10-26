import json
from pathlib import Path
from typing import Any
from tqdm import tqdm  # type: ignore

from judgment_parser import (
    ECJProcessor,
    ECJProcessor15,
    ECJTextProcessor,
)


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
        languages["EN"] = eng_path

    # Check for French version
    fra_path = judgment_dir / "fra_judgment.html"
    if fra_path.exists():
        languages["FR"] = fra_path

    return languages


def process_paragraphs(celex: str, judgment_dir: Path) -> dict[int, str]:
    """Process a single judgment, choosing the best language version."""

    # Get available languages
    languages = get_available_languages(judgment_dir)

    if not languages:
        print(f"No language versions found for {celex}")
        return {}

    # Parse each available language
    parsed_versions = {}
    for lang, path in languages.items():
        try:
            processor = ECJProcessor(str(path))
            paragraphs = processor.read_paragraphs()
        except Exception as e:
            paragraphs = {}

        parsed_versions[lang] = paragraphs

    # Choose the version with most paragraphs
    best_lang = max(parsed_versions.keys(), key=lambda lang: len(parsed_versions[lang]))
    best_paragraphs = parsed_versions[best_lang]

    return best_paragraphs


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


def construct_judgments_json(
    judgments_dir: str, par_to_par_path: str, output_path: str
) -> None:
    cj_judgments = find_cj_judgments(judgments_dir)
    metadata = load_metadata(par_to_par_path)
    judgments_data = {}
    failed_count = 0

    for celex in tqdm(
        cj_judgments, desc=f"Processing judgments (Failed: {failed_count})"
    ):
        judgment_dir = Path(judgments_dir) / celex

        celex_meta = metadata.get(celex, {})

        paragraphs = process_paragraphs(celex, judgment_dir)

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
    base_dir = Path(__file__).parent
    judgments_dir = base_dir / "judgments"
    par_to_par_path = base_dir / "data" / "par-to-par.json"
    output_path = base_dir / "data" / "judgments.json"
    construct_judgments_json(str(judgments_dir), str(par_to_par_path), str(output_path))


if __name__ == "__main__":
    main()
    # parse_judgment_with_processor(Path("judgments/62010CJ0412/eng_judgment.html"))
    # print(process_judgment("61980CJ0211", Path("judgments/61980CJ0211")))
    # processor: ECJProcessor = ECJProcessor("judgments/61980CJ0211/eng_judgment.html")
    # paragraphs = processor.read_paragraphs()
    # print(paragraphs)
