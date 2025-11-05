import json
from pathlib import Path
from text_cleaner import TextCleaner
from tqdm import tqdm  # type: ignore


def clean_judgments(
    input_path: str | Path,
    output_path: str | Path,
    remove_paragraph_numbers: bool = True,
    remove_citations: bool = True,
    remove_dates: bool = True,
    min_length: int = 0,
) -> None:
    """
    Clean all paragraphs in judgments.json.

    Args:
        input_path: Path to input judgments.json file
        output_path: Path to output cleaned judgments file
        remove_paragraph_numbers: Whether to remove paragraph numbers
        remove_citations: Whether to remove legal citations
        remove_dates: Whether to remove dates
        min_length: Minimum length for text to be kept (0 = no minimum)
    """

    # Initialize cleaner
    cleaner = TextCleaner()

    # Load judgments
    print(f"Loading judgments from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        judgments = json.load(f)

    print(f"Loaded {len(judgments)} judgments")

    # Clean each judgment
    cleaned_judgments = {}
    total_paragraphs = 0
    cleaned_paragraphs = 0

    for celex_id, judgment in tqdm(judgments.items(), desc="Cleaning judgments"):
        cleaned_judgment = {
            "paragraphs": {},
            "meta": judgment.get("meta", {}),
        }

        # Clean each paragraph
        for para_num, para_text in judgment["paragraphs"].items():
            total_paragraphs += 1

            # Clean the paragraph
            cleaned_text = cleaner.clean_text(
                para_text,
                reference_text=None,
                remove_paragraph_numbers=remove_paragraph_numbers,
                remove_citations=remove_citations,
                remove_dates=remove_dates,
                mask_quotes=False,
                min_length=min_length,
            )

            # Only keep if not empty (after cleaning)
            if cleaned_text:
                cleaned_judgment["paragraphs"][para_num] = cleaned_text
                cleaned_paragraphs += 1

        cleaned_judgments[celex_id] = cleaned_judgment

    # Save cleaned judgments
    print(f"\nSaving cleaned judgments to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_judgments, f, ensure_ascii=False, indent=2)

    print(f"\nCleaning complete!")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Cleaned paragraphs: {cleaned_paragraphs}")
    print(f"Removed paragraphs: {total_paragraphs - cleaned_paragraphs}")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "judgments.json"
    output_path = base_dir / "data" / "judgments_cleaned.json"

    # Run cleaning
    clean_judgments(
        input_path=input_path,
        output_path=output_path,
        remove_paragraph_numbers=True,
        remove_citations=True,
        remove_dates=True,
        min_length=0,
    )
