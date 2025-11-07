import csv
import json
from pathlib import Path
from tqdm import tqdm  # type: ignore


def create_par_to_par_with_context(
    input_csv_path: str | Path,
    judgments_path: str | Path,
    output_path: str | Path,
    separator: str = " <SEP> ",
) -> None:
    """
    Create a version of par-to-par-cleaned where texts include preceding and proceeding paragraphs.

    Args:
        input_csv_path: Path to input par-to-par-cleaned.csv file
        judgments_path: Path to judgments_cleaned.json file
        output_path: Path to output CSV file
        separator: Separator to use between preceding, paragraph, and proceeding text
    """
    # Load judgments
    print(f"Loading judgments from {judgments_path}...")
    with open(judgments_path, "r", encoding="utf-8") as f:
        judgments = json.load(f)

    # Build index: (celex, paragraph_number) -> text
    # Also store sorted paragraph numbers per judgment for context lookup
    paragraph_index: dict[tuple[str, int], str] = {}
    judgment_paragraphs: dict[str, list[int]] = {}

    for celex, judgment in tqdm(judgments.items(), desc="Indexing paragraphs"):
        paragraphs = judgment.get("paragraphs", {})
        if not paragraphs:
            continue

        # Get sorted paragraph numbers
        par_numbers = sorted([int(k) for k in paragraphs.keys()])
        judgment_paragraphs[celex] = par_numbers

        # Index paragraphs
        for par_num_str, text in paragraphs.items():
            par_num = int(par_num_str)
            paragraph_index[(celex, par_num)] = text

    print(
        f"Indexed {len(paragraph_index)} paragraphs from {len(judgment_paragraphs)} judgments"
    )

    def get_context_text(celex: str, par_num: int) -> str:
        """Get text with preceding and proceeding paragraphs."""
        if celex not in judgment_paragraphs:
            # If judgment not found, return empty string
            return ""

        par_numbers = judgment_paragraphs[celex]
        if par_num not in par_numbers:
            # If paragraph not found, return empty string
            return ""

        # Find index of current paragraph
        current_idx = par_numbers.index(par_num)

        # Get preceding paragraph
        preceding = ""
        if current_idx > 0:
            prev_par_num = par_numbers[current_idx - 1]
            preceding = paragraph_index.get((celex, prev_par_num), "")

        # Get current paragraph
        current = paragraph_index.get((celex, par_num), "")
        if not current:
            # If current paragraph text is missing, return empty
            return ""

        # Get proceeding paragraph
        proceeding = ""
        if current_idx < len(par_numbers) - 1:
            next_par_num = par_numbers[current_idx + 1]
            proceeding = paragraph_index.get((celex, next_par_num), "")

        # Combine with separator: preceding<SEP>paragraph<SEP>proceeding
        # Even if preceding or proceeding is empty, include separators
        return separator.join([preceding, current, proceeding])

    # Load CSV
    print(f"Loading par-to-par data from {input_csv_path}...")
    with open(input_csv_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("CSV file appears to be empty or malformed")

        # Process rows
        output_rows = []

        for row in tqdm(reader, desc="Adding context to paragraphs"):
            output_row = row.copy()

            # Process TEXT_FROM
            celex_from = str(row.get("CELEX_FROM", ""))
            number_from_str = row.get("NUMBER_FROM", "")
            if celex_from and number_from_str:
                try:
                    number_from = int(number_from_str)
                    context_text = get_context_text(celex_from, number_from)
                    if context_text:
                        output_row["TEXT_FROM"] = context_text
                    # If context_text is empty, keep original TEXT_FROM
                except ValueError:
                    # Skip if number_from is not a valid integer, keep original
                    pass

            # Process TEXT_TO
            celex_to = str(row.get("CELEX_TO", ""))
            number_to_str = row.get("NUMBER_TO", "")
            if celex_to and number_to_str:
                try:
                    number_to = int(number_to_str)
                    context_text = get_context_text(celex_to, number_to)
                    if context_text:
                        output_row["TEXT_TO"] = context_text
                    # If context_text is empty, keep original TEXT_TO
                except ValueError:
                    # Skip if number_to is not a valid integer, keep original
                    pass

            output_rows.append(output_row)

    # Save output CSV
    print(f"\nSaving par-to-par with context to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nComplete!")
    print(f"Processed {len(output_rows)} rows")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_csv_path = base_dir / "data" / "par-to-par-cleaned.csv"
    judgments_path = base_dir / "data" / "judgments_cleaned.json"
    output_path = base_dir / "data" / "par-to-par-cleaned-with-context.csv"

    # Run processing
    create_par_to_par_with_context(
        input_csv_path=input_csv_path,
        judgments_path=judgments_path,
        output_path=output_path,
    )
