import csv
from pathlib import Path
from text_cleaner import TextCleaner
from tqdm import tqdm  # type: ignore


def clean_par_to_par(
    input_path: str | Path,
    output_path: str | Path,
    remove_paragraph_numbers: bool = True,
    remove_citations: bool = True,
    remove_dates: bool = True,
    min_length: int = 0,
) -> None:
    """
    Clean all text columns in par-to-par-og.csv.

    Args:
        input_path: Path to input par-to-par-og.csv file
        output_path: Path to output cleaned CSV file
        remove_paragraph_numbers: Whether to remove paragraph numbers
        remove_citations: Whether to remove legal citations
        remove_dates: Whether to remove dates
        min_length: Minimum length for text to be kept (0 = no minimum)
    """

    # Initialize cleaner
    cleaner = TextCleaner()

    # Load CSV
    print(f"Loading par-to-par data from {input_path}...")

    total_rows = 0
    cleaned_rows = 0
    text_columns = ["TEXT_FROM", "TEXT_TO"]

    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("CSV file appears to be empty or malformed")

        # Process rows
        cleaned_rows_data = []

        for row in tqdm(reader, desc="Cleaning par-to-par data"):
            total_rows += 1
            cleaned_row = row.copy()

            # Clean text columns
            for text_col in text_columns:
                if text_col in row and row[text_col]:
                    original_text = row[text_col]

                    # Clean the text
                    cleaned_text = cleaner.clean_text(
                        original_text,
                        reference_text=None,
                        remove_paragraph_numbers=remove_paragraph_numbers,
                        remove_citations=remove_citations,
                        remove_dates=remove_dates,
                        mask_quotes=False,
                        min_length=min_length,
                    )

                    cleaned_row[text_col] = cleaned_text

            # Only keep rows where at least one text column has content after cleaning
            if any(cleaned_row.get(col, "").strip() for col in text_columns):
                cleaned_rows_data.append(cleaned_row)
                cleaned_rows += 1

    # Save cleaned CSV
    print(f"\nSaving cleaned par-to-par data to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows_data)

    print(f"\nCleaning complete!")
    print(f"Total rows: {total_rows}")
    print(f"Cleaned rows: {cleaned_rows}")
    print(f"Removed rows: {total_rows - cleaned_rows}")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "par-to-par-og.csv"
    output_path = base_dir / "data" / "par-to-par-cleaned.csv"

    # Run cleaning
    clean_par_to_par(
        input_path=input_path,
        output_path=output_path,
        remove_paragraph_numbers=True,
        remove_citations=True,
        remove_dates=True,
        min_length=0,
    )
