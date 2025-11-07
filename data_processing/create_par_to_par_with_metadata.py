import csv
import json
from pathlib import Path
from tqdm import tqdm  # type: ignore


def format_metadata_value(value) -> str:
    """Format a metadata value for inclusion in text."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    if isinstance(value, dict):
        # Format dict as "key: value" pairs
        parts = [f"{k}: {v}" for k, v in value.items() if v]
        return "; ".join(parts)
    return str(value)


def create_par_to_par_with_metadata(
    input_csv_path: str | Path,
    judgments_path: str | Path,
    output_path: str | Path,
    separator: str = "<SEP>",
) -> None:
    """
    Create a version of par-to-par-cleaned where texts include metadata.

    Args:
        input_csv_path: Path to input par-to-par-cleaned.csv file
        judgments_path: Path to judgments_cleaned.json file
        output_path: Path to output CSV file
        separator: Separator to use between metadata fields and paragraph
    """
    # Load judgments
    print(f"Loading judgments from {judgments_path}...")
    with open(judgments_path, "r", encoding="utf-8") as f:
        judgments = json.load(f)

    # Build index: (celex, paragraph_number) -> text
    # Also store metadata per judgment
    paragraph_index: dict[tuple[str, int], str] = {}
    judgment_metadata: dict[str, dict] = {}

    for celex, judgment in tqdm(
        judgments.items(), desc="Indexing paragraphs and metadata"
    ):
        paragraphs = judgment.get("paragraphs", {})
        if not paragraphs:
            continue

        # Store metadata
        meta = judgment.get("meta", {}).get("meta", {})
        judgment_metadata[celex] = meta

        # Index paragraphs
        for par_num_str, text in paragraphs.items():
            par_num = int(par_num_str)
            paragraph_index[(celex, par_num)] = text

    print(
        f"Indexed {len(paragraph_index)} paragraphs from {len(judgment_metadata)} judgments"
    )

    def get_metadata_text(celex: str, par_num: int) -> str:
        """Get text with metadata prepended."""
        # Get paragraph text
        paragraph_text = paragraph_index.get((celex, par_num), "")
        if not paragraph_text:
            return ""

        # Get metadata
        meta = judgment_metadata.get(celex, {})
        if not meta:
            return paragraph_text

        # Extract and format metadata fields
        metadata_parts = []

        # subject_matter
        subject_matter = format_metadata_value(meta.get("subject_matter"))
        if subject_matter:
            metadata_parts.append(f"Subject matter: {subject_matter}")

        # authentic_language
        authentic_language = format_metadata_value(meta.get("authentic_language"))
        if authentic_language:
            metadata_parts.append(f"Authentic language: {authentic_language}")

        # application_date
        application_date = format_metadata_value(meta.get("application_date"))
        if application_date:
            metadata_parts.append(f"Application date: {application_date}")

        # date
        date = format_metadata_value(meta.get("date"))
        if date:
            metadata_parts.append(f"Date: {date}")

        # case_law_about
        case_law_about = format_metadata_value(meta.get("case_law_about"))
        if case_law_about:
            metadata_parts.append(f"Case law about: {case_law_about}")

        # advocate_general
        advocate_general = format_metadata_value(meta.get("advocate_general"))
        if advocate_general:
            metadata_parts.append(f"Advocate general: {advocate_general}")

        # rapporteur
        rapporteur = format_metadata_value(meta.get("rapporteur"))
        if rapporteur:
            metadata_parts.append(f"Rapporteur: {rapporteur}")

        # defendant
        defendant = format_metadata_value(meta.get("defendant"))
        if defendant:
            metadata_parts.append(f"Defendant: {defendant}")

        # full_title
        full_title = format_metadata_value(meta.get("full_title"))
        if full_title:
            metadata_parts.append(f"Full title: {full_title}")

        # keywords
        keywords = format_metadata_value(meta.get("keywords"))
        if keywords:
            metadata_parts.append(f"Keywords: {keywords}")

        # Combine metadata and paragraph: metadata1<SEP>metadata2<SEP>...<SEP>paragraph
        if metadata_parts:
            metadata_text = separator.join(metadata_parts)
            return separator.join([metadata_text, paragraph_text])
        else:
            return paragraph_text

    # Load CSV
    print(f"Loading par-to-par data from {input_csv_path}...")
    with open(input_csv_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("CSV file appears to be empty or malformed")

        # Process rows
        output_rows = []

        for row in tqdm(reader, desc="Adding metadata to paragraphs"):
            output_row = row.copy()

            # Process TEXT_FROM
            celex_from = str(row.get("CELEX_FROM", ""))
            number_from_str = row.get("NUMBER_FROM", "")
            if celex_from and number_from_str:
                try:
                    number_from = int(number_from_str)
                    metadata_text = get_metadata_text(celex_from, number_from)
                    if metadata_text:
                        output_row["TEXT_FROM"] = metadata_text
                    # If metadata_text is empty, keep original TEXT_FROM
                except ValueError:
                    # Skip if number_from is not a valid integer, keep original
                    pass

            # Process TEXT_TO
            celex_to = str(row.get("CELEX_TO", ""))
            number_to_str = row.get("NUMBER_TO", "")
            if celex_to and number_to_str:
                try:
                    number_to = int(number_to_str)
                    metadata_text = get_metadata_text(celex_to, number_to)
                    if metadata_text:
                        output_row["TEXT_TO"] = metadata_text
                    # If metadata_text is empty, keep original TEXT_TO
                except ValueError:
                    # Skip if number_to is not a valid integer, keep original
                    pass

            output_rows.append(output_row)

    # Save output CSV
    print(f"\nSaving par-to-par with metadata to {output_path}...")
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
    output_path = base_dir / "data" / "par-to-par-cleaned-with-metadata.csv"

    # Run processing
    create_par_to_par_with_metadata(
        input_csv_path=input_csv_path,
        judgments_path=judgments_path,
        output_path=output_path,
    )
