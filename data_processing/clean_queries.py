import csv
from pathlib import Path
from tqdm import tqdm  # type: ignore

from .text_cleaner import TextCleaner


def clean_queries(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    cleaner = TextCleaner()

    print(f"Reading queries from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="\t")
        header = next(reader)

        rows = list(reader)
        print(f"Loaded {len(rows)} queries")

    print("Cleaning queries...")
    cleaned_rows = []
    for celex, par_num, query in tqdm(rows, desc="Cleaning"):
        cleaned_query = cleaner.clean_text(query)
        cleaned_rows.append([celex, par_num, cleaned_query])

    print(f"Writing cleaned queries to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(header)
        writer.writerows(cleaned_rows)

    print(f"Done! Cleaned {len(cleaned_rows)} queries")


def clean_par_to_par(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    cleaner = TextCleaner()

    print(f"Reading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        print(f"Loaded {len(rows)} rows")

    print("Cleaning text columns...")
    cleaned_rows = []
    for row in tqdm(rows, desc="Cleaning"):
        cleaned_row = row.copy()
        cleaned_row["TEXT_FROM"] = cleaner.clean_text(row["TEXT_FROM"])
        cleaned_row["TEXT_TO"] = cleaner.clean_text(
            row["TEXT_TO"], remove_citations=False, remove_dates=False
        )
        cleaned_rows.append(cleaned_row)

    print(f"Writing cleaned data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        fieldnames = rows[0].keys() if rows else []
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Done! Cleaned {len(cleaned_rows)} rows")


def main() -> None:
    input_path = Path("data/evaluation/queries.tsv")
    output_path = Path("data/evaluation/queries_cleaned.tsv")
    clean_queries(input_path, output_path)
    clean_par_to_par("data/par-to-par-og.csv", "data/par-to-par-cleaned.csv")


if __name__ == "__main__":
    main()
