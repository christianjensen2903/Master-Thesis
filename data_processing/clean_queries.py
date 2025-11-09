import csv
from pathlib import Path
from tqdm import tqdm  # type: ignore

from text_cleaner import TextCleaner


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


def main() -> None:
    input_path = Path("data/evaluation/queries.tsv")
    output_path = Path("data/evaluation/queries_cleaned.tsv")
    clean_queries(input_path, output_path)


if __name__ == "__main__":
    main()
