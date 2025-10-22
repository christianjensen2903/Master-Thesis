import json
import pandas as pd
from typing import Dict


def load_cleaned_judgments(file_path: str) -> Dict[str, Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        judgments = json.load(f)

    cleaned_paragraphs = {}
    for judgment in judgments:
        celex_id = judgment["celex_id"]
        paragraphs = judgment.get("paragraphs", {})
        cleaned_paragraphs[celex_id] = paragraphs

    return cleaned_paragraphs


def update_par_to_par(
    par_to_par_file: str, cleaned_judgments: Dict[str, Dict[str, str]]
) -> None:
    df = pd.read_csv(par_to_par_file)

    # Update TEXT_FROM column
    for idx, row in df.iterrows():
        celex_from = row["CELEX_FROM"]
        paragraph_from = str(row["NUMBER_FROM"])

        if (
            celex_from in cleaned_judgments
            and paragraph_from in cleaned_judgments[celex_from]
        ):
            df.at[idx, "TEXT_FROM"] = cleaned_judgments[celex_from][paragraph_from]

    # Update TEXT_TO column
    for idx, row in df.iterrows():
        celex_to = row["CELEX_TO"]
        paragraph_to = str(row["NUMBER_TO"])

        if (
            celex_to in cleaned_judgments
            and paragraph_to in cleaned_judgments[celex_to]
        ):
            df.at[idx, "TEXT_TO"] = cleaned_judgments[celex_to][paragraph_to]

    df.to_csv(par_to_par_file, index=False)


if __name__ == "__main__":
    cleaned_judgments = load_cleaned_judgments("../data/judgments_cleaned.json")
    update_par_to_par("../data/par-to-par.csv", cleaned_judgments)
