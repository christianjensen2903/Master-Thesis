import pandas as pd
import json
from datetime import datetime
from data_processing.text_cleaner import TextCleaner
from tqdm import tqdm
import os


def main():
    """Apply verbatim cleaning only to query texts from judgments after 2018-01-01 using cited texts as reference."""

    print("=" * 80)
    print("APPLYING VERBATIM CLEANING TO QUERY TEXTS (POST-2018 ONLY)")
    print("=" * 80)

    # Check if files exist
    if not os.path.exists("data/par-to-par.csv"):
        print("Error: data/par-to-par.csv not found")
        return

    if not os.path.exists("data/judgments_cleaned.json"):
        print("Error: data/judgments_cleaned.json not found")
        return

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv("data/par-to-par.csv")
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    with open("data/judgments_cleaned.json", "r") as f:
        judgments = {judgment["celex_id"]: judgment for judgment in json.load(f)}

    print(f"   Loaded {len(df)} paragraph pairs")
    print(f"   Loaded {len(judgments)} judgments")

    # Initialize cleaner
    cleaner = TextCleaner()

    # Track processed paragraphs and statistics
    processed_paragraphs = set()
    verbatim_masked_count = 0
    total_pairs = len(df)

    print(
        f"\n2. Processing {total_pairs} paragraph pairs (only queries from judgments after 2018-01-01)..."
    )

    # Process each paragraph pair
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Applying verbatim cleaning"
    ):
        celex_from = row["CELEX_FROM"]
        para_num_from = str(row["NUMBER_FROM"])
        text_from = row["TEXT_FROM"]
        text_to = row["TEXT_TO"]
        date_from = row["DATE_FROM"]

        para_id = f"{celex_from}::{para_num_from}"

        # Skip if already processed
        if para_id in processed_paragraphs:
            continue

        # Only process queries from judgments dated after 2018-01-01
        if date_from < pd.Timestamp("2018-01-01"):
            continue

        # Check if the judgment exists
        if celex_from not in judgments:
            continue

        judgment = judgments[celex_from]

        # Check if the paragraph exists
        if para_num_from not in judgment["paragraphs"]:
            continue

        # Apply verbatim cleaning to the query text using cited text as reference
        cleaned_text = cleaner.clean_text(
            text_from,
            reference_text=text_to,
            mask_verbatim=True,
            remove_citations=False,  # Already cleaned
            remove_dates=False,  # Already cleaned
            remove_paragraph_numbers=False,  # Already cleaned
            min_length=10,
        )

        # Check if verbatim masking was applied
        if "<VERBATIM_TEXT>" in cleaned_text:
            verbatim_masked_count += 1

        # Update the judgment paragraph
        judgment["paragraphs"][para_num_from] = cleaned_text
        processed_paragraphs.add(para_id)

    # Save the updated judgments
    output_path = "data/judgments_cleaned_verbatim.json"
    print(f"\n3. Saving updated judgments to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(list(judgments.values()), f, indent=2)

    print(f"\n" + "=" * 80)
    print("VERBATIM CLEANING COMPLETED!")
    print("=" * 80)
    print(f"✅ Processed {len(processed_paragraphs)} paragraphs")
    print(f"✅ Applied verbatim masking to {verbatim_masked_count} paragraphs")
    print(f"✅ Updated judgments saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
