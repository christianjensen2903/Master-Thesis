#!/usr/bin/env python3
"""
Script to sample paragraphs from judgments.json and demonstrate text cleaning.

This script randomly samples paragraphs from the judgments.json file and shows
the before and after results of text cleaning.
"""

import json
import random
import argparse
from typing import Any
from pathlib import Path

from data_processing.text_cleaner import TextCleaner


def load_judgments(file_path: str) -> list[dict[str, Any]]:
    """Load judgments from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_paragraphs(
    judgments: list[dict[str, Any]], num_samples: int
) -> list[tuple[str, str, str]]:
    """
    Sample random paragraphs from judgments.

    Returns:
        List of tuples: (celex_id, paragraph_number, paragraph_text)
    """
    samples = []

    # Collect all paragraphs from all judgments
    all_paragraphs = []
    for judgment in judgments:
        celex_id = judgment.get("celex_id", "Unknown")
        paragraphs = judgment.get("paragraphs", {})

        for para_num, para_text in paragraphs.items():
            all_paragraphs.append((celex_id, para_num, para_text))

    # Sample randomly
    if len(all_paragraphs) < num_samples:
        print(
            f"Warning: Only {len(all_paragraphs)} paragraphs available, sampling all of them."
        )
        return all_paragraphs

    return random.sample(all_paragraphs, num_samples)


def display_cleaning_result(
    celex_id: str, para_num: str, original: str, cleaned: str
) -> None:
    """Display the before and after cleaning results."""
    print("=" * 100)
    print(f"JUDGMENT: {celex_id} | PARAGRAPH: {para_num}")
    print("=" * 100)

    print("\n📄 ORIGINAL TEXT:")
    print("-" * 50)
    print(original)

    print("\n🧹 CLEANED TEXT:")
    print("-" * 50)
    print(cleaned)

    print(f"\n📊 STATISTICS:")
    print(f"  Original length: {len(original)} characters")
    print(f"  Cleaned length:  {len(cleaned)} characters")
    print(
        f"  Reduction:       {len(original) - len(cleaned)} characters ({((len(original) - len(cleaned)) / len(original) * 100):.1f}%)"
    )

    print("\n" + "=" * 100 + "\n")


def main() -> None:
    """Main function to run the paragraph sampling and cleaning."""
    parser = argparse.ArgumentParser(
        description="Sample paragraphs from judgments.json and show text cleaning results"
    )
    parser.add_argument(
        "--file",
        default="data/judgments.json",
        help="Path to judgments.json file (default: data/judgments.json)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of paragraphs to sample (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum length for paragraphs to be included (default: 50)",
    )
    parser.add_argument(
        "--no-citations", action="store_true", help="Don't remove citations"
    )
    parser.add_argument("--no-dates", action="store_true", help="Don't remove dates")
    parser.add_argument(
        "--no-paragraph-numbers",
        action="store_true",
        help="Don't remove paragraph numbers",
    )
    parser.add_argument(
        "--no-quotes", action="store_true", help="Don't mask quoted text"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Check if file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return

    print(f"Loading judgments from: {args.file}")
    judgments = load_judgments(args.file)
    print(f"Loaded {len(judgments)} judgments")

    # Sample paragraphs
    print(f"Sampling {args.samples} paragraphs...")
    samples = sample_paragraphs(judgments, args.samples)

    # Filter by minimum length
    if args.min_length > 0:
        samples = [
            (celex, num, text)
            for celex, num, text in samples
            if len(text.strip()) >= args.min_length
        ]
        print(
            f"After filtering by minimum length ({args.min_length}): {len(samples)} paragraphs"
        )

    if not samples:
        print("No paragraphs found matching the criteria.")
        return

    # Initialize text cleaner
    cleaner = TextCleaner()

    # Configure cleaning options
    cleaning_options = {
        "remove_citations": not args.no_citations,
        "remove_dates": not args.no_dates,
        "remove_paragraph_numbers": not args.no_paragraph_numbers,
        "mask_quotes": not args.no_quotes,
        "min_length": args.min_length,
    }

    print(f"\nCleaning options: {cleaning_options}")
    print(f"\nProcessing {len(samples)} paragraphs...\n")

    # Process each sample
    for i, (celex_id, para_num, original_text) in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] Processing {celex_id} paragraph {para_num}")

        # Clean the text
        cleaned_text = cleaner.clean_text(original_text, **cleaning_options)

        # Display results
        display_cleaning_result(celex_id, para_num, original_text, cleaned_text)

    print("✅ Processing complete!")


if __name__ == "__main__":
    main()
