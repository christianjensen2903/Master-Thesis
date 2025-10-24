#!/usr/bin/env python3
"""
Script to clean par-to-par-2.xlsx data and save as par-to-par.csv.

This script:
1. Loads the par-to-par-2.xlsx data
2. Applies text cleaner to TEXT_FROM and TEXT_TO columns
3. Removes self loops (where CELEX_FROM == CELEX_TO)
4. Removes temporal violations (where DATE_TO > DATE_FROM)
5. Saves the cleaned data as par-to-par.csv
"""

import pandas as pd
from datetime import datetime
from data_processing.text_cleaner import TextCleaner
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load the Excel file and return as DataFrame."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_excel(file_path)
    logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df


def apply_text_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaner to TEXT_FROM and TEXT_TO columns."""
    logger.info("Applying text cleaner to text columns...")
    cleaner = TextCleaner()

    # Create a copy to avoid modifying original
    df_cleaned = df.copy()

    # Clean TEXT_FROM column
    logger.info("Cleaning TEXT_FROM column...")
    df_cleaned["TEXT_FROM_CLEANED"] = df_cleaned.apply(
        lambda row: cleaner.clean_text(
            str(row["TEXT_FROM"]) if pd.notna(row["TEXT_FROM"]) else "", min_length=10
        ),
        axis=1,
    )

    # Clean TEXT_TO column
    logger.info("Cleaning TEXT_TO column...")
    df_cleaned["TEXT_TO_CLEANED"] = df_cleaned.apply(
        lambda row: cleaner.clean_text(
            str(row["TEXT_TO"]) if pd.notna(row["TEXT_TO"]) else "", min_length=10
        ),
        axis=1,
    )

    # Replace original text columns with cleaned versions
    df_cleaned["TEXT_FROM"] = df_cleaned["TEXT_FROM_CLEANED"]
    df_cleaned["TEXT_TO"] = df_cleaned["TEXT_TO_CLEANED"]

    # Drop the temporary cleaned columns
    df_cleaned = df_cleaned.drop(["TEXT_FROM_CLEANED", "TEXT_TO_CLEANED"], axis=1)

    logger.info("Text cleaning completed")
    return df_cleaned


def remove_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where CELEX_FROM == CELEX_TO (self loops)."""
    initial_count = len(df)
    df_filtered = df[df["CELEX_FROM"] != df["CELEX_TO"]]
    removed_count = initial_count - len(df_filtered)

    logger.info(f"Removed {removed_count} self loops (CELEX_FROM == CELEX_TO)")
    logger.info(f"Remaining rows: {len(df_filtered)}")

    return df_filtered


def remove_temporal_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where DATE_TO > DATE_FROM (temporal violations)."""
    # Convert date columns to datetime
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])

    initial_count = len(df)
    # Keep only rows where DATE_TO <= DATE_FROM
    df_filtered = df[df["DATE_TO"] <= df["DATE_FROM"]]
    removed_count = initial_count - len(df_filtered)

    logger.info(f"Removed {removed_count} temporal violations (DATE_TO > DATE_FROM)")
    logger.info(f"Remaining rows: {len(df_filtered)}")

    return df_filtered


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned DataFrame to CSV."""
    logger.info(f"Saving cleaned data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(df)} rows to {output_path}")


def main():
    """Main function to process the data."""
    input_file = "data/par-to-par-2.xlsx"
    output_file = "data/par-to-par.csv"

    try:
        # Load data
        df = load_data(input_file)

        # Apply text cleaner
        df_cleaned = apply_text_cleaner(df)

        # Remove self loops
        df_no_self_loops = remove_self_loops(df_cleaned)

        # Remove temporal violations
        df_final = remove_temporal_violations(df_no_self_loops)

        # Save cleaned data
        save_cleaned_data(df_final, output_file)

        # Print summary statistics
        logger.info("=" * 50)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Original rows: {len(df)}")
        logger.info(f"After text cleaning: {len(df_cleaned)}")
        logger.info(f"After removing self loops: {len(df_no_self_loops)}")
        logger.info(f"After removing temporal violations: {len(df_final)}")
        logger.info(f"Final output saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
