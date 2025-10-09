from __future__ import annotations

"""Sample paragraph citation pairs before and after a cutoff year.

This script reads an Excel file containing paragraph-to-paragraph citation data,
splits rows into two groups based on a date column relative to a cutoff date
(default 2015-01-01), samples a specified number from each group, and writes the
sample to JSON or JSONL with additional annotation fields for manual labeling.

Example
-------
python sample_citation_pairs.py \
    --input data/par-to-par-2.xlsx \
    --cutoff 2015-01-01 \
    --sample-per-group 50 \
    --output data/sampled_par_pairs.jsonl \
    --output-format jsonl
"""

from pathlib import Path
import argparse
import json
import logging
from typing import Any

import pandas as pd  # type: ignore


def setup_logging(verbosity: int) -> None:
    """Configure root logger based on verbosity level.

    Parameters
    ----------
    verbosity
        0 for WARNING, 1 for INFO, 2+ for DEBUG.
    """

    level = (
        logging.WARNING
        if verbosity <= 0
        else logging.INFO if verbosity == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Sample paragraph citation pairs before/after a cutoff date"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/par-to-par-2.xlsx"),
        help="Path to input Excel file",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=0,
        help="Excel sheet name or index (default first sheet)",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2015-01-01",
        help="Cutoff date in ISO format YYYY-MM-DD",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=None,
        help="Name of date column to split on (auto-detect if omitted)",
    )
    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=50,
        help="Number of rows to sample from each group",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sampled_par_pairs.jsonl"),
        help="Output file path",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format: 'json' (array) or 'jsonl' (one JSON object per line)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    return parser.parse_args()


def find_date_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Find an appropriate date column in the dataframe.

    The function prefers `preferred` if provided and present. Otherwise, it
    searches for common variants in a case-insensitive manner.

    Parameters
    ----------
    df
        Input dataframe.
    preferred
        Optional explicit column name to use if present.

    Returns
    -------
    str
        Selected column name.

    Raises
    ------
    ValueError
        If no suitable date column is found.
    """

    logger = logging.getLogger(__name__)
    columns_lower = {c.lower(): c for c in df.columns}

    if preferred is not None:
        # Exact or case-insensitive match
        if preferred in df.columns:
            return preferred
        if preferred.lower() in columns_lower:
            return columns_lower[preferred.lower()]
        logger.warning(
            "Preferred date column '%s' not found; falling back to auto-detection",
            preferred,
        )

    candidates = [
        "date_from",
        "date",
        "date_to",
        "decision_date",
        "judgment_date",
        "doc_date",
    ]
    for cand in candidates:
        if cand in columns_lower:
            return columns_lower[cand]

    # Fallback: any column with 'date' substring
    date_like = [orig for low, orig in columns_lower.items() if "date" in low]
    if date_like:
        logger.info("Auto-selected date-like column '%s'", date_like[0])
        return date_like[0]

    raise ValueError("No suitable date column found. Specify with --date-col.")


def ensure_datetime(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to datetime, coercing errors to NaT.

    Parameters
    ----------
    series
        Input series.

    Returns
    -------
    pd.Series
        Series converted to datetime64[ns].
    """

    return pd.to_datetime(series, errors="coerce")


def add_split_column(df: pd.DataFrame, date_col: str, cutoff_iso: str) -> pd.DataFrame:
    """Add a 'split' column with values 'pre_cutoff' or 'post_cutoff'.

    Parameters
    ----------
    df
        Input dataframe.
    date_col
        Column name to use for the split.
    cutoff_iso
        Cutoff date in ISO format (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Dataframe with a new 'split' column.
    """

    df = df.copy()
    cutoff = pd.Timestamp(cutoff_iso)
    dates = ensure_datetime(df[date_col])
    # Use numpy.where for vectorized conditional assignment
    import numpy as np  # type: ignore

    df["split"] = np.where(dates < cutoff, "pre_cutoff", "post_cutoff")
    return df


def sample_group(
    df: pd.DataFrame, group_value: str, n: int, random_state: int
) -> pd.DataFrame:
    """Sample up to n rows from a dataframe group.

    Parameters
    ----------
    df
        Source dataframe must include a 'split' column.
    group_value
        Value in the 'split' column to filter by.
    n
        Desired sample size.
    random_state
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe (size <= n).
    """

    logger = logging.getLogger(__name__)
    group_df = df[df["split"] == group_value]
    available = len(group_df)
    if available == 0:
        logger.warning("No rows found for split '%s'", group_value)
        return group_df
    if available <= n:
        logger.info(
            "Split '%s' has only %d rows; returning all", group_value, available
        )
        return group_df.sample(n=available, random_state=random_state)
    return group_df.sample(n=n, random_state=random_state)


def convert_datetimes_to_iso(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all datetime-like columns to ISO strings for JSON serialization.

    Parameters
    ----------
    df
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with datetime columns converted to string.
    """

    result = df.copy()
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].dt.strftime("%Y-%m-%d")
    return result


def add_annotation_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add empty annotation fields to the dataframe.

    Adds the following columns:
    - is_correct: None
    - annotator: None
    - notes: None

    Parameters
    ----------
    df
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with added annotation columns.
    """

    annotated = df.copy()
    annotated["is_correct"] = None
    annotated["annotator"] = None
    annotated["notes"] = None
    return annotated


def write_json(records: list[dict[str, Any]], path: Path, fmt: str) -> None:
    """Write records to disk as JSON or JSONL.

    Parameters
    ----------
    records
        List of JSON-serializable dicts.
    path
        Destination file path.
    fmt
        Either 'json' or 'jsonl'.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("sample_citation_pairs")

    logger.info("Reading input Excel: %s (sheet %s)", args.input, args.sheet)
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    date_col = find_date_column(df, preferred=args.date_col)
    logger.info("Using date column: %s", date_col)

    # Convert chosen date column to datetime for robust comparison
    df[date_col] = ensure_datetime(df[date_col])
    before_na = df[date_col].isna().sum()
    if before_na:
        logger.warning(
            "%d rows have invalid dates in '%s' and will be included based on NaT handling",
            before_na,
            date_col,
        )

    df = add_split_column(df, date_col=date_col, cutoff_iso=args.cutoff)
    pre_sample = sample_group(
        df,
        group_value="pre_cutoff",
        n=args.sample_per_group,
        random_state=args.random_state,
    )
    post_sample = sample_group(
        df,
        group_value="post_cutoff",
        n=args.sample_per_group,
        random_state=args.random_state,
    )

    sampled = pd.concat([pre_sample, post_sample], ignore_index=True)
    sampled = add_annotation_fields(sampled)

    # Add a stable row id if not present
    if "id" not in sampled.columns:
        sampled.insert(0, "id", range(1, len(sampled) + 1))

    serializable = convert_datetimes_to_iso(sampled)
    records: list[dict[str, Any]] = serializable.to_dict(orient="records")

    logger.info(
        "Writing %d sampled records to %s (%s)",
        len(records),
        args.output,
        args.output_format,
    )
    write_json(records, args.output, args.output_format)
    logger.info("Done")


if __name__ == "__main__":
    main()
