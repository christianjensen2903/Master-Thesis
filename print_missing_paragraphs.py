import argparse
import csv
import os
import shutil
from typing import Iterable


def load_missing_paragraphs(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "celex": row.get("celex", ""),
                    "paragraph_number": row.get("paragraph_number", ""),
                    "issue": row.get("issue", ""),
                }
            )
        return rows


def filter_rows(
    rows: Iterable[dict[str, str]], celex: str | None, paragraph: int | None
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for r in rows:
        if celex and r.get("celex") != celex:
            continue
        if paragraph is not None:
            try:
                par_val = int(str(r.get("paragraph_number", "")).strip())
            except ValueError:
                continue
            if par_val != paragraph:
                continue
        result.append(r)
    return result


def print_missing_paragraph_info(celex: str, paragraph_number: str, issue: str) -> None:
    term_size = shutil.get_terminal_size(fallback=(120, 40))
    total_width = max(term_size.columns, 80)

    print("=" * total_width)
    print(f"Missing Paragraph Details")
    print("=" * total_width)
    print(f"CELEX ID: {celex}")
    print(f"Paragraph Number: {paragraph_number}")
    print(f"Issue: {issue}")
    print("=" * total_width)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print missing paragraphs from validation results"
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("artifacts", "missing_paragraphs.csv"),
        help="Path to missing paragraphs CSV (default: artifacts/missing_paragraphs.csv)",
    )
    parser.add_argument("--celex", default=None, help="Filter by CELEX ID")
    parser.add_argument(
        "--paragraph",
        type=int,
        default=None,
        help="Filter by paragraph number",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max rows to print (default: 10). Use -1 for all",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return 1

    rows = load_missing_paragraphs(args.csv)
    rows = filter_rows(rows, args.celex, args.paragraph)

    if not rows:
        print("No rows matched the filters.")
        return 0

    to_show = rows if args.limit is not None and args.limit < 0 else rows[: args.limit]

    for idx, r in enumerate(to_show, start=1):
        celex = r.get("celex", "")
        par = r.get("paragraph_number", "")
        issue = r.get("issue", "")

        print()
        print_missing_paragraph_info(celex, par, issue)
        print()

        if idx < len(to_show):
            print()

    remaining = len(rows) - len(to_show)
    if remaining > 0:
        print(f"\n... {remaining} more rows not shown (use --limit -1 to show all).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
