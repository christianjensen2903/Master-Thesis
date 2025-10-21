import argparse
import csv
import os
import shutil
from typing import Iterable


def load_mismatches(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "celex": row.get("celex", ""),
                    "paragraph_number": row.get("paragraph_number", ""),
                    "excel_text": row.get("excel_text", ""),
                    "html_parser_text": row.get("html_parser_text", ""),
                    "similarity_ratio": row.get("similarity_ratio", ""),
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


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        add_len = len(w) + (1 if current else 0)
        if current_len + add_len > width:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += add_len
    if current:
        lines.append(" ".join(current))
    return lines if lines else [""]


def print_side_by_side(
    left: str, right: str, left_title: str, right_title: str, similarity_ratio: int
) -> None:
    term_size = shutil.get_terminal_size(fallback=(120, 40))
    total_width = max(term_size.columns, 80)
    gutter = 3
    left_width = (total_width - gutter) // 2
    right_width = total_width - gutter - left_width

    left_lines = wrap_text(left, left_width)
    right_lines = wrap_text(right, right_width)
    max_lines = max(len(left_lines), len(right_lines))

    header_left = (left_title[:left_width]).ljust(left_width)
    header_right = (right_title[:right_width]).ljust(right_width)
    print(
        header_left
        + (" " * gutter)
        + header_right
        + f" (Similarity: {similarity_ratio}%)"
    )
    print(("-" * left_width) + (" " * gutter) + ("-" * right_width))

    for i in range(max_lines):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        print(l.ljust(left_width) + (" " * gutter) + r.ljust(right_width))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print Excel vs HTML parser texts side-by-side for mismatches"
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("artifacts", "html_vs_excel_mismatches.csv"),
        help="Path to mismatches CSV (default: artifacts/html_vs_excel_mismatches.csv)",
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

    rows = load_mismatches(args.csv)
    rows = filter_rows(rows, args.celex, args.paragraph)

    if not rows:
        print("No rows matched the filters.")
        return 0

    to_show = rows if args.limit is not None and args.limit < 0 else rows[: args.limit]

    for idx, r in enumerate(to_show, start=1):
        celex = r.get("celex", "")
        par = r.get("paragraph_number", "")
        title_l = f"Excel (CELEX={celex}, ¶{par})"
        title_r = "HTML Parser"
        print()
        print_side_by_side(
            r.get("excel_text", ""),
            r.get("html_parser_text", ""),
            title_l,
            title_r,
            r.get("similarity_ratio", ""),
        )
        print()
        if idx < len(to_show):
            print("=" * shutil.get_terminal_size(fallback=(120, 40)).columns)

    remaining = len(rows) - len(to_show)
    if remaining > 0:
        print(f"\n... {remaining} more rows not shown (use --limit -1 to show all).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
