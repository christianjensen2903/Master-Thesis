#!/usr/bin/env python3
"""
Script to add a verified test case to test_cases.json.

Usage:
    python add_test_case.py <celex>

Example:
    python add_test_case.py 62010CJ0454

This will:
1. Parse the case using ECJProcessor
2. Show you the extracted paragraphs for verification
3. Add to test_cases.json
"""

import sys
import argparse
import json
from pathlib import Path
from html_parser import JudgementParser


def find_case_file(celex: str) -> str | None:
    """Find the HTML file for a given CELEX number."""
    cases_dir = Path("cases")
    summaries_dir = Path("summaries")

    # Try cases directory first
    case_file = cases_dir / f"{celex}.html"
    if case_file.exists():
        return str(case_file)

    # Try summaries directory
    summary_file = summaries_dir / f"{celex}.html"
    if summary_file.exists():
        return str(summary_file)

    return None


def extract_paragraphs(path: str) -> dict[int, str]:
    """Extract paragraphs from a case file."""
    parser = JudgementParser()
    return parser.extract_paragraphs(path)


def show_verification_info(celex: str, paragraphs: dict[int, str]) -> None:
    """Display information for manual verification."""
    print("\n" + "=" * 80)
    print(f"CELEX: {celex}")
    print("=" * 80)
    print(f"Total paragraphs: {len(paragraphs)}")

    if not paragraphs:
        print("⚠️  WARNING: No paragraphs extracted!")
        return

    # Check consecutive numbering
    expected_keys = set(range(1, len(paragraphs) + 1))
    actual_keys = set(paragraphs.keys())

    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing:
            print(f"⚠️  WARNING: Missing paragraph numbers: {sorted(missing)}")
        if extra:
            print(f"⚠️  WARNING: Unexpected paragraph numbers: {sorted(extra)}")
    else:
        print("✓ Paragraph numbering is consecutive from 1")

    # Check for empty paragraphs
    empty_paragraphs = [num for num, text in paragraphs.items() if not text.strip()]
    if empty_paragraphs:
        print(f"⚠️  WARNING: Found empty paragraphs: {empty_paragraphs}")
    else:
        print("✓ No empty paragraphs")

    # Show sample paragraphs
    print("\n" + "-" * 80)
    print("SAMPLE PARAGRAPHS (please verify these are correct):")
    print("-" * 80)

    # First paragraph
    if 1 in paragraphs:
        print(f"\n[Paragraph 1]")
        print(paragraphs[1][:200] + ("..." if len(paragraphs[1]) > 200 else ""))

    # Middle paragraph
    if len(paragraphs) > 5:
        middle = len(paragraphs) // 2
        if middle in paragraphs:
            print(f"\n[Paragraph {middle}]")
            print(
                paragraphs[middle][:200]
                + ("..." if len(paragraphs[middle]) > 200 else "")
            )

    # Last paragraph
    last = max(paragraphs.keys())
    print(f"\n[Paragraph {last}]")
    print(paragraphs[last][:200] + ("..." if len(paragraphs[last]) > 200 else ""))

    print("\n" + "=" * 80)


def add_to_test_cases(celex: str, path: str, paragraphs: dict[int, str]) -> None:
    """Add a test case to test_cases.json."""
    test_cases_file = Path("test_cases.json")

    # Load existing test cases
    if test_cases_file.exists():
        with open(test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    else:
        test_cases = {}

    # Check if case already exists
    if celex in test_cases:
        print(f"\nℹ️  {celex} already exists in test_cases.json; overwriting.")

    # Convert paragraph numbers to strings (JSON requires string keys)
    paragraphs_str = {str(k): v for k, v in sorted(paragraphs.items())}

    # Add the test case
    test_cases[celex] = {
        "path": path,
        "paragraphs": paragraphs_str,
    }

    # Save to file with nice formatting
    with open(test_cases_file, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Successfully added {celex} to test_cases.json")
    print(f"Total test cases: {len(test_cases)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a verified test case to test_cases.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 62010CJ0454
  %(prog)s 61972CJ0077 --path cases/61972CJ0077.html
  
After running this script, run the tests to verify:
  pytest test_html_parser.py -v
        """,
    )

    parser.add_argument("celex", help="CELEX number of the case (e.g., 62010CJ0454)")

    parser.add_argument(
        "--path",
        help="Path to HTML file (if not provided, will search in cases/ and summaries/)",
    )

    # Confirmation prompts have been removed to support non-interactive use

    args = parser.parse_args()

    # Find the file
    if args.path:
        path = args.path
    else:
        path = find_case_file(args.celex)
        if path is None:
            print(f"ERROR: Could not find HTML file for {args.celex}")
            print("Searched in: cases/ and summaries/")
            print("\nPlease specify the path manually using --path")
            sys.exit(1)

    if not Path(path).exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Processing {args.celex} from {path}...")
    print("Using parser: ECJProcessor15")

    # Extract paragraphs
    try:
        paragraphs = extract_paragraphs(path)
    except Exception as e:
        print(f"ERROR: Failed to parse case: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not paragraphs:
        print("ERROR: No paragraphs were extracted from the case")
        sys.exit(1)

    # Show verification info
    show_verification_info(args.celex, paragraphs)

    # Add to test cases
    add_to_test_cases(args.celex, path, paragraphs)

    print("\nNext steps:")
    print("  1. Run tests: pytest test_html_parser.py -v")
    print(
        f"  2. Verify test passes: pytest test_html_parser.py::test_paragraph_extraction[{args.celex}] -v"
    )


if __name__ == "__main__":
    main()
