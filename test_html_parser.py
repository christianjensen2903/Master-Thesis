"""
Test suite for HTML parser to ensure correct extraction of case paragraphs.

Test cases are stored in test_cases.json. Use add_test_case.py to add new cases
after you've manually verified they parse correctly.
"""

import json
from pathlib import Path
from typing import Any
import pytest  # type: ignore
from html_parser import ECJProcessor


def load_test_cases() -> dict[str, dict[str, Any]]:
    """Load test cases from JSON file."""
    test_cases_file = Path(__file__).parent / "test_cases.json"
    with open(test_cases_file, "r", encoding="utf-8") as f:
        return json.load(f)


TEST_CASES = load_test_cases()


@pytest.mark.parametrize("celex", TEST_CASES.keys())  # type: ignore
def test_paragraph_extraction(celex: str) -> None:
    """Test that paragraphs are correctly extracted from HTML files."""
    test_case = TEST_CASES[celex]
    expected_paragraphs = test_case["paragraphs"]

    # Parse the HTML file
    parser = ECJProcessor(test_case["path"])  # type: ignore
    actual_paragraphs = parser.read_paragraphs()

    # Convert keys to strings for comparison (JSON keys are always strings)
    actual_paragraphs_str = {str(k): v for k, v in actual_paragraphs.items()}

    # Test 1: Check number of paragraphs
    expected_count = len(expected_paragraphs)
    actual_count = len(actual_paragraphs_str)
    assert (
        actual_count == expected_count
    ), f"{celex}: Expected {expected_count} paragraphs, but got {actual_count}"

    # Test 2: Check paragraph numbers match
    expected_nums = set(expected_paragraphs.keys())
    actual_nums = set(actual_paragraphs_str.keys())
    assert actual_nums == expected_nums, (
        f"{celex}: Paragraph numbers don't match.\n"
        f"Missing: {expected_nums - actual_nums}\n"
        f"Extra: {actual_nums - expected_nums}"
    )

    # Test 3: Check each paragraph content matches exactly
    mismatches = []
    for para_num in expected_nums:
        expected_text = expected_paragraphs[para_num]
        actual_text = actual_paragraphs_str[para_num]

        if expected_text != actual_text:
            mismatches.append(
                {
                    "paragraph": para_num,
                    "expected": (
                        expected_text[:100] + "..."
                        if len(expected_text) > 100
                        else expected_text
                    ),
                    "actual": (
                        actual_text[:100] + "..."
                        if len(actual_text) > 100
                        else actual_text
                    ),
                }
            )

    assert not mismatches, (
        f"{celex}: Found {len(mismatches)} paragraph(s) with mismatched content:\n"
        + "\n".join(
            f"Paragraph {m['paragraph']}:\n"
            f"  Expected: {m['expected']}\n"
            f"  Actual:   {m['actual']}"
            for m in mismatches[:3]  # Show first 3 mismatches
        )
        + (f"\n... and {len(mismatches) - 3} more" if len(mismatches) > 3 else "")
    )


def test_test_cases_file_exists() -> None:
    """Verify that test_cases.json exists and is valid JSON."""
    test_cases_file = Path(__file__).parent / "test_cases.json"
    assert test_cases_file.exists(), "test_cases.json file not found"

    with open(test_cases_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), "test_cases.json should contain a JSON object"
    assert len(data) > 0, "test_cases.json should contain at least one test case"


def test_all_cases_have_required_fields() -> None:
    """Validate that all test cases have the required fields."""
    required_fields = ["path", "paragraphs"]

    for celex, test_case in TEST_CASES.items():
        for field in required_fields:
            assert (
                field in test_case
            ), f"Test case {celex} is missing required field: {field}"

        # Check that paragraphs is a dict with string keys
        assert isinstance(
            test_case["paragraphs"], dict
        ), f"Test case {celex}: 'paragraphs' should be a dictionary"

        # Check that all paragraph numbers are valid
        for para_num in test_case["paragraphs"].keys():
            assert (
                para_num.isdigit()
            ), f"Test case {celex}: paragraph number '{para_num}' is not a digit"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
