import json
import os
from typing import Any
from dataclasses import dataclass

from text_cleaner import TextCleaner


@dataclass
class TestCase:
    """Represents a single test case for the TextCleaner."""

    name: str
    input_text: str
    expected_output: str
    description: str
    language: str = "eng"
    reference_text: str | None = None
    cleaning_options: dict[str, Any] | None = None


class TextCleanerTester:
    """Test framework for TextCleaner with comprehensive validation."""

    def __init__(self):
        self.cleaner = TextCleaner()
        self.test_cases: list[TestCase] = []
        self.results: list[dict[str, Any]] = []

    def load_test_cases(self, test_cases_file: str) -> None:
        """Load test cases from a JSON file."""
        with open(test_cases_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case_data in data["test_cases"]:
            # Convert cleaning_options from dict to proper format if present
            cleaning_options = case_data.get("cleaning_options", None)
            if cleaning_options:
                # Convert boolean strings to actual booleans if needed
                for key, value in cleaning_options.items():
                    if isinstance(value, str) and value.lower() in ["true", "false"]:
                        cleaning_options[key] = value.lower() == "true"

            test_case = TestCase(
                name=case_data["name"],
                input_text=case_data["input_text"],
                expected_output=case_data["expected_output"],
                description=case_data["description"],
                language=case_data.get("language", "eng"),
                reference_text=case_data.get("reference_text", None),
                cleaning_options=cleaning_options,
            )
            self.test_cases.append(test_case)

    def run_single_test(self, test_case: TestCase) -> dict[str, Any]:
        """Run a single test case and return results."""
        options = test_case.cleaning_options or {}

        # Run the cleaner
        if test_case.reference_text:
            cleaned = self.cleaner.clean_text(
                test_case.input_text,
                reference_text=test_case.reference_text,
                **options,
            )
        else:
            cleaned = self.cleaner.clean_text(test_case.input_text, **options)

        # Check if test passed
        passed = cleaned == test_case.expected_output

        return {
            "name": test_case.name,
            "description": test_case.description,
            "language": test_case.language,
            "input_text": test_case.input_text,
            "expected_output": test_case.expected_output,
            "actual_output": cleaned,
            "passed": passed,
            "error": None,
        }

    def run_all_tests(self) -> None:
        """Run all test cases and collect results."""
        print("Running TextCleaner tests...")
        print("=" * 80)

        for i, test_case in enumerate(self.test_cases, 1):
            print(f"Test {i}/{len(self.test_cases)}: {test_case.name}")
            result = self.run_single_test(test_case)
            self.results.append(result)

            if result["passed"]:
                print("  ✓ PASSED")
            else:
                print("  ✗ FAILED")
                if result["error"]:
                    print(f"    Error: {result['error']}")
                else:
                    print(f"    Expected: {result['expected_output']}")
                    print(f"    Actual:   {result['actual_output']}")
            print()

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        failed_tests = total_tests - passed_tests

        # Group by language
        by_language = {}
        for result in self.results:
            lang = result["language"]
            if lang not in by_language:
                by_language[lang] = {"total": 0, "passed": 0, "failed": 0}
            by_language[lang]["total"] += 1
            if result["passed"]:
                by_language[lang]["passed"] += 1
            else:
                by_language[lang]["failed"] += 1

        # Failed tests details
        failed_tests_details = [r for r in self.results if not r["passed"]]

        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (
                    (passed_tests / total_tests * 100) if total_tests > 0 else 0
                ),
            },
            "by_language": by_language,
            "failed_tests": failed_tests_details,
            "all_results": self.results,
        }

    def print_report(self, report: dict[str, Any]) -> None:
        """Print a formatted test report."""
        print("\n" + "=" * 80)
        print("TEXTCLEANER TEST REPORT")
        print("=" * 80)

        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")

        print("\nBy Language:")
        for lang, stats in report["by_language"].items():
            print(
                f"  {lang.upper()}: {stats['passed']}/{stats['total']} ({stats['passed']/stats['total']*100:.1f}%)"
            )

        if report["failed_tests"]:
            print(f"\nFailed Tests ({len(report['failed_tests'])}):")
            for failed in report["failed_tests"]:
                print(f"  - {failed['name']}: {failed['description']}")
                if failed["error"]:
                    print(f"    Error: {failed['error']}")
                else:
                    print(f"    Expected: {repr(failed['expected_output'])}")
                    print(f"    Actual:   {repr(failed['actual_output'])}")

    def save_report(self, report: dict[str, Any], output_path: str) -> None:
        """Save the test report to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {output_path}")


def main() -> int:
    tester = TextCleanerTester()

    # Load test cases from JSON file
    test_cases_file = "data_processing/text_cleaner_test_cases.json"
    if not os.path.exists(test_cases_file):
        print(f"Test cases file not found: {test_cases_file}")
        return 1

    tester.load_test_cases(test_cases_file)

    tester.run_all_tests()

    report = tester.generate_report()
    tester.print_report(report)

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "text_cleaner_test_results.json")
    tester.save_report(report, report_path)

    return 0 if report["summary"]["failed_tests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
