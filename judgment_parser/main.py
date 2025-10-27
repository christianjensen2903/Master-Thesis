"""
Main judgment parser that routes to appropriate format-specific parsers.

This module contains the main JudgementParser class that automatically
detects the format of a judgment document and routes to the appropriate
specialized parser.
"""

import glob
import os
import sys
from bs4 import BeautifulSoup as bs

from .base import BaseJudgementParser
from .modern import ModernJudgementParser
from .legacy import LegacyEurLexParser, LegacySingleParagraphParser
from .specialized import (
    DtDdParser,
    NestedDocumentParser,
    ReportForHearingParser,
    OperativePartParser,
)


class JudgementParser:
    """Main parser that routes to the appropriate format-specific parser."""

    def __init__(self) -> None:
        self.parsers: list[BaseJudgementParser] = [
            NestedDocumentParser(),
            LegacyEurLexParser(),
            DtDdParser(),
            ModernJudgementParser(),
            ReportForHearingParser(),
            LegacySingleParagraphParser(),
            OperativePartParser(),
        ]

    def _normalize_path(self, path: str) -> str:
        """Handle paths with commas by taking the first segment."""
        if "," in path:
            print("Comma in path")
            path = path.split(",")[0]
            print(path)
        return path

    def _load_html(self, path: str) -> bs | None:
        """Load and parse the HTML file."""
        normalized_path = self._normalize_path(path)
        if not os.path.exists(normalized_path):
            return None

        with open(normalized_path, encoding="utf-8") as file:
            content = file.read()
            parser_feature = "xml" if self._is_xml_document(content) else "lxml"
            return bs(content, parser_feature)

    def _is_xml_document(self, content: str) -> bool:
        """Heuristically determine if the document is XML/XHTML.

        Uses simple checks for XML declaration and common XHTML namespace markers
        to select the XML parser when appropriate in order to avoid warnings and
        ensure correct parsing.
        """
        import re

        stripped = content.lstrip()
        if stripped.startswith("<?xml"):
            return True
        if "http://www.w3.org/1999/xhtml" in content:
            return True
        # Some EUR-Lex files are XHTML without an explicit XML declaration
        if re.search(
            r"<html[^>]+xmlns=\"http://www.w3.org/1999/xhtml\"", content, re.IGNORECASE
        ):
            return True
        return False

    def extract_paragraphs(self, path: str) -> dict[int, str]:
        """Extract paragraphs using the appropriate parser."""
        soup = self._load_html(path)
        if not soup:
            return {}

        # Try each parser in order
        for parser in self.parsers:
            if parser.can_parse(soup):
                return parser.extract_paragraphs(soup)

        # No parser could handle this format
        return {}

    def extract_paragraphs_from_celex(self, celex_id: str) -> dict[int, str]:
        """Extract paragraphs from a judgment using CELEX ID with automatic language selection.

        Language priority: English > French > Native (remaining)
        When multiple versions exist, chooses the one with most paragraphs.
        """
        judgment_path = self._find_best_judgment(celex_id)
        if not judgment_path:
            return {}

        return self.extract_paragraphs(judgment_path)

    def _find_best_judgment(self, celex_id: str) -> str | None:
        """Find the best judgment file for a given CELEX ID.

        Returns the path to the best judgment file, or None if no judgment found.
        """
        judgments_dir = os.path.join("judgments", celex_id)
        if not os.path.exists(judgments_dir):
            return None

        # Language priority: English > French > Native (remaining)
        language_priority = ["eng", "fra"]

        # Find all available judgment files
        available_judgments = []
        for file in os.listdir(judgments_dir):
            if file.endswith("_judgment.html"):
                lang = file.split("_")[0]
                file_path = os.path.join(judgments_dir, file)
                available_judgments.append((lang, file_path))

        if not available_judgments:
            return None

        # If only one judgment, return it
        if len(available_judgments) == 1:
            return available_judgments[0][1]

        # Try languages in priority order
        for preferred_lang in language_priority:
            for lang, file_path in available_judgments:
                if lang == preferred_lang:
                    # Check if this is a good choice by parsing and counting paragraphs
                    paragraphs = self._count_paragraphs_in_file(file_path)
                    if paragraphs > 0:  # Valid judgment with content
                        return file_path

        # If no preferred language found, or all preferred languages have no content,
        # compare all available judgments and pick the one with most paragraphs
        best_judgment = None
        max_paragraphs = 0

        for lang, file_path in available_judgments:
            paragraphs = self._count_paragraphs_in_file(file_path)
            if paragraphs > max_paragraphs:
                max_paragraphs = paragraphs
                best_judgment = file_path

        return best_judgment

    def _count_paragraphs_in_file(self, file_path: str) -> int:
        """Count the number of paragraphs in a judgment file without full parsing."""
        try:
            soup = self._load_html(file_path)
            if not soup:
                return 0

            # Try each parser to see which one can handle this format
            for parser in self.parsers:
                if parser.can_parse(soup):
                    paragraphs = parser.extract_paragraphs(soup)
                    return len(paragraphs)

            return 0
        except Exception:
            return 0


if __name__ == "__main__":
    # Get all HTML files from the cases folder
    judgment_files = glob.glob("judgments/**/*.html", recursive=True)

    if not judgment_files:
        print("No judgment files found.")
        sys.exit(1)

    parser = JudgementParser()

    # Test the new CELEX-based functionality
    test_celex = "61974CJ0008"
    print(f"Testing automatic judgment selection for CELEX: {test_celex}")

    # Use the new method that automatically selects the best judgment
    paragraphs = parser.extract_paragraphs_from_celex(test_celex)

    if paragraphs:
        print(f"Successfully extracted {len(paragraphs)} paragraphs from {test_celex}")
        print("First few paragraphs:")
        for number, text in list(paragraphs.items())[:3]:  # Show first 3 paragraphs
            print(f"{number}: {text[:100]}...")
            print()
    else:
        print(f"No paragraphs found for CELEX: {test_celex}")

    # Also test with a few more CELEX IDs to demonstrate the functionality
    test_celex_ids = ["61974CJ0008", "61975CJ0001", "61976CJ0001"]

    for celex_id in test_celex_ids:
        print(f"\nTesting CELEX: {celex_id}")
        paragraphs = parser.extract_paragraphs_from_celex(celex_id)
        if paragraphs:
            print(f"  Found {len(paragraphs)} paragraphs")
        else:
            print(f"  No paragraphs found")
