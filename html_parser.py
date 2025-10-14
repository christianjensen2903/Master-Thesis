"""
HTML Parser for CJEU Judgment Documents

This module provides a flexible parser for extracting structured paragraphs from
CJEU (Court of Justice of the European Union) judgment HTML files in different formats.

Architecture:
- JudgementParser: Main parser that routes to format-specific parsers
- BaseJudgementParser: Abstract base class for format-specific parsers
- LegacyEurLexParser: Handles older EUR-Lex format (1950s-1990s)
- ModernJudgementParser: Handles modern CJEU format with CSS classes (2000s+)

Supported Formats:
1. Legacy EUR-Lex: Simple HTML with <h2> sections and numbered <p> tags
   Example: cases/61972CJ0077.html

2. Modern CJEU: Structured HTML with CSS classes (C01PointAltN, C01PointnumeroteAltN)
   Example: cases/62010CJ0454.html

Usage:
    parser = JudgementParser()
    paragraphs = parser.extract_paragraphs("path/to/case.html")
    # Returns: {1: "text...", 2: "text...", ...}
"""

import glob
import os
import os.path
import random
import re
import sys
from abc import ABC, abstractmethod

from bs4 import BeautifulSoup as bs
from bs4 import Tag


class BaseJudgementParser(ABC):
    """Base class for judgment parsers."""

    @abstractmethod
    def can_parse(self, soup: bs) -> bool:
        """Check if this parser can handle the given HTML structure."""
        pass

    @abstractmethod
    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract numbered paragraphs from the judgment."""
        pass

    def _get_text(self, tag: Tag) -> str:
        """Extract cleaned text from a tag."""
        return " ".join(tag.stripped_strings)


class ModernJudgementParser(BaseJudgementParser):
    """Parser for modern CJEU judgment format (with CSS classes)."""

    # CSS classes that indicate the end of judgment content
    STOP_CLASSES = [
        "C04Titre1",
        "S40Titre",
        "C41DispositifIntroduction",
        "C30Dispositifalinea",
        "C77Signatures",
        "C42FootnoteLangue",
    ]

    # CSS classes for numbered paragraphs (various formats)
    NUMBERED_PARAGRAPH_CLASSES = [
        "C01PointnumeroteAltN",
        "C01PointAltN",
    ]

    def can_parse(self, soup: bs) -> bool:
        """Check if this is the modern format with CSS classes."""
        # Check for numbered paragraph classes
        for cls in self.NUMBERED_PARAGRAPH_CLASSES:
            if soup.find("p", class_=cls):
                return True
        # Check for table format with point IDs
        if soup.find("p", attrs={"id": re.compile(r"point\d")}):
            return True
        return False

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        # Find all paragraph elements
        all_paragraphs = soup.find_all("p")
        if not all_paragraphs:
            return {}

        # Find where the judgment content starts
        start_paragraph = self._find_starting_paragraph(all_paragraphs)
        if not start_paragraph:
            return {}

        # Check for table-based format (newer format)
        if start_paragraph.get("count"):
            return self._extract_table_format(soup)

        # Collect all numbered paragraph markers
        numbered_paragraphs = self._collect_numbered_paragraphs(start_paragraph)

        # Extract text for each numbered paragraph
        return self._extract_texts_from_numbered_paragraphs(numbered_paragraphs)

    def _collect_numbered_paragraphs(self, start_paragraph: Tag) -> list[Tag]:
        """Collect all paragraphs with the numbered class starting from the given paragraph."""
        numbered_paragraphs = []
        current = start_paragraph

        while current:
            if self._has_any_numbered_class(current):
                numbered_paragraphs.append(current)

            next_sibling = current.find_next_sibling("p")
            if not next_sibling:
                break
            current = next_sibling

        return numbered_paragraphs

    def _extract_texts_from_numbered_paragraphs(
        self, numbered_paragraphs: list[Tag]
    ) -> dict[int, str]:
        """Extract text from numbered paragraphs and their following siblings."""
        result = {}
        for i, paragraph in enumerate(numbered_paragraphs, start=1):
            full_text = self._extract_paragraph_with_siblings(paragraph)
            if full_text:
                result[i] = full_text
        return result

    def _extract_paragraph_with_siblings(self, paragraph: Tag) -> str:
        """Extract text from a numbered paragraph including all following sibling paragraphs."""
        text_parts = [self._get_text(paragraph)]
        sibling_texts = self._collect_sibling_texts(paragraph)
        text_parts.extend(sibling_texts)
        return " ".join(text_parts)

    def _collect_sibling_texts(self, paragraph: Tag) -> list[str]:
        """Collect text from all sibling paragraphs until the next numbered paragraph or stop class."""
        text_parts = []
        sibling = paragraph.find_next_sibling("p")

        while sibling:
            # Stop if we hit another numbered paragraph or a stop class
            if self._has_any_numbered_class(sibling):
                break
            if self._has_any_stop_class(sibling):
                break

            # Add sibling text
            sibling_text = self._get_text(sibling)
            if sibling_text:
                text_parts.append(sibling_text)

            sibling = sibling.find_next_sibling("p")

        return text_parts

    def _extract_table_format(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from the newer table-based format."""
        pars = []
        pars_temp = soup.find_all("p", attrs={"id": re.compile(r"point\d")})

        for p in pars_temp:
            row_parent = p.find_parent("tr")
            if row_parent:
                p_text = row_parent.find("p", attrs={"class": "normal"})
                if p_text:
                    pars.append(self._get_text(p_text))

        return {par_no: text for par_no, text in enumerate(pars, start=1)}

    def _find_starting_paragraph(self, paragraphs: list[Tag]) -> Tag | None:
        # Look for first numbered paragraph
        for p in paragraphs:
            if self._has_any_numbered_class(p):
                return p

        # If no numbered paragraph found, return first paragraph as fallback
        return paragraphs[0] if paragraphs else None

    def _has_class(self, tag: Tag, class_name: str) -> bool:
        """Check if a tag has a specific CSS class."""
        raw_classes = tag.get("class")
        if not raw_classes:
            return False
        classes = [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
        return class_name in classes

    def _has_any_stop_class(self, tag: Tag) -> bool:
        """Check if a tag has any of the stop classes."""
        raw_classes = tag.get("class")
        if not raw_classes:
            return False
        classes = [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
        return any(stop_class in classes for stop_class in self.STOP_CLASSES)

    def _has_any_numbered_class(self, tag: Tag) -> bool:
        """Check if a tag has any of the numbered paragraph classes."""
        raw_classes = tag.get("class")
        if not raw_classes:
            return False
        classes = [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
        return any(
            num_class in classes for num_class in self.NUMBERED_PARAGRAPH_CLASSES
        )


class LegacyEurLexParser(BaseJudgementParser):
    """Parser for legacy EUR-Lex format (older cases, simple HTML structure)."""

    def can_parse(self, soup: bs) -> bool:
        """Check if this is the legacy EUR-Lex format."""
        # Look for the Grounds section with anchor tag
        grounds_section = soup.find("a", attrs={"name": "MO"})
        if grounds_section:
            return True
        return False

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from the Grounds section."""
        # Find the Grounds section
        grounds_anchor = soup.find("a", attrs={"name": "MO"})
        if not grounds_anchor:
            return {}

        # Find the h2 header and the em tag containing the paragraphs
        h2_tag = grounds_anchor.find_next_sibling("h2")
        if not h2_tag:
            return {}

        # Find the em tag after the h2
        em_tag = h2_tag.find_next_sibling("em")
        if not em_tag:
            return {}

        # Find all p tags within the em section
        p_tags = em_tag.find_all("p", recursive=False)

        paragraphs = {}
        current_number = None
        current_text_parts = []

        for p_tag in p_tags:
            text = self._get_text(p_tag).strip()
            if not text:
                continue

            # Check if this paragraph starts with a number
            match = re.match(r"^(\d+)\s+(.+)", text)

            if match:
                # Save previous paragraph if exists
                if current_number is not None and current_text_parts:
                    paragraphs[current_number] = " ".join(current_text_parts)

                # Start new paragraph
                current_number = int(match.group(1))
                current_text_parts = [match.group(2)]
            elif current_number is not None:
                # This is a subsection or continuation
                # Check if it's a subsection header (all caps, starts with "AS TO")
                if text.isupper() and ("AS TO" in text or "ON THE" in text):
                    # Add as continuation with some separation
                    current_text_parts.append(text)
                else:
                    # Regular continuation
                    current_text_parts.append(text)

        # Don't forget the last paragraph
        if current_number is not None and current_text_parts:
            paragraphs[current_number] = " ".join(current_text_parts)

        return paragraphs


class JudgementParser:
    """Main parser that routes to the appropriate format-specific parser."""

    def __init__(self) -> None:
        self.parsers: list[BaseJudgementParser] = [
            LegacyEurLexParser(),
            ModernJudgementParser(),
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
            return bs(file.read(), "lxml")

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


if __name__ == "__main__":
    # Get all HTML files from the cases folder
    case_files = glob.glob("cases/*.html")

    if not case_files:
        print("No case files found in the cases folder.")
        sys.exit(1)

    # Randomly select a case file
    # random_case = random.choice(case_files)
    random_case = "cases/61972CJ0077.html"
    print(f"Processing random case: {random_case}\n")

    parser = JudgementParser()
    paragraphs = parser.extract_paragraphs(random_case)

    for number, text in paragraphs.items():
        print(f"{number}:")
        print(text)
        print("\n" + "=" * 100 + "\n")
