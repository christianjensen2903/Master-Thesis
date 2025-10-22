"""
Modern CJEU judgment parser for cases with CSS classes (2000s+).

This module handles the modern CJEU format with structured HTML and CSS classes
like C01PointAltN and C01PointnumeroteAltN.
"""

import re
from bs4 import BeautifulSoup as bs
from bs4 import Tag

from .base import BaseJudgementParser


class ModernJudgementParser(BaseJudgementParser):
    """Parser for modern CJEU judgment format (with CSS classes)."""

    # CSS classes that indicate the end of judgment content
    STOP_CLASSES = [
        "C04Titre1",
        "C03Titre1",  # e.g., Conclusion heading
        "S40Titre",
        "C41DispositifIntroduction",
        "C30Dispositifalinea",
        "C77Signatures",
        "C42FootnoteLangue",  # some docs use 42
        "C40FootnoteLangue",  # some docs use 40 (e.g., 62009CC0397)
        "Cfootnotetext",  # individual footnote paragraphs
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
        # Table-based format: presence of <p id="pointN"> markers (and often class="count")
        if soup.find("p", attrs={"id": re.compile(r"^point\d+$")}):
            return self._extract_table_format(soup)

        # Find all paragraph elements
        all_paragraphs = soup.find_all("p")
        if not all_paragraphs:
            return {}

        # Analyze document pattern to determine if it has mixed numbering patterns
        self._analyze_document_pattern(all_paragraphs)

        # Find where the judgment content starts
        start_paragraph = self._find_starting_paragraph(all_paragraphs)
        if not start_paragraph:
            return {}

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

    def _analyze_document_pattern(self, all_paragraphs: list[Tag]) -> None:
        """Analyze the document to detect if it has mixed numbering patterns."""
        import re

        period_pattern_count = 0
        no_period_pattern_count = 0

        # Count paragraphs with different numbering patterns
        for p in all_paragraphs:
            raw_classes = p.get("class")
            if not raw_classes:
                continue
            classes = (
                [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
            )

            # Only analyze paragraphs with numbered classes
            if not any(
                num_class in classes for num_class in self.NUMBERED_PARAGRAPH_CLASSES
            ):
                continue

            text = self._get_text(p).strip()
            if re.match(r"^\s*\d+\.\s+", text):
                period_pattern_count += 1
            elif re.match(r"^\s*\d+\s+", text):
                no_period_pattern_count += 1

        # Mark as analyzed and determine if it has mixed patterns
        self._document_pattern_analyzed = True
        self._document_has_mixed_patterns = (
            period_pattern_count > 0 and no_period_pattern_count > 0
        )

    def _extract_texts_from_numbered_paragraphs(
        self, numbered_paragraphs: list[Tag]
    ) -> dict[int, str]:
        """Extract text from numbered paragraphs and their following siblings."""
        result = {}
        expected_number = 1

        for paragraph in numbered_paragraphs:
            # Extract the actual paragraph number from the text
            actual_number = self._extract_paragraph_number(paragraph)

            # Skip paragraphs that don't have the expected sequential number
            if actual_number != expected_number:
                continue

            full_text = self._extract_paragraph_with_siblings(paragraph)
            if full_text:
                result[expected_number] = full_text
                expected_number += 1

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

    def _extract_paragraph_number(self, paragraph: Tag) -> int | None:
        """Extract the paragraph number from the paragraph text.

        Returns the actual number found in the text, or None if no valid number is found.
        """
        import re

        text = self._get_text(paragraph).strip()

        # Try to match patterns like "1.", "1 ", "1)", etc.
        patterns = [
            r"^\s*(\d+)\.\s+",  # "1. " pattern
            r"^\s*(\d+)\s+",  # "1 " pattern
            r"^\s*(\d+)\)\s+",  # "1) " pattern
        ]

        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                return int(match.group(1))

        return None

    def _extract_table_format(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from the newer table-based format."""
        paragraphs: dict[int, str] = {}
        point_markers = soup.find_all("p", attrs={"id": re.compile(r"^point\d+$")})
        expected_number = 1

        for marker in point_markers:
            # Skip nested markers that belong to sub-points inside a parent paragraph's content cell
            if self._is_nested_point_marker(marker):
                continue
            row_parent = marker.find_parent("tr")
            if not row_parent:
                continue

            # Only consider the immediate TDs of this row to avoid nested table TDs
            tds = row_parent.find_all("td", recursive=False)
            if len(tds) < 2:
                continue

            # Extract the actual paragraph number from the marker ID
            marker_id = marker.get("id", "")
            if not isinstance(marker_id, str):
                continue
            match = re.match(r"^point(\d+)$", marker_id)
            if not match:
                continue

            actual_number = int(match.group(1))

            # Skip if the number is not the expected sequential number
            if actual_number != expected_number:
                continue

            # Content is in the last TD of the row
            content_td = tds[-1]

            # Simpler and more robust: take all text within the content cell using get_text
            # Using a separator preserves inline content from nested tags and links
            combined: str = content_td.get_text(" ", strip=True)
            if combined:
                paragraphs[expected_number] = combined
                expected_number += 1

        return paragraphs

    def _is_nested_point_marker(self, marker: Tag) -> bool:
        """Return True if the marker is inside the content cell of another numbered row.

        Heuristic: if any ancestor TD has a previous sibling TD that itself contains
        a `p` with id matching ^point\\d+$, then this marker is nested inside that
        row's content cell and should not be treated as a top-level paragraph marker.
        """
        for td in marker.find_parents("td"):
            prev_td = td.find_previous_sibling("td")
            if not prev_td:
                continue
            if prev_td.find("p", attrs={"id": re.compile(r"^point\d+$")}):
                return True
        return False

    def _find_starting_paragraph(self, paragraphs: list[Tag]) -> Tag | None:
        # Look for first numbered paragraph, but skip summary sections
        for p in paragraphs:
            if self._has_any_numbered_class(p):
                # Skip summary items (C45MotClenumerote class)
                if self._has_class(p, "C45MotClenumerote"):
                    continue
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
        # Explicit stop classes
        if any(stop_class in classes for stop_class in self.STOP_CLASSES):
            return True

        # Generic section headings (e.g., Titres/Titre*) should also stop collection
        return self._is_section_heading(tag)

    def _has_any_numbered_class(self, tag: Tag) -> bool:
        """Check if a tag has any of the numbered paragraph classes."""
        raw_classes = tag.get("class")
        if not raw_classes:
            return False
        classes = [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)

        # Check if it has the numbered class
        has_numbered_class = any(
            num_class in classes for num_class in self.NUMBERED_PARAGRAPH_CLASSES
        )

        if not has_numbered_class:
            return False

        # Check if this is a subparagraph rather than a main numbered paragraph
        text = self._get_text(tag).strip()
        import re

        # Only treat as a numbered paragraph if it actually starts with a number
        # This prevents paragraphs with the numbered class but no number from being treated as numbered
        if re.match(r"^\s*\d+\s+", text):
            return True
        else:
            return False

    def _is_section_heading(self, tag: Tag) -> bool:
        """Heuristically detect section headings by class name.

        Many CJEU documents mark major/minor headings with class names containing
        'Titre' or 'Titresansnumero'. These should not be appended to numbered
        paragraph bodies and should terminate sibling aggregation.
        """
        raw_classes = tag.get("class")
        if not raw_classes:
            return False
        classes = [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
        for class_name in classes:
            if ("Titre" in class_name) or ("Titresansnumero" in class_name):
                return True
        return False
