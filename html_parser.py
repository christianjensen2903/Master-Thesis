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
from bs4.element import NavigableString


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
        paragraphs: list[str] = []
        point_markers = soup.find_all("p", attrs={"id": re.compile(r"^point\d+$")})

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

            # Content is in the last TD of the row
            content_td = tds[-1]

            # Simpler and more robust: take all text within the content cell using get_text
            # Using a separator preserves inline content from nested tags and links
            combined: str = content_td.get_text(" ", strip=True)
            paragraphs.append(combined)

        return {par_no: text for par_no, text in enumerate(paragraphs, start=1)}

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

        # Special case handling for documents with mixed patterns
        # This is a heuristic to handle cases like 62004TJ0406 that have both "1." and "1 " patterns
        text = self._get_text(tag).strip()
        import re

        # If the document has both patterns, prefer the "1 " pattern (sub-paragraphs)
        # This is detected by checking if there are paragraphs with both patterns in the document
        if hasattr(self, "_document_pattern_analyzed"):
            if self._document_has_mixed_patterns:
                # For mixed pattern documents, only include "1 " pattern (not "1.")
                return bool(re.match(r"^\s*\d+\s+", text)) and not bool(
                    re.match(r"^\s*\d+\.\s+", text)
                )

        # Default behavior: include all numbered paragraphs
        return True

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

        # Find the h2 header and the em tag containing the paragraphs
        h2_tag = soup.find("h2", text=re.compile(r"grounds", re.IGNORECASE))

        if not h2_tag:
            return {}

        # Find the em tag after the h2; some simplified pages do not wrap in <em>
        em_tag = h2_tag.find_next_sibling("em")

        # Collect paragraph tags: prefer <em> container, otherwise scan sibling <p> until next section
        p_tags: list[Tag] = []
        if em_tag:
            p_tags = list(em_tag.find_all("p", recursive=False))
        else:
            sibling = h2_tag.find_next_sibling()
            while sibling:
                if isinstance(sibling, Tag):
                    # Stop at next major section heading or operative part anchor
                    if sibling.name == "h2":
                        break
                    if sibling.name == "p":
                        p_tags.append(sibling)
                sibling = sibling.find_next_sibling()

        paragraphs: dict[int, str] = {}
        current_number: int | None = None

        for p_tag in p_tags:
            text = self._get_text(p_tag).strip()
            if not text:
                continue

            # Check if this paragraph starts with a number
            match = re.match(r"^\s*(\d+)[\.\)]?\s*(.+)", text)

            if match:
                proposed_num = int(match.group(1))
                proposed_text = match.group(2)

                if not current_number or proposed_num > current_number:
                    current_number = proposed_num
                    paragraphs[current_number] = proposed_text
                elif current_number is not None:
                    paragraphs[current_number] += " " + proposed_text
            elif current_number in paragraphs:
                paragraphs[current_number] += " " + text

        return paragraphs


class NestedDocumentParser(BaseJudgementParser):
    """Parser for documents with nested HTML content inside document_content div."""

    def can_parse(self, soup: bs) -> bool:
        """Check if this document has nested HTML content."""
        # Look for document_content div with nested HTML
        doc_content = soup.find("div", id="document_content")
        if not doc_content:
            return False

        # Check if there's a nested HTML tag inside (case insensitive)
        nested_html = doc_content.find("html") or doc_content.find("HTML")
        if not nested_html:
            return False

        # Check if the nested HTML has numbered paragraphs (case insensitive)
        nested_body = nested_html.find("body") or nested_html.find("BODY")
        if not nested_body:
            return False

        # Look for numbered paragraph classes in the nested content
        for cls in ["C01PointnumeroteAltN", "C01PointAltN"]:
            if nested_body.find("p", class_=cls):
                return True

        # Also check if the content is actually HTML text that needs to be parsed
        all_ps = nested_body.find_all("p")
        if len(all_ps) == 0:
            body_text = str(nested_body)
            # Try to parse the body content as HTML
            from bs4 import BeautifulSoup

            inner_soup = BeautifulSoup(body_text, "html.parser")

            # Check for numbered paragraph classes in the re-parsed content
            for cls in ["C01PointnumeroteAltN", "C01PointAltN"]:
                if inner_soup.find("p", class_=cls):
                    return True

        return False

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from nested HTML content."""
        doc_content = soup.find("div", id="document_content")
        if not doc_content:
            return {}

        nested_html = doc_content.find("html") or doc_content.find("HTML")
        if not nested_html:
            return {}

        nested_body = nested_html.find("body") or nested_html.find("BODY")
        if not nested_body:
            return {}

        # Check if we need to re-parse the body content as HTML
        body_ps = nested_body.find_all("p")
        if len(body_ps) == 0:
            # The body content is HTML text that needs to be re-parsed
            body_text = str(nested_body)
            from bs4 import BeautifulSoup

            inner_soup = BeautifulSoup(body_text, "html.parser")
            # Use the ModernJudgementParser logic on the re-parsed content
            modern_parser = ModernJudgementParser()
            return modern_parser.extract_paragraphs(inner_soup)
        else:
            # Create a new BeautifulSoup object from the nested body content
            from bs4 import BeautifulSoup

            body_soup = BeautifulSoup(str(nested_body), "html.parser")
            # Use the ModernJudgementParser logic on the nested content
            modern_parser = ModernJudgementParser()
            return modern_parser.extract_paragraphs(body_soup)


class DtDdParser(BaseJudgementParser):
    """Parser for cases that use <dt> and <dd> tags for numbered paragraphs."""

    # Localized tokens for the judgment heading across EU languages (lowercased)
    JUDGMENT_HEADINGS: list[str] = [
        # EN
        "judgment",
        # FR
        "arrêt",
        # DE
        "urteil",
        # ES
        "sentencia",
        # IT
        "sentenza",
        # NL
        "arrest",
        # PT
        "acórdão",
        # PL
        "wyrok",
        # RO
        "hotărâre",
        # BG
        "решение",
        # CS
        "rozsudek",
        # DA
        "dom",
        # ET
        "kohtuotsus",
        # FI
        "tuomio",
        # EL
        "απόφαση",
        # HU
        "ítélet",
        # LV
        "spriedums",
        # LT
        "sprendimas",
        # MT (often Italian term is used)
        "sentenza",
        # SK
        "rozsudok",
        # SL
        "sodba",
        # SV
        "dom",
    ]

    # Localized variants of the introduction to the operative part
    STOP_PHRASES: list[str] = [
        # EN
        "on those grounds",
        # FR
        "par ces motifs",
        # DE
        "aus diesen gründen",
        # ES
        "por estos motivos",
        "por esos motivos",
        # IT
        "per questi motivi",
        # NL
        "om die redenen",
        # PT
        "pelos fundamentos expostos",
        # PL
        "z tych względów",
        # RO
        "pentru aceste motive",
        # BG
        "по тези съображения",
        # CS
        "z těchto důvodů",
        # DA
        "af disse grunde",
        # ET
        "nendel põhjustel",
        # FI
        "näillä perusteilla",
        # EL
        "για τους λόγους αυτούς",
        # HU
        "ezen indokok alapján",
        # LV
        "šo apsvērumu dēļ",
        # LT
        "dėl šių priežasčių",
        # MT
        "għal dawn ir-raġunijiet",
        # SK
        "z týchto dôvodov",
        # SL
        "iz teh razlogov",
        # SV
        "på dessa grunder",
    ]

    def _strip_after_operative_intro(self, text: str) -> str:
        """Strip any content from the start of an operative-intro phrase to the end."""
        if not text:
            return text
        # Build a single regex alternation out of localized phrases; allow comma/colon/dash after
        escaped = [re.escape(p) for p in self.STOP_PHRASES]
        pattern = r"\s*(?:" + "|".join(escaped) + r")[\s\u00A0]*[,:\-–]?\s*.*$"
        return re.sub(pattern, "", text, 0, re.IGNORECASE | re.UNICODE).strip()

    def can_parse(self, soup: bs) -> bool:
        """Check if this document uses dt/dd format for numbered paragraphs."""
        # Look for dt tags that contain numbers
        dt_tags = soup.find_all("dt")
        for dt in dt_tags:
            text = self._get_text(dt).strip()
            # Check if it starts with a number
            if re.match(r"^\s*\d+\s*$", text):
                return True
        return False

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from dt/dd structure starting at H3 'Judgment'."""
        paragraphs: dict[int, str] = {}

        # Locate the Judgment heading
        judgment_heading = self._find_judgment_heading(soup)
        if not judgment_heading:
            return {}

        # Find the first outer paragraph start (dt == 1)
        first_outer: Tag | None = None
        for dt in judgment_heading.find_all_next("dt"):
            if self._get_dt_number(dt) == 1:
                first_outer = dt
                break

        if first_outer is None:
            return {}

        # Build the sequence of outer dt anchors using inner/outer disambiguation
        outer_dts = self._find_outer_dt_sequence(first_outer)
        if not outer_dts:
            return {}

        # Collect texts for each outer paragraph
        for idx, dt_anchor in enumerate(outer_dts, start=1):
            next_dt = outer_dts[idx] if idx < len(outer_dts) else None
            text = self._collect_text_between(dt_anchor, next_dt)
            if text:
                paragraphs[idx] = text

        # Ensure the final paragraph keeps text before the localized operative-intro phrase
        if paragraphs:
            last_idx = max(paragraphs.keys())
            paragraphs[last_idx] = self._strip_after_operative_intro(
                paragraphs[last_idx]
            )

        return paragraphs

    def _find_judgment_heading(self, soup: bs) -> Tag | None:
        """Locate the localized 'Judgment' heading that marks the start of numbered paragraphs."""
        # Prefer <h3> first
        for tag_name in ("h3", "h2", "h4"):
            for heading in soup.find_all(tag_name):
                heading_text = self._get_text(heading).strip().lower()
                if any(token in heading_text for token in self.JUDGMENT_HEADINGS):
                    return heading
        return None

    def _get_dt_number(self, dt_tag: Tag) -> int | None:
        """Return the integer value of a dt tag if it is a bare number (no punctuation)."""
        text = self._get_text(dt_tag).strip()
        match = re.fullmatch(r"\d+", text)
        if not match:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _collect_text_between(self, start_dt: Tag, end_dt: Tag | None) -> str:
        """Collect text after start_dt until end_dt or the first <br> tag."""
        parts: list[str] = []
        for node in start_dt.next_elements:
            # Stop at the next expected <dt>
            if isinstance(node, Tag) and end_dt is not None and node is end_dt:
                break

            # Stop at a line break
            if isinstance(node, Tag) and node.name == "b":
                break

            if isinstance(node, NavigableString):
                parent = node.parent
                if not isinstance(parent, Tag):
                    continue
                if parent.name == "dt":
                    # Skip the numeric marker itself
                    continue
                text = str(node).strip()
                if text:
                    parts.append(text)

        combined = " ".join(parts)
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined

    def _find_outer_dt_sequence(self, first_outer: Tag) -> list[Tag]:
        """Return ordered list of outer dt anchors using inner/outer counters.

        Logic:
        - Outer counter can only increase by 1.
        - Inner counter must be strictly increasing.
        - If a dt number equals both the next outer and a valid next inner, look ahead:
          if the dt immediately following the next <b> has the same number, the current
          dt is inner; otherwise, it's outer.
        """
        outers: list[Tag] = [first_outer]
        first_num = self._get_dt_number(first_outer) or 1
        outer_expected: int = first_num + 1
        inner_last: int = 0

        for dt in first_outer.find_all_next("dt"):
            if dt is first_outer:
                continue
            num = self._get_dt_number(dt)
            if num is None:
                continue

            if num == outer_expected:
                # Ambiguous: could be inner (strictly increasing) or the next outer
                if num > inner_last and self._next_b_followed_by_dt_number(
                    dt, outer_expected
                ):
                    # Treat as inner
                    inner_last = num
                    continue
                # Treat as outer
                outers.append(dt)
                outer_expected += 1
                continue

            # Strictly increasing inner numbering
            if num > inner_last and num != outer_expected:
                inner_last = num
                continue

            # Otherwise ignore (not valid progression for inner nor the next outer)
            continue

        return outers

    def _next_b_followed_by_dt_number(self, start: Tag, number: int) -> bool:
        """Check if the first <dt> after the next <b> has the given number."""
        b_tag = start.find_next("b")
        if not b_tag:
            return False
        next_dt = b_tag.find_next("dt")
        if not next_dt:
            return False
        return self._get_dt_number(next_dt) == number


class OperativePartParser(BaseJudgementParser):
    """Parser for pages that only contain an Operative part rendered via tables."""

    # Some documents use C12DispositifIntroduction, others use C09DispositifIntroduction
    HEADING_CLASSES = [
        "C12DispositifIntroduction",
        "C09DispositifIntroduction",
    ]

    def _find_operative_heading(self, soup: bs) -> Tag | None:
        """Locate the operative part heading, supporting multiple class variants."""
        # Prefer class-based detection
        for class_name in self.HEADING_CLASSES:
            heading = soup.find("p", class_=class_name)
            if heading:
                return heading

        # Fallback to text-based detection
        for p_tag in soup.find_all("p"):
            text = " ".join(p_tag.stripped_strings)
            if text and "operative part" in text.lower():
                return p_tag

        return None

    def can_parse(self, soup: bs) -> bool:
        """Detects an Operative part section (tables or numeric-only paragraphs)."""
        # Look for the Operative part heading
        heading = self._find_operative_heading(soup)
        if not heading:
            return False

        # Check if any following <p> contains a <table> with numbered first cell
        for p_tag in heading.find_all_next("p"):
            table = p_tag.find("table")
            if not table:
                # Stop if we've reached a new major section after the operative part
                raw_classes = p_tag.get("class")
                if raw_classes:
                    class_list: list[str] = (
                        [raw_classes]
                        if isinstance(raw_classes, str)
                        else list(raw_classes)
                    )
                    if ("C04Titre1" in class_list) or ("C77Signatures" in class_list):
                        break
                continue

            first_td = table.find("td")
            if not first_td:
                continue
            first_td_text = " ".join(first_td.stripped_strings)
            if re.match(r"^\s*\d+\.?\s*$", first_td_text):
                return True

        # Fallback: detect numeric-only paragraph pattern like '1.' followed by text
        numeric_seen = False
        for p_tag in heading.find_all_next("p"):
            # Stop at next major sections
            raw_classes = p_tag.get("class")
            if raw_classes:
                class_list2: list[str] = (
                    [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
                )
                if ("C04Titre1" in class_list2) or ("C77Signatures" in class_list2):
                    break

            text = " ".join(p_tag.stripped_strings)
            if re.match(r"^\s*\d+\.?\s*$", text):
                numeric_seen = True
                continue
            if numeric_seen and text:
                return True

        return False

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract numbered operative points from tables or numeric-only paragraphs."""
        heading = self._find_operative_heading(soup)
        if not heading:
            return {}

        points: dict[int, str] = {}

        # First attempt: table-based extraction
        found_table = False
        for p_tag in heading.find_all_next("p"):
            # Stop conditions after the operative part
            raw_classes = p_tag.get("class")
            class_list: list[str] = []
            if raw_classes:
                class_list = (
                    [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
                )
            if any(c in {"C04Titre1", "C77Signatures"} for c in class_list):
                break

            table = p_tag.find("table")
            if not table:
                continue

            found_table = True
            # Each table typically contains one TR with three TDs: number, spacer, text
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if not tds:
                    continue
                number_text = " ".join(tds[0].stripped_strings)
                match = re.search(r"(\d+)", number_text)
                if not match:
                    continue
                number = int(match.group(1))

                # Text is usually in the last TD; extract clean text
                text_td = tds[-1]
                text = " ".join(text_td.stripped_strings).strip()
                if text:
                    points[number] = text

        if found_table and points:
            ordered_numbers = sorted(points.keys())
            return {i: points[i] for i in ordered_numbers}

        # Fallback: numeric-only paragraph extraction (e.g., '1.' then text in subsequent <p>)
        current_number: int | None = None
        current_text_parts: list[str] = []
        for p_tag in heading.find_all_next("p"):
            raw_classes = p_tag.get("class")
            class_list3: list[str] = []
            if raw_classes:
                class_list3 = (
                    [raw_classes] if isinstance(raw_classes, str) else list(raw_classes)
                )
            if any(c in {"C04Titre1", "C77Signatures"} for c in class_list3):
                break

            text = " ".join(p_tag.stripped_strings).strip()
            if not text:
                continue

            match_num = re.match(r"^(\d+)\.\s*$", text)
            if match_num:
                # Commit previous
                if current_number is not None and current_text_parts:
                    points[current_number] = " ".join(current_text_parts)
                current_number = int(match_num.group(1))
                current_text_parts = []
                continue

            # Continuation of current number
            if current_number is not None:
                current_text_parts.append(text)

        # Commit last
        if current_number is not None and current_text_parts:
            points[current_number] = " ".join(current_text_parts)

        if not points:
            return {}
        ordered_numbers = sorted(points.keys())
        return {i: points[i] for i in ordered_numbers}


class JudgementParser:
    """Main parser that routes to the appropriate format-specific parser."""

    def __init__(self) -> None:
        self.parsers: list[BaseJudgementParser] = [
            NestedDocumentParser(),
            LegacyEurLexParser(),
            DtDdParser(),
            ModernJudgementParser(),
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


if __name__ == "__main__":
    # Get all HTML files from the cases folder
    judgment_files = glob.glob("judgments/**/*.html", recursive=True)

    if not judgment_files:
        print("No judgment files found.")
        sys.exit(1)

    # Randomly select a judgment file
    random_case = random.choice(
        [file for file in judgment_files if any(x in file for x in ["CJ", "FJ", "TJ"])]
    )

    parser = JudgementParser()

    # Continue until it finds a case which can be parsed by DtDdParser
    # dtd_parser = DtDdParser()
    # soup = parser._load_html(random_case)
    # while soup is None or not dtd_parser.can_parse(soup):
    #     random_case = random.choice(
    #         [
    #             file
    #             for file in judgment_files
    #             if any(x in file for x in ["CJ", "FJ", "TJ"])
    #         ]
    #     )
    #     soup = parser._load_html(random_case)

    # 61976CJ0085
    random_case = "judgments/62007CJ0521/eng_judgment.html"

    paragraphs = parser.extract_paragraphs(random_case)

    for number, text in list(paragraphs.items()):
        print(f"{number}:")
        print(text)
        print("\n" + "=" * 100 + "\n")

    print(f"Processed random case: {random_case}\n")
    celex = random_case.split("/")[-2].split(".")[0]
    print(f"CELEX: {celex}\n")
