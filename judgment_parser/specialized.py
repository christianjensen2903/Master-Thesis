"""
Specialized parsers for specific judgment formats.

This module contains parsers for various specialized formats including
dt/dd structures, nested documents, reports for hearing, and operative parts.
"""

import re
from bs4 import BeautifulSoup as bs
from bs4 import Tag
from bs4.element import NavigableString

from .base import BaseJudgementParser


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
            from .modern import ModernJudgementParser

            modern_parser = ModernJudgementParser()
            return modern_parser.extract_paragraphs(inner_soup)
        else:
            # Create a new BeautifulSoup object from the nested body content
            from bs4 import BeautifulSoup

            body_soup = BeautifulSoup(str(nested_body), "html.parser")
            # Use the ModernJudgementParser logic on the nested content
            from .modern import ModernJudgementParser

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


class ReportForHearingParser(BaseJudgementParser):
    """Parser for Reports for the Hearing that use table-based format with coj-count and coj-normal classes."""

    def can_parse(self, soup: bs) -> bool:
        """Check if this document uses the Report for the Hearing format."""
        # Look for the specific table structure with coj-count and coj-normal classes
        coj_count_tags = soup.find_all("p", class_="coj-count")
        coj_normal_tags = soup.find_all("p", class_="coj-normal")

        # Must have both types of tags
        if not coj_count_tags or not coj_normal_tags:
            return False

        # Check if we have numbered paragraphs (coj-count should contain numbers)
        numbered_paragraphs = 0
        for tag in coj_count_tags:
            text = self._get_text(tag).strip()
            # Check if it's a simple number like "1.", "2.", etc.
            if re.match(r"^\s*\d+\.?\s*$", text):
                numbered_paragraphs += 1

        # Need at least a few numbered paragraphs to be considered a report
        return numbered_paragraphs >= 3

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from the Report for the Hearing format, starting from the Judgment section."""
        paragraphs: dict[int, str] = {}

        # Find the Judgment heading to start extraction from there
        judgment_heading = self._find_judgment_heading(soup)
        if not judgment_heading:
            return {}

        # Find all tables that come after the Judgment heading
        tables = judgment_heading.find_all_next("table")

        for table in tables:
            # Look for the specific table structure with coj-count and coj-normal
            coj_count = table.find("p", class_="coj-count")
            coj_normal = table.find("p", class_="coj-normal")

            if coj_count and coj_normal:
                # Extract the paragraph number
                count_text = self._get_text(coj_count).strip()

                # Only process simple numbers (1, 2, 3, etc.) not sub-numbers like (1), (2), etc.
                # Check if it's a simple number without parentheses or quotes
                if re.match(r"^\s*\d+\s*$", count_text):
                    paragraph_num = int(count_text.strip())

                    # Extract all text from the coj-normal cell and its sibling tables
                    normal_text = self._extract_full_paragraph_text(coj_normal, table)

                    if normal_text.strip():
                        # If we already have this paragraph number, append the text
                        if paragraph_num in paragraphs:
                            paragraphs[paragraph_num] += " " + normal_text
                        else:
                            paragraphs[paragraph_num] = normal_text

        return paragraphs

    def _extract_full_paragraph_text(
        self, coj_normal_tag: Tag, parent_table: Tag
    ) -> str:
        """Extract all text from a coj-normal tag, including sibling paragraphs and nested sub-points."""
        text_parts: list[str] = []

        # Find the parent td element that contains the content
        parent_td = coj_normal_tag.find_parent("td")
        if not parent_td:
            # Fallback to just the main text
            main_text = self._get_text(coj_normal_tag)
            if main_text.strip():
                text_parts.append(main_text.strip())
            return " ".join(text_parts)

        # Check if there are nested tables with subpoints
        nested_tables = parent_td.find_all("table")
        has_subpoint_tables = False

        for nested_table in nested_tables:
            if nested_table == parent_table:
                continue
            nested_count = nested_table.find("p", class_="coj-count")
            nested_normal = nested_table.find("p", class_="coj-normal")
            if nested_count and nested_normal:
                has_subpoint_tables = True
                break

        if has_subpoint_tables:
            # Check if the subpoint tables contain the same content as main paragraphs
            # by comparing the first few words of the main text with subpoint text
            main_text = self._get_text(coj_normal_tag).strip()
            first_subpoint_text = ""

            for nested_table in nested_tables:
                if nested_table == parent_table:
                    continue
                nested_count = nested_table.find("p", class_="coj-count")
                nested_normal = nested_table.find("p", class_="coj-normal")
                if nested_count and nested_normal:
                    first_subpoint_text = self._get_text(nested_normal).strip()
                    break

            # Check if there's significant overlap between main text and subpoint text
            # This indicates duplication (like in 61980CJ0158)
            has_duplication = False
            if main_text and first_subpoint_text:
                # Check if the subpoint text is much longer than the main text
                # and contains the main text content - this indicates duplication
                if len(first_subpoint_text) > len(main_text) * 2:
                    # The subpoint text is much longer, check if it contains the main text
                    if main_text.lower() in first_subpoint_text.lower():
                        has_duplication = True
                else:
                    # Simple heuristic: if the subpoint text contains a significant portion of the main text
                    main_words = main_text.split()[:10]  # First 10 words
                    subpoint_words = first_subpoint_text.split()[:10]
                    if len(main_words) > 3 and len(subpoint_words) > 3:
                        # Check if there's significant overlap
                        overlap = len(set(main_words) & set(subpoint_words))
                        if overlap > len(main_words) * 0.3:  # 30% overlap threshold
                            has_duplication = True

            if has_duplication:
                # Case like 61980CJ0158: subpoints contain the main content, avoid duplication
                # Get the main introductory text (usually the first coj-normal paragraph)
                if main_text:
                    text_parts.append(main_text)

                # Collect subpoints from nested tables
                for nested_table in nested_tables:
                    if nested_table == parent_table:
                        continue
                    nested_count = nested_table.find("p", class_="coj-count")
                    nested_normal = nested_table.find("p", class_="coj-normal")
                    if nested_count and nested_normal:
                        nested_count_text = self._get_text(nested_count).strip()
                        nested_normal_text = self._get_text(nested_normal).strip()
                        if nested_normal_text:
                            text_parts.append(
                                f"{nested_count_text} {nested_normal_text}"
                            )
            else:
                # Case like 62012CJ0377: subpoints are additional content, include both
                # Collect all coj-normal paragraphs first, but exclude those that are in nested tables
                all_normal_paragraphs = parent_td.find_all("p", class_="coj-normal")
                for p_tag in all_normal_paragraphs:
                    # Skip if this paragraph is inside a nested table (subpoint)
                    if p_tag.find_parent("table") != parent_table:
                        continue
                    p_text = self._get_text(p_tag).strip()
                    if p_text:
                        text_parts.append(p_text)

                # Then add subpoints from nested tables
                for nested_table in nested_tables:
                    if nested_table == parent_table:
                        continue
                    nested_count = nested_table.find("p", class_="coj-count")
                    nested_normal = nested_table.find("p", class_="coj-normal")
                    if nested_count and nested_normal:
                        nested_count_text = self._get_text(nested_count).strip()
                        nested_normal_text = self._get_text(nested_normal).strip()
                        if nested_normal_text:
                            text_parts.append(
                                f"{nested_count_text} {nested_normal_text}"
                            )
        else:
            # If no subpoint tables, collect all coj-normal paragraphs
            all_normal_paragraphs = parent_td.find_all("p", class_="coj-normal")
            for p_tag in all_normal_paragraphs:
                p_text = self._get_text(p_tag).strip()
                if p_text:
                    text_parts.append(p_text)

        return " ".join(text_parts)

    def _find_judgment_heading(self, soup: bs) -> Tag | None:
        """Find the Judgment heading to start extraction from there."""
        # Look for the "Judgment" heading (case insensitive)
        judgment_tags = soup.find_all("p", class_="coj-sum-title-1")
        for tag in judgment_tags:
            # Use get_text() to handle nested elements like <span>
            text = tag.get_text().strip()
            if re.match(r"^Judgment$", text, re.IGNORECASE):
                return tag

        # Fallback: look for any heading containing "Judgment" (case insensitive)
        for tag in judgment_tags:
            text = tag.get_text().strip()
            if re.search(r"Judgment", text, re.IGNORECASE):
                return tag

        return None


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
