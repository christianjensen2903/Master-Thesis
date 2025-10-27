"""
Legacy EUR-Lex parsers for older judgment formats.

This module handles older CJEU cases with simple HTML structure
from the 1950s-1990s period.
"""

import re
from bs4 import BeautifulSoup as bs
from bs4 import Tag

from .base import BaseJudgementParser, NumberingPattern


class LegacyEurLexParser(BaseJudgementParser):
    """Parser for legacy EUR-Lex format (older cases, simple HTML structure)."""

    GROUNDS_PATTERNS = [
        r"grounds",  # English
        r"motifs",  # French
        r"gründe",  # German
        r"motivos",  # Spanish
        r"motivi",  # Italian
        r"gronden",  # Dutch
        r"fundamentação",  # Portuguese
        r"uzasadnienie",  # Polish
        r"fundamentare",  # Romanian
        r"обосновение",  # Bulgarian
        r"odůvodnění",  # Czech
        r"põhjendused",  # Estonian
        r"perustelut",  # Finnish
        r"αιτιολόγηση",  # Greek
    ]

    COSTS_PATTERNS = [
        r"decision on costs",  # English
        r"décision sur les dépens",  # French
        r"entscheidung über die kosten",  # German
        r"decisión sobre las costas",  # Spanish
        r"decisione sulle spese",  # Italian
        r"beslissing over de kosten",  # Dutch
        r"decisão sobre as custas",  # Portuguese
        r"decyzja w sprawie kosztów",  # Polish
        r"decizie privind cheltuielile",  # Romanian
        r"решение за разходите",  # Bulgarian
        r"rozhodnutí o nákladech",  # Czech
        r"otsus kuludest",  # Estonian
        r"päätös kuluista",  # Finnish
        r"απόφαση για τα έξοδα",  # Greek
    ]

    def can_parse(self, soup: bs) -> bool:
        """Check if this is the legacy EUR-Lex format."""

        p_tags = self._collect_grounds_p_tags(soup)
        return len(p_tags) > 1

    def _detect_numbering_pattern(self, tag: Tag) -> str:
        text = self._get_text(tag).strip()
        if re.match(r"^`?\s*\d+\.", text):
            return NumberingPattern.DOT
        elif re.match(r"^`?\s*\d+\)", text):
            return NumberingPattern.PARENTHESIS
        elif re.match(r"^`?\s*\d+", text):
            return NumberingPattern.SPACE
        return NumberingPattern.SPACE

    def _matches_pattern(self, text: str, pattern_type: str) -> bool:
        """Check if text matches the specified numbering pattern."""

        if pattern_type == NumberingPattern.DOT:
            return bool(re.match(r"^`?\s*\d+\.", text))
        elif pattern_type == NumberingPattern.SPACE:
            return bool(
                re.match(r"^`?\s*\d+", text)
                and not re.match(r"^`?\s*\d+\.", text)
                and not re.match(r"^`?\s*\d+\)", text)
            )
        elif pattern_type == NumberingPattern.PARENTHESIS:
            return bool(re.match(r"^`?\s*\d+\)", text))
        return False

    def _get_pattern_match(self, text: str, pattern_type: str) -> re.Match[str] | None:
        """Get the regex match object for the specified numbering pattern."""

        if pattern_type == NumberingPattern.DOT:
            return re.match(r"^`?\s*(\d+)\.", text)
        elif pattern_type == NumberingPattern.SPACE:
            match = re.match(r"^`?\s*(\d+)", text)
            if (
                match
                and not re.match(r"^`?\s*\d+\.", text)
                and not re.match(r"^`?\s*\d+\)", text)
            ):
                return match
            return None
        elif pattern_type == NumberingPattern.PARENTHESIS:
            return re.match(r"^`?\s*(\d+)\)", text)
        return None

    def _is_last_number_of_kind(
        self,
        current_index: int,
        proposed_num: int,
        p_tags: list[Tag],
        pattern: str,
    ) -> bool:
        """Check if the proposed number is the last occurrence of its kind in the remaining paragraphs."""
        # Look ahead in the remaining paragraphs to see if this number appears again
        for i in range(current_index + 1, len(p_tags)):
            future_text = self._get_text(p_tags[i]).strip()
            if not future_text:
                continue

            # Check if this future paragraph starts with the same number
            future_match = self._get_pattern_match(future_text, pattern)
            if future_match:
                future_num = int(future_match.group(1))
                if future_num == proposed_num:
                    return False  # Found another occurrence, so this is not the last

        return True  # No more occurrences found, this is the last

    def _has_next_plus_one(
        self,
        current_index: int,
        current_outer_counter: int,
        p_tags: list[Tag],
        pattern: str,
    ) -> bool:
        """Check if there's a next +1 paragraph in the remaining paragraphs."""
        for i in range(current_index + 1, len(p_tags)):
            future_text = self._get_text(p_tags[i]).strip()
            if not future_text:
                continue

            # Check if this future paragraph starts with the next +1 number
            future_match = self._get_pattern_match(future_text, pattern)
            if future_match:
                future_num = int(future_match.group(1))
                if future_num == current_outer_counter + 1:
                    return True  # Found a next +1

        return False  # No next +1 found

    def _collect_grounds_p_tags(self, soup: bs) -> list[Tag]:
        """Collect paragraphs from the Grounds section only."""
        grounds_h2_tag = None
        for pattern in self.GROUNDS_PATTERNS:
            h2_tags = soup.find_all("h2")
            for tag in h2_tags:
                if tag.string and re.match(pattern, tag.string, re.IGNORECASE):
                    grounds_h2_tag = tag
                    break
            if grounds_h2_tag:
                break

        if not grounds_h2_tag:
            return []

        em_tag = grounds_h2_tag.find_next_sibling("em")
        p_tags: list[Tag] = []
        if em_tag:
            p_tags = list(em_tag.find_all("p", recursive=False))
        else:
            sibling = grounds_h2_tag.find_next_sibling()
            while sibling:
                if isinstance(sibling, Tag):
                    # Stop at next major section heading
                    if sibling.name == "h2":
                        break
                    if sibling.name == "p":
                        p_tags.append(sibling)
                sibling = sibling.find_next_sibling()

        return p_tags

    def _collect_costs_p_tags(self, soup: bs) -> list[Tag]:
        """Collect paragraphs from the Decision on costs section only."""
        costs_h2_tag = None
        for pattern in self.COSTS_PATTERNS:
            h2_tags = soup.find_all("h2")
            for tag in h2_tags:
                if tag.string and re.match(pattern, tag.string, re.IGNORECASE):
                    costs_h2_tag = tag
                    break
            if costs_h2_tag:
                break

        if not costs_h2_tag:
            return []

        em_tag = costs_h2_tag.find_next_sibling("em")
        p_tags: list[Tag] = []
        if em_tag:
            p_tags = list(em_tag.find_all("p", recursive=False))
        else:
            sibling = costs_h2_tag.find_next_sibling()
            while sibling:
                if isinstance(sibling, Tag):
                    # Stop at next major section heading
                    if sibling.name == "h2":
                        break
                    if sibling.name == "p":
                        p_tags.append(sibling)
                sibling = sibling.find_next_sibling()

        return p_tags

    def _collect_p_tags(self, soup: bs) -> list[Tag]:
        """Collect paragraphs from both Grounds and Decision on costs sections."""
        all_p_tags: list[Tag] = []
        all_p_tags.extend(self._collect_grounds_p_tags(soup))
        all_p_tags.extend(self._collect_costs_p_tags(soup))
        return all_p_tags

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from the Grounds and Decision on costs sections."""
        # Process grounds section first
        grounds_paragraphs = self._extract_paragraphs_from_section(
            soup, self._collect_grounds_p_tags
        )

        # Process costs section
        costs_paragraphs = self._extract_paragraphs_from_section(
            soup, self._collect_costs_p_tags
        )

        # Combine both sections
        all_paragraphs = {}
        all_paragraphs.update(grounds_paragraphs)
        all_paragraphs.update(costs_paragraphs)

        return all_paragraphs

    def _extract_paragraphs_from_section(
        self, soup: bs, collect_method
    ) -> dict[int, str]:
        """Extract paragraphs from a specific section using the given collection method."""
        p_tags = collect_method(soup)

        if not p_tags:
            return {}

        paragraphs: dict[int, str] = {}
        outer_counter: int = 0
        inner_counter: int = 0
        mode: str = "outer"  # "outer" or "inner"
        inner_pattern: str | None = None
        outer_pattern: str | None = None

        for current_index, p_tag in enumerate(p_tags):
            text = self._get_text(p_tag).strip()
            if not text:
                continue

            # Check if this paragraph starts with a number
            match = re.match(r"^`?\s*(\d+)\s*[\.\)]?\s*(.+)", text)

            if match:
                proposed_num = int(match.group(1))
                proposed_text = match.group(2)

                if outer_pattern is None:
                    outer_pattern = self._detect_numbering_pattern(p_tag)

                if mode == "outer":

                    if outer_counter == 0:
                        outer_counter = proposed_num
                        paragraphs[outer_counter] = text

                        continue
                    elif proposed_num == outer_counter + 1:
                        # Check if this matches the outer pattern
                        if self._matches_pattern(text, outer_pattern):
                            outer_counter = proposed_num
                            paragraphs[outer_counter] = text
                        else:
                            # Doesn't match outer pattern, treat as inner
                            paragraphs[outer_counter] += " " + text
                            mode = "inner"
                            inner_counter = proposed_num

                            if inner_pattern is None:
                                inner_pattern = self._detect_numbering_pattern(p_tag)
                    elif proposed_num == outer_counter + 2:
                        # Only match +2 if there's no next +1
                        has_next_plus_one = self._has_next_plus_one(
                            current_index, outer_counter, p_tags, outer_pattern
                        )
                        if not has_next_plus_one:
                            # Check if this matches the outer pattern
                            if self._matches_pattern(text, outer_pattern):
                                outer_counter = proposed_num
                                paragraphs[outer_counter] = text
                            else:
                                # Doesn't match outer pattern, treat as inner
                                paragraphs[outer_counter] += " " + text
                                mode = "inner"
                                inner_counter = proposed_num

                                if inner_pattern is None:
                                    inner_pattern = self._detect_numbering_pattern(
                                        p_tag
                                    )
                        else:
                            # There's a next +1, so treat this as inner content
                            paragraphs[outer_counter] += " " + text
                            mode = "inner"
                            inner_counter = proposed_num

                            if inner_pattern is None:
                                inner_pattern = self._detect_numbering_pattern(p_tag)
                    else:
                        if inner_pattern is None:
                            inner_pattern = self._detect_numbering_pattern(p_tag)
                        # Check if this matches inner pattern
                        if self._matches_pattern(text, inner_pattern):
                            paragraphs[outer_counter] += " " + text
                            mode = "inner"
                            inner_counter = proposed_num
                        else:
                            # Doesn't match inner pattern either, add to current paragraph
                            paragraphs[outer_counter] += " " + text

                    if proposed_text.strip().endswith(":"):
                        mode = "inner"
                else:

                    could_be_inner = (
                        inner_pattern
                        and self._matches_pattern(text, inner_pattern)
                        and (proposed_num == inner_counter + 1)
                    )
                    could_be_outer = (
                        proposed_num == outer_counter + 1
                        or (
                            proposed_num == outer_counter + 2
                            and not self._has_next_plus_one(
                                current_index, outer_counter, p_tags, outer_pattern
                            )
                        )
                    ) and self._matches_pattern(text, outer_pattern)

                    if could_be_inner and could_be_outer:
                        # If end with : it should be outer
                        if proposed_text.strip().endswith(":"):
                            mode = "outer"
                            outer_counter = proposed_num
                            paragraphs[outer_counter] = text
                        elif self._is_last_number_of_kind(
                            current_index, proposed_num, p_tags, outer_pattern
                        ):
                            mode = "outer"
                            outer_counter = proposed_num
                            paragraphs[outer_counter] = text
                            if proposed_text.strip().endswith(":"):
                                mode = "inner"
                        else:
                            paragraphs[outer_counter] += " " + text
                            inner_counter = proposed_num
                    elif could_be_outer:
                        mode = "outer"
                        outer_counter = proposed_num
                        paragraphs[outer_counter] = text
                        if proposed_text.strip().endswith(":"):
                            mode = "inner"
                    else:
                        if inner_pattern is None:
                            inner_pattern = self._detect_numbering_pattern(p_tag)
                        paragraphs[outer_counter] += " " + text
                        inner_counter = proposed_num

            elif outer_counter in paragraphs:
                # Non-numbered paragraph - add to current paragraph
                paragraphs[outer_counter] += " " + text

        return paragraphs


class LegacySingleParagraphParser(BaseJudgementParser):
    """Parser for legacy EUR-Lex format where grounds are in one big paragraph."""

    def can_parse(self, soup: bs) -> bool:
        """Check if this is the legacy single paragraph format."""
        # Look for the Grounds section with anchor tag
        grounds_section = soup.find("a", attrs={"name": "MO"})
        if not grounds_section:
            return False

        # Find the h2 tag that follows the anchor
        h2_tag = grounds_section.find_next_sibling("h2")
        if not h2_tag:
            return False

        em_tag = h2_tag.find_next_sibling("em")
        if not em_tag:
            return False

        # Check if there's only one paragraph in the em tag
        p_tags = em_tag.find_all("p", recursive=False)
        if len(p_tags) != 1:
            return False

        # Check if the single paragraph contains numbered content
        single_p = p_tags[0]
        text = self._get_text(single_p).strip()

        # Look for patterns like "1 BY AN ORDER" or "2 THOSE QUESTIONS"
        # Also check if there are multiple numbered paragraphs in the text
        if re.match(r"^\s*\d+\s+[A-Z]", text):
            # Count how many numbered paragraphs are in this single text
            numbered_matches = re.findall(r"\d+\s+[A-Z]", text)
            if len(numbered_matches) > 1:
                return True

        return False

    def extract_paragraphs(self, soup: bs) -> dict[int, str]:
        """Extract paragraphs from the single grounds paragraph."""
        # Find the Grounds section
        grounds_section = soup.find("a", attrs={"name": "MO"})
        if not grounds_section:
            return {}

        h2_tag = grounds_section.find_next_sibling("h2")
        if not h2_tag:
            return {}

        em_tag = h2_tag.find_next_sibling("em")
        if not em_tag:
            return {}

        # Get the single paragraph
        p_tags = em_tag.find_all("p", recursive=False)
        if len(p_tags) != 1:
            return {}

        single_paragraph = p_tags[0]
        full_text = self._get_text(single_paragraph)

        # Split the text into numbered paragraphs
        paragraphs = self._split_into_numbered_paragraphs(full_text)

        return paragraphs

    def _split_into_numbered_paragraphs(self, text: str) -> dict[int, str]:
        """Split the text into numbered paragraphs using a simple counter approach."""
        paragraphs: list[str] = []

        # Find all numbers in the text
        number_matches = list(re.finditer(r" \d+ ", text))

        expected_paragraph_num = 2

        last_pos = 0

        for match in number_matches:
            found_num = int(match.group())

            # Check if this is the next expected paragraph number
            if found_num == expected_paragraph_num:
                start_pos = match.start()

                paragraph_text = text[last_pos:start_pos].strip()

                # Clean up the text
                paragraph_text = re.sub(r"\s+", " ", paragraph_text).strip()

                paragraphs.append(paragraph_text)

                expected_paragraph_num += 1
                last_pos = start_pos

        paragraphs.append(text[last_pos:].strip())

        return {i + 1: p for i, p in enumerate(paragraphs)}
