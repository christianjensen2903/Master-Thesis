import glob
import os
import os.path
import random
import re
import sys

from bs4 import BeautifulSoup as bs
from bs4 import Tag


class JudgementParser:
    """Parser for extracting structured paragraphs from CJEU judgment HTML files."""

    # CSS classes that indicate the end of judgment content
    STOP_CLASSES = [
        "C04Titre1",
        "S40Titre",
        "C41DispositifIntroduction",
        "C30Dispositifalinea",
        "C77Signatures",
        "C42FootnoteLangue",
    ]

    # CSS class for numbered paragraphs (post-2010 format)
    NUMBERED_PARAGRAPH_CLASS = "C01PointnumeroteAltN"

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
        soup = self._load_html(path)
        if not soup:
            return {}

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
            if self._has_class(current, self.NUMBERED_PARAGRAPH_CLASS):
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
            if self._has_class(sibling, self.NUMBERED_PARAGRAPH_CLASS):
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
        for p in paragraphs:
            # Look for "Judgment" or "Arrêt" markers
            if p.text.strip() not in ("Judgment", "Arrêt"):
                continue

            # Find the next non-empty paragraph after the marker
            next_p = p.find_next_sibling("p")
            while next_p and not next_p.text.strip():
                next_p = next_p.find_next_sibling("p")

            if next_p:
                return next_p

        # If no marker found, return first paragraph as fallback
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

    def _get_text(self, tag: Tag) -> str:
        """Extract cleaned text from a tag."""
        return " ".join(tag.stripped_strings)


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
