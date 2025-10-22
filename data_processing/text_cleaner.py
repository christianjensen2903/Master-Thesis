import re


class TextCleaner:
    """A comprehensive text cleaner for legal documents based on the cleaning logic from clean_data.ipynb."""

    def __init__(self):
        self._setup_citation_patterns()
        self._setup_date_patterns()

    def _setup_citation_patterns(self) -> None:
        """Setup regex patterns for legal citations."""
        # Dashes commonly seen in legal cites: -, ‐, -, ‒, –, —, −, (optional soft hyphen)
        DASH = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212\u00AD]"

        COURT = rf"(?:[CTF]\s*{DASH}?\s*)?"
        YEAR = r"\d{2,4}"
        CASE_NUMBER = rf"\d+/{YEAR}(?:\s*[A-Z]{{1,4}})?"  # 347/88, 131/12 P, 404/15 PPU

        # Sub-patterns
        _case_joined_all = (
            rf"\bJoined Cases?\s+(?:{COURT}{CASE_NUMBER}"
            rf"(?:\s*(?:,|and|&)\s*{COURT}{CASE_NUMBER})+)"
            r"[^.,;)]*"
        )
        _case_joined_trailing_year = (
            rf"\bJoined Cases?\s+(?:{COURT}\d+(?:\s*(?:,|and|&)\s*{COURT}\d+)+\s*/\s*{YEAR})"
            r"[^.,;)]*"
        )
        _case_joined_range = (
            rf"\bJoined Cases?\s+(?:{COURT}\d+\s*(?:{DASH}|to|through)\s*{COURT}\d+/\s*{YEAR})"
            r"[^.,;)]*"
        )
        _case_single = rf"\bCase\s+{COURT}{CASE_NUMBER}[^.,;)]*"

        _case_bare = rf"(?<!\w){COURT}{CASE_NUMBER}(?!\w)[^.,;)]*"

        _ecr = r"\bECR\s+(?:[IVX]+[-\u2013\u2014\u2212]?\d{1,5}|\d{1,5})\b"

        _ecli = r"\bEU:(?:C|T|F):\d{4}:\d+\b"

        _para = (
            r"\bpara(?:graph)?s?\.?\s+\d+"
            r"(?:\s*(?:[-\u2013\u2014\u2212]|to|through)\s*\d+)?"
            r"(?:\s*,\s*\d+)*"
            r"(?:\s*(?:and|&)\s*\d+)?\b"
        )

        # Party-v-party (EU style) case titles without numbers
        _TOKEN = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ0-9'''-]*"
        _CONNECTOR = r"(?:and|&|Others|of|the)"
        _NAME = rf"{_TOKEN}(?:\s+(?:{_TOKEN}|{_CONNECTOR})){{0,7}}"
        _END_BOUNDARY = r"(?=(?:\s*[,\.;\)])|(?:\s*$)|$)"
        _party_v_party = rf"\b{_NAME}\s+v\.?\s+{_NAME}{_END_BOUNDARY}"

        self._MASTER = re.compile(
            rf"(?P<CASE>{_party_v_party}|{_case_joined_all}|{_case_joined_trailing_year}|{_case_joined_range}|{_case_single}|{_case_bare})"
            rf"|(?P<ECR>{_ecr})"
            rf"|(?P<ECLI>{_ecli})"
            rf"|(?P<PARAGRAPH>{_para})",
            flags=re.IGNORECASE,
        )

    def _setup_date_patterns(self) -> None:
        """Setup regex patterns for dates."""
        months = (
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
            r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        )

        date_patterns_raw = [
            # Numeric dates (DD/MM/YYYY, MM-DD-YYYY, YYYY/MM/DD, etc.)
            r"\b\d{1,2}[./\-\s]\d{1,2}[./\-\s](?:19|20)\d{2}\b",  # 12/05/2020, 12-05-2020, 12.05.2020
            r"\b(?:19|20)\d{2}[./\-\s]\d{1,2}[./\-\s]\d{1,2}\b",  # 2020-05-12, 2020/5/12
            # Month name + day + year (comma usually present; allow optional)
            rf"\b{months}\s+\d{{1,2}}(?:st|nd|rd|th)?\,?\s+(?:19|20)\d{{2}}\b",  # March 3, 2021 / March 3 2021
            # Day + month name + year (comma optional)
            rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{months}\,?\s*(?:19|20)\d{{2}}\b",  # 3 March 2021 / 3rd March, 2021
            # Year + month name + day
            rf"\b(?:19|20)\d{{2}}\s+{months}\s+\d{{1,2}}(?:st|nd|rd|th)?\b",  # 2021 March 3
            # Standalone year
            r"\b(?:19|20)\d{2}\b",
        ]

        # Compile with IGNORECASE to match 'march'/'MARCH' etc.
        self.date_patterns = [re.compile(p, re.IGNORECASE) for p in date_patterns_raw]

    def _repl_citation(self, m: re.Match) -> str:
        """Replacement function for citation patterns."""
        g = m.lastgroup
        if g == "CASE":
            return "<CASE>"
        if g == "ECR":
            return "<ECR>"
        if g == "ECLI":
            return "<ECLI>"
        return "<PARAGRAPH>"

    def remove_paragraph_numbers(self, text: str) -> str:
        """Remove paragraph numbers from the beginning of text."""
        if not isinstance(text, str):
            return text
        return re.sub(r"^\s*\d+[\.\)]?\s*", "", text).strip()

    def remove_citations(self, text: str) -> str:
        """Remove legal citations from text."""
        if not isinstance(text, str):
            return text
        return self._MASTER.sub(self._repl_citation, text).strip()

    def remove_dates(self, text: str) -> str:
        """Remove dates from text."""
        if not isinstance(text, str):
            return text
        for pattern in self.date_patterns:
            text = re.sub(pattern, "<DATE>", text)
        return text.strip()

    def mask_quoted_text(self, text: str, reference_text: str | None = None) -> str:
        """Mask quoted text that appears in both the current text and reference text."""
        if not isinstance(text, str):
            return text

        if reference_text is None:
            return text

        # Find all quoted spans in text
        quotes = re.findall(r'"[^"]*"', text)
        for q in quotes:
            if q in reference_text:
                text = text.replace(q, "<QUOTED_TEXT>")
        return text

    def clean_text(
        self,
        text: str,
        reference_text: str | None = None,
        remove_paragraph_numbers: bool = True,
        remove_citations: bool = True,
        remove_dates: bool = True,
        mask_quotes: bool = True,
        min_length: int = 0,
    ) -> str:
        """
        Clean text using all available cleaning methods.

        Args:
            text: The text to clean
            reference_text: Optional reference text for quote masking
            remove_paragraph_numbers: Whether to remove paragraph numbers
            remove_citations: Whether to remove legal citations
            remove_dates: Whether to remove dates
            mask_quotes: Whether to mask quoted text
            min_length: Minimum length for text to be kept (0 = no minimum)

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return text

        # Apply cleaning steps in order
        if remove_paragraph_numbers:
            text = self.remove_paragraph_numbers(text)

        if mask_quotes and reference_text:
            text = self.mask_quoted_text(text, reference_text)

        if remove_citations:
            text = self.remove_citations(text)

        if remove_dates:
            text = self.remove_dates(text)

        # Remove very short texts
        if min_length > 0 and len(text.strip()) < min_length:
            return ""

        return text.strip()

    def clean_pair(self, text_from: str, text_to: str, **kwargs) -> tuple[str, str]:
        """
        Clean a pair of texts, applying quote masking between them.

        Args:
            text_from: First text
            text_to: Second text
            **kwargs: Additional arguments passed to clean_text

        Returns:
            Tuple of (cleaned_text_from, cleaned_text_to)
        """
        # Clean text_from with text_to as reference for quote masking
        cleaned_from = self.clean_text(text_from, reference_text=text_to, **kwargs)

        # Clean text_to with text_from as reference for quote masking
        cleaned_to = self.clean_text(text_to, reference_text=text_from, **kwargs)

        return cleaned_from, cleaned_to


# Example usage
if __name__ == "__main__":
    # Example text with various elements to clean
    sample_text = """
    25 In that connection it must be observed that, as the Court held in its judgment of 30 October 1974 in Case 188/73 Grassi v Council, paragraph 38, the appointing authority has a wide discretion in the matter of recruitment.
    
    The Court has already held, in its judgment in Case 31/80 L'Oréal v De Nieuwe AMCK, paragraph 25, that Article 11(2) of Regulation No 17/62.
    
    See Commission v Technische Glaswerke Ilmenau, paragraph 50; Sweden and Others v API and Commission, paragraph 9.
    """

    cleaner = TextCleaner()

    print("Original text:")
    print(sample_text)
    print("\n" + "=" * 80 + "\n")

    print("Cleaned text:")
    cleaned = cleaner.clean_text(sample_text)
    print(cleaned)
