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

        # ECLI and ECR patterns - defined early for use in case patterns
        _ecli_pattern = r"EU:(?:C|T|F):\d{4}:\d+"
        # ECR references can be: "ECR I-123", "ECR 123", or year in brackets "[1989]"
        # Use DASH pattern to handle all dash variations
        _ecr_ref = rf"ECR\s+(?:[IVX]+{DASH}?\d{{1,5}}|\d{{1,5}})"
        _ecr_year_bracket = r"\[(?:19|20)\d{2}\]"
        # Match years in parens (including double parens with spaces)
        _year_in_parens = r"\(+\s*(?:19|20)\d{2}\s*\)+"
        # Optional ECR can be: year_in_parens + ECR, or [YEAR] + ECR, or just ECR/ECLI
        _optional_ecr = rf"(?:\s*(?:{_year_in_parens}\s*)?(?:{_ecr_year_bracket}\s+)?(?:{_ecr_ref}|{_ecli_pattern}))?"

        # Sub-patterns
        _case_joined_all = (
            rf"\bJoined Cases?\s+(?:{COURT}{CASE_NUMBER}"
            rf"(?:\s*(?:,|and|&)\s*{COURT}{CASE_NUMBER})+)(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ\s\-'']+)?"
            rf"(?=\s*[,\.;\)\[\]]|\s*$|$)"
        )
        _case_joined_trailing_year = (
            rf"\bJoined Cases?\s+(?:{COURT}\d+(?:\s*(?:,|and|&)\s*{COURT}\d+)+\s*/\s*{YEAR})"
            rf"(?=\s*[,\.;\)]|\s*$|$)"
        )
        _case_joined_range = (
            rf"\bJoined Cases?\s+(?:{COURT}\d+\s*(?:{DASH}|to|through)\s*{COURT}\d+/\s*{YEAR})"
            rf"(?=\s*[,\.;\)]|\s*$|$)"
        )

        # Case patterns with optional ECR/ECLI at the end to avoid double matching
        _case_single_with_v = rf"Case\s+{COURT}{CASE_NUMBER}\s+[A-Za-zÀ-ÖØ-öø-ÿ\s\-'']+(?:v\.?|contre)\s+[A-Za-zÀ-ÖØ-öø-ÿ\s\-'']+{_optional_ecr}(?=\s*[,\.;\)\(\[]|\s*$|$)"
        # Match case with names: require actual capital letter, then match names + optional ECR
        # Use greedy matching but stop before paragraph/comma
        _case_single_with_names = rf"Case\s+{COURT}{CASE_NUMBER}\s+(?=(?-i:[A-ZÀ-ÖØ-Þ]))(?:[A-Za-zÀ-ÖØ-öø-ÿ\-'']+\s+)*(?:[A-Za-zÀ-ÖØ-öø-ÿ\-'']+)\s*{_optional_ecr}(?=\s*[,\.;]|\s+para(?:graph)?|\s*$|$)"
        _case_single = rf"Case\s+{COURT}{CASE_NUMBER}{_optional_ecr}(?=\s*[,\.;\)]|\s+[a-z]|\s*$|$)"

        _case_bare = rf"(?<!Case\s)(?<!case\s)(?<!Regulation\sNo\s)(?<!Directive\s)(?<!Directive\s\[)(?<!Law\s)(?<!of\s)(?<!\[)(?<!\w){COURT}{CASE_NUMBER}(?!\w)(?!\])"

        # Standalone ECR/ECLI patterns (for cases not matched by case patterns above)
        _ecr = rf"\b{_ecr_ref}\b"
        _ecr_year_brackets = _ecr_year_bracket
        _ecli = rf"\b{_ecli_pattern}\b"

        _para = (
            r"\bpara(?:graph)?s?\.?\s+\d+"
            r"(?:\s*(?:[-\u2013\u2014\u2212]|to|through)\s*\d+)?"
            r"(?:\s*,\s*\d+)*"
            r"(?:\s*(?:and|&)\s*\d+)?\b"
        )

        # French legal patterns - improved to handle articles
        _affaire_with_v = rf"\b(?:l'|les?\s+)?(?:affaire|affaires)\s+(?:jointes?\s+)?{COURT}{CASE_NUMBER}\s+[A-Za-zÀ-ÖØ-öø-ÿ\s]+(?:v\.?|contre)\s+[A-Za-zÀ-ÖØ-öø-ÿ\s]+(?=\s*[,\.;\)]|\s*$|$)"
        _affaire_with_names = rf"\b(?:l'|les?\s+)?(?:affaire|affaires)\s+(?:jointes?\s+)?{COURT}{CASE_NUMBER}\s+(?:[A-Z][a-z]+\s+(?:and\s+|et\s+)?[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+|\s+et\s+[A-Z][a-z]+)*)(?=\s*[,\.;\)]|\s*$|$)"
        _affaire = rf"\b(?:l'|les?\s+)?(?:affaire|affaires)\s+(?:jointes?\s+)?{COURT}{CASE_NUMBER}(?=\s*[,\.;\)]|\s+[a-z]|\s*$|$)"
        _affaires_jointes = rf"\b(?:les?\s+)?affaires\s+jointes?\s+(?:{COURT}{CASE_NUMBER}(?:\s*(?:,|et|&)\s*{COURT}{CASE_NUMBER})+)(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ\s]+)?(?=\s*[,\.;\)]|\s*$|$)"
        _arret = rf"\b(?:l'|les?\s+)?(?:arrêt|arrêts)\s+(?:de\s+la\s+)?(?:Cour|Tribunal|Conseil)\s+(?:du\s+\d{{1,2}}\s+\w+\s+\d{{4}})?"
        _paragraphe = (
            r"\b(?:l'|les?\s+)?(?:para|paragraphe|point)s?\.?\s+\d+"
            r"(?:\s*(?:[-\u2013\u2014\u2212]|à|jusqu'?à)\s*\d+)?"
            r"(?:\s*,\s*\d+)*"
            r"(?:\s*(?:et|&)\s*\d+)?\b"
        )
        _alinea = (
            r"\b(?:l'|les?\s+)?(?:alinéa|al\.)\s+\d+"
            r"(?:\s*(?:[-\u2013\u2014\u2212]|à|jusqu'?à)\s*\d+)?"
            r"(?:\s*,\s*\d+)*"
            r"(?:\s*(?:et|&)\s*\d+)?\b"
        )

        # Party-v-party (EU style) case titles without numbers
        # Make NAME pattern more restrictive - must start with TOKEN (not connector)
        # Use negative lookahead to prevent matching common words as party names
        # Extended to include Latin Extended-A (U+0100-U+017F) and Latin Extended-B (U+0180-U+024F)
        _TOKEN = r"[A-ZÀ-ÖØ-Þ\u0100-\u024F][A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024F0-9'''-]*"
        _PARTY_CONNECTOR = r"(?:and|&|Others|et)"
        # Negative lookahead to prevent matching common words that aren't party names
        _NOT_COMMON_WORD = r"(?!(?:and|or|in|of|the|to|for|with|by|at|from|as|on|that|this|it|is|was|be|has|have|had|but|not|are|were|been|being|also|see|judgments?|judgment|into|than|over|after|before|between|under)\b)"
        _NAME = (
            rf"{_NOT_COMMON_WORD}{_TOKEN}(?:\s+(?:{_TOKEN}|{_PARTY_CONNECTOR})){{0,5}}"
        )
        # For SHORT names in comma-separated contexts, be even more restrictive
        _NAME_SHORT = rf"{_NOT_COMMON_WORD}{_TOKEN}(?:\s+{_TOKEN}){{0,2}}"
        _END_BOUNDARY = r"(?=\s*[,\.;\)]|\s*$|$)"
        _party_v_party = rf"\b{_NAME}\s+(?:v\.?|contre)\s+{_NAME}{_END_BOUNDARY}"

        # Party names in parentheses with case numbers or before ECLI
        # Party name with v pattern followed by parentheses with case number/ECLI
        # Must match full name including multi-word names
        _party_v_in_parens = rf"\b{_NAME}\s+(?:v\.?|contre)\s+{_NAME}\s+\({COURT}{CASE_NUMBER}\s*,?\s*(?:{_ecli_pattern})?\s*\)"

        # Party name in parentheses with case number/ECLI
        _party_in_parens = (
            rf"\({_NAME}\s+{COURT}{CASE_NUMBER}\s*,?\s*(?:{_ecli_pattern})?\s*\)"
        )

        # Party name followed by comma and ECLI/case - use full NAME pattern for better matching
        _party_before_ecli = rf"(?:^|\b|(?<=\s))(?!judgments?\s)(?!see\s)(?!in\s){_NAME}\s*,\s*{_ecli_pattern}"
        _party_before_case = rf"(?:^|\b|(?<=\s))(?!judgments?\s)(?!see\s)(?!voir\s)(?!arrêt\s)(?!arrêts\s){_NAME},\s*{COURT}{CASE_NUMBER}"

        # Party names with slash separator (Commission/Grèce, Commission/Italie) followed by case number
        _party_slash_party = rf"\b[A-ZÀ-ÖØ-Þ\u0100-\u024F][A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024F]+\s*/\s*[A-ZÀ-ÖØ-Þ\u0100-\u024F][A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024F]+\s*,\s*{COURT}{CASE_NUMBER}"

        self._MASTER = re.compile(
            rf"(?P<CASE>{_case_joined_all}|{_affaires_jointes}|{_case_joined_trailing_year}|{_case_joined_range}|{_case_single_with_v}|{_case_single_with_names}|{_case_single}|{_affaire_with_v}|{_affaire_with_names}|{_affaire}|{_party_v_in_parens}|{_party_v_party}|{_case_bare}|{_arret}|{_party_in_parens}|{_party_slash_party}|{_party_before_case}|{_party_before_ecli})"
            rf"|(?P<ECR>{_ecr}|{_ecr_year_brackets})"
            rf"|(?P<ECLI>{_ecli})"
            rf"|(?P<PARAGRAPH>{_para}|{_paragraphe}|{_alinea})",
            flags=re.IGNORECASE,
        )

    def _setup_date_patterns(self) -> None:
        """Setup regex patterns for dates."""
        months = (
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
            r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
            r"janv(?:ier)?|fév(?:rier)?|mars|avr(?:il)?|mai|"
            r"juin|juil(?:let)?|août|sept(?:embre)?|"
            r"oct(?:obre)?|nov(?:embre)?|déc(?:embre)?)"
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
            # Standalone year (but NOT after /, No, or inside brackets)
            r"(?<![/\[\d])(?<!No\s)\b(?:19|20)\d{2}\b(?![\]/)])",
        ]

        # Compile with IGNORECASE to match 'march'/'MARCH' etc.
        self.date_patterns = [re.compile(p, re.IGNORECASE) for p in date_patterns_raw]

    def _repl_citation(self, m: re.Match) -> str:
        """Replacement function for citation patterns."""
        g = m.lastgroup
        if g == "CASE":
            # Check if the match starts with "In " and preserve it
            matched_text = m.group(0)
            if matched_text.startswith("In "):
                return "In <CASE>"
            return "<CASE>"
        if g == "ECR":
            return "<CASE>"
        if g == "ECLI":
            return "<CASE>"
        return "<PARAGRAPH>"

    def remove_paragraph_numbers(self, text: str) -> str:
        """Remove paragraph numbers from the beginning of text."""
        if not isinstance(text, str):
            return text

        # Preserve trailing spaces
        trailing = ""
        if text.endswith(" "):
            stripped = text.rstrip()
            trailing = text[len(stripped) :]

        # Remove paragraph numbers at the beginning, but be more careful about what follows
        # Don't remove if it's part of a larger number like "19 of [Regulation"
        if re.match(r"^\s*\d+\s+of\s+\[", text):
            return text.strip() + trailing
        return re.sub(r"^\s*\d+[\.\)]?\s*", "", text).strip() + trailing

    def remove_citations(self, text: str) -> str:
        """Remove legal citations from text."""
        if not isinstance(text, str):
            return text
        return self._MASTER.sub(self._repl_citation, text).strip()

    def remove_dates(self, text: str) -> str:
        """Remove dates from text."""
        if not isinstance(text, str):
            return text

        # Protect specific contexts where dates should be preserved
        protected = []

        # Protect standalone OJ references like "(OJ 2015 L 105, p. 1)"
        oj_pattern = r"\(OJ\s+\d{4}[^)]+\)"

        def protect_oj(m):
            protected.append(m.group(0))
            return f"__PROTECTED_{len(protected)-1}__"

        text = re.sub(oj_pattern, protect_oj, text)

        # Now remove dates
        for pattern in self.date_patterns:
            text = re.sub(pattern, "<DATE>", text)

        # Restore protected text
        for i, orig in enumerate(protected):
            text = text.replace(f"__PROTECTED_{i}__", orig)

        return text.strip()

    def normalize_whitespace(self, text: str) -> str:

        # Replace all whitespace characters (spaces, tabs, newlines, etc.) with single space
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def clean_spacing(self, text: str) -> str:
        """Clean up spacing around punctuation."""
        if not isinstance(text, str):
            return text

        # Remove extra spaces around punctuation
        text = re.sub(r"\s+([,.])", r"\1", text)
        text = re.sub(r"([,.])\s+", r"\1 ", text)
        # Remove spaces before closing parentheses
        text = re.sub(r"\s+\)", ")", text)
        # Remove spaces after opening parentheses
        text = re.sub(r"\(\s+", "(", text)

        # Handle quoted text: remove spaces INSIDE quotes but preserve spaces OUTSIDE
        # Match pattern: (space) ' (spaces) TEXT (spaces) ' (spaces)
        # Keep: (space) 'TEXT' (spaces)
        def clean_quote_spacing(m):
            before_space = m.group(1) if m.group(1) else ""
            content = m.group(2).strip()
            after_space = m.group(3) if m.group(3) else ""
            return f"{before_space}'{content}'{after_space}"

        text = re.sub(r"(\s)?'\s*([^']+?)\s*'(\s)?", clean_quote_spacing, text)
        # Collapse multiple spaces to 2 maximum
        text = re.sub(r" {3,}", "  ", text)

        return text.strip()

    def remove_duplicate_tags(self, text: str) -> str:
        """Remove duplicate consecutive tags."""
        if not isinstance(text, str):
            return text

        # Track if we had duplicate CASE tags
        had_duplicates = "<CASE>, <CASE>" in text or "<CASE> <CASE>" in text

        # Remove duplicate consecutive <CASE> tags (multiple passes to handle nested cases)
        # This handles <CASE><CASE>, <CASE> <CASE>, <CASE>, <CASE> etc.
        while True:
            orig = text
            text = re.sub(r"<CASE>\s*<CASE>", "<CASE>", text)
            text = re.sub(r"<CASE>,\s*<CASE>", "<CASE>", text)
            text = re.sub(r"<CASE>\s*\(\s*<CASE>\s*\)", "<CASE>", text)
            text = re.sub(r"<CASE>\s*\[\s*<CASE>\s*\]", "<CASE>", text)
            # Handle <CASE> followed by date/year in parens and another <CASE>
            text = re.sub(r"<CASE>\s*\(+\s*<DATE>\s*\)+\s*<CASE>", "<CASE>", text)
            # Handle <DATE> inside <CASE> patterns more generally
            text = re.sub(r"<CASE>\s*<DATE>\s*<CASE>", "<CASE>", text)
            if text == orig:
                break

        # If we had duplicate CASE tags and collapsed them, also remove connector words
        # between citation tags, as they were connecting the duplicates
        # E.g., "<CASE>, <PARAGRAPH>, et <CASE>" -> "<CASE>, <PARAGRAPH>, <CASE>"
        if had_duplicates:
            text = re.sub(
                r"<CASE>,\s+<PARAGRAPH>,\s+(?:and|et|&)\s+<CASE>",
                "<CASE>, <PARAGRAPH>, <CASE>",
                text,
            )

        return text.strip()

    def fix_false_positives(self, text: str) -> str:
        """Fix false positive matches in Law/Regulation/Directive numbers."""
        if not isinstance(text, str):
            return text
        # Fix duplicate date tags: ((<DATE>)) -> <DATE>
        text = re.sub(r"\(\(<DATE>\)\)", "<DATE>", text)
        # Fix duplicate date tags: <DATE> <CASE> -> <CASE>
        text = re.sub(r"<DATE>\s+<CASE>", "<CASE>", text)
        # Fix duplicate case tags: <CASE> <CASE> -> <CASE>
        text = re.sub(r"<CASE>\s+<CASE>", "<CASE>", text)
        return text.strip()

    def clean_text(
        self,
        text: str,
        remove_paragraph_numbers: bool = True,
        remove_citations: bool = True,
        remove_dates: bool = True,
    ) -> str:
        """
        Clean text using all available cleaning methods.

        Args:
            text: The text to clean
            remove_paragraph_numbers: Whether to remove paragraph numbers
            remove_citations: Whether to remove legal citations
            remove_dates: Whether to remove dates

        Returns:
            Cleaned text
        """

        # Normalize whitespace first
        text = self.normalize_whitespace(text)

        # Apply cleaning steps in order
        if remove_paragraph_numbers:
            text = self.remove_paragraph_numbers(text)

        if remove_citations:
            text = self.remove_citations(text)

        if remove_dates:
            text = self.remove_dates(text)

        text = self.clean_spacing(text)

        # Remove duplicate tags
        text = self.remove_duplicate_tags(text)

        # Fix false positives
        text = self.fix_false_positives(text)

        return text.strip()


# Example usage
if __name__ == "__main__":
    # Example text with various elements to clean (including extra whitespace)
    sample_text = """
    25 In that   connection it must be observed that,    as the Court held in its judgment of 30 October 1974 in Case 188/73 Grassi v Council, paragraph 38, the appointing authority has a wide discretion in the matter of recruitment.
    
    The Court has already held,		in its judgment in Case 31/80 L'Oréal v De Nieuwe AMCK, paragraph 25, that Article 11(2) of Regulation No 17/62.
    
    See Commission v Technische Glaswerke Ilmenau, paragraph 50; Sweden and Others v API and Commission, paragraph 9.
    """

    cleaner = TextCleaner()

    print("Original text:")
    print(sample_text)
    print("\n" + "=" * 80 + "\n")

    print("Cleaned text:")
    cleaned = cleaner.clean_text(sample_text)
    print(cleaned)
