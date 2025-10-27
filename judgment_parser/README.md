# Judgment Parser Module

This module provides a flexible parser for extracting structured paragraphs from CJEU (Court of Justice of the European Union) judgment HTML files in different formats.

## Architecture

- **JudgementParser**: Main parser that routes to format-specific parsers
- **BaseJudgementParser**: Abstract base class for format-specific parsers
- **LegacyEurLexParser**: Handles older EUR-Lex format (1950s-1990s)
- **ModernJudgementParser**: Handles modern CJEU format with CSS classes (2000s+)
- **Specialized parsers**: Handle various other formats (dt/dd, nested docs, etc.)

## Supported Formats

1. **Legacy EUR-Lex**: Simple HTML with `<h2>` sections and numbered `<p>` tags
2. **Modern CJEU**: Structured HTML with CSS classes (C01PointAltN, C01PointnumeroteAltN)
3. **DtDd format**: Cases using `<dt>` and `<dd>` tags for numbered paragraphs
4. **Nested documents**: Documents with nested HTML content
5. **Reports for hearing**: Table-based format with coj-count and coj-normal classes
6. **Operative parts**: Pages containing only operative parts rendered via tables

## Usage

```python
from judgment_parser import JudgementParser

parser = JudgementParser()
paragraphs = parser.extract_paragraphs("path/to/case.html")
# Returns: {1: "text...", 2: "text...", ...}
```

## Testing

Run the test suite:

```bash
pytest judgment_parser/test_judgment_parser.py -v
```

Add a new test case:

```bash
python -m judgment_parser.add_test_case <celex_id>
```

## Module Structure

```
judgment_parser/
├── __init__.py          # Module exports
├── base.py              # Base classes and utilities
├── modern.py            # Modern CJEU format parser
├── legacy.py            # Legacy EUR-Lex parsers
├── specialized.py       # Specialized format parsers
├── main.py              # Main parser and routing logic
├── test_judgment_parser.py  # Test suite
├── add_test_case.py     # Script to add test cases
├── test_cases.json      # Test case data
└── README.md            # This file
```
