"""
Judgment Parser Module

This module provides a flexible parser for extracting structured paragraphs from
CJEU (Court of Justice of the European Union) judgment HTML files in different formats.

Architecture:
- JudgementParser: Main parser that routes to format-specific parsers
- BaseJudgementParser: Abstract base class for format-specific parsers
- LegacyEurLexParser: Handles older EUR-Lex format (1950s-1990s)
- ModernJudgementParser: Handles modern CJEU format with CSS classes (2000s+)
- Specialized parsers: Handle various other formats (dt/dd, nested docs, etc.)

Supported Formats:
1. Legacy EUR-Lex: Simple HTML with <h2> sections and numbered <p> tags
2. Modern CJEU: Structured HTML with CSS classes (C01PointAltN, C01PointnumeroteAltN)
3. DtDd format: Cases using <dt> and <dd> tags for numbered paragraphs
4. Nested documents: Documents with nested HTML content
5. Reports for hearing: Table-based format with coj-count and coj-normal classes
6. Operative parts: Pages containing only operative parts rendered via tables

Usage:
    from judgment_parser import JudgementParser

    parser = JudgementParser()
    paragraphs = parser.extract_paragraphs("path/to/case.html")
    # Returns: {1: "text...", 2: "text...", ...}
"""

from .main import JudgementParser
from .base import BaseJudgementParser, NumberingPattern
from .modern import ModernJudgementParser
from .legacy import LegacyEurLexParser, LegacySingleParagraphParser
from .specialized import (
    DtDdParser,
    NestedDocumentParser,
    ReportForHearingParser,
    OperativePartParser,
)

__all__ = [
    "JudgementParser",
    "BaseJudgementParser",
    "NumberingPattern",
    "ModernJudgementParser",
    "LegacyEurLexParser",
    "LegacySingleParagraphParser",
    "DtDdParser",
    "NestedDocumentParser",
    "ReportForHearingParser",
    "OperativePartParser",
]
