"""
Base classes and utilities for judgment parsers.

This module provides the abstract base class for all judgment parsers
and common utility functions used across different parser implementations.
"""

import re
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


class NumberingPattern:
    """Enum-like class for numbering patterns."""

    DOT = "dot"
    SPACE = "space"
    PARENTHESIS = "parenthesis"
