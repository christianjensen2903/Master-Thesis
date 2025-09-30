from __future__ import annotations

"""Reusable text preprocessing utilities for retrievers.

This module provides composable functions to build preprocessing pipelines,
including lowercasing, punctuation removal, stopword filtering, and stemming.

All functions have explicit type hints and PEP 257 compliant docstrings.
"""

from typing import Callable, Iterable
import re


def lowercase() -> Callable[[str], str]:
    """Return a function that lowercases input text.

    Returns
    -------
    Callable[[str], str]
        Function that transforms text to lowercase.
    """

    def _fn(text: str) -> str:
        return text.lower()

    return _fn


def remove_punctuation(pattern: str | None = None) -> Callable[[str], str]:
    """Return a function that removes punctuation using a regex.

    Parameters
    ----------
    pattern
        Regular expression pattern to match characters to remove. If None,
        defaults to removing all non-word and non-space characters.

    Returns
    -------
    Callable[[str], str]
        Function that strips punctuation from text.
    """

    regex = re.compile(pattern or r"[^\w\s]+", flags=re.UNICODE)

    def _fn(text: str) -> str:
        return regex.sub(" ", text)

    return _fn


def regex_sub(pattern: str, repl: str = " ") -> Callable[[str], str]:
    """Return a function that applies a regex substitution.

    Parameters
    ----------
    pattern
        Pattern to match.
    repl
        Replacement string.

    Returns
    -------
    Callable[[str], str]
        Function that replaces pattern occurrences with `repl`.
    """

    regex = re.compile(pattern, flags=re.UNICODE)

    def _fn(text: str) -> str:
        return regex.sub(repl, text)

    return _fn


def stopword_filter(stopwords: Iterable[str]) -> Callable[[str], str]:
    """Return a function that removes stopwords from whitespace-tokenized text.

    Parameters
    ----------
    stopwords
        Collection of stopwords to remove (case-sensitive to input text).

    Returns
    -------
    Callable[[str], str]
        Function that filters stopwords and rejoins tokens with single spaces.
    """

    stop_set = set(stopwords)

    def _fn(text: str) -> str:
        tokens = text.split()
        kept = [t for t in tokens if t not in stop_set]
        return " " + " ".join(kept) if kept else ""

    return _fn


def porter_stemmer() -> Callable[[str], str]:
    """Return a function that applies NLTK's PorterStemmer to tokens.

    Notes
    -----
    Requires `nltk` to be installed. The function lazily imports NLTK.

    Returns
    -------
    Callable[[str], str]
        Function that stems whitespace-tokenized tokens.
    """

    from nltk.stem import PorterStemmer  # type: ignore

    stemmer = PorterStemmer()

    def _fn(text: str) -> str:
        tokens = text.split()
        return " " + " ".join(stemmer.stem(t) for t in tokens) if tokens else ""

    return _fn


def snowball_stemmer(language: str = "english") -> Callable[[str], str]:
    """Return a function that applies SnowballStemmer to tokens.

    Parameters
    ----------
    language
        Language for the stemmer (e.g., "english", "danish").

    Returns
    -------
    Callable[[str], str]
        Function that stems whitespace-tokenized tokens.
    """

    from nltk.stem import SnowballStemmer  # type: ignore

    stemmer = SnowballStemmer(language)

    def _fn(text: str) -> str:
        tokens = text.split()
        return " " + " ".join(stemmer.stem(t) for t in tokens) if tokens else ""

    return _fn


def compose(*funcs: Callable[[str], str]) -> Callable[[str], str]:
    """Compose multiple preprocessing steps into a single callable.

    Steps are applied in the provided order.

    Parameters
    ----------
    *funcs
        One or more functions of the form `(str) -> str`.

    Returns
    -------
    Callable[[str], str]
        A function that applies each step sequentially.
    """

    def _fn(text: str) -> str:
        result = text
        for f in funcs:
            result = f(result)
        return result

    return _fn
