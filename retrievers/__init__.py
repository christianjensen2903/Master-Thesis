"""Retrievers package providing base and concrete implementations.

This package exposes a simple, typed interface for retrieval models and
includes TF-IDF and BM25 implementations optimized for batch querying.
"""

from .base import BaseRetriever  # type: ignore
from .tfidf import TFIDFRetriever  # type: ignore
from .bm25 import BM25Retriever  # type: ignore
from . import preprocess as preprocess_utils  # type: ignore

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever",
    "BM25Retriever",
    "preprocess_utils",
]
