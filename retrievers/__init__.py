"""Retrievers package providing base and concrete implementations.

This package exposes a simple, typed interface for retrieval models and
includes TF-IDF and BM25 implementations optimized for batch querying.
"""

from .base import BaseRetriever
from .tfidf import TFIDFRetriever
from .bm25 import BM25Retriever
from . import preprocess as preprocess_utils
from .sentence_bert import SentenceBERTRetriever

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever",
    "BM25Retriever",
    "SentenceBERTRetriever",
    "preprocess_utils",
]
