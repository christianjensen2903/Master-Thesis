"""
Preprocessing module for GNN training and evaluation.

This module provides tools to precompute embeddings and build flexible graphs
from legal documents and judgments.
"""

from preprocessing.precompute_embeddings import EmbeddingPreprocessor
from preprocessing.graph_builder import (
    BaseGraphBuilder,
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
)

__all__ = [
    "EmbeddingPreprocessor",
    "BaseGraphBuilder",
    "HomogeneousGraphBuilder",
    "HeterogeneousGraphBuilder",
]
