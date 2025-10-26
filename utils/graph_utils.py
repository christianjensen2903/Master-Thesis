"""Graph utility functions for citation graph manipulation."""

import numpy as np


def filter_graph_to_train(
    citation_graph: dict[int, list[int]],
    paragraph_set: np.ndarray,
) -> dict[int, list[int]]:
    """
    Filter citation graph to only include edges from training paragraphs.

    This prevents data leakage by ensuring that test paragraph citation
    patterns are not visible during model fitting or inference.

    Args:
        citation_graph: Dictionary mapping source paragraph ID to list of cited paragraph IDs
        paragraph_set: Array indicating "train"/"test" for each paragraph

    Returns:
        Filtered citation graph with only training edges

    Example:
        >>> citation_graph = {0: [1, 2], 1: [2], 3: [1]}
        >>> paragraph_set = np.array(["train", "train", "test", "test"])
        >>> filtered = filter_graph_to_train(citation_graph, paragraph_set)
        >>> # Result: {0: [1, 2], 1: [2]} - excludes edge from paragraph 3 (test)
    """
    return {
        src_pid: cited_pids
        for src_pid, cited_pids in citation_graph.items()
        if src_pid < len(paragraph_set) and paragraph_set[src_pid] == "train"
    }


def count_edges(citation_graph: dict[int, list[int]]) -> int:
    """
    Count total number of edges in a citation graph.

    Args:
        citation_graph: Dictionary mapping source to list of targets

    Returns:
        Total number of edges
    """
    return sum(len(cited_pids) for cited_pids in citation_graph.values())


def validate_no_test_edges(
    citation_graph: dict[int, list[int]],
    paragraph_set: np.ndarray,
) -> None:
    """
    Validate that citation graph contains no edges from test paragraphs.

    Raises ValueError if test edges are found (data leakage detected).

    Args:
        citation_graph: Citation graph to validate
        paragraph_set: Array indicating "train"/"test" for each paragraph

    Raises:
        ValueError: If edges from test paragraphs are found
    """
    test_edges = []
    for src_pid in citation_graph:
        if src_pid < len(paragraph_set) and paragraph_set[src_pid] == "test":
            test_edges.append(src_pid)

    if test_edges:
        raise ValueError(
            f"⚠️  DATA LEAKAGE DETECTED: Found {len(test_edges)} edges from test paragraphs! "
            f"First few test sources: {test_edges[:5]}"
        )

    print(f"✓ Validation passed: No test edges found in citation graph")
