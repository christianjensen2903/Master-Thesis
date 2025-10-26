import numpy as np
from datetime import datetime


def build_temporal_dag(
    citation_graph: dict[int, list[int]],
    paragraph_dates: np.ndarray,
) -> dict[int, list[int]]:
    """
    Build a temporal DAG from citation graph by enforcing older → newer edges only.

    This ensures causality: a paragraph can only cite older paragraphs,
    and can only be influenced by older paragraphs through message passing.

    Args:
        citation_graph: Full citation graph {src_pid: [tgt_pids]}
        paragraph_dates: Array of dates for each paragraph

    Returns:
        Temporal DAG with only edges from older to newer paragraphs

    Example:
        Para A (2015) ──cites──> Para X (2010) ✓ (older cited)
        Para B (2020) ──cites──> Para A (2015) ✓ (older cited)
        Para X (2010) ──cites──> Para A (2015) ✗ (removed: newer cited)
    """
    temporal_dag: dict[int, list[int]] = {}

    for src_pid, target_pids in citation_graph.items():
        src_date = paragraph_dates[src_pid]

        # Only keep edges to OLDER paragraphs (or same date)
        valid_targets = []
        for tgt_pid in target_pids:
            tgt_date = paragraph_dates[tgt_pid]

            # Source must be newer than or equal to target
            if src_date >= tgt_date:
                valid_targets.append(tgt_pid)

        if valid_targets:
            temporal_dag[src_pid] = valid_targets

    return temporal_dag


def build_temporal_subgraph(
    temporal_dag: dict[int, list[int]],
    query_pid: int,
    paragraph_dates: np.ndarray,
) -> dict[int, list[int]]:
    """
    Build subgraph containing only paragraphs older than the query paragraph.

    This is used during inference to ensure the model can only "see" the past.

    Args:
        temporal_dag: Full temporal DAG
        query_pid: Query paragraph ID
        paragraph_dates: Array of dates for each paragraph

    Returns:
        Subgraph containing only edges between paragraphs older than query

    Example:
        Query: Para Q (2020)
        Subgraph includes:
          - Para A (2015) ──> Para X (2010) ✓
          - Para B (2018) ──> Para A (2015) ✓
        Excludes:
          - Para Q (2020) ──> anything ✗ (query has no outgoing)
          - Para Z (2021) ──> anything ✗ (future paragraph)
    """
    query_date = paragraph_dates[query_pid]

    subgraph: dict[int, list[int]] = {}

    for src_pid, target_pids in temporal_dag.items():
        src_date = paragraph_dates[src_pid]

        # Only include edges where source is older than query
        # (or is the query itself, but query won't have outgoing edges in temporal_dag)
        if src_date < query_date:
            # Further filter targets to only those older than query
            valid_targets = [
                tgt_pid
                for tgt_pid in target_pids
                if paragraph_dates[tgt_pid] < query_date
            ]

            if valid_targets:
                subgraph[src_pid] = valid_targets

    return subgraph


def get_temporal_candidates(
    query_pid: int,
    paragraph_dates: np.ndarray,
) -> np.ndarray:
    """
    Get all candidate paragraphs that are strictly older than the query.

    Args:
        query_pid: Query paragraph ID
        paragraph_dates: Array of dates for each paragraph

    Returns:
        Array of paragraph IDs that are older than query
    """
    query_date = paragraph_dates[query_pid]

    # Find all paragraphs with earlier dates
    older_mask = paragraph_dates < query_date
    candidate_pids = np.where(older_mask)[0]

    return candidate_pids


def validate_temporal_dag(
    temporal_dag: dict[int, list[int]],
    paragraph_dates: np.ndarray,
) -> bool:
    """
    Validate that a graph is a proper temporal DAG.

    Checks:
    1. All edges go from newer/equal to older paragraphs
    2. No cycles exist (DAG property)

    Args:
        temporal_dag: Graph to validate
        paragraph_dates: Array of dates for each paragraph

    Returns:
        True if valid temporal DAG, False otherwise
    """
    # Check temporal ordering
    for src_pid, target_pids in temporal_dag.items():
        src_date = paragraph_dates[src_pid]

        for tgt_pid in target_pids:
            tgt_date = paragraph_dates[tgt_pid]

            # Source must be newer than or equal to target
            if src_date < tgt_date:
                print(
                    f"❌ Temporal violation: {src_pid} ({src_date}) → {tgt_pid} ({tgt_date})"
                )
                print(f"   Source is older than target!")
                return False

    # Check for cycles (simple DFS-based cycle detection)
    visited = set()
    rec_stack = set()

    def has_cycle(node: int) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in temporal_dag.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                print(f"❌ Cycle detected involving node {node} → {neighbor}")
                return True

        rec_stack.remove(node)
        return False

    for node in temporal_dag.keys():
        if node not in visited:
            if has_cycle(node):
                return False

    print("✓ Valid temporal DAG")
    return True


def print_temporal_graph_stats(
    temporal_dag: dict[int, list[int]],
    paragraph_dates: np.ndarray,
    paragraph_set: np.ndarray | None = None,
) -> None:
    """Print statistics about the temporal graph."""
    num_nodes = len(paragraph_dates)
    num_edges = sum(len(targets) for targets in temporal_dag.values())
    num_source_nodes = len(temporal_dag)

    print("\n" + "=" * 60)
    print("Temporal DAG Statistics")
    print("=" * 60)
    print(f"Total nodes: {num_nodes}")
    print(f"Nodes with outgoing edges: {num_source_nodes}")
    print(f"Total edges: {num_edges}")
    print(f"Average out-degree: {num_edges / num_source_nodes:.2f}")

    # Date range
    valid_dates = paragraph_dates[~np.isnat(paragraph_dates)]
    if len(valid_dates) > 0:
        min_date = np.min(valid_dates)
        max_date = np.max(valid_dates)
        print(f"Date range: {min_date} to {max_date}")

    # Breakdown by set
    if paragraph_set is not None:
        train_mask = paragraph_set == "train"
        test_mask = paragraph_set == "test"

        train_edges = sum(
            len(targets)
            for src, targets in temporal_dag.items()
            if src < len(train_mask) and train_mask[src]
        )
        test_edges = sum(
            len(targets)
            for src, targets in temporal_dag.items()
            if src < len(test_mask) and test_mask[src]
        )

        print(f"\nTrain nodes: {np.sum(train_mask)}")
        print(f"Train edges: {train_edges}")
        print(f"Test nodes: {np.sum(test_mask)}")
        print(f"Test edges: {test_edges}")

    print("=" * 60 + "\n")
