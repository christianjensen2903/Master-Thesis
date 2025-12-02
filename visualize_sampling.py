import os
import torch
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
from matplotlib.patches import FancyBboxPatch
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.data import Data, HeteroData  # type: ignore
from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
)
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected  # type: ignore


class SamplingVisualizer:
    def __init__(self, preprocessed_dir: str, graph_type: str = "heterogeneous"):
        self.preprocessed_dir = preprocessed_dir
        self.graph_type = graph_type
        self.device = torch.device("cpu")  # Use CPU for visualization

    def load_graph(self, train_cutoff_year=None):
        """Load the graph data and compute nodes with positives."""
        if self.graph_type == "heterogeneous":
            builder = HeterogeneousGraphBuilder(self.preprocessed_dir)
            graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
            ).to(self.device)

            # Find nodes with positive examples (same as trainer)
            cite_edge_index = graph_data["paragraph", "cites", "paragraph"].edge_index
            nodes_with_positives = cite_edge_index[0].unique()

            print(f"Heterogeneous graph loaded:")
            print(f"  Total paragraph nodes: {graph_data['paragraph'].num_nodes}")
            print(f"  Nodes with citations: {len(nodes_with_positives)}")
            print(
                f"  Percentage with positives: {len(nodes_with_positives) / graph_data['paragraph'].num_nodes * 100:.1f}%"
            )

            return graph_data, True, nodes_with_positives
        else:
            builder = HomogeneousGraphBuilder(self.preprocessed_dir)
            graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
            ).to(self.device)

            # Find nodes with positive examples (same as trainer)
            edge_index = graph_data.edge_index
            nodes_with_positives = edge_index[0].unique()

            print(f"Homogeneous graph loaded:")
            print(f"  Total nodes: {graph_data.num_nodes}")
            print(f"  Nodes with citations: {len(nodes_with_positives)}")
            print(
                f"  Percentage with positives: {len(nodes_with_positives) / graph_data.num_nodes * 100:.1f}%"
            )

            graph_data = ToUndirected()(graph_data)

            return graph_data, False, nodes_with_positives

    def process_batch(self, batch, is_hetero, nodes_with_positives=None):
        """Process a batch similar to the trainer."""
        if is_hetero:
            batch_size = batch["paragraph"].batch_size

            if ("paragraph", "cites", "paragraph") in batch.edge_types:
                cite_edge_index = batch["paragraph", "cites", "paragraph"].edge_index
            else:
                return None

            # Check which anchor nodes have positives
            anchor_node_ids = None
            if hasattr(batch["paragraph"], "n_id"):
                anchor_node_ids = batch["paragraph"].n_id[:batch_size]

            # Extract time information
            anchor_times = None
            all_times = None
            if hasattr(batch["paragraph"], "time"):
                anchor_times = batch["paragraph"].time[:batch_size]
                all_times = batch["paragraph"].time

            # Get case structure for masking
            if ("paragraph", "belongs_to", "case") in batch.edge_types:
                par_to_case = batch["paragraph", "belongs_to", "case"].edge_index
                case_to_par = batch["case", "contains", "paragraph"].edge_index

                anchor_mask = par_to_case[0] < batch_size
                anchor_cases = par_to_case[1, anchor_mask].unique()

                case_mask = torch.isin(case_to_par[0], anchor_cases)
                paragraphs_in_anchor_cases = case_to_par[1, case_mask].unique()
            else:
                paragraphs_in_anchor_cases = torch.arange(batch_size)

            # Identify masked edges
            cite_src, cite_tgt = cite_edge_index
            leakage_mask = torch.isin(
                cite_src, paragraphs_in_anchor_cases
            ) | torch.isin(cite_tgt, paragraphs_in_anchor_cases)
            masked_edges = cite_edge_index[:, leakage_mask]
            kept_edges = cite_edge_index[:, ~leakage_mask]

            return {
                "batch_size": batch_size,
                "cite_edge_index": cite_edge_index,
                "masked_edges": masked_edges,
                "kept_edges": kept_edges,
                "anchor_times": anchor_times,
                "all_times": all_times,
                "paragraphs_in_anchor_cases": paragraphs_in_anchor_cases,
                "total_nodes": batch["paragraph"].x.shape[0],
                "anchor_node_ids": anchor_node_ids,
                "nodes_with_positives": nodes_with_positives,
            }
        else:
            batch_size = batch.batch_size
            edge_index = batch.edge_index

            # Check which anchor nodes have positives
            anchor_node_ids = None
            if hasattr(batch, "n_id"):
                anchor_node_ids = batch.n_id[:batch_size]

            # Extract time information
            anchor_times = None
            all_times = None
            if hasattr(batch, "time"):
                anchor_times = batch.time[:batch_size]
                all_times = batch.time

            # Identify masked edges
            src, tgt = edge_index
            leakage_mask = (src < batch_size) | (tgt < batch_size)
            masked_edges = edge_index[:, leakage_mask]
            kept_edges = edge_index[:, ~leakage_mask]

            return {
                "batch_size": batch_size,
                "cite_edge_index": edge_index,
                "masked_edges": masked_edges,
                "kept_edges": kept_edges,
                "anchor_times": anchor_times,
                "all_times": all_times,
                "total_nodes": batch.x.shape[0],
                "anchor_node_ids": anchor_node_ids,
                "nodes_with_positives": nodes_with_positives,
            }

    def compute_temporal_mask(self, anchor_times, positive_times):
        """Compute which positives are valid as negatives based on time."""
        if anchor_times is None or positive_times is None:
            return None

        # positive_j is valid negative for anchor_i if positive_time_j < anchor_time_i
        time_mask = positive_times.unsqueeze(0) < anchor_times.unsqueeze(1)
        return time_mask

    def visualize_batch(self, batch_data, output_path="sampling_vis.png"):
        """Create a comprehensive visualization of the sampling behavior."""
        if batch_data is None:
            print("No valid batch data to visualize")
            return

        batch_size = batch_data["batch_size"]
        total_nodes = batch_data["total_nodes"]
        cite_edges = batch_data["cite_edge_index"]
        masked_edges = batch_data["masked_edges"]
        kept_edges = batch_data["kept_edges"]

        # Find positive pairs (edges where source is in batch)
        src, tgt = cite_edges
        positive_mask = src < batch_size
        positive_src = src[positive_mask]
        positive_tgt = tgt[positive_mask]

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        # gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Full graph structure
        # ax1 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(1, 1, 1)
        self._plot_graph_structure(
            ax1,
            batch_size,
            total_nodes,
            cite_edges,
            masked_edges,
            kept_edges,
            batch_data,
        )

        # 2. Positive pairs
        # ax2 = fig.add_subplot(gs[1, 0])
        # self._plot_positive_pairs(ax2, positive_src, positive_tgt, batch_size)

        # # 3. Temporal filtering (if available)
        # ax3 = fig.add_subplot(gs[1, 1])
        # if batch_data["anchor_times"] is not None:
        #     self._plot_temporal_filtering(
        #         ax3,
        #         positive_src,
        #         positive_tgt,
        #         batch_data["anchor_times"],
        #         batch_data["all_times"],
        #     )
        # else:
        #     ax3.text(0.5, 0.5, "No temporal info", ha="center", va="center")
        #     ax3.axis("off")

        # # 4. Contrastive pairs matrix
        # ax4 = fig.add_subplot(gs[1, 2])
        # self._plot_contrastive_matrix(
        #     ax4,
        #     positive_src,
        #     positive_tgt,
        #     batch_data["anchor_times"],
        #     batch_data["all_times"],
        # )

        # 5. Statistics
        # ax5 = fig.add_subplot(gs[2, :])
        # self._plot_statistics(ax5, batch_data, positive_src, positive_tgt)

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")
        plt.close()

    def _plot_graph_structure(
        self,
        ax,
        batch_size,
        total_nodes,
        cite_edges,
        masked_edges,
        kept_edges,
        batch_data,
    ):
        """Plot the graph structure showing anchor nodes and edges as a DAG."""
        ax.set_title(
            "Graph Structure (DAG): Anchor Nodes and Edge Masking",
            fontsize=14,
            fontweight="bold",
        )

        # Limit nodes for visualization
        max_nodes = total_nodes  # min(total_nodes, 80)
        anchor_nodes = list(range(batch_size))
        neighbor_nodes = list(range(batch_size, max_nodes))
        all_viz_nodes = anchor_nodes + neighbor_nodes

        # Build NetworkX graph for layout computation
        G = nx.DiGraph()
        G.add_nodes_from(all_viz_nodes)

        # Add all edges (we'll color them differently later)
        for i in range(cite_edges.shape[1]):
            src, tgt = cite_edges[0, i].item(), cite_edges[1, i].item()
            if src < max_nodes and tgt < max_nodes:
                G.add_edge(src, tgt)

        # Use timestamp-based layout
        # Sort nodes by their timestamps

        all_times = batch_data.get("all_times")
        if all_times is None:
            raise ValueError("No timestamps available")

        # Get timestamps for visualization nodes
        node_times = {}
        for node in all_viz_nodes:
            if node < len(all_times):
                node_times[node] = all_times[node].item()
            else:
                # Fallback if node index is out of range
                node_times[node] = 0

        # Normalize timestamps to [0, 1] for y-position
        times = [node_times[n] for n in all_viz_nodes]
        if times:
            min_time = min(times)
            max_time = max(times)
            time_range = max_time - min_time if max_time > min_time else 1
        else:
            time_range = 1
            min_time = 0

        # Group nodes by timestamp (with some tolerance for floating point)
        time_groups = {}
        for node in all_viz_nodes:
            time = node_times[node]
            # Round to nearest 0.01 to group similar timestamps
            time_key = (
                round((time - min_time) / time_range * 100) if time_range > 0 else 0
            )
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(node)

        # Create positions
        # Better horizontal spreading: force-directed within each time layer
        positions = {}

        for time_key, nodes in sorted(time_groups.items()):
            y = 1.0 - (time_key / 100.0)

            # Subgraph: only edges among these nodes
            subG = nx.DiGraph()
            subG.add_nodes_from(nodes)
            for src, tgt in G.edges():
                if src in nodes and tgt in nodes:
                    subG.add_edge(src, tgt)

            if len(nodes) == 1:
                # Only one node → center it
                positions[nodes[0]] = (0.5, y)
                continue

            # Compute horizontal positions using force-layout in 2D
            try:
                # Request 2D positions (NetworkX internals expect 2D).
                sub_pos = nx.spring_layout(
                    subG,
                    dim=2,  # <- use 2D to avoid IndexError
                    k=0.4,  # preferred spacing (tweak if needed)
                    iterations=50,
                    seed=42,
                )
                # Extract x-coordinates (index 0) for horizontal placement
                xs = [sub_pos[n][0] for n in nodes]
            except Exception:
                # Fallback: small random jitter around an even spread
                base = np.linspace(0.1, 0.9, num=len(nodes))
                jitter = (np.random.RandomState(42).rand(len(nodes)) - 0.5) * 0.08
                xs = list(base + jitter)

            # Normalize x coordinates into [0.05, 0.95]
            xmin, xmax = min(xs), max(xs)
            scale = (xmax - xmin) if xmax > xmin else 1.0

            for idx, n in enumerate(nodes):
                # if xs came from spring_layout (float) or fallback array
                x_val = xs[idx] if isinstance(xs, (list, np.ndarray)) else xs[idx]
                x = 0.05 + 0.9 * ((x_val - xmin) / scale)
                positions[n] = (x, y)

        # Create edge sets for coloring
        masked_edge_set = set()
        if masked_edges.shape[1] > 0:
            for i in range(masked_edges.shape[1]):
                src, tgt = masked_edges[0, i].item(), masked_edges[1, i].item()
                if src < max_nodes and tgt < max_nodes:
                    masked_edge_set.add((src, tgt))

        kept_edge_set = set()
        if kept_edges.shape[1] > 0:
            for i in range(kept_edges.shape[1]):
                src, tgt = kept_edges[0, i].item(), kept_edges[1, i].item()
                if src < max_nodes and tgt < max_nodes:
                    kept_edge_set.add((src, tgt))

        # Draw masked edges (red, dashed) - draw first so they're in back
        for src, tgt in masked_edge_set:
            if src in positions and tgt in positions:
                ax.annotate(
                    "",
                    xy=positions[tgt],
                    xytext=positions[src],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="red",
                        alpha=0.3,
                        linestyle="--",
                        linewidth=0.8,
                        shrinkA=5,
                        shrinkB=5,
                    ),
                    zorder=1,
                )

        # Draw kept edges (green) - draw on top
        for src, tgt in kept_edge_set:
            if src in positions and tgt in positions:
                ax.annotate(
                    "",
                    xy=positions[tgt],
                    xytext=positions[src],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="green",
                        alpha=0.6,
                        linewidth=1.2,
                        shrinkA=5,
                        shrinkB=5,
                    ),
                    zorder=2,
                )

        # Draw nodes on top of edges
        # Draw neighbor nodes first
        for node in neighbor_nodes:
            if node in positions:
                circle = plt.Circle(
                    positions[node],
                    0.015,
                    color="lightblue",
                    alpha=0.8,
                    zorder=3,
                    edgecolor="darkblue",
                    linewidth=0.5,
                )
                ax.add_patch(circle)

        # Draw anchor nodes on top
        for node in anchor_nodes:
            if node in positions:
                circle = plt.Circle(
                    positions[node],
                    0.02,
                    color="red",
                    alpha=0.8,
                    zorder=4,
                    edgecolor="darkred",
                    linewidth=0.8,
                )
                ax.add_patch(circle)

        # Add node labels for anchor nodes (optional, can be removed if too cluttered)
        if batch_size <= 20:
            for node in anchor_nodes:
                if node in positions:
                    ax.text(
                        positions[node][0],
                        positions[node][1],
                        str(node),
                        fontsize=6,
                        ha="center",
                        va="center",
                        color="white",
                        weight="bold",
                        zorder=5,
                    )

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label=f"Anchor Nodes (n={batch_size})",
                markeredgecolor="darkred",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lightblue",
                markersize=8,
                label=f"Neighbor Nodes (n={len(neighbor_nodes)})",
                markeredgecolor="darkblue",
            ),
            Line2D(
                [0],
                [0],
                color="g",
                linewidth=2,
                label=f"Kept Edges (n={kept_edges.shape[1]})",
            ),
            Line2D(
                [0],
                [0],
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Masked Edges (n={masked_edges.shape[1]})",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        # Add note about DAG structure
        note_text = "DAG Layout: Citations flow top → bottom\n(older papers at top, newer at bottom)"

        # Add note about filtering if applicable
        if batch_data.get("nodes_with_positives") is not None:
            note_text += (
                "\n\n✓ Anchors sampled from filtered set\n(only nodes with ≥1 citation)"
            )

        ax.text(
            0.02,
            0.02,
            note_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")

    def _plot_positive_pairs(self, ax, positive_src, positive_tgt, batch_size):
        """Plot positive pairs (anchor -> cited document)."""
        ax.set_title("Positive Pairs (Citation Edges)", fontsize=12, fontweight="bold")

        if len(positive_src) == 0:
            ax.text(0.5, 0.5, "No positive pairs", ha="center", va="center")
            ax.axis("off")
            return

        # Count pairs per anchor
        unique_anchors, counts = torch.unique(positive_src, return_counts=True)

        # Bar plot
        ax.barh(
            unique_anchors.cpu().numpy(), counts.cpu().numpy(), color="green", alpha=0.7
        )
        ax.set_xlabel("Number of Positive Pairs")
        ax.set_ylabel("Anchor Node ID")
        ax.set_title(
            f"Positive Pairs per Anchor (Total: {len(positive_src)})", fontsize=10
        )
        ax.grid(axis="x", alpha=0.3)

    def _plot_temporal_filtering(
        self, ax, positive_src, positive_tgt, anchor_times, all_times
    ):
        """Visualize temporal filtering of negatives."""
        ax.set_title("Temporal Filtering", fontsize=12, fontweight="bold")

        if anchor_times is None or all_times is None:
            ax.text(0.5, 0.5, "No temporal info", ha="center", va="center")
            ax.axis("off")
            return

        # Get times for positive pairs
        pair_anchor_times = anchor_times[positive_src]
        pair_positive_times = all_times[positive_tgt]

        # Plot scatter: anchor time vs positive time
        ax.scatter(
            pair_anchor_times.cpu().numpy(),
            pair_positive_times.cpu().numpy(),
            alpha=0.6,
            s=50,
        )

        # Add diagonal line (anchor_time = positive_time)
        time_min = min(anchor_times.min().item(), pair_positive_times.min().item())
        time_max = max(anchor_times.max().item(), pair_positive_times.max().item())
        ax.plot([time_min, time_max], [time_min, time_max], "r--", label="Equal Time")

        ax.set_xlabel("Anchor Time")
        ax.set_ylabel("Positive Time")
        ax.legend()
        ax.grid(alpha=0.3)

        # Add text about temporal constraint
        valid_negatives = pair_positive_times < pair_anchor_times
        ax.text(
            0.05,
            0.95,
            f"Valid as negatives: {valid_negatives.sum().item()}/{len(valid_negatives)}\n"
            f"(positives must be before anchors)",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=8,
        )

    def _plot_contrastive_matrix(
        self, ax, positive_src, positive_tgt, anchor_times, all_times
    ):
        """Plot the contrastive similarity matrix structure."""
        ax.set_title("Contrastive Matrix Structure", fontsize=12, fontweight="bold")

        if len(positive_src) == 0:
            ax.text(0.5, 0.5, "No positive pairs", ha="center", va="center")
            ax.axis("off")
            return

        # Limit to first 50 pairs for visualization
        n_pairs = min(50, len(positive_src))
        src_sample = positive_src[:n_pairs]
        tgt_sample = positive_tgt[:n_pairs]

        # Create matrix showing structure
        matrix = np.zeros((n_pairs, n_pairs))

        # Mark diagonal (positive pairs)
        np.fill_diagonal(matrix, 2)

        # Mark same-anchor pairs (should be masked)
        for i in range(n_pairs):
            for j in range(n_pairs):
                if i != j and src_sample[i] == src_sample[j]:
                    matrix[i, j] = 1

        # Apply temporal filtering if available
        if anchor_times is not None and all_times is not None:
            pair_anchor_times = anchor_times[src_sample]
            pair_positive_times = all_times[tgt_sample]

            for i in range(n_pairs):
                for j in range(n_pairs):
                    # If not diagonal and not same-anchor
                    if matrix[i, j] == 0:
                        # Check temporal constraint
                        if pair_positive_times[j] >= pair_anchor_times[i]:
                            matrix[i, j] = -1  # Invalid negative

        # Plot matrix
        cmap = plt.cm.colors.ListedColormap(["white", "yellow", "green", "gray"])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

        # Add colorbar
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="white", label="Valid Negative"),
            Patch(facecolor="gray", label="Invalid (Temporal)"),
            Patch(facecolor="yellow", label="Same Anchor"),
            Patch(facecolor="green", label="Positive (Diagonal)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )

        ax.set_xlabel("Positive Index")
        ax.set_ylabel("Anchor Index")
        ax.set_title(f"Matrix Structure (first {n_pairs} pairs)", fontsize=10)

    def _plot_statistics(self, ax, batch_data, positive_src, positive_tgt):
        """Plot key statistics about the batch."""
        ax.axis("off")

        # Check if anchors are from filtered set
        anchor_info = ""
        if (
            batch_data.get("nodes_with_positives") is not None
            and batch_data.get("anchor_node_ids") is not None
        ):
            nodes_with_positives = batch_data["nodes_with_positives"]
            anchor_node_ids = batch_data["anchor_node_ids"]

            # Check which anchors are in the filtered set
            anchors_in_filtered = (
                torch.isin(anchor_node_ids, nodes_with_positives).sum().item()
            )

            anchor_info = f"""
        Anchor Node Filtering:
        • Anchors sampled from filtered set: {anchors_in_filtered}/{len(anchor_node_ids)}
        • All anchors guaranteed to have ≥1 positive: {'✓ Yes' if anchors_in_filtered == len(anchor_node_ids) else '✗ No'}
        • Total nodes with positives in graph: {len(nodes_with_positives)}
            """

        stats_text = f"""
        BATCH STATISTICS
        {'=' * 60}
        
        Graph Structure:
        • Total nodes in batch: {batch_data['total_nodes']}
        • Anchor nodes (batch_size): {batch_data['batch_size']}
        • Neighbor nodes: {batch_data['total_nodes'] - batch_data['batch_size']}
        {anchor_info}
        Edges:
        • Total citation edges: {batch_data['cite_edge_index'].shape[1]}
        • Masked edges (leakage prevention): {batch_data['masked_edges'].shape[1]}
        • Kept edges (for GNN): {batch_data['kept_edges'].shape[1]}
        • Masking ratio: {batch_data['masked_edges'].shape[1] / batch_data['cite_edge_index'].shape[1] * 100:.1f}%
        
        Positive Pairs:
        • Total positive pairs: {len(positive_src)}
        • Unique anchors with positives: {len(torch.unique(positive_src))}
        • Avg positives per anchor: {len(positive_src) / len(torch.unique(positive_src)):.2f}
        • Max positives for one anchor: {torch.bincount(positive_src).max().item() if len(positive_src) > 0 else 0}
        • Anchors without positives: {batch_data['batch_size'] - len(torch.unique(positive_src))}
        
        Contrastive Learning:
        • In-batch negatives per anchor: {len(positive_src) - 1} (before filtering)
        • Same-anchor pairs masked: Yes
        • Temporal filtering: {'Yes' if batch_data['anchor_times'] is not None else 'No'}
        """

        if batch_data["anchor_times"] is not None and len(positive_src) > 0:
            pair_anchor_times = batch_data["anchor_times"][positive_src]
            pair_positive_times = batch_data["all_times"][positive_tgt]
            valid_as_negatives = (
                pair_positive_times.unsqueeze(0) < pair_anchor_times.unsqueeze(1)
            ).sum(dim=1)

            stats_text += f"""
        Temporal Statistics:
        • Avg valid negatives per anchor: {valid_as_negatives.float().mean():.1f}
        • Min valid negatives: {valid_as_negatives.min().item()}
        • Max valid negatives: {valid_as_negatives.max().item()}
            """

        # Add warning if some anchors don't have positives
        if batch_data["batch_size"] - len(torch.unique(positive_src)) > 0:
            stats_text += f"""
        
        ⚠️  WARNING: {batch_data['batch_size'] - len(torch.unique(positive_src))} anchor nodes have NO positive pairs!
        These nodes will be skipped during training.
        Consider filtering input_nodes to only include nodes with positives.
            """

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

    def visualize_multiple_batches(
        self,
        graph_data,
        is_hetero,
        nodes_with_positives,
        num_batches=3,
        batch_size=16,
        num_hops=2,
        output_dir="sampling_visualizations",
    ):
        """Visualize multiple batches to check consistency."""
        os.makedirs(output_dir, exist_ok=True)

        # Create data loader with filtered input nodes (same as trainer)
        num_neighbors = [-1] * (num_hops + 1) if num_hops > 0 else [-1]

        if is_hetero:
            input_nodes = ("paragraph", nodes_with_positives)
        else:
            input_nodes = nodes_with_positives

        loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=input_nodes,
            shuffle=True,
            time_attr=(
                "time"
                if (
                    hasattr(graph_data["paragraph"], "time")
                    if is_hetero
                    else hasattr(graph_data, "time")
                )
                else None
            ),
            subgraph_type="bidirectional",
        )

        print(f"\nVisualizing {num_batches} batches...")
        print(
            f"Sampling only from nodes with positives (n={len(nodes_with_positives)})..."
        )

        for i, batch in enumerate(loader):
            if i >= num_batches:
                break

            print(f"\nProcessing batch {i + 1}/{num_batches}...")
            batch_data = self.process_batch(batch, is_hetero, nodes_with_positives)

            if batch_data is not None:
                output_path = os.path.join(output_dir, f"batch_{i + 1}.png")
                self.visualize_batch(batch_data, output_path)
            else:
                print(f"  Skipping batch {i + 1} (no valid data)")

        print(f"\nAll visualizations saved to {output_dir}/")


def main():
    """Main function to run visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize GNN sampling behavior")
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        required=True,
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="heterogeneous",
        choices=["heterogeneous", "homogeneous"],
        help="Type of graph to visualize",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=3,
        help="Number of batches to visualize",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for sampling",
    )
    parser.add_argument(
        "--num_hops",
        type=int,
        default=2,
        help="Number of hops for neighbor sampling",
    )
    parser.add_argument(
        "--train_cutoff_year",
        type=int,
        default=None,
        help="Cutoff year for training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sampling_visualizations",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GNN Sampling Visualization")
    print("=" * 80)
    print(f"Graph type: {args.graph_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num hops: {args.num_hops}")
    print(f"Num batches: {args.num_batches}")

    # Initialize visualizer
    visualizer = SamplingVisualizer(args.preprocessed_dir, args.graph_type)

    # Load graph
    print("\nLoading graph...")
    graph_data, is_hetero, nodes_with_positives = visualizer.load_graph(
        args.train_cutoff_year
    )

    if is_hetero:
        print(
            f"Loaded heterogeneous graph with {graph_data['paragraph'].x.shape[0]} paragraphs"
        )
    else:
        print(f"Loaded homogeneous graph with {graph_data.x.shape[0]} nodes")

    # Visualize multiple batches
    visualizer.visualize_multiple_batches(
        graph_data,
        is_hetero,
        nodes_with_positives,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        num_hops=args.num_hops,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
