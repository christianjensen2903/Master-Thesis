import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.data import Data, HeteroData  # type: ignore
from torch_geometric.transforms import ToUndirected  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import wandb

from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
    SemanticGraphBuilder,
)


def compute_contrastive_stats(
    sim_matrix: torch.Tensor,
    diagonal_mask: torch.Tensor,
) -> dict:
    """Compute statistics for monitoring contrastive learning."""
    stats = {}

    positive_sims = torch.diagonal(sim_matrix)
    stats["pos_sim_mean"] = positive_sims.mean().item()
    stats["pos_sim_std"] = positive_sims.std().item()
    stats["pos_sim_min"] = positive_sims.min().item()
    stats["pos_sim_max"] = positive_sims.max().item()

    valid_mask = ~torch.isinf(sim_matrix) & ~diagonal_mask
    if not valid_mask.any():
        return {
            **stats,
            "neg_sim_mean": 0.0,
            "neg_sim_std": 0.0,
            "neg_sim_max": 0.0,
            "num_negatives_mean": 0.0,
            "margin_mean": 0.0,
            "pos_rank_mean": 1.0,
        }

    negative_sims = sim_matrix[valid_mask]
    stats["neg_sim_mean"] = negative_sims.mean().item()
    stats["neg_sim_std"] = negative_sims.std().item()
    stats["neg_sim_max"] = negative_sims.max().item()

    num_valid = valid_mask.sum(dim=1).float()
    stats["num_negatives_mean"] = num_valid.mean().item()
    stats["num_negatives_min"] = num_valid.min().item()
    stats["num_negatives_max"] = num_valid.max().item()

    max_neg = torch.where(
        valid_mask, sim_matrix, torch.full_like(sim_matrix, float("-inf"))
    ).max(dim=1)[0]
    margin = positive_sims - max_neg
    stats["margin_mean"] = margin.mean().item()
    stats["margin_min"] = margin.min().item()

    ranks = (sim_matrix > positive_sims.unsqueeze(1)).sum(dim=1) + 1
    stats["pos_rank_mean"] = ranks.float().mean().item()
    stats["pos_rank_median"] = ranks.float().median().item()

    for k in [1, 5, 10]:
        if k <= sim_matrix.size(1):
            stats[f"acc@{k}"] = (ranks <= k).float().mean().item()

    return stats


def degree_regularization_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Degree regularization loss (CaseLink style).

    Encourages node embeddings to be less dependent on node degree,
    preventing popular nodes from dominating the representation space.
    """
    num_nodes = embeddings.size(0)

    # Compute degree
    degree = torch.zeros(num_nodes, device=embeddings.device)
    ones = torch.ones(edge_index.size(1), device=embeddings.device)
    degree.scatter_add_(0, edge_index[0], ones)
    degree.scatter_add_(0, edge_index[1], ones)

    # Normalize embeddings
    emb_norm = F.normalize(embeddings, p=2, dim=1)

    # Compute correlation between embedding magnitude and degree
    degree_weight = torch.log1p(degree).unsqueeze(1)
    degree_weighted_emb = emb_norm * degree_weight

    # The loss encourages decorrelation between embedding and degree
    correlation = (emb_norm * degree_weighted_emb).sum(dim=1).mean()

    return correlation


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
    anchor_times: torch.Tensor | None = None,
    positive_times: torch.Tensor | None = None,
    anchor_indices: torch.Tensor | None = None,
    positive_indices: torch.Tensor | None = None,
    return_stats: bool = False,
    # Degree regularization options
    edge_index: torch.Tensor | None = None,
    all_embeddings: torch.Tensor | None = None,
    degree_reg_weight: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    In-batch contrastive loss with temporal filtering, false negative masking,
    and optional degree regularization (CaseLink style).

    Args:
        anchor: Anchor embeddings [N, D]
        positive: Positive embeddings [N, D]
        temperature: Softmax temperature
        anchor_times: Timestamps for anchors
        positive_times: Timestamps for positives
        anchor_indices: Original indices of anchors (for false negative masking)
        positive_indices: Original indices of positives
        return_stats: Return detailed statistics
        edge_index: Edge index for degree computation (required if degree_reg_weight > 0)
        all_embeddings: All node embeddings for degree regularization
        degree_reg_weight: Weight for degree regularization loss (0 = disabled)
    """
    sim_matrix = torch.mm(anchor, positive.t()) / temperature
    batch_size = sim_matrix.size(0)

    diagonal_mask = torch.eye(
        batch_size, batch_size, dtype=torch.bool, device=sim_matrix.device
    )

    # Mask false negatives (same source pointing to same target)
    if anchor_indices is not None and positive_indices is not None:
        same_anchor = anchor_indices.unsqueeze(1) == anchor_indices.unsqueeze(0)
        same_target = positive_indices.unsqueeze(1) == positive_indices.unsqueeze(0)
        fn_mask = (same_anchor.float() @ same_target.float()) > 0

        sim_matrix = sim_matrix.masked_fill(fn_mask & ~diagonal_mask, float("-inf"))

    # Apply temporal masking
    if anchor_times is not None and positive_times is not None:
        time_mask = positive_times.unsqueeze(0) < anchor_times.unsqueeze(1)

        sim_matrix = sim_matrix.masked_fill(~(time_mask | diagonal_mask), float("-inf"))

    labels = torch.arange(batch_size, device=sim_matrix.device)
    nce_loss = F.cross_entropy(sim_matrix, labels)

    # Add degree regularization if enabled
    total_loss = nce_loss
    deg_reg = None
    if degree_reg_weight > 0 and edge_index is not None:
        if all_embeddings is not None:
            deg_reg = degree_regularization_loss(all_embeddings, edge_index)
        else:
            # Fallback to using anchor/positive (may not include all nodes)
            combined = torch.cat([anchor, positive], dim=0)
            deg_reg = degree_regularization_loss(combined, edge_index)
        total_loss = nce_loss + degree_reg_weight * deg_reg

    if not return_stats:
        return total_loss

    stats = compute_contrastive_stats(sim_matrix, diagonal_mask)
    stats["nce_loss"] = nce_loss.item()
    if deg_reg is not None:
        stats["deg_reg"] = deg_reg.item()
        stats["deg_reg_weighted"] = (degree_reg_weight * deg_reg).item()
    stats["total_loss"] = total_loss.item()

    return total_loss, stats


class GNNTrainer:
    def __init__(
        self,
        graph_builder: "HomogeneousGraphBuilder | HeterogeneousGraphBuilder | SemanticGraphBuilder",
        output_path: str = "output/gnn",
        batch_size: int = 16,
        epochs: int = 5,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_hops: int = 2,
        checkpoint_interval: int = 25,
        wandb_project: str | None = "gnn-training",
        wandb_name: str | None = None,
        warmup_epochs: int = 3,
        eval_every_n_epochs: int = 1,
        gradient_clip_val: float | None = None,
        log_every_n_batches: int = 100,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        # Degree regularization (CaseLink-style)
        degree_reg_weight: float = 0.0,
    ):
        self.graph_builder = graph_builder
        self.output_path = output_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.num_hops = num_hops
        self.checkpoint_interval = checkpoint_interval
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.warmup_epochs = warmup_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_batches = log_every_n_batches
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.degree_reg_weight = degree_reg_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_hetero = isinstance(graph_builder, HeterogeneousGraphBuilder)

        print(f"Using device: {self.device}")
        print(f"Graph builder: {type(graph_builder).__name__}")
        if degree_reg_weight > 0:
            print(f"Degree regularization enabled (weight={degree_reg_weight})")

    def _process_hetero_batch(self, batch: HeteroData) -> dict | None:
        """Process a heterogeneous batch."""
        batch_size = batch["paragraph"].batch_size
        x = batch["paragraph"].x.clone()

        if hasattr(batch["paragraph"], "x_query"):
            x[:batch_size] = batch["paragraph"].x_query[:batch_size]

        anchor_times = getattr(batch["paragraph"], "time", None)
        if anchor_times is not None:
            anchor_times = anchor_times[:batch_size]

        if ("paragraph", "cites", "paragraph") not in batch.edge_types:
            return None

        cite_edge_index = batch["paragraph", "cites", "paragraph"].edge_index

        # Mask citation edges to prevent leakage
        if ("paragraph", "belongs_to", "case") in batch.edge_types:
            par_to_case = batch["paragraph", "belongs_to", "case"].edge_index
            case_to_par = batch["case", "contains", "paragraph"].edge_index
            anchor_cases = par_to_case[1, par_to_case[0] < batch_size].unique()
            paragraphs_in_anchor_cases = case_to_par[
                1, torch.isin(case_to_par[0], anchor_cases)
            ].unique()
        else:
            paragraphs_in_anchor_cases = torch.arange(batch_size, device=self.device)

        src, tgt = cite_edge_index
        leakage_mask = torch.isin(src, paragraphs_in_anchor_cases) | torch.isin(
            tgt, paragraphs_in_anchor_cases
        )

        modified_batch = batch.clone()
        modified_batch["paragraph", "cites", "paragraph"].edge_index = cite_edge_index[
            :, ~leakage_mask
        ]
        modified_batch["paragraph"].x = x

        return {
            "batch_size": batch_size,
            "modified_batch": modified_batch,
            "edge_index": cite_edge_index,
            "anchor_times": anchor_times,
            "all_times": getattr(batch["paragraph"], "time", None),
        }

    def _process_homo_batch(self, batch: Data) -> dict:
        """Process a homogeneous batch.

        Uses the graph builder's masking strategy to prevent information leakage.
        """
        batch_size = batch.batch_size
        x = batch.x.clone()

        if hasattr(batch, "x_query"):
            x[:batch_size] = batch.x_query[:batch_size]

        anchor_times = batch.time[:batch_size] if hasattr(batch, "time") else None
        edge_attr = getattr(batch, "edge_attr", None)

        # Use the graph builder's masking logic
        masked_edge_index, masked_edge_attr = (
            self.graph_builder.mask_edges_for_training(
                batch.edge_index, edge_attr, batch_size
            )
        )

        # Check if using citation_pairs for loss (SemanticGraphBuilder)
        use_citation_pairs = (
            hasattr(batch, "citation_pairs") and batch.citation_pairs.size(1) > 0
        )

        # Remap citation_pairs to local subgraph indices if available
        citation_pairs_local = None
        n_id = getattr(batch, "n_id", None)
        if use_citation_pairs and n_id is not None:
            citation_pairs = batch.citation_pairs
            # Create reverse mapping: global index -> local index
            global_to_local = {
                int(global_idx): local_idx
                for local_idx, global_idx in enumerate(n_id.tolist())
            }

            # Remap citation pairs to local indices
            local_src = []
            local_tgt = []
            for i in range(citation_pairs.size(1)):
                src_global = int(citation_pairs[0, i])
                tgt_global = int(citation_pairs[1, i])
                if src_global in global_to_local and tgt_global in global_to_local:
                    local_src.append(global_to_local[src_global])
                    local_tgt.append(global_to_local[tgt_global])

            if local_src:
                citation_pairs_local = torch.tensor(
                    [local_src, local_tgt],
                    dtype=torch.long,
                    device=batch.x.device,
                )

        return {
            "batch_size": batch_size,
            "x": x,
            "edge_index": batch.edge_index,
            "masked_edge_index": masked_edge_index,
            "masked_edge_attr": masked_edge_attr,
            "edge_attr": edge_attr,
            "date_feature": batch.date_feature,
            "language": getattr(batch, "language", None),
            "anchor_times": anchor_times,
            "all_times": getattr(batch, "time", None),
            "node_type": getattr(batch, "node_type", None),
            # Citation pairs for loss computation (SemanticGraphBuilder)
            "citation_pairs": citation_pairs_local,
            "use_citation_pairs": use_citation_pairs,
            # Case metadata embeddings
            "subject_matter": getattr(batch, "subject_matter", None),
            "keywords": getattr(batch, "keywords", None),
            "case_law_about": getattr(batch, "case_law_about", None),
        }

    def _get_embeddings(
        self, model: nn.Module, batch_data: dict
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings from the model.

        For dual encoders, returns (query_embeddings, doc_embeddings).
        For single encoders, returns all embeddings.
        """
        if self.is_hetero:
            out = model(batch_data["modified_batch"])
            return out["paragraph"] if isinstance(out, dict) else out

        # Check if model is a dual encoder
        is_dual = hasattr(model, "encode_query") and hasattr(model, "encode_document")

        if is_dual:
            batch_size = batch_data["batch_size"]
            x = batch_data["x"]
            date_feature = batch_data.get("date_feature")
            language = batch_data.get("language")
            subject_matter = batch_data.get("subject_matter")
            keywords = batch_data.get("keywords")
            case_law_about = batch_data.get("case_law_about")

            # Query encoding for anchor nodes (no edges needed)
            query_emb = model.encode_query(
                x[:batch_size],
                date_feature=(
                    date_feature[:batch_size] if date_feature is not None else None
                ),
                language=language[:batch_size] if language is not None else None,
                subject_matter=(
                    subject_matter[:batch_size] if subject_matter is not None else None
                ),
                keywords=keywords[:batch_size] if keywords is not None else None,
                case_law_about=(
                    case_law_about[:batch_size] if case_law_about is not None else None
                ),
            )

            # Document encoding for all nodes (with edges)
            doc_emb = model.encode_document(
                x,
                batch_data["masked_edge_index"],
                date_feature=date_feature,
                edge_attr=batch_data.get("masked_edge_attr"),
                language=language,
                subject_matter=subject_matter,
                keywords=keywords,
                case_law_about=case_law_about,
            )

            return query_emb, doc_emb
        else:
            out = model(
                batch_data["x"],
                batch_data["masked_edge_index"],
                date_feature=batch_data.get("date_feature"),
                edge_attr=batch_data.get("masked_edge_attr"),
                language=batch_data.get("language"),
                subject_matter=batch_data.get("subject_matter"),
                keywords=batch_data.get("keywords"),
                case_law_about=batch_data.get("case_law_about"),
            )
            return out["paragraph"] if isinstance(out, dict) else out

    def _compute_loss(
        self,
        model: nn.Module,
        batch_data: dict,
        return_stats: bool = False,
    ):
        """Compute contrastive loss for a batch.

        Supports two modes for finding training pairs:
        1. Default: from edge_index where edge_attr == 0 (cites)
        2. Citation pairs mode: from citation_pairs tensor (CaseLink-style)
        """
        batch_size = batch_data["batch_size"]
        edge_index = batch_data["edge_index"]
        edge_attr = batch_data.get("edge_attr")
        all_times = batch_data.get("all_times")
        use_citation_pairs = batch_data.get("use_citation_pairs", False)
        citation_pairs = batch_data.get("citation_pairs")
        node_type = batch_data.get("node_type")

        emb_result = self._get_embeddings(model, batch_data)

        # Check if dual encoder (returns tuple) or single encoder
        is_dual = isinstance(emb_result, tuple)
        if is_dual:
            query_emb, doc_emb = emb_result
        else:
            doc_emb = emb_result

        # Find training pairs
        if (
            use_citation_pairs
            and citation_pairs is not None
            and citation_pairs.size(1) > 0
        ):
            # CaseLink-style: use citation_pairs tensor (already in local indices)
            cite_src, cite_tgt = citation_pairs

            # Filter to pairs where source is in input batch
            input_mask = cite_src < batch_size

            # Filter to paragraph nodes only (if node_type available)
            if node_type is not None:
                src_is_par = node_type[cite_src] == 0
                tgt_is_par = node_type[cite_tgt] == 0
                input_mask = input_mask & src_is_par & tgt_is_par

            if input_mask.sum() == 0:
                return (None, None) if return_stats else None

            batch_src = cite_src[input_mask]
            batch_tgt = cite_tgt[input_mask]
        else:
            # Default: find pairs from edge_index
            src, tgt = edge_index
            if edge_attr is not None:
                input_mask = (src < batch_size) & (edge_attr == 0)
            else:
                input_mask = src < batch_size

            if input_mask.sum() == 0:
                return (None, None) if return_stats else None

            batch_src, batch_tgt = src[input_mask], tgt[input_mask]

        if is_dual:
            anchor_emb = query_emb[batch_src]
            positive_emb = doc_emb[batch_tgt]
        else:
            anchor_emb = doc_emb[batch_src]
            positive_emb = doc_emb[batch_tgt]

        pair_times = (
            (all_times[batch_src], all_times[batch_tgt])
            if all_times is not None
            else (None, None)
        )

        result = info_nce_loss(
            anchor_emb,
            positive_emb,
            self.temperature,
            anchor_times=pair_times[0],
            positive_times=pair_times[1],
            anchor_indices=batch_src,
            positive_indices=batch_tgt,
            return_stats=return_stats,
            # Degree regularization parameters
            edge_index=(
                batch_data["masked_edge_index"] if self.degree_reg_weight > 0 else None
            ),
            all_embeddings=doc_emb if self.degree_reg_weight > 0 else None,
            degree_reg_weight=self.degree_reg_weight,
        )

        if return_stats:
            loss, stats = result
            stats["emb_norm_mean"] = doc_emb.norm(dim=1).mean().item()
            stats["emb_norm_std"] = doc_emb.norm(dim=1).std().item()
            if is_dual:
                stats["query_emb_norm_mean"] = query_emb.norm(dim=1).mean().item()
            stats["num_pairs"] = batch_src.size(0)
            # Log language embedding stats
            if hasattr(model, "language_embedding") and model.use_language:
                lang_emb_weight = model.language_embedding.embedding.weight
                stats["lang_emb_norm_mean"] = lang_emb_weight.norm(dim=1).mean().item()
                stats["lang_emb_norm_std"] = lang_emb_weight.norm(dim=1).std().item()
            return loss, stats
        return result

    def _run_epoch(
        self,
        model: nn.Module,
        loader: NeighborLoader,
        optimizer: torch.optim.Optimizer | None,
        scheduler: LambdaLR | None,
        epoch: int,
        batch_counter: int,
        training: bool = True,
    ) -> tuple[float, int, dict]:
        """Run one epoch of training or validation."""
        model.train() if training else model.eval()
        total_loss, num_batches = 0.0, 0
        stats_accum: dict[str, list] = {}

        desc = "Training" if training else "Validation"
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_idx, batch in enumerate(
                tqdm(loader, desc=f"{desc} batches", leave=False, dynamic_ncols=True)
            ):
                # Move batch to device (NeighborLoader keeps data on CPU)
                batch = batch.to(self.device)

                batch_data = (
                    self._process_hetero_batch(batch)
                    if self.is_hetero
                    else self._process_homo_batch(batch)
                )
                if batch_data is None:
                    continue

                result = self._compute_loss(model, batch_data, return_stats=True)
                if result is None or result[0] is None:
                    continue

                loss, batch_stats = result

                if training and optimizer is not None and scheduler is not None:
                    optimizer.zero_grad()
                    loss.backward()

                    if self.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip_val
                        )

                    optimizer.step()
                    scheduler.step()
                    batch_counter += 1

                    if self.wandb_project and batch_idx % self.log_every_n_batches == 0:
                        wandb.log(
                            {
                                "train/batch_loss": loss.item(),
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "train/batch": batch_counter,
                                **{f"train/{k}": v for k, v in batch_stats.items()},
                            }
                        )

                total_loss += loss.item()
                num_batches += 1

                for k, v in batch_stats.items():
                    stats_accum.setdefault(k, []).append(v)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_stats = {k: sum(v) / len(v) for k, v in stats_accum.items() if v}

        return avg_loss, batch_counter, avg_stats

    def _build_graph(self, train_cutoff_year: int | None = None):
        """Build graph using the configured graph builder.

        Note: Graph stays on CPU for NeighborLoader compatibility.
        Batches are moved to GPU in _run_epoch.
        """
        graph = self.graph_builder.build_graph(train_cutoff_year=train_cutoff_year)
        if self.is_hetero:
            return ToUndirected()(graph)
        return graph

    def _get_input_nodes(self, graph_data):
        """Get input nodes (nodes with citation edges or citation pairs)."""
        if self.is_hetero:
            cite_edges = graph_data["paragraph", "cites", "paragraph"].edge_index
            nodes = cite_edges[0].unique()
            print(
                f"  Paragraph nodes with citations: {len(nodes)} / {graph_data['paragraph'].num_nodes}"
            )
            return ("paragraph", nodes), nodes
        else:
            # Use citation_pairs if available (CaseLink-style), otherwise edge_index
            if (
                hasattr(graph_data, "citation_pairs")
                and graph_data.citation_pairs.size(1) > 0
            ):
                nodes = graph_data.citation_pairs[0].unique()
                # Filter to paragraph nodes only
                if hasattr(graph_data, "node_type"):
                    par_mask = graph_data.node_type[nodes] == 0
                    nodes = nodes[par_mask]
                num_par = getattr(graph_data, "num_par_nodes", graph_data.num_nodes)
                print(f"  Paragraph nodes with citations: {len(nodes)} / {num_par}")
            else:
                cites_edges = graph_data.edge_index[:, graph_data.edge_attr == 0]
                nodes = cites_edges[0].unique()
                print(f"  Nodes with citations: {len(nodes)} / {graph_data.num_nodes}")
            return nodes, nodes

    def _create_loader(
        self, graph_data, input_nodes, shuffle: bool = True
    ) -> NeighborLoader:
        """Create a NeighborLoader."""
        num_neighbors = [-1] * (self.num_hops + 1) if self.num_hops > 0 else [-1]
        return NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=shuffle,
            time_attr="time",
            subgraph_type="bidirectional",
        )

    def _print_epoch_stats(
        self,
        epoch: int,
        train_loss: float,
        train_stats: dict,
        val_loss: float | None = None,
        val_stats: dict | None = None,
    ):
        """Print epoch statistics."""
        print(f"\nEpoch {epoch + 1}/{self.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")

        if train_stats:
            # Show NCE loss and deg reg separately if both present
            if "nce_loss" in train_stats:
                print(f"  NCE Loss: {train_stats.get('nce_loss', 0):.4f}")
            if "deg_reg" in train_stats:
                print(
                    f"  Deg Reg: {train_stats.get('deg_reg', 0):.4f} "
                    f"(weighted: {train_stats.get('deg_reg_weighted', 0):.4f})"
                )
            print(
                f"  Pos Sim: {train_stats.get('pos_sim_mean', 0):.3f} ± {train_stats.get('pos_sim_std', 0):.3f}"
            )
            print(f"  Neg Sim: {train_stats.get('neg_sim_mean', 0):.3f}")
            print(f"  Margin:  {train_stats.get('margin_mean', 0):.3f}")
            print(f"  Acc@1:   {train_stats.get('acc@1', 0):.2%}")

        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
            if val_stats:
                if "nce_loss" in val_stats:
                    print(f"  Val NCE Loss: {val_stats.get('nce_loss', 0):.4f}")
                if "deg_reg" in val_stats:
                    print(f"  Val Deg Reg: {val_stats.get('deg_reg', 0):.4f}")
                print(f"  Val Acc@1: {val_stats.get('acc@1', 0):.2%}")

    def train(
        self,
        gnn_model: nn.Module,
        train_cutoff_year: int | None = None,
        val_cutoff_year: int | None = None,
    ) -> nn.Module:
        """Train GNN model with optional validation.

        Args:
            gnn_model: The GNN model to train
            train_cutoff_year: Only include data before this year for training
            val_cutoff_year: Only include data before this year for validation
                (validation uses nodes between train_cutoff and val_cutoff)
        """
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        graph_type = "Heterogeneous" if self.is_hetero else "Homogeneous"
        print("\n" + "=" * 80)
        print(f"Training GNN with {graph_type} Graph")
        print("=" * 80)

        # Build training graph
        train_graph = self._build_graph(train_cutoff_year)

        print("\nFiltering nodes with positive examples...")
        input_nodes, nodes_with_positives = self._get_input_nodes(train_graph)
        train_loader = self._create_loader(train_graph, input_nodes)

        # Build validation graph if val_cutoff_year provided
        val_loader = None
        val_nodes_with_positives = None
        if val_cutoff_year is not None:
            print("\nBuilding validation graph...")
            val_graph = self._build_graph(val_cutoff_year)
            val_input_nodes, val_nodes_with_positives = self._get_input_nodes(val_graph)

            # Filter val nodes to only those after train cutoff
            if not self.is_hetero and train_cutoff_year:
                cutoff_ts = self.graph_builder._date_to_timestamp(
                    f"{train_cutoff_year}-01-01"
                )
                time_mask = val_graph.time[val_nodes_with_positives] > cutoff_ts
                val_nodes_with_positives = val_nodes_with_positives[time_mask]
                val_input_nodes = val_nodes_with_positives
                print(f"  Val nodes after filtering: {len(val_nodes_with_positives)}")

            val_loader = self._create_loader(val_graph, val_input_nodes, shuffle=False)

        # Initialize wandb
        if self.wandb_project:
            config = {
                k: v
                for k, v in self.__dict__.items()
                if not k.startswith("_") and k != "device"
            }
            config["num_nodes_with_positives"] = len(nodes_with_positives)
            if val_nodes_with_positives is not None:
                config["num_val_nodes_with_positives"] = len(val_nodes_with_positives)
            wandb.init(project=self.wandb_project, name=self.wandb_name, config=config)

        # Setup model and optimizer
        model = gnn_model.to(self.device)
        optimizer = AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Scheduler with warmup and cosine decay
        steps_per_epoch = len(train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        if self.wandb_project:
            wandb.watch(model, log="all", log_freq=100)

        print(f"\nStarting training for {self.epochs} epochs...")
        if self.early_stopping_patience is not None:
            print(
                f"  Early stopping: patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}"
            )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        batch_counter = 0

        for epoch in range(self.epochs):
            train_loss, batch_counter, train_stats = self._run_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                batch_counter,
                training=True,
            )

            val_loss, val_stats = None, None
            if val_loader and (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss, _, val_stats = self._run_epoch(
                    model, val_loader, None, None, epoch, 0, training=False
                )

            self._print_epoch_stats(epoch, train_loss, train_stats, val_loss, val_stats)

            # Logging and checkpointing
            if self.wandb_project:
                log_dict = {"train/epoch_loss": train_loss, "epoch": epoch + 1}
                log_dict.update({f"train_epoch/{k}": v for k, v in train_stats.items()})
                if val_loss is not None:
                    log_dict["val/epoch_loss"] = val_loss
                    log_dict.update(
                        {f"val_epoch/{k}": v for k, v in (val_stats or {}).items()}
                    )
                wandb.log(log_dict)

            if val_loss is not None:
                if val_loss < best_val_loss - self.early_stopping_min_delta:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best val loss: {best_val_loss:.4f} -> {val_loss:.4f}"
                    )
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if self.early_stopping_patience is not None:
                        print(
                            f"  No improvement for {epochs_without_improvement}/{self.early_stopping_patience} epochs"
                        )
                        if epochs_without_improvement >= self.early_stopping_patience:
                            print(
                                f"\n⚡ Early stopping triggered after {epoch + 1} epochs"
                            )
                            break

            if (epoch + 1) % self.checkpoint_interval == 0:
                path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), path)
                print(f"  ✓ Checkpoint saved: {path}")

        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")

        if os.path.exists(f"{self.output_path}/best_model.pt"):
            print(f"\nLoading best model (val loss: {best_val_loss:.4f})...")
            model.load_state_dict(torch.load(f"{self.output_path}/best_model.pt"))

        if self.wandb_project:
            wandb.finish()

        return model
