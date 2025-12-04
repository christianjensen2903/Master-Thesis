"""
Analyze model performance by difficulty bins using per-query results.

This script:
1. Loads per-query results from evaluator output
2. Joins with verbatim percentages
3. Groups by difficulty bins and computes aggregate metrics
4. Plots results

Usage:
    # First run evaluators with per-query output:
    # (evaluator automatically stores per_query_results)
    
    # Then analyze:
    python data_analysis/analyze_by_difficulty.py \
        --results artifacts/dense_per_query.json \
        --model-name Dense
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_verbatim_cache(cache_path: str) -> dict[tuple[str, int], float]:
    """Load verbatim percentages from cache and return as dict."""
    with open(cache_path) as f:
        pairs = json.load(f)

    # Build mapping: (celex, number) -> verbatim_pct
    # Take max verbatim_pct if query appears multiple times (multiple citations)
    query_to_verbatim: dict[tuple[str, int], float] = {}
    for pair in pairs:
        key = (pair["query_key"][0], pair["query_key"][1])
        pct = pair["verbatim_pct"]
        if key not in query_to_verbatim:
            query_to_verbatim[key] = pct
        else:
            # Use max verbatim percentage across all citations from this query
            query_to_verbatim[key] = max(query_to_verbatim[key], pct)

    return query_to_verbatim


def load_per_query_results(results_path: str) -> list[dict]:
    """Load per-query results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def bin_results_by_difficulty(
    per_query_results: list[dict],
    query_to_verbatim: dict[tuple[str, int], float],
    bin_edges: list[float] | None = None,
) -> dict[str, list[dict]]:
    """Bin per-query results by verbatim percentage."""
    if bin_edges is None:
        # 10% bins for finer granularity
        bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]

    # Create bin names
    bin_names = []
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        # Cap upper at 100 for display
        upper_display = min(int(upper * 100), 100)
        name = f"{int(lower*100)}-{upper_display}%"
        bin_names.append(name)

    bins: dict[str, list[dict]] = {name: [] for name in bin_names}

    matched = 0
    unmatched = 0

    for result in per_query_results:
        celex = result["query_celex"]
        number = result["query_number"]
        key = (celex, number)

        if key not in query_to_verbatim:
            unmatched += 1
            continue

        matched += 1
        pct = query_to_verbatim[key]

        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= pct < bin_edges[i + 1]:
                bins[bin_names[i]].append({**result, "verbatim_pct": pct})
                break

    print(f"Matched {matched} queries, {unmatched} unmatched")
    return bins


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) == 0:
        return 0.0, 0.0

    values_arr = np.array(values)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    means = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)
        means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def compute_bin_metrics(
    bins: dict[str, list[dict]],
    k_values: list[int] = [5, 10, 100],
    compute_ci: bool = True,
) -> dict[str, dict]:
    """Compute aggregate metrics for each bin with optional confidence intervals."""
    results = {}

    for bin_name, queries in bins.items():
        if not queries:
            results[bin_name] = {
                "map": 0.0,
                "map_ci": (0.0, 0.0),
                "recall": {k: 0.0 for k in k_values},
                "n_queries": 0,
            }
            continue

        ap_scores = [q["ap"] for q in queries]
        recall_scores = {k: [] for k in k_values}

        for q in queries:
            for k in k_values:
                # Handle both int and str keys in recall dict
                recall_val = q["recall"].get(k, q["recall"].get(str(k), 0.0))
                recall_scores[k].append(recall_val)

        map_mean = float(np.mean(ap_scores))
        map_ci = bootstrap_ci(ap_scores) if compute_ci else (map_mean, map_mean)

        results[bin_name] = {
            "map": map_mean,
            "map_ci": map_ci,
            "recall": {k: float(np.mean(recall_scores[k])) for k in k_values},
            "n_queries": len(queries),
        }

    return results


def print_summary(all_results: dict[str, dict[str, dict]]) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Get bin names from first model
    first_model = list(all_results.keys())[0]
    bin_names = list(all_results[first_model].keys())

    # Sort bins by lower bound
    def get_lower(name: str) -> float:
        return float(name.split("-")[0])

    bin_names = sorted(bin_names, key=get_lower)

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for bin_name in bin_names:
            metrics = results[bin_name]
            r10 = metrics["recall"].get(10, metrics["recall"].get("10", 0))
            ci = metrics.get("map_ci", (metrics["map"], metrics["map"]))
            print(
                f"  {bin_name}: MAP={metrics['map']:.3f} "
                f"[{ci[0]:.3f}, {ci[1]:.3f}], "
                f"R@10={r10:.3f}, n={metrics['n_queries']}"
            )


def plot_results(
    all_results: dict[str, dict[str, dict]],
    output_dir: str,
    k_values: list[int] = [5, 10, 100],
) -> None:
    """Generate plots for the results in academic style."""
    import seaborn as sns

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set academic style
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10

    # Get bin names sorted
    first_model = list(all_results.keys())[0]
    bin_names = list(all_results[first_model].keys())

    def get_lower(name: str) -> float:
        return float(name.split("-")[0])

    bin_names = sorted(bin_names, key=get_lower)
    x = np.arange(len(bin_names))

    # Distinct color palette - ensure good contrast between all models
    colors = {
        "TF-IDF": "#D62728",  # red
        "BOW": "#FF7F0E",  # orange
        "DPR": "#1F77B4",  # blue
        "Dense": "#1F77B4",  # blue (alias)
        "Homogeneous GNN": "#2CA02C",  # green
        "HomoGNN": "#2CA02C",  # green (alias)
        "MLP": "#9467BD",  # purple
        "CaseLink": "#8C564B",  # brown (distinct color)
    }

    markers = {
        "TF-IDF": "s",
        "BOW": "^",
        "DPR": "o",
        "Dense": "o",
        "Homogeneous GNN": "D",
        "HomoGNN": "D",
        "MLP": "v",
        "CaseLink": "s",
    }

    # Single MAP plot with academic styling
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    for model_name, results in all_results.items():
        map_scores = [results[bn]["map"] for bn in bin_names]
        color = colors.get(model_name, "#7F7F7F")
        marker = markers.get(model_name, "o")

        # Get confidence intervals if available
        if "map_ci" in results[bin_names[0]]:
            ci_lower = np.array([results[bn]["map_ci"][0] for bn in bin_names])
            ci_upper = np.array([results[bn]["map_ci"][1] for bn in bin_names])

            # Plot shaded confidence region
            ax.fill_between(
                x,
                ci_lower,
                ci_upper,
                color=color,
                alpha=0.2,
            )

        # Plot line with markers
        ax.plot(
            x,
            map_scores,
            marker=marker,
            linewidth=2,
            markersize=7,
            label=model_name,
            color=color,
            alpha=0.9,
        )

    # Labels with academic formatting
    ax.set_xlabel("Verbatim Overlap (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Average Precision", fontsize=12, fontweight="bold")
    ax.set_title(
        "Retrieval Performance by Citation Difficulty",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, fontsize=9, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=10)

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Y-axis limits
    ax.set_ylim(0, None)

    # Legend inside plot in lower right corner
    ax.legend(
        loc="lower right",
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add annotation about difficulty direction (positioned to not overlap with legend)
    ax.annotate(
        "← Harder",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        color="#7F7F7F",
        style="italic",
    )
    ax.annotate(
        "Easier →",
        xy=(0.88, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        color="#7F7F7F",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path / "difficulty_map.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved MAP plot to {output_path / 'difficulty_map.png'}")

    # Sample counts plot with academic styling
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    counts = [all_results[first_model][bn]["n_queries"] for bn in bin_names]

    # Color gradient from red (hard) to green (easy)
    colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bin_names)))

    bars = ax.bar(
        range(len(bin_names)),
        counts,
        color=colors_bar,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Verbatim Overlap (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Queries", fontsize=12, fontweight="bold")
    ax.set_title(
        "Query Distribution by Citation Difficulty",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(range(len(bin_names)))
    ax.set_xticklabels(bin_names, fontsize=9, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        output_path / "difficulty_sample_counts.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved sample counts to {output_path / 'difficulty_sample_counts.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze results by difficulty bins")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Paths to per-query result JSON files (model_name:path format or just path)",
    )
    parser.add_argument(
        "--verbatim-cache",
        default="artifacts/citation_pairs_with_verbatim.json",
        help="Path to verbatim percentage cache",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/difficulty_results.json",
        help="Output path for aggregated results JSON",
    )
    args = parser.parse_args()

    # Load verbatim cache
    print(f"Loading verbatim cache from {args.verbatim_cache}")
    query_to_verbatim = load_verbatim_cache(args.verbatim_cache)
    print(f"Loaded verbatim percentages for {len(query_to_verbatim)} queries")

    all_results: dict[str, dict[str, dict]] = {}

    for result_spec in args.results:
        # Parse model_name:path or just path
        if ":" in result_spec:
            model_name, path = result_spec.split(":", 1)
        else:
            path = result_spec
            model_name = Path(path).stem.replace("_per_query", "")

        print(f"\nProcessing {model_name} from {path}")
        per_query = load_per_query_results(path)
        print(f"Loaded {len(per_query)} per-query results")

        bins = bin_results_by_difficulty(per_query, query_to_verbatim)
        metrics = compute_bin_metrics(bins)
        all_results[model_name] = metrics

    # Print summary
    print_summary(all_results)

    # Save aggregated results
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved aggregated results to {args.output_json}")

    # Plot
    plot_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
