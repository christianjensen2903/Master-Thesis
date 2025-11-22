import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def compute_time_diff_days(date_str, application_date_str):
    """Compute time difference in days between document date and application date."""
    if not date_str or not application_date_str:
        return None
    try:
        date = datetime.fromisoformat(date_str)
        app_date = datetime.fromisoformat(application_date_str)
        return (date - app_date).days
    except (ValueError, AttributeError):
        return None


def main(preprocessed_dir="data/preprocessed", output_dir="artifacts"):
    preprocessed_path = Path(preprocessed_dir)

    # Load data
    print("Loading data...")
    with open(preprocessed_path / "paragraph_metadata.pkl", "rb") as f:
        par_metadata = pickle.load(f)
    with open(preprocessed_path / "citations.pkl", "rb") as f:
        citations = pickle.load(f)

    # Calculate time differences
    par_ids = []
    time_diffs = []

    for meta in par_metadata:
        date_str = meta.get("date")
        app_date_str = meta.get("meta", {}).get("application_date")
        diff = compute_time_diff_days(date_str, app_date_str)

        if diff is not None and diff > 0:
            par_ids.append(meta["id"])
            time_diffs.append(diff)

    # Count citations per paragraph
    citation_counts = defaultdict(int)
    for src_id, tgt_id in citations:
        if src_id.startswith("par:") and tgt_id.startswith("par:"):
            citation_counts[tgt_id] += 1  # incoming citations

    citations_array = np.array([citation_counts.get(pid, 0) for pid in par_ids])
    time_diffs_array = np.array(time_diffs)

    # Remove time differences above 10 years
    max_days = 10 * 365
    time_mask = time_diffs_array <= max_days
    citations_array = citations_array[time_mask]
    time_diffs_array = time_diffs_array[time_mask]

    print(f"Removed {(~time_mask).sum()} points with time diff > 10 years")

    # Remove top 1% citation outliers
    upper_bound = np.percentile(citations_array, 99)
    if upper_bound > 0:
        mask = citations_array <= upper_bound
        citations_array = citations_array[mask]
        time_diffs_array = time_diffs_array[mask]
        print(f"Removed {(~mask).sum()} outliers (citations > {upper_bound:.0f})")

    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(time_diffs_array, citations_array)
    spearman_r, spearman_p = stats.spearmanr(time_diffs_array, citations_array)

    # Print results
    print(f"\n{'='*50}")
    print("CORRELATION: Time Difference vs Citation Count")
    print(f"{'='*50}")
    print(f"\nSample size: {len(time_diffs_array)} paragraphs")
    print(f"\nPearson correlation:  r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}")

    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    # Interpret strength
    abs_r = abs(spearman_r)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if spearman_r > 0 else "negative"
    significant = "Yes" if spearman_p < 0.05 else "No"

    print(f"\nCorrelation strength: {strength}")
    print(f"Direction: {direction}")
    print(f"Statistically significant (p < 0.05): {significant}")

    if spearman_r > 0:
        print(
            f"\n→ Paragraphs with longer time since application tend to have MORE citations"
        )
    else:
        print(
            f"\n→ Paragraphs with longer time since application tend to have FEWER citations"
        )

    # Create scatter plot
    print(f"\nGenerating plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sample if too many points
    if len(time_diffs_array) > 5000:
        idx = np.random.choice(len(time_diffs_array), 5000, replace=False)
        plot_x = time_diffs_array[idx]
        plot_y = citations_array[idx]
    else:
        plot_x = time_diffs_array
        plot_y = citations_array

    # Scatter plot
    ax.scatter(plot_x, plot_y, alpha=0.3, s=15, color="steelblue")

    # Add trend line
    z = np.polyfit(time_diffs_array, citations_array, 1)
    p = np.poly1d(z)
    x_line = np.linspace(time_diffs_array.min(), time_diffs_array.max(), 100)
    ax.plot(
        x_line, p(x_line), color="red", linewidth=2, label=f"Trend (r={spearman_r:.3f})"
    )

    ax.set_xlabel("Time Difference (days since application)", fontsize=12)
    ax.set_ylabel("Number of Citations", fontsize=12)
    ax.set_title(
        f"Time Difference vs Citations\nSpearman ρ = {spearman_r:.3f}, p = {spearman_p:.2e}",
        fontsize=14,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "time_diff_citations_correlation.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", default="data/preprocessed")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    main(args.preprocessed_dir, args.output_dir)
