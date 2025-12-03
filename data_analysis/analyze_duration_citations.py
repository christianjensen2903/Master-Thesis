"""
Analyze whether cases with longer duration (date - application_date) get cited more.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt


def compute_duration_days(
    application_date_str: str | None, date_str: str | None
) -> int | None:
    """Compute case duration in days."""
    if not application_date_str or not date_str:
        return None
    try:
        app_date = datetime.strptime(application_date_str, "%Y-%m-%d")
        case_date = datetime.strptime(date_str, "%Y-%m-%d")
        return (case_date - app_date).days
    except (ValueError, TypeError):
        return None


def main(
    json_path: str = "data/par-to-par.json",
    csv_path: str = "data/par-to-par-cleaned.csv",
    max_duration_years: float = 5,
    output_dir: str = "artifacts",
):
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load case metadata
    print(f"Loading case data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    # Compute duration for each case
    case_durations: dict[str, int] = {}
    missing_dates = 0
    invalid_durations = 0

    for celex, case in case_data.items():
        meta = case.get("meta", {})
        application_date = meta.get("application_date")
        case_date = meta.get("date")

        duration = compute_duration_days(application_date, case_date)
        if duration is None:
            missing_dates += 1
        elif duration < 0:
            invalid_durations += 1
        else:
            case_durations[celex] = duration

    print(f"Total cases: {len(case_data)}")
    print(f"Cases with valid durations: {len(case_durations)}")
    print(f"Missing dates: {missing_dates}")
    print(f"Invalid (negative) durations: {invalid_durations}")

    # Load citation data
    print(f"\nLoading citations from {csv_path}...")
    citations_df = pd.read_csv(csv_path)
    print(f"Total citation pairs: {len(citations_df)}")
    print(f"Citation columns: {list(citations_df.columns)}")

    # Extract case CELEX from paragraph IDs (CELEX_TO is the cited case)
    # Count citations per case (incoming citations)
    case_citation_counts: dict[str, int] = defaultdict(int)

    for celex_to in citations_df["CELEX_TO"]:
        case_citation_counts[celex_to] += 1

    print(f"Cases with at least one citation: {len(case_citation_counts)}")

    # Build arrays for analysis
    durations = []
    citations = []
    celex_ids = []

    for celex, duration in case_durations.items():
        durations.append(duration)
        citations.append(case_citation_counts.get(celex, 0))
        celex_ids.append(celex)

    durations = np.array(durations)
    citations = np.array(citations)

    print(f"\n{'='*60}")
    print("INITIAL DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Cases with duration data: {len(durations)}")
    print(f"Duration range: {durations.min()} to {durations.max()} days")
    print(
        f"Duration range: {durations.min()/365:.1f} to {durations.max()/365:.1f} years"
    )
    print(f"Citation range: {citations.min()} to {citations.max()}")

    # Cap duration at max_duration_years
    max_days = int(max_duration_years * 365)
    duration_mask = durations <= max_days
    durations_capped = durations[duration_mask]
    citations_capped = citations[duration_mask]

    removed_outliers = (~duration_mask).sum()
    print(
        f"\nRemoved {removed_outliers} cases with duration > {max_duration_years} years"
    )

    # Statistics after capping
    print(f"\n{'='*60}")
    print(f"DATA AFTER CAPPING AT {max_duration_years} YEARS")
    print(f"{'='*60}")
    print(f"Cases remaining: {len(durations_capped)}")
    print(
        f"Duration: mean={durations_capped.mean()/365:.2f} years, median={np.median(durations_capped)/365:.2f} years"
    )
    print(
        f"Citations: mean={citations_capped.mean():.2f}, median={np.median(citations_capped):.0f}, max={citations_capped.max()}"
    )

    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(durations_capped, citations_capped)
    spearman_r, spearman_p = stats.spearmanr(durations_capped, citations_capped)

    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Pearson correlation:  r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}")

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
        print(f"\n→ Cases with longer duration tend to get MORE citations")
    else:
        print(f"\n→ Cases with longer duration tend to get FEWER citations")

    # Bin analysis: group by duration ranges
    print(f"\n{'='*60}")
    print("CITATION COUNTS BY DURATION RANGE")
    print(f"{'='*60}")

    duration_ranges = [
        (0, 180, "0-6 months"),
        (181, 365, "6-12 months"),
        (366, 730, "1-2 years"),
        (731, 1095, "2-3 years"),
        (1096, 1460, "3-4 years"),
        (1461, 1825, "4-5 years"),
    ]

    for min_days, max_days_bin, label in duration_ranges:
        mask = (durations_capped >= min_days) & (durations_capped <= max_days_bin)
        if mask.sum() > 0:
            subset_citations = citations_capped[mask]
            print(
                f"{label:15s}: n={mask.sum():6d}, "
                f"mean_citations={subset_citations.mean():6.2f}, "
                f"median={np.median(subset_citations):5.0f}, "
                f"total={subset_citations.sum():8d}"
            )

    # Create visualizations
    print(f"\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Scatter plot with trend line
    ax1 = axes[0, 0]
    if len(durations_capped) > 5000:
        idx = np.random.choice(len(durations_capped), 5000, replace=False)
        plot_x = durations_capped[idx] / 365  # Convert to years
        plot_y = citations_capped[idx]
    else:
        plot_x = durations_capped / 365
        plot_y = citations_capped

    ax1.scatter(plot_x, plot_y, alpha=0.3, s=15, color="steelblue")

    # Add trend line
    z = np.polyfit(durations_capped / 365, citations_capped, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max_duration_years, 100)
    ax1.plot(x_line, p(x_line), color="red", linewidth=2, label=f"Trend line")

    ax1.set_xlabel("Case Duration (years)", fontsize=12)
    ax1.set_ylabel("Number of Citations", fontsize=12)
    ax1.set_title(
        f"Case Duration vs Citations (Spearman ρ = {spearman_r:.3f})", fontsize=14
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot by duration range
    ax2 = axes[0, 1]
    boxplot_data = []
    boxplot_labels = []

    for min_days, max_days_bin, label in duration_ranges:
        mask = (durations_capped >= min_days) & (durations_capped <= max_days_bin)
        if mask.sum() > 0:
            boxplot_data.append(citations_capped[mask])
            boxplot_labels.append(label.replace(" ", "\n"))

    bp = ax2.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax2.set_xlabel("Duration Range", fontsize=12)
    ax2.set_ylabel("Number of Citations", fontsize=12)
    ax2.set_title("Citations Distribution by Duration Range", fontsize=14)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Mean citations by duration range (bar plot)
    ax3 = axes[1, 0]
    mean_citations = [np.mean(d) for d in boxplot_data]
    x_pos = range(len(mean_citations))
    bars = ax3.bar(x_pos, mean_citations, color="steelblue", edgecolor="navy")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(boxplot_labels, rotation=45, ha="right")
    ax3.set_xlabel("Duration Range", fontsize=12)
    ax3.set_ylabel("Mean Citations", fontsize=12)
    ax3.set_title("Mean Citations by Duration Range", fontsize=14)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, mean_citations):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 4: Duration distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(
        durations_capped / 365, bins=50, color="steelblue", edgecolor="navy", alpha=0.7
    )
    ax4.set_xlabel("Case Duration (years)", fontsize=12)
    ax4.set_ylabel("Number of Cases", fontsize=12)
    ax4.set_title("Distribution of Case Durations", fontsize=14)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_file = output_path / "duration_citations_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_file}")

    # Summary
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    if abs_r < 0.1:
        print(
            f"There is NO meaningful relationship between case duration and citations."
        )
    elif spearman_r > 0:
        print(
            f"There is a {strength} POSITIVE relationship: longer cases tend to get more citations."
        )
    else:
        print(
            f"There is a {strength} NEGATIVE relationship: longer cases tend to get fewer citations."
        )

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze case duration vs citations")
    parser.add_argument("--json-path", default="data/par-to-par.json")
    parser.add_argument("--csv-path", default="data/par-to-par-cleaned.csv")
    parser.add_argument("--max-duration-years", type=float, default=5)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    main(args.json_path, args.csv_path, args.max_duration_years, args.output_dir)
