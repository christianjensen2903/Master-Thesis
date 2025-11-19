import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def compute_normalized_time_diff(
    date_str: str | None,
    application_date_str: str | None,
    max_years: float = 5.0,
) -> float:
    """Compute normalized time difference between date and application_date.

    Args:
        date_str: Document date (ISO format)
        application_date_str: Application date (ISO format)
        max_years: Maximum years for normalization (default 5.0)

    Returns:
        Normalized time difference in [0, 1], or 0.0 if dates are missing/invalid
    """
    if not date_str or not application_date_str:
        return 0.0

    try:
        date = datetime.fromisoformat(date_str)
        app_date = datetime.fromisoformat(application_date_str)

        # Calculate difference in days
        diff_days = (date - app_date).days

        # Normalize: clamp to [0, max_years * 365] and divide by max
        max_days = max_years * 365.0
        normalized = max(0.0, min(diff_days / max_days, 1.0))

        return normalized
    except (ValueError, AttributeError):
        return 0.0


def compute_time_diff_days(
    date_str: str | None,
    application_date_str: str | None,
    max_years: float = 5.0,
) -> float:
    """Compute time difference in days, capped at max_years.

    Args:
        date_str: Document date (ISO format)
        application_date_str: Application date (ISO format)
        max_years: Maximum years to cap at (default 5.0)

    Returns:
        Time difference in days (capped at max_years * 365), or 0.0 if dates are missing/invalid
    """
    if not date_str or not application_date_str:
        return 0.0

    try:
        date = datetime.fromisoformat(date_str)
        app_date = datetime.fromisoformat(application_date_str)

        # Calculate difference in days
        diff_days = (date - app_date).days

        # Cap at max_years
        max_days = max_years * 365.0
        capped = max(0.0, min(diff_days, max_days))

        return capped
    except (ValueError, AttributeError):
        return 0.0


def analyze_time_diff_citations(
    preprocessed_dir: str = "data/preprocessed",
    output_dir: str = "artifacts",
):
    """Analyze the relationship between time difference and citation frequency."""

    preprocessed_path = Path(preprocessed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load paragraph metadata
    print("Loading paragraph metadata...")
    with open(preprocessed_path / "paragraph_metadata.pkl", "rb") as f:
        par_metadata = pickle.load(f)

    # Load citations
    print("Loading citations...")
    with open(preprocessed_path / "citations.pkl", "rb") as f:
        citations = pickle.load(f)

    print(f"Loaded {len(par_metadata)} paragraphs and {len(citations)} citation edges")

    # Calculate time difference for each paragraph
    print("Calculating time differences...")
    par_id_to_idx = {meta["id"]: i for i, meta in enumerate(par_metadata)}

    time_diffs = []
    time_diffs_days = []
    par_ids = []

    for meta in par_metadata:
        date_str = meta.get("date")
        case_meta = meta.get("meta", {})
        application_date_str = case_meta.get("application_date")

        time_diff_norm = compute_normalized_time_diff(
            date_str, application_date_str, max_years=5.0
        )
        time_diff_days = compute_time_diff_days(
            date_str, application_date_str, max_years=5.0
        )

        time_diffs.append(time_diff_norm)
        time_diffs_days.append(time_diff_days)
        par_ids.append(meta["id"])

    time_diffs = np.array(time_diffs)
    time_diffs_days = np.array(time_diffs_days)

    # Count citations for each paragraph (incoming and outgoing)
    print("Counting citations...")
    incoming_citations = defaultdict(int)
    outgoing_citations = defaultdict(int)
    total_citations = defaultdict(int)

    for src_id, tgt_id in citations:
        if src_id.startswith("par:") and tgt_id.startswith("par:"):
            outgoing_citations[src_id] += 1
            incoming_citations[tgt_id] += 1
            total_citations[src_id] += 1
            total_citations[tgt_id] += 1

    # Create arrays for citation counts
    incoming_counts = np.array(
        [incoming_citations.get(par_id, 0) for par_id in par_ids]
    )
    outgoing_counts = np.array(
        [outgoing_citations.get(par_id, 0) for par_id in par_ids]
    )
    total_counts = np.array([total_citations.get(par_id, 0) for par_id in par_ids])

    # Create DataFrame for analysis
    df = pd.DataFrame(
        {
            "par_id": par_ids,
            "time_diff_normalized": time_diffs,
            "time_diff_days": time_diffs_days,
            "incoming_citations": incoming_counts,
            "outgoing_citations": outgoing_counts,
            "total_citations": total_counts,
            "is_cited": (incoming_counts > 0).astype(int),
            "is_citing": (outgoing_counts > 0).astype(int),
        }
    )

    # Filter out paragraphs with missing time difference (0.0)
    df_valid = df[df["time_diff_days"] > 0].copy()

    print(f"\nStatistics:")
    print(f"Total paragraphs: {len(df)}")
    print(f"Paragraphs with valid time difference: {len(df_valid)}")
    print(f"Paragraphs with incoming citations: {df['is_cited'].sum()}")
    print(f"Paragraphs with outgoing citations: {df['is_citing'].sum()}")
    print(f"\nTime difference statistics (days, capped at 5 years):")
    print(f"  Mean: {df_valid['time_diff_days'].mean():.2f} days")
    print(f"  Median: {df_valid['time_diff_days'].median():.2f} days")
    print(f"  Std: {df_valid['time_diff_days'].std():.2f} days")
    print(f"  Min: {df_valid['time_diff_days'].min():.2f} days")
    print(f"  Max: {df_valid['time_diff_days'].max():.2f} days")
    print(f"  Q25: {df_valid['time_diff_days'].quantile(0.25):.2f} days")
    print(f"  Q75: {df_valid['time_diff_days'].quantile(0.75):.2f} days")

    # Analyze relationship between time difference and citations
    print(f"\n=== Analysis: Time Difference vs Citation Frequency ===")

    # Correlation analysis
    corr_incoming = df_valid["time_diff_days"].corr(df_valid["incoming_citations"])
    corr_outgoing = df_valid["time_diff_days"].corr(df_valid["outgoing_citations"])
    corr_total = df_valid["time_diff_days"].corr(df_valid["total_citations"])

    print(f"\nCorrelation coefficients:")
    print(f"  Time diff vs Incoming citations: {corr_incoming:.4f}")
    print(f"  Time diff vs Outgoing citations: {corr_outgoing:.4f}")
    print(f"  Time diff vs Total citations: {corr_total:.4f}")

    # Compare cited vs non-cited paragraphs
    cited_pars = df_valid[df_valid["is_cited"] == 1]
    non_cited_pars = df_valid[df_valid["is_cited"] == 0]

    if len(cited_pars) > 0 and len(non_cited_pars) > 0:
        print(f"\nTime difference comparison:")
        print(f"  Cited paragraphs (n={len(cited_pars)}):")
        print(f"    Mean: {cited_pars['time_diff_days'].mean():.2f} days")
        print(f"    Median: {cited_pars['time_diff_days'].median():.2f} days")
        print(f"  Non-cited paragraphs (n={len(non_cited_pars)}):")
        print(f"    Mean: {non_cited_pars['time_diff_days'].mean():.2f} days")
        print(f"    Median: {non_cited_pars['time_diff_days'].median():.2f} days")

        # Statistical test
        statistic, p_value = stats.mannwhitneyu(
            cited_pars["time_diff_days"],
            non_cited_pars["time_diff_days"],
            alternative="two-sided",
        )
        print(f"\n  Mann-Whitney U test:")
        print(f"    Statistic: {statistic:.2f}")
        print(f"    p-value: {p_value:.4f}")
        print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Bin analysis
    print(f"\n=== Binned Analysis ===")
    df_valid["time_diff_bin"] = pd.cut(
        df_valid["time_diff_days"],
        bins=[0, 365, 730, 1095, 1460, 1825],  # 0-1, 1-2, 2-3, 3-4, 4-5 years
        labels=["0-1yr", "1-2yr", "2-3yr", "3-4yr", "4-5yr"],
    )

    bin_stats = (
        df_valid.groupby("time_diff_bin", observed=True)
        .agg(
            {
                "incoming_citations": ["mean", "median", "count"],
                "is_cited": "mean",
            }
        )
        .round(4)
    )

    print("\nTime difference bins vs citation metrics:")
    print(bin_stats)

    # Create visualizations
    print(f"\nCreating visualizations...")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 12))

    # 1. Distribution of time differences
    ax1 = plt.subplot(3, 3, 1)
    df_valid["time_diff_days"].hist(bins=50, ax=ax1, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Time Difference (days, capped at 5 years)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Time Differences")
    ax1.axvline(
        df_valid["time_diff_days"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df_valid['time_diff_days'].mean():.0f}",
    )
    ax1.legend()

    # 2. Time difference vs incoming citations (scatter)
    ax2 = plt.subplot(3, 3, 2)
    sample_size = min(5000, len(df_valid))
    sample_df = df_valid.sample(n=sample_size, random_state=42)
    ax2.scatter(
        sample_df["time_diff_days"], sample_df["incoming_citations"], alpha=0.3, s=10
    )
    ax2.set_xlabel("Time Difference (days)")
    ax2.set_ylabel("Incoming Citations")
    ax2.set_title(
        f"Time Difference vs Incoming Citations\n(correlation: {corr_incoming:.3f})"
    )

    # 3. Time difference vs outgoing citations (scatter)
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(
        sample_df["time_diff_days"], sample_df["outgoing_citations"], alpha=0.3, s=10
    )
    ax3.set_xlabel("Time Difference (days)")
    ax3.set_ylabel("Outgoing Citations")
    ax3.set_title(
        f"Time Difference vs Outgoing Citations\n(correlation: {corr_outgoing:.3f})"
    )

    # 4. Box plot: cited vs non-cited
    ax4 = plt.subplot(3, 3, 4)
    box_data = [
        non_cited_pars["time_diff_days"].values,
        cited_pars["time_diff_days"].values,
    ]
    ax4.boxplot(box_data, tick_labels=["Not Cited", "Cited"])
    ax4.set_ylabel("Time Difference (days)")
    ax4.set_title("Time Difference: Cited vs Non-Cited")

    # 5. Binned analysis: mean incoming citations
    ax5 = plt.subplot(3, 3, 5)
    bin_means = df_valid.groupby("time_diff_bin", observed=True)[
        "incoming_citations"
    ].mean()
    bin_means.plot(kind="bar", ax=ax5, color="steelblue", edgecolor="black")
    ax5.set_xlabel("Time Difference Bin")
    ax5.set_ylabel("Mean Incoming Citations")
    ax5.set_title("Mean Incoming Citations by Time Difference Bin")
    ax5.tick_params(axis="x", rotation=45)

    # 6. Binned analysis: citation rate
    ax6 = plt.subplot(3, 3, 6)
    bin_citation_rate = df_valid.groupby("time_diff_bin", observed=True)[
        "is_cited"
    ].mean()
    bin_citation_rate.plot(kind="bar", ax=ax6, color="coral", edgecolor="black")
    ax6.set_xlabel("Time Difference Bin")
    ax6.set_ylabel("Citation Rate")
    ax6.set_title("Citation Rate by Time Difference Bin")
    ax6.tick_params(axis="x", rotation=45)

    # 7. Distribution comparison: cited vs non-cited
    ax7 = plt.subplot(3, 3, 7)
    non_cited_pars["time_diff_days"].hist(
        bins=30,
        ax=ax7,
        alpha=0.5,
        label="Not Cited",
        color="lightblue",
        edgecolor="black",
    )
    cited_pars["time_diff_days"].hist(
        bins=30, ax=ax7, alpha=0.5, label="Cited", color="orange", edgecolor="black"
    )
    ax7.set_xlabel("Time Difference (days)")
    ax7.set_ylabel("Frequency")
    ax7.set_title("Time Difference Distribution: Cited vs Non-Cited")
    ax7.legend()

    # 8. Cumulative distribution
    ax8 = plt.subplot(3, 3, 8)
    sorted_td = np.sort(df_valid["time_diff_days"])
    cumulative = np.arange(1, len(sorted_td) + 1) / len(sorted_td)
    ax8.plot(sorted_td, cumulative, linewidth=2)
    ax8.set_xlabel("Time Difference (days)")
    ax8.set_ylabel("Cumulative Probability")
    ax8.set_title("Cumulative Distribution of Time Differences")
    ax8.grid(True, alpha=0.3)

    # 9. Correlation heatmap
    ax9 = plt.subplot(3, 3, 9)
    corr_matrix = df_valid[
        [
            "time_diff_days",
            "incoming_citations",
            "outgoing_citations",
            "total_citations",
        ]
    ].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        ax=ax9,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax9.set_title("Correlation Matrix")

    plt.tight_layout()
    output_file = output_path / "time_diff_citations_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_file}")

    # Save detailed statistics
    stats_file = output_path / "time_diff_citations_stats.txt"
    with open(stats_file, "w") as f:
        f.write("Time Difference and Citation Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total paragraphs: {len(df)}\n")
        f.write(f"Paragraphs with valid time difference: {len(df_valid)}\n")
        f.write(f"Paragraphs with incoming citations: {df['is_cited'].sum()}\n")
        f.write(f"Paragraphs with outgoing citations: {df['is_citing'].sum()}\n\n")

        f.write("Time Difference Statistics (days, capped at 5 years):\n")
        f.write(f"  Mean: {df_valid['time_diff_days'].mean():.2f}\n")
        f.write(f"  Median: {df_valid['time_diff_days'].median():.2f}\n")
        f.write(f"  Std: {df_valid['time_diff_days'].std():.2f}\n")
        f.write(f"  Min: {df_valid['time_diff_days'].min():.2f}\n")
        f.write(f"  Max: {df_valid['time_diff_days'].max():.2f}\n")
        f.write(f"  Q25: {df_valid['time_diff_days'].quantile(0.25):.2f}\n")
        f.write(f"  Q75: {df_valid['time_diff_days'].quantile(0.75):.2f}\n\n")

        f.write("Correlation Coefficients:\n")
        f.write(f"  Time diff vs Incoming citations: {corr_incoming:.4f}\n")
        f.write(f"  Time diff vs Outgoing citations: {corr_outgoing:.4f}\n")
        f.write(f"  Time diff vs Total citations: {corr_total:.4f}\n\n")

        if len(cited_pars) > 0 and len(non_cited_pars) > 0:
            f.write("Cited vs Non-Cited Comparison:\n")
            f.write(f"  Cited paragraphs (n={len(cited_pars)}):\n")
            f.write(f"    Mean: {cited_pars['time_diff_days'].mean():.2f} days\n")
            f.write(f"    Median: {cited_pars['time_diff_days'].median():.2f} days\n")
            f.write(f"  Non-cited paragraphs (n={len(non_cited_pars)}):\n")
            f.write(f"    Mean: {non_cited_pars['time_diff_days'].mean():.2f} days\n")
            f.write(
                f"    Median: {non_cited_pars['time_diff_days'].median():.2f} days\n\n"
            )

            f.write("Mann-Whitney U Test:\n")
            f.write(f"  Statistic: {statistic:.2f}\n")
            f.write(f"  p-value: {p_value:.4f}\n")
            f.write(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}\n\n")

        f.write("Binned Analysis:\n")
        f.write(str(bin_stats))

    print(f"Saved statistics to {stats_file}")

    # Save DataFrame for further analysis
    csv_file = output_path / "time_diff_citations_data.csv"
    df_valid.to_csv(csv_file, index=False)
    print(f"Saved data to {csv_file}")

    return df_valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze time difference and citation relationship"
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="data/preprocessed",
        help="Directory containing preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for results",
    )

    args = parser.parse_args()

    analyze_time_diff_citations(
        preprocessed_dir=args.preprocessed_dir, output_dir=args.output_dir
    )
