from pathlib import Path
from collections import Counter
import json
import matplotlib.pyplot as plt


def extract_year_from_celex(celex_id: str) -> int | None:
    """Extract year from CELEX ID format: 3YYYY[L|R]NNNN"""
    if len(celex_id) < 6:
        return None

    try:
        # CELEX format: 3YYYY[L|R]NNNN
        # First digit is 3, next 4 digits are year (e.g., 1958, 1960)
        year_part = celex_id[1:5]
        year = int(year_part)
        return year
    except (ValueError, IndexError):
        return None


def analyze_zero_articles(
    legal_acts_file: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    """Analyze legal acts and return frequency by year for all and those with 0 articles."""
    print(f"Loading legal acts from {legal_acts_file}...")
    with open(legal_acts_file, "r", encoding="utf-8") as f:
        legal_acts = json.load(f)

    print(f"Total legal acts loaded: {len(legal_acts)}")

    all_years = []
    zero_article_years = []

    # Filter valid years (1950-2025)
    MIN_YEAR = 1950
    MAX_YEAR = 2025

    for celex_id, act_data in legal_acts.items():
        year = extract_year_from_celex(celex_id)
        if year and MIN_YEAR <= year <= MAX_YEAR:
            all_years.append(year)

            # Check if articles list is empty
            articles = act_data.get("articles", [])
            if len(articles) == 0:
                zero_article_years.append(year)

    all_freq = dict(Counter(all_years))
    zero_article_freq = dict(Counter(zero_article_years))

    print(
        f"Legal acts with 0 articles: {len(zero_article_years)} out of {len(all_years)}"
    )

    return all_freq, zero_article_freq


def plot_frequencies(
    all_freq: dict[int, int], zero_article_freq: dict[int, int], output_path: Path
) -> None:
    """Plot frequency of all legal acts and those with 0 articles by year."""
    # Get all years from both dictionaries
    all_years = sorted(set(list(all_freq.keys()) + list(zero_article_freq.keys())))

    # Prepare data for plotting
    years = []
    all_counts = []
    zero_article_counts = []

    for year in all_years:
        years.append(year)
        all_counts.append(all_freq.get(year, 0))
        zero_article_counts.append(zero_article_freq.get(year, 0))

    # Create the plot
    plt.figure(figsize=(14, 8))
    plt.plot(
        years, all_counts, label="All legal acts", linewidth=2, marker="o", markersize=3
    )
    plt.plot(
        years,
        zero_article_counts,
        label="Legal acts with 0 articles",
        linewidth=2,
        marker="s",
        markersize=3,
    )

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Frequency of Legal Acts by Year", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main() -> None:
    legal_acts_file = Path("data/legal_acts_simple.json")
    output_file = Path("artifacts/legal_acts_zero_articles_analysis_simple.png")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Analyze
    all_freq, zero_article_freq = analyze_zero_articles(legal_acts_file)

    # Plot
    plot_frequencies(all_freq, zero_article_freq, output_file)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total legal acts: {sum(all_freq.values())}")
    print(f"Total with 0 articles: {sum(zero_article_freq.values())}")
    print(
        f"Percentage with 0 articles: {sum(zero_article_freq.values()) / sum(all_freq.values()) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
