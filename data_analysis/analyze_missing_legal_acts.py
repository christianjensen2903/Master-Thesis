from pathlib import Path
from collections import Counter
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


def load_celex_ids(celex_file: Path) -> set[str]:
    """Load CELEX IDs from file."""
    celex_ids = set()
    with open(celex_file, "r", encoding="utf-8") as f:
        for line in f:
            celex_id = line.strip()
            if celex_id:
                celex_ids.add(celex_id)
    return celex_ids


def get_existing_legal_acts(legal_acts_dir: Path) -> set[str]:
    """Get set of CELEX IDs that exist as HTML files."""
    existing = set()
    if not legal_acts_dir.exists():
        return existing
    
    for html_file in legal_acts_dir.glob("*.html"):
        celex_id = html_file.stem  # filename without extension
        existing.add(celex_id)
    
    return existing


def analyze_missing_legal_acts(
    celex_file: Path, legal_acts_dir: Path
) -> tuple[dict[int, int], dict[int, int]]:
    """Analyze missing legal acts and return frequency by year for all and missing."""
    # Load all CELEX IDs
    all_celex_ids = load_celex_ids(celex_file)
    print(f"Total CELEX IDs in file: {len(all_celex_ids)}")
    
    # Get existing legal acts
    existing_celex_ids = get_existing_legal_acts(legal_acts_dir)
    print(f"Existing legal acts: {len(existing_celex_ids)}")
    
    # Find missing
    missing_celex_ids = all_celex_ids - existing_celex_ids
    print(f"Missing legal acts: {len(missing_celex_ids)}")
    
    # Extract years and count frequencies
    all_years = []
    missing_years = []
    
    # Filter valid years (1950-2025)
    MIN_YEAR = 1950
    MAX_YEAR = 2025
    
    for celex_id in all_celex_ids:
        year = extract_year_from_celex(celex_id)
        if year and MIN_YEAR <= year <= MAX_YEAR:
            all_years.append(year)
            if celex_id in missing_celex_ids:
                missing_years.append(year)
    
    all_freq = dict(Counter(all_years))
    missing_freq = dict(Counter(missing_years))
    
    return all_freq, missing_freq


def plot_frequencies(
    all_freq: dict[int, int], missing_freq: dict[int, int], output_path: Path
) -> None:
    """Create visualization of frequencies over time."""
    # Get all years and sort
    all_years = sorted(set(list(all_freq.keys()) + list(missing_freq.keys())))
    
    # Prepare data
    all_counts = [all_freq.get(year, 0) for year in all_years]
    missing_counts = [missing_freq.get(year, 0) for year in all_years]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All legal acts frequency
    ax1.plot(all_years, all_counts, marker='o', linewidth=2, markersize=4, color='#2E86AB', label='All Legal Acts')
    ax1.fill_between(all_years, all_counts, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Frequency of All Legal Acts Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Missing legal acts frequency
    ax2.plot(all_years, missing_counts, marker='o', linewidth=2, markersize=4, color='#A23B72', label='Missing Legal Acts')
    ax2.fill_between(all_years, missing_counts, alpha=0.3, color='#A23B72')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Frequency of Missing Legal Acts Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format x-axis to show years properly
    for ax in [ax1, ax2]:
        ax.set_xlim(min(all_years) - 1, max(all_years) + 1)
        # Rotate x-axis labels if many years
        if len(all_years) > 20:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also create a combined plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(14, 6))
    ax3.plot(all_years, all_counts, marker='o', linewidth=2, markersize=4, color='#2E86AB', label='All Legal Acts', alpha=0.8)
    ax3.plot(all_years, missing_counts, marker='s', linewidth=2, markersize=4, color='#A23B72', label='Missing Legal Acts', alpha=0.8)
    ax3.fill_between(all_years, all_counts, alpha=0.2, color='#2E86AB')
    ax3.fill_between(all_years, missing_counts, alpha=0.2, color='#A23B72')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Frequency of All Legal Acts vs Missing Legal Acts Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(min(all_years) - 1, max(all_years) + 1)
    if len(all_years) > 20:
        ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    combined_path = output_path.parent / f"{output_path.stem}_combined{output_path.suffix}"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined visualization saved to: {combined_path}")


def print_statistics(all_freq: dict[int, int], missing_freq: dict[int, int]) -> None:
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_all = sum(all_freq.values())
    total_missing = sum(missing_freq.values())
    
    print(f"\nTotal legal acts: {total_all}")
    print(f"Total missing legal acts: {total_missing}")
    print(f"Percentage missing: {(total_missing / total_all * 100):.2f}%")
    
    if all_freq:
        print(f"\nYear range: {min(all_freq.keys())} - {max(all_freq.keys())}")
        print(f"Year with most legal acts: {max(all_freq.items(), key=lambda x: x[1])}")
        if missing_freq:
            print(f"Year with most missing acts: {max(missing_freq.items(), key=lambda x: x[1])}")
    
    print("\nTop 10 years by total legal acts:")
    sorted_all = sorted(all_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    for year, count in sorted_all:
        missing = missing_freq.get(year, 0)
        print(f"  {year}: {count} total ({missing} missing, {count - missing} present)")


if __name__ == "__main__":
    # Define paths
    workspace = Path(__file__).parent.parent
    celex_file = workspace / "artifacts" / "regulations_directives_celex.txt"
    legal_acts_dir = workspace / "legal_acts"
    output_dir = workspace / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "legal_acts_frequency_analysis.png"
    
    print("Analyzing missing legal acts...")
    all_freq, missing_freq = analyze_missing_legal_acts(celex_file, legal_acts_dir)
    
    print_statistics(all_freq, missing_freq)
    
    print("\nGenerating visualizations...")
    plot_frequencies(all_freq, missing_freq, output_path)
    
    print("\nAnalysis complete!")

