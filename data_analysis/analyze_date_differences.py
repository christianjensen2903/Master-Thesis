import json
from datetime import datetime
from collections import Counter
import statistics
from pathlib import Path


def analyze_date_differences(json_path: str) -> None:
    """Analyze the distribution of differences between application_date and date."""
    print(f"Loading {json_path}...")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    differences_days: list[int] = []
    missing_application_date = 0
    missing_date = 0
    invalid_dates = 0
    total_entries = len(data)

    print(f"Processing {total_entries} entries...")

    for celex, case in data.items():
        meta = case.get("meta", {})
        application_date_str = meta.get("application_date")
        date_str = meta.get("date")

        if not application_date_str:
            missing_application_date += 1
            continue

        if not date_str:
            missing_date += 1
            continue

        try:
            app_date = datetime.strptime(application_date_str, "%Y-%m-%d")
            case_date = datetime.strptime(date_str, "%Y-%m-%d")

            diff_days = (case_date - app_date).days

            if diff_days < 0:
                print(f"Warning: Negative difference for {celex}: {diff_days} days")

            differences_days.append(diff_days)
        except (ValueError, TypeError) as e:
            invalid_dates += 1
            continue

    print(f"\n=== Summary ===")
    print(f"Total entries: {total_entries}")
    print(f"Valid date pairs: {len(differences_days)}")
    print(f"Missing application_date: {missing_application_date}")
    print(f"Missing date: {missing_date}")
    print(f"Invalid dates: {invalid_dates}")

    if not differences_days:
        print("No valid date differences found!")
        return

    print(f"\n=== Distribution Statistics (in days) ===")
    print(f"Mean: {statistics.mean(differences_days):.2f}")
    print(f"Median: {statistics.median(differences_days):.2f}")
    print(f"Mode: {statistics.mode(differences_days)}")
    print(f"Min: {min(differences_days)}")
    print(f"Max: {max(differences_days)}")
    print(f"Standard deviation: {statistics.stdev(differences_days):.2f}")

    # Percentiles
    sorted_diffs = sorted(differences_days)
    print(f"\n=== Percentiles ===")
    print(f"25th percentile: {sorted_diffs[len(sorted_diffs) // 4]}")
    print(f"50th percentile (median): {sorted_diffs[len(sorted_diffs) // 2]}")
    print(f"75th percentile: {sorted_diffs[3 * len(sorted_diffs) // 4]}")
    print(f"90th percentile: {sorted_diffs[9 * len(sorted_diffs) // 10]}")
    print(f"95th percentile: {sorted_diffs[19 * len(sorted_diffs) // 20]}")
    print(f"99th percentile: {sorted_diffs[99 * len(sorted_diffs) // 100]}")

    # Distribution by ranges
    print(f"\n=== Distribution by Ranges ===")
    ranges = [
        (0, 30, "0-30 days"),
        (31, 90, "31-90 days"),
        (91, 180, "91-180 days"),
        (181, 365, "181-365 days"),
        (366, 730, "1-2 years"),
        (731, 1095, "2-3 years"),
        (1096, 1825, "3-5 years"),
        (1826, float("inf"), ">5 years"),
    ]

    for min_days, max_days, label in ranges:
        count = sum(1 for d in differences_days if min_days <= d <= max_days)
        percentage = (count / len(differences_days)) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

    # Top 10 most common differences
    print(f"\n=== Top 10 Most Common Differences (days) ===")
    diff_counter = Counter(differences_days)
    for days, count in diff_counter.most_common(10):
        percentage = (count / len(differences_days)) * 100
        print(f"{days} days: {count} occurrences ({percentage:.2f}%)")


if __name__ == "__main__":
    json_path = Path(__file__).parent / "data" / "par-to-par-new2.json"
    analyze_date_differences(str(json_path))
