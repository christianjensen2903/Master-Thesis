import argparse
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag
from typing import Any
import requests


def load_html(case_id: str) -> BeautifulSoup | None:
    url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{case_id}"
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_sections_from_soup(soup: BeautifulSoup) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # Find h2 headings and extract the text until the next h2
    for h2 in soup.find_all("h2"):
        em = h2.find_next("em")
        if em:
            key = h2.get_text(strip=True).lower().replace(" ", "_")
            results[key] = em.get_text(strip=True, separator="\n") if em else ""
    return results


def enrich_case(case_id: str, case_data: dict) -> dict:
    meta = case_data.get("meta", {}) or {}

    # if "EN" not in files:
    #     # skip non-English cases
    #     return case_data

    soup = load_html(case_id)

    if not soup:
        print(f"Failed to load HTML for case {case_id}")
        return case_data

    sections = extract_sections_from_soup(soup)
    meta["sections"] = sections

    case_data["meta"] = meta
    return case_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich cases by extracting labeled sections from HTML files (EN only)."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input JSON file containing cases",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write the enriched JSON output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = args.input
    out_path = args.output

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated = {}
    for case_id, case_data in list(data.items())[:10]:
        updated[case_id] = enrich_case(case_id, case_data)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=4)

    print(f"Done. Wrote enriched data to {out_path}")


if __name__ == "__main__":
    main()
