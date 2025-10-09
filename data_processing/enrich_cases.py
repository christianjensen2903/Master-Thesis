import argparse
import json
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Any
from tqdm import tqdm


def load_html(case_id: str) -> BeautifulSoup | None:
    file_path = Path(f"html/{case_id}.html")
    if not file_path.exists():
        print(f"HTML file for case {case_id} not found.")
        return None

    with file_path.open("r", encoding="utf-8") as file:
        html_content = file.read()

    return BeautifulSoup(html_content, "html.parser")


def extract_sections_from_soup(soup: BeautifulSoup) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # Find h2 headings and extract the text until the next h2
    for h2 in soup.find_all("h2"):
        em = h2.find_next("em")
        if em:
            key = h2.get_text(strip=True).lower().replace(" ", "_")
            results[key] = em.get_text(strip=True, separator="\n") if em else ""
    return results


def extract_keywords(soup: BeautifulSoup) -> str:
    # Find p tag with class "index"
    p = soup.find("p", class_="index")
    if p:
        return p.get_text(strip=True, separator="\n") if p else ""
    return ""


def enrich_case(case_id: str, case_data: dict) -> dict:
    meta = case_data.get("meta", {}) or {}

    soup = load_html(case_id)

    if not soup:
        print(f"Failed to load HTML for case {case_id}")
        return case_data

    sections = extract_sections_from_soup(soup)
    # if "keywords" not in sections:
    #     sections["keywords"] = extract_keywords(soup)
    meta["sections"] = sections

    # print(sections)

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
    for case_id, case_data in tqdm(list(data.items()), desc="Enriching cases"):
        updated[case_id] = enrich_case(case_id, case_data)
    # enrich_case("62014CJ0526", data["62014CJ0526"])

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=4)

    print(f"Done. Wrote enriched data to {out_path}")


if __name__ == "__main__":
    main()
