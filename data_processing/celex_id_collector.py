from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from tqdm import tqdm  # type: ignore

from eur_lex_client import EurLexClient


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_celex_ids_from_search_xml(xml_text: str) -> list[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as err:
        raise ValueError(f"Invalid XML provided: {err}") from err

    celex_ids: list[str] = []
    for elem in root.iter():
        if _local(elem.tag) == "ID_CELEX":
            for child in elem:
                if _local(child.tag) == "VALUE" and child.text:
                    value = child.text.strip()
                    if value and value not in celex_ids:
                        celex_ids.append(value)
                    break
    return celex_ids


class CelexIdCollector:
    """Retrieves all CJ case IDs from EUR-Lex for years 1954-2025."""

    def __init__(self, output_file: str | Path = "cj_cases.txt") -> None:
        load_dotenv()

        username = os.environ.get("EUR_LEX_USERNAME")
        password = os.environ.get("EUR_LEX_PASSWORD")

        if not username or not password:
            raise RuntimeError(
                "Missing EUR_LEX_USERNAME or EUR_LEX_PASSWORD in environment/.env"
            )

        self.client = EurLexClient(username=username, password=password)
        self.output_file = Path(output_file)
        self.page_size: Final[int] = 20

    def retrieve_all_cases(self) -> None:
        all_case_ids: set[str] = set()
        years = range(2022, 2027)  # 2022–2026 inclusive

        # Create progress bar for years
        with tqdm(total=len(years), desc="Processing years", unit="year") as pbar:
            for year in years:
                pbar.set_description(f"Processing year {year}")
                year_case_ids = self._retrieve_cases_for_year(year)
                all_case_ids.update(year_case_ids)
                pbar.set_postfix(
                    cases_found=len(year_case_ids), total_cases=len(all_case_ids)
                )
                pbar.update(1)

        # Save all case IDs to file
        self._save_case_ids_to_file(all_case_ids)
        print(f"Total unique cases retrieved: {len(all_case_ids)}")
        print(f"Case IDs saved to: {self.output_file}")

    def _retrieve_cases_for_year(self, year: int) -> set[str]:
        query = f"FM_CODED = JUDG AND DTS_SUBDOM = EU_CASE_LAW AND DTS = 6 AND DTT = CJ AND DTA = {year}"
        year_case_ids: set[str] = set()
        current_page = 1

        while True:
            try:
                xml_response = self.client.search(
                    query=query,
                    page=current_page,
                    page_size=self.page_size,
                    language="en",
                )

                celex_ids = parse_celex_ids_from_search_xml(xml_response)

                if not celex_ids:
                    break

                # Add new case IDs to the set
                new_ids = [cid for cid in celex_ids if cid not in year_case_ids]
                year_case_ids.update(new_ids)

                # If we got fewer results than page size, we've reached the last page
                if len(celex_ids) < self.page_size:
                    break

                current_page += 1

            except Exception as e:
                print(
                    f"Error retrieving cases for year {year}, page {current_page}: {e}"
                )
                break

        return year_case_ids

    def _save_case_ids_to_file(self, case_ids: set[str]) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        sorted_case_ids = sorted(case_ids)

        with open(self.output_file, "w", encoding="utf-8") as f:
            for case_id in sorted_case_ids:
                f.write(f"{case_id}\n")


def main() -> None:
    """Main function to run the case retriever."""
    retriever = CelexIdCollector(output_file="cj_cases_2022-2026.txt")
    retriever.retrieve_all_cases()


if __name__ == "__main__":
    main()
