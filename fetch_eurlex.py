from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from eur_lex_client import EurLexClient
from eur_lex_rdf import (
    parse_celex_ids_from_search_xml,
    fetch_rdf_for_celex,
    save_rdf_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch EUR-Lex SOAP search results as XML"
    )
    parser.add_argument(
        "--query",
        required=False,
        default="FM_CODED = JUDG AND DTS_SUBDOM = EU_CASE_LAW AND DTS = 6 AND DTA = 2023",
        help="EUR-Lex expert query (default: the user's provided query)",
    )
    parser.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    parser.add_argument(
        "--page-size", type=int, default=10, help="Page size (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eurlex_rdf"),
        help="Output directory to save the RDF files",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()

    username = os.environ.get("EUR_LEX_USERNAME")
    password = os.environ.get("EUR_LEX_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Missing EUR_LEX_USERNAME or EUR_LEX_PASSWORD in environment/.env"
        )

    args = parse_args()

    client = EurLexClient(
        username=username,
        password=password,
    )

    xml_response = client.search(
        query=args.query,
        page=args.page,
        page_size=args.page_size,
        language="en",
    )

    celex_ids = parse_celex_ids_from_search_xml(xml_response)
    print(f"Found {len(celex_ids)} CELEX ids in search results")
    for celex in celex_ids:
        try:
            rdf_bytes = fetch_rdf_for_celex(celex=celex, language="EN")
            out_path = save_rdf_bytes(
                data=rdf_bytes,
                out_dir=args.output,
                celex=celex,
            )
            print(f"Saved RDF for {celex} -> {out_path}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed to fetch RDF for {celex}: {e}")


if __name__ == "__main__":
    main()
