from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from eur_lex_client import EurLexClient
from eur_lex_rdf import (
    parse_celex_ids_from_search_xml,
    fetch_rdf_for_celex,
    save_rdf_bytes,
)


# TODO: FM_CODED = JUDG AND DD >= 20170101 ORDER BY DD ASC
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch EUR-Lex SOAP search results as XML"
    )
    parser.add_argument(
        "--query",
        required=False,
        default="FM_CODED = JUDG",
        help="EUR-Lex expert query (default: the user's provided query)",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Starting page number (default: 1)",
    )
    parser.add_argument(
        "--page-size", type=int, default=20, help="Page size (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eurlex_rdf"),
        help="Output directory to save the RDF files",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Optional cap on number of pages to fetch (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between page requests (default: 0.0)",
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = EurLexClient(
        username=username,
        password=password,
    )

    total_saved: int = 0
    seen_celex_ids: set[str] = set()
    current_page: int = args.page
    pages_fetched: int = 0

    while True:
        logging.info("Requesting page %s (page_size=%s)", current_page, args.page_size)
        xml_response = client.search(
            query=args.query,
            page=current_page,
            page_size=args.page_size,
            language="en",
        )

        celex_ids = parse_celex_ids_from_search_xml(xml_response)
        if not celex_ids:
            logging.info("No CELEX ids returned on page %s. Stopping.", current_page)
            break

        new_ids = [cid for cid in celex_ids if cid not in seen_celex_ids]
        logging.info(
            "Found %s CELEX ids on page %s (%s new, %s duplicates)",
            len(celex_ids),
            current_page,
            len(new_ids),
            len(celex_ids) - len(new_ids),
        )
        for celex in new_ids:
            try:
                rdf_bytes = fetch_rdf_for_celex(celex=celex, language="EN")
                out_path = save_rdf_bytes(
                    data=rdf_bytes,
                    out_dir=args.output,
                    celex=celex,
                )
                total_saved += 1
                logging.info("Saved RDF for %s -> %s", celex, out_path)
            except Exception as e:  # noqa: BLE001
                logging.error("Failed to fetch RDF for %s: %s", celex, e)
            finally:
                seen_celex_ids.add(celex)

        pages_fetched += 1
        # Stop if fewer than a full page of results were returned (likely last page)
        if len(celex_ids) < args.page_size:
            logging.info(
                "Last page reached at page %s (returned %s < page_size)",
                current_page,
                len(celex_ids),
            )
            break

        # Respect optional page cap
        if args.max_pages and pages_fetched >= args.max_pages:
            logging.info("Reached max pages limit: %s", args.max_pages)
            break

        current_page += 1
        if args.delay > 0:
            time.sleep(args.delay)

    logging.info("Total RDF files saved: %s", total_saved)


if __name__ == "__main__":
    main()
