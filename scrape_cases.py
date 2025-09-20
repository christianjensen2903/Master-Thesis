import argparse
import json
import time
from pathlib import Path

import requests


BASE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{case_id}"

HEADERS = {"User-Agent": "CaseHTMLFetcher/1.0 (+contact@example.com)"}


def fetch_html_bytes(case_id: str) -> bytes:
    url = BASE_URL.format(case_id=case_id)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.content  # save bytes to avoid encoding issues


def save_html(case_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.html"
    html = fetch_html_bytes(case_id)
    out_path.write_bytes(html)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and save raw EUR-Lex HTML pages for case IDs in a JSON file."
    )
    p.add_argument(
        "input", type=Path, help="Path to input JSON mapping case_id -> case_data"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("html"),
        help="Directory to write .html files (default: ./html)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of cases to fetch (0 = no limit)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between requests (default: 0.5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    case_items = list(data.items())
    if args.limit and args.limit > 0:
        case_items = case_items[: args.limit]

    for i, (case_id, _case_data) in enumerate(case_items, 1):
        try:
            out_path = save_html(case_id, args.out_dir)
            print(f"[{i}/{len(case_items)}] Saved {case_id} -> {out_path}")
        except requests.HTTPError as e:
            print(f"[{i}/{len(case_items)}] HTTP error for {case_id}: {e}")
        except requests.RequestException as e:
            print(f"[{i}/{len(case_items)}] Request error for {case_id}: {e}")
        except Exception as e:
            print(f"[{i}/{len(case_items)}] Unexpected error for {case_id}: {e}")

        # polite pacing
        if args.delay > 0:
            time.sleep(args.delay)

    print("Done.")


if __name__ == "__main__":
    main()
