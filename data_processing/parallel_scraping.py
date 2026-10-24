import asyncio
from pathlib import Path
from typing import Iterable
import dotenv
import os
import random

from tqdm import tqdm  # type: ignore

from case_scraper import CaseScraper


async def _bounded_scrape(
    semaphore: asyncio.Semaphore,
    scraper: CaseScraper,
    case_id: str,
    out_dir: Path,
    base_delay: float,
    jitter: float,
) -> tuple[str, bool, str]:
    async with semaphore:
        try:
            # Add a randomized delay before each request to reduce burstiness
            if base_delay > 0.0 or jitter > 0.0:
                sleep_seconds = max(0.0, base_delay + random.uniform(0.0, jitter))
                await asyncio.sleep(sleep_seconds)
            await scraper.scrape_case(case_id=case_id, out_dir=out_dir)
            return case_id, True, ""
        except Exception as exc:  # noqa: BLE001
            return case_id, False, str(exc)


async def scrape_many_async(
    case_ids: Iterable[str],
    output_dir: Path,
    max_concurrency: int = 32,
    proxy: str | None = None,
    base_delay: float = 0.2,
    jitter: float = 0.3,
) -> None:
    scraper = CaseScraper(proxy=proxy)
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)

    successful = 0
    failed = 0

    tasks = [
        asyncio.create_task(
            _bounded_scrape(semaphore, scraper, case_id, output_dir, base_delay, jitter)
        )
        for case_id in case_ids
    ]
    with tqdm(
        total=len(tasks), desc="Scraping cases (async)", dynamic_ncols=True
    ) as pbar:
        for coro in asyncio.as_completed(tasks):
            _, ok, _err = await coro
            if ok:
                successful += 1
            else:
                failed += 1
            pbar.set_postfix(success=successful, failed=failed)
            pbar.update(1)


def main() -> None:
    # Optional CLI arg: path to a file with one CELEX id per line.
    # Defaults to the cited-cases list produced by collect_cited_celexes.py.
    import sys

    celex_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cited_cases_to_scrape.txt")
    output_dir = Path("judgments")
    max_concurrency = 1
    base_delay = 0.05  # seconds
    jitter = 0.15  # additional random delay in [0, jitter]

    dotenv.load_dotenv()
    # Proxy disabled for now — scrape directly.
    proxy = None

    with celex_file.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    # Skip cases whose folder already exists AND is non-empty. scrape_case
    # creates the folder before fetching, so a fully-failed case (WAF/timeout)
    # leaves an empty folder behind; skipping only non-empty folders lets those
    # be retried on a re-run instead of being locked in as permanently "done".
    def _already_done(cid: str) -> bool:
        d = output_dir / cid
        return d.exists() and any(d.iterdir())

    before = len(ids)
    ids = [cid for cid in ids if not _already_done(cid)]
    skipped = before - len(ids)
    print(f"Skipping {skipped} already-scraped cases; {len(ids)} remaining.")

    if not ids:
        return

    asyncio.run(
        scrape_many_async(
            ids,
            output_dir,
            max_concurrency=max_concurrency,
            proxy=proxy,
            base_delay=base_delay,
            jitter=jitter,
        )
    )


if __name__ == "__main__":
    main()
