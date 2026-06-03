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
    celex_file = Path("cj_cases_2022-2026.txt")
    output_dir = Path("judgments")
    max_concurrency = 8
    base_delay = 0.2  # seconds
    jitter = 0.3  # additional random delay in [0, jitter]

    dotenv.load_dotenv()
    proxy = os.getenv("PROXY")

    with celex_file.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    # Skip cases whose folder already exists — assume they're done.
    # Delete a case folder (or specific files inside it) to force re-scraping.
    before = len(ids)
    ids = [cid for cid in ids if not (output_dir / cid).exists()]
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
