import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import aiofiles  # type: ignore
import aiohttp
import dotenv
from fake_useragent import UserAgent


dotenv.load_dotenv()


ua = UserAgent()


# The EUR-Lex summary page for a case is available by appending "_SUM" to the CELEX id
# Example: https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:61990CJ0003_SUM
BASE_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{case_id}_SUM"
)

# Basic retry/backoff settings (tunable)
MAX_RETRIES = 5
INITIAL_BACKOFF = 0.5  # seconds
MAX_BACKOFF = 10.0

proxy = os.getenv("PROXY")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the summary scraper.

    Returns
    -------
    argparse.Namespace
        The parsed arguments namespace containing input path, output directory, and runtime options.
    """
    parser = argparse.ArgumentParser(
        description='Concurrent EUR-Lex summary ("_SUM") HTML downloader with proxy/user-agent rotation.'
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input JSON mapping case_id -> case_data (values unused)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("summaries"),
        help="Directory to write summary .html files (default: summaries)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on number of cases to fetch (0 = no limit)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Base seconds to sleep between requests per task (can be randomized)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--randomize-delay",
        action="store_true",
        help="Randomize per-request delay (adds jitter)",
    )
    return parser.parse_args()


async def save_bytes(path: Path, data: bytes) -> None:
    """Persist raw bytes to disk asynchronously.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : bytes
        Raw bytes to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)


async def fetch_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
) -> bytes:
    """Fetch a URL with retry and exponential backoff.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Active HTTP session.
    url : str
        URL to fetch.
    headers : dict
        Request headers to include.

    Returns
    -------
    bytes
        Response content as bytes.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(
                url,
                headers=headers,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                content = await resp.read()
                return content
        except (aiohttp.ClientResponseError,) as e:
            status = getattr(e, "status", None)
            # For 4xx errors (except maybe 429) don't retry many times
            if status and 400 <= status < 500 and status != 429:
                raise
            msg = f"HTTP {status}" if status else str(e)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - broad to implement backoff policy
            msg = str(e)

        # If we get here, we will retry (unless last attempt)
        if attempt == MAX_RETRIES:
            raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {msg}")
        # jittered backoff
        jitter = random.uniform(0, backoff * 0.3)
        sleep_for = min(MAX_BACKOFF, backoff) + jitter
        await asyncio.sleep(sleep_for)
        backoff *= 2  # exponential


async def worker(
    name: int,
    queue: asyncio.Queue,
    out_dir: Path,
    session: aiohttp.ClientSession,
    delay: float,
    randomize_delay: bool,
    total: int,
) -> None:
    """Worker coroutine that dequeues case ids and downloads their summary HTML.

    Parameters
    ----------
    name : int
        Worker identifier for logging.
    queue : asyncio.Queue
        Queue of tuples (index, case_id) to process.
    out_dir : Path
        Output directory for saved HTML files.
    session : aiohttp.ClientSession
        Shared HTTP session.
    delay : float
        Base delay between requests per worker.
    randomize_delay : bool
        Whether to add jitter to the delay.
    total : int
        Total number of items to process (for progress logging).
    """
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, case_id = item
        url = BASE_URL.format(case_id=case_id)
        headers = {"User-Agent": ua.random, "Accept": "text/html,application/xhtml+xml"}
        try:
            content = await fetch_with_retries(session, url, headers)
            out_path = out_dir / f"{case_id}.html"
            await save_bytes(out_path, content)
            print(f"[{idx}/{total}] Worker-{name} saved {case_id} -> {out_path}")
        except Exception as e:  # noqa: BLE001 - log and continue to next item
            print(f"[{idx}/{total}] Worker-{name} error for {case_id}: {e}")
        finally:
            # polite pacing per worker
            sleep_time = delay
            if randomize_delay and delay > 0:
                sleep_time = random.uniform(0, delay * 2)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            queue.task_done()


async def main_async(args: argparse.Namespace) -> None:
    """Entrypoint for the asynchronous download pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    data = json.loads(args.input.read_text(encoding="utf-8"))
    case_items = list(data.keys())
    if args.limit and args.limit > 0:
        case_items = case_items[: args.limit]
    total = len(case_items)
    if total == 0:
        print("No cases found.")
        return

    conn = aiohttp.TCPConnector(
        limit=args.concurrency * 2, ttl_dns_cache=300, force_close=False
    )
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        queue: asyncio.Queue = asyncio.Queue()
        for idx, case_id in enumerate(case_items, 1):
            queue.put_nowait((idx, case_id))

        workers = []
        for n in range(args.concurrency):
            workers.append(
                asyncio.create_task(
                    worker(
                        n + 1,
                        queue,
                        out_dir,
                        session,
                        args.delay,
                        args.randomize_delay,
                        total,
                    )
                )
            )

        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)


def main() -> None:
    """CLI entrypoint for the summary scraper."""
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
