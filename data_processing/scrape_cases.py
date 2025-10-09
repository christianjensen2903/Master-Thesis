import argparse
import asyncio
import json
import random
from pathlib import Path
import dotenv
import os
import aiohttp
import aiofiles  # type: ignore
from fake_useragent import UserAgent


dotenv.load_dotenv()


ua = UserAgent()

BASE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{case_id}"

# Basic retry/backoff settings (tunable)
MAX_RETRIES = 5
INITIAL_BACKOFF = 0.5  # seconds
MAX_BACKOFF = 10.0

proxy = os.getenv("PROXY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concurrent EUR-Lex HTML downloader with proxy/user-agent rotation."
    )
    p.add_argument(
        "input", type=Path, help="Path to input JSON mapping case_id -> case_data"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("html"),
        help="Directory to write .html files",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on number of cases to fetch (0 = no limit)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Base seconds to sleep between requests per task (can be randomized)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    p.add_argument(
        "--user-agents-file",
        type=Path,
        default=None,
        help="Optional file with user-agents (one per line)",
    )
    p.add_argument(
        "--randomize-delay",
        action="store_true",
        help="Randomize per-request delay (adds jitter)",
    )

    return p.parse_args()


async def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)


async def fetch_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
) -> bytes:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Note: proxy can be None
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
            # otherwise fall through to retry
            msg = f"HTTP {status}" if status else str(e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
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
):
    i = 0
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, case_id = item
        i += 1
        url = BASE_URL.format(case_id=case_id)
        headers = {"User-Agent": ua.random, "Accept": "text/html,application/xhtml+xml"}
        try:
            content = await fetch_with_retries(session, url, headers)
            out_path = out_dir / f"{case_id}.html"
            await save_bytes(out_path, content)
            print(f"[{idx}/{total}] Worker-{name} saved {case_id} -> {out_path}")
        except Exception as e:
            print(f"[{idx}/{total}] Worker-{name} error for {case_id}: {e}")
        finally:
            # polite pacing per worker
            sleep_time = delay
            if randomize_delay and delay > 0:
                sleep_time = random.uniform(0, delay * 2)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            queue.task_done()


async def main_async(args: argparse.Namespace):
    data = json.loads(args.input.read_text(encoding="utf-8"))
    case_items = list(data.keys())
    if args.limit and args.limit > 0:
        case_items = case_items[: args.limit]
    total = len(case_items)
    if total == 0:
        print("No cases found.")
        return

    # connection limits: tune connector and session settings
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

        # workers
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

        # wait until done
        await queue.join()
        # stop workers
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
