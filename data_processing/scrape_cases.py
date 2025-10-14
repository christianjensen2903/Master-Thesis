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
from collections.abc import Iterable


dotenv.load_dotenv()


ua = UserAgent()

BASE_URL_TEMPLATE = (
    "https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/?uri=CELEX:{case_id}"
)

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
    p.add_argument(
        "--languages",
        type=str,
        default=(
            "EN,FR,DE,IT,ES,NL,PL,PT,RO,BG,CS,DA,ET,EL,GA,HR,LV,LT,HU,MT,SK,SL,FI,SV"
        ),
        help=(
            "Comma-separated language codes to try in order (default tries EN then major fallbacks)"
        ),
    )
    p.add_argument(
        "--rescrape-errors",
        action="store_true",
        help=(
            "If set, only re-download items whose existing HTML contains the EUR-Lex 'document does not exist' error"
        ),
    )
    p.add_argument(
        "--scan-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to scan for .html files; if provided, case IDs are taken from filenames. With --rescrape-errors, only files that look like the EUR-Lex error page are queued."
        ),
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
    # If the loop exits without returning, signal failure explicitly
    raise RuntimeError("Exhausted retries without receiving content")


def parse_languages_arg(languages_arg: str) -> list[str]:
    return [lang.strip().upper() for lang in languages_arg.split(",") if lang.strip()]


def build_url(case_id: str, lang: str) -> str:
    return BASE_URL_TEMPLATE.format(lang=lang, case_id=case_id)


def html_is_error_document(content: bytes) -> bool:
    # EUR-Lex uses a stable error container id regardless of UI language
    return (
        b'id="errorDocumentView"' in content
        or b"The requested document does not exist" in content
    )


async def try_languages_for_case(
    session: aiohttp.ClientSession,
    case_id: str,
    languages: Iterable[str],
) -> tuple[bytes | None, str | None]:
    headers_base = {
        "Accept": "text/html,application/xhtml+xml",
    }
    for lang in languages:
        headers = {
            **headers_base,
            "User-Agent": ua.random,
            "Accept-Language": f"{lang.lower()}-*,{lang.lower()};q=0.9,en;q=0.6",
        }
        url = build_url(case_id, lang)
        try:
            content = await fetch_with_retries(session, url, headers)
        except Exception:
            content = b""
        if content and not html_is_error_document(content):
            return content, lang
    return None, None


def discover_case_ids_from_dir(scan_dir: Path, only_errors: bool) -> list[str]:
    case_ids: list[str] = []
    if not scan_dir.exists():
        return case_ids
    for p in sorted(scan_dir.glob("*.html")):
        try:
            if only_errors:
                content = p.read_bytes()
                if not html_is_error_document(content):
                    continue
            case_ids.append(p.stem)
        except Exception:
            # Skip unreadable files silently
            continue
    return case_ids


async def worker(
    name: int,
    queue: asyncio.Queue,
    out_dir: Path,
    session: aiohttp.ClientSession,
    delay: float,
    randomize_delay: bool,
    total: int,
    languages: list[str],
    rescrape_errors: bool,
) -> None:
    i = 0
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, case_id = item
        i += 1
        try:
            out_path = out_dir / f"{case_id}.html"
            if rescrape_errors and out_path.exists():
                try:
                    existing = out_path.read_bytes()
                    if existing and not html_is_error_document(existing):
                        print(
                            f"[{idx}/{total}] Worker-{name} skip {case_id}: existing file is valid"
                        )
                        continue
                except Exception:
                    pass

            content, used_lang = await try_languages_for_case(
                session=session, case_id=case_id, languages=languages
            )
            if content is None:
                raise RuntimeError("No available language produced a valid document")

            await save_bytes(out_path, content)
            lang_note = f" (lang={used_lang})" if used_lang else ""
            print(
                f"[{idx}/{total}] Worker-{name} saved {case_id}{lang_note} -> {out_path}"
            )
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


async def main_async(args: argparse.Namespace) -> None:
    if args.scan_dir is not None:
        case_items = discover_case_ids_from_dir(args.scan_dir, args.rescrape_errors)
    else:
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

        languages = parse_languages_arg(args.languages)

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
                        languages,
                        args.rescrape_errors,
                    )
                )
            )

        # wait until done
        await queue.join()
        # stop workers
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
