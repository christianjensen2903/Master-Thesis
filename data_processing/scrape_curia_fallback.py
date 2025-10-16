import argparse
import asyncio
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import aiohttp
import aiofiles  # type: ignore
from bs4 import BeautifulSoup, Tag  # type: ignore
from fake_useragent import UserAgent  # type: ignore


SEARCH_URL = (
    "https://curia.europa.eu/juris/liste.jsf?language=en&jur=C%2CT%2CF&num={case_no}"
)


ua = UserAgent()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch Curia HTML judgments for CELEX cases that are still invalid after EUR-Lex scraping."
        )
    )
    p.add_argument(
        "input",
        type=Path,
        help=(
            "Path to JSON mapping CELEX -> any, or a text file with one CELEX per line"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cases"),
        help="Directory to write downloaded HTML files as {celex}.html",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of CELEX IDs to process (0 = no cap)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Number of concurrent workers",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Base delay in seconds between requests per worker",
    )
    p.add_argument(
        "--randomize-delay",
        action="store_true",
        help="Add jitter to the per-request delay",
    )
    return p.parse_args()


def load_celex_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # JSON mapping CELEX -> {} or list
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return list(data.keys())
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    # Fallback: newline-delimited CELEX
    return [line.strip() for line in text.splitlines() if line.strip()]


def celex_to_curia_case_number(celex: str) -> str | None:
    """Convert CELEX like 62015TJ0585 -> Curia number like T-585/15.

    Patterns observed:
    - CJ: 6YYYYCJNNNN -> C-NNNN/YY
    - TJ (General Court): 6YYYYTJNNNN -> T-NNNN/YY
    - FJ (Civil Service Tribunal, F): 6YYYYFJNNNN -> F-NNNN/YY
    """
    m = re.match(r"^6(\d{4})(CJ|TJ|FJ)(\d{3,4})$", celex)
    if not m:
        return None
    year_full, court, number = m.groups()
    yy = year_full[-2:]
    # Drop any leading zeros in number for display
    number_int = int(number)
    prefix = {"CJ": "C", "TJ": "T", "FJ": "F"}.get(court)
    if not prefix:
        return None
    return f"{prefix}-{number_int}/{yy}"


def build_search_url(case_no: str) -> str:
    return SEARCH_URL.format(case_no=case_no)


@dataclass
class CuriaResult:
    celex: str
    content: bytes | None
    error: str | None


async def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)


async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    headers = {"User-Agent": ua.random, "Accept": "text/html,application/xhtml+xml"}
    async with session.get(url, headers=headers, allow_redirects=True) as resp:
        resp.raise_for_status()
        return await resp.read()


def _attr_to_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):  # BeautifulSoup may return a list for some attributes
        if not value:
            return None
        return str(value[0])
    return str(value)


def extract_document_link_from_search(html: bytes) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    # Prefer image link with title "View html documents"
    img = soup.find("img", attrs={"title": re.compile(r"View html documents", re.I)})
    if img is not None:
        parent = img.parent
        if isinstance(parent, Tag) and parent.name == "a":
            href_raw = parent.get("href")
            href = _attr_to_str(href_raw)
            if href and "document.jsf" in href:
                return href
    # Fallback: any anchor to document.jsf
    a = soup.find("a", href=re.compile(r"document\.jsf", re.I))
    if isinstance(a, Tag):
        href_any = a.get("href")
        href2 = _attr_to_str(href_any)
        if href2:
            return href2
    return None


async def fetch_curia_document_for_celex(
    session: aiohttp.ClientSession, celex: str
) -> CuriaResult:
    try:
        case_no = celex_to_curia_case_number(celex)
        if not case_no:
            return CuriaResult(
                celex=celex, content=None, error="Unrecognized CELEX format"
            )
        search_url = build_search_url(case_no)
        search_html = await fetch_bytes(session, search_url)
        doc_link = extract_document_link_from_search(search_html)
        if not doc_link:
            return CuriaResult(
                celex=celex, content=None, error="No document link found"
            )
        # Ensure absolute URL
        if doc_link.startswith("/"):
            doc_url = "https://curia.europa.eu" + doc_link
        elif doc_link.startswith("http"):
            doc_url = doc_link
        else:
            doc_url = "https://curia.europa.eu/juris/" + doc_link
        content = await fetch_bytes(session, doc_url)
        return CuriaResult(celex=celex, content=content, error=None)
    except Exception as e:
        return CuriaResult(celex=celex, content=None, error=str(e))


async def worker(
    name: int,
    queue: asyncio.Queue,
    out_dir: Path,
    session: aiohttp.ClientSession,
    delay: float,
    randomize_delay: bool,
    total: int,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, celex = item
        try:
            out_path = out_dir / f"{celex}.html"
            if out_path.exists():
                queue.task_done()
                continue
            res = await fetch_curia_document_for_celex(session, celex)
            if res.content:
                await save_bytes(out_path, res.content)
                print(f"[{idx}/{total}] Worker-{name} saved {celex} -> {out_path}")
            else:
                print(f"[{idx}/{total}] Worker-{name} failed {celex}: {res.error}")
        finally:
            sleep_time = delay
            if randomize_delay and delay > 0:
                sleep_time = random.uniform(0, delay * 2)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            queue.task_done()


async def main_async(args: argparse.Namespace) -> None:
    celex_ids = load_celex_ids(args.input)
    if args.limit and args.limit > 0:
        celex_ids = celex_ids[: args.limit]
    total = len(celex_ids)
    if total == 0:
        print("No CELEX IDs to process.")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        queue: asyncio.Queue = asyncio.Queue()
        for idx, celex in enumerate(celex_ids, 1):
            queue.put_nowait((idx, celex))

        workers = [
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
            for n in range(args.concurrency)
        ]

        await queue.join()
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
