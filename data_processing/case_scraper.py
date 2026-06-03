import asyncio
import random
from pathlib import Path
import dotenv
import aiohttp
import aiofiles  # type: ignore
from fake_useragent import UserAgent
import xml.etree.ElementTree as ET
import os

dotenv.load_dotenv()

cookies: dict[str, str] = {}

# EUR-Lex content URLs use ISO 639-1 (2-letter, uppercase) codes,
# not the 3-letter codes used as our on-disk filename prefixes.
_URL_LANG = {
    "eng": "EN",
    "fra": "FR",
    "deu": "DE",
    "spa": "ES",
    "ita": "IT",
    "nld": "NL",
    "por": "PT",
    "ell": "EL",
    "dan": "DA",
    "swe": "SV",
    "fin": "FI",
    "pol": "PL",
    "ces": "CS",
    "slk": "SK",
    "slv": "SL",
    "hun": "HU",
    "ron": "RO",
    "bul": "BG",
    "hrv": "HR",
    "est": "ET",
    "lav": "LV",
    "lit": "LT",
    "mlt": "MT",
    "gle": "GA",
}


def _url_lang(lang: str) -> str:
    """Map a 3-letter ISO 639-3 code (or already-2-letter code) to EUR-Lex's URL code."""
    if len(lang) == 2:
        return lang.upper()
    return _URL_LANG.get(lang.lower(), lang.upper())


# Markers that identify EUR-Lex's anti-bot challenge page. The page is served
# with HTTP 200, so we have to recognize it by body content. Any of these
# substrings (case-insensitive) in the response means the WAF caught us and
# the body is NOT real content — discard it so the file stays missing and a
# later re-run can retry from a fresh IP.
_WAF_MARKERS: tuple[bytes, ...] = (
    b"not a robot",
    b"JavaScript is disabled",
)


def _looks_like_waf(content: bytes) -> bool:
    if not content:
        return False
    # Cheap fast-path: the challenge page is tiny (<2 KB). Skip the substring
    # scan on responses that are obviously real content.
    if len(content) > 8192:
        return False
    lowered = content.lower()
    return any(marker.lower() in lowered for marker in _WAF_MARKERS)


class CaseScraper:

    def __init__(self, proxy: str | None = None, proxy_pool_size: int = 20000):
        """
        proxy: a proxy URL. May contain "{n}" as a placeholder; if present, a
               random integer in [1, proxy_pool_size] is substituted per request.
               Example for Webshare rotating residential (per-username sessions):
                   http://USER-{n}:PASS@p.webshare.io:80
        """
        self.ua = UserAgent()
        self.proxy_template = proxy
        self.proxy_pool_size = proxy_pool_size

    def _proxy_url(self) -> str | None:
        if not self.proxy_template:
            return None
        if "{n}" in self.proxy_template:
            n = random.randint(1, self.proxy_pool_size)
            return self.proxy_template.replace("{n}", str(n))
        return self.proxy_template

    async def save_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)

    async def fetch_single(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict,
    ) -> tuple[bytes | None, int | None]:
        try:
            async with session.get(
                url,
                headers=headers,
                cookies=cookies,
                proxy=self._proxy_url(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                content = await resp.read()
                if _looks_like_waf(content):
                    # Treat WAF challenge pages as a failure so they don't
                    # get saved to disk and the case can be retried later.
                    return None, resp.status
                return content, resp.status
        except Exception:
            return None, None

    def _build_headers(self, accept: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": accept,
            "User-Agent": self.ua.random,
            "Referer": "https://eur-lex.europa.eu/",
        }
        return headers

    @staticmethod
    def extract_authentic_language(metadata_content: bytes) -> str | None:
        try:
            root = ET.fromstring(metadata_content)
            found = root.findtext(".//RESOURCE_LEGAL_USES_ORIGINALLY_LANGUAGE/OP-CODE")
            return found if found else None
        except ET.ParseError:
            return None

    async def fetch_summary(
        self,
        session: aiohttp.ClientSession,
        case_id: str,
    ) -> bytes | None:
        """Fetch summary content with fallbacks: EN _SUM, EN _INF, FR _SUM."""
        # TODO: Fetch native if not available in other languages
        headers = self._build_headers("text/html,application/xhtml+xml")
        attempts = [
            ("EN", "_SUM"),
            ("EN", "_INF"),
            ("EN", "_RES"),
            ("FR", "_SUM"),
        ]
        for lang, suffix in attempts:
            url = (
                f"https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/"
                f"?uri=CELEX:{case_id}{suffix}"
            )
            content, status = await self.fetch_single(session, url, headers)
            if content and status != 404:
                return content
        return None

    async def fetch_metadata(
        self,
        session: aiohttp.ClientSession,
        case_id: str,
        lang: str,
    ) -> bytes | None:
        """Fetch metadata content for a specific language."""
        headers = self._build_headers("text/html,application/xhtml+xml,application/xml")
        url = f"https://eur-lex.europa.eu/legal-content/{_url_lang(lang)}/TXT/XML/?uri=CELEX:{case_id}"
        content, status = await self.fetch_single(session, url, headers)
        if content and status != 404:
            return content
        return None

    async def fetch_judgment(
        self,
        session: aiohttp.ClientSession,
        case_id: str,
        lang: str,
    ) -> bytes | None:
        """Fetch judgment content for a specific language."""
        headers = self._build_headers("text/html,application/xhtml+xml")
        url = f"https://eur-lex.europa.eu/legal-content/{_url_lang(lang)}/TXT/HTML/?uri=CELEX:{case_id}"
        content, status = await self.fetch_single(session, url, headers)
        if content and status != 404:
            return content
        return None

    async def scrape_legal_act(self, celex_id: str, out_dir: Path) -> None:
        conn = aiohttp.TCPConnector(limit=6, ttl_dns_cache=300, force_close=False)
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            headers = self._build_headers("text/html,application/xhtml+xml")
            url = f"https://eur-lex.europa.eu/legal-content/en/TXT/HTML/?uri=CELEX:{celex_id}"
            content, status = await self.fetch_single(session, url, headers)
            if content and status != 404:
                await self.save_bytes(out_dir / f"{celex_id}.html", content)
            else:
                print(f"Skipped {celex_id}.html - no valid content found")

    async def scrape_case(self, case_id: str, out_dir: Path) -> None:
        # TODO: Clean function by removing reading of files
        conn = aiohttp.TCPConnector(limit=6, ttl_dns_cache=300, force_close=False)
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            out_dir.mkdir(parents=True, exist_ok=True)
            celex_dir = out_dir / case_id
            celex_dir.mkdir(parents=True, exist_ok=True)

            summary_path = celex_dir / "summary.html"
            if not summary_path.exists():
                summary_content = await self.fetch_summary(session, case_id)
                if not summary_content:
                    print(f"Skipped {case_id}/summary.html - no valid content found")
                else:
                    await self.save_bytes(summary_path, summary_content)

            # Fetch and save both EN and FR metadata
            metadata_content = None
            for lang in ["eng", "fra"]:
                metadata_path = celex_dir / f"{lang.lower()}_metadata.xml"
                if not metadata_path.exists():
                    metadata_content_lang = await self.fetch_metadata(
                        session, case_id, lang
                    )
                    if not metadata_content_lang:
                        print(f"Skipped {metadata_path} - no valid content found")
                    else:
                        await self.save_bytes(metadata_path, metadata_content_lang)
                        metadata_content = metadata_content or metadata_content_lang
                else:
                    if metadata_content is None:
                        with open(metadata_path, "rb") as f:
                            metadata_content = f.read()

            content_found = False
            for lang in ["eng", "fra"]:
                judgment_path = celex_dir / f"{lang}_judgment.html"
                if not judgment_path.exists():
                    content = await self.fetch_judgment(session, case_id, lang)
                    if not content:
                        continue
                    content_found = True
                    await self.save_bytes(judgment_path, content)

            if not content_found:
                if metadata_content:
                    authentic_lang = self.extract_authentic_language(metadata_content)
                    if not authentic_lang:
                        print(
                            f"Skipped {case_id}/judgment.html - no valid content found"
                        )
                        return
                    judgment_path = celex_dir / f"{authentic_lang}_judgment.html"
                    if not judgment_path.exists():
                        content = await self.fetch_judgment(
                            session, case_id, authentic_lang
                        )
                        if not content:
                            print(
                                f"Skipped {case_id}/judgment.html - no valid content found"
                            )
                            return
                        await self.save_bytes(judgment_path, content)
                else:
                    print(
                        f"Skipped {case_id}/judgment.html - no metadata available for authentic language detection"
                    )


def main() -> None:
    # proxy = os.getenv("PROXY")
    proxy = None
    scraper = CaseScraper(proxy=proxy)
    try:
        # asyncio.run(
        #     scraper.scrape_case(case_id="61954CC0001", out_dir=Path("judgments"))
        # )
        asyncio.run(
            scraper.scrape_legal_act(celex_id="31960L0921", out_dir=Path("legal_acts"))
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
