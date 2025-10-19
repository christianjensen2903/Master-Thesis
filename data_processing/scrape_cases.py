import asyncio
from pathlib import Path
import dotenv
import aiohttp
import aiofiles  # type: ignore
from fake_useragent import UserAgent
import xml.etree.ElementTree as ET
import os

dotenv.load_dotenv()


class CaseScraper:

    def __init__(self, proxy: str | None = None):
        self.ua = UserAgent()
        self.proxy = proxy

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
                proxy=self.proxy,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                content = await resp.read()
                return content, resp.status
        except Exception:
            return None, None

    def _build_headers(self, accept: str) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": accept, "User-Agent": self.ua.random}
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
        """Fetch summary content in English."""
        headers = self._build_headers("text/html,application/xhtml+xml")
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{case_id}_SUM"
        content, status = await self.fetch_single(session, url, headers)
        if status == 404:
            return None

        return content

    async def fetch_metadata(
        self,
        session: aiohttp.ClientSession,
        case_id: str,
    ) -> bytes | None:
        """Fetch metadata content in English."""
        headers = self._build_headers("text/html,application/xhtml+xml,application/xml")
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/XML/?uri=CELEX:{case_id}"
        content, status = await self.fetch_single(session, url, headers)
        if status == 404:
            return None

        return content

    async def fetch_judgment(
        self,
        session: aiohttp.ClientSession,
        case_id: str,
        lang: str,
    ) -> bytes | None:
        """Fetch judgment content for a specific language."""
        headers = self._build_headers("text/html,application/xhtml+xml")
        url = f"https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/?uri=CELEX:{case_id}"
        content, status = await self.fetch_single(session, url, headers)
        if content and status != 404:
            return content
        return None

    async def scrape_case(self, case_id: str, out_dir: Path) -> None:
        conn = aiohttp.TCPConnector(limit=6, ttl_dns_cache=300, force_close=False)
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            out_dir.mkdir(parents=True, exist_ok=True)
            celex_dir = out_dir / case_id
            celex_dir.mkdir(parents=True, exist_ok=True)

            summary_content = await self.fetch_summary(session, case_id)
            if not summary_content:
                print(f"Skipped {case_id}/summary.html - no valid content found")
                return

            await self.save_bytes(celex_dir / "summary.html", summary_content)

            metadata_content = await self.fetch_metadata(session, case_id)
            if not metadata_content:
                print(f"Skipped {case_id}/metadata.xml - no valid content found")
                return

            await self.save_bytes(celex_dir / "metadata.xml", metadata_content)

            authentic_lang = self.extract_authentic_language(metadata_content)

            languages_to_try = set(["eng", "fra"])
            if authentic_lang:
                languages_to_try.add(authentic_lang.lower())

            for lang in languages_to_try:
                content = await self.fetch_judgment(session, case_id, lang)
                if not content:
                    continue

                await self.save_bytes(celex_dir / f"{lang}_judgment.html", content)


def main() -> None:
    # proxy = os.getenv("PROXY")
    proxy = None
    scraper = CaseScraper(proxy=proxy)
    try:
        asyncio.run(scraper.scrape_case(case_id="62014CJ0005", out_dir=Path("html")))
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
