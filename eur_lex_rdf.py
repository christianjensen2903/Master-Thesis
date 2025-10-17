from __future__ import annotations

import httpx
from pathlib import Path
from typing import Final
import xml.etree.ElementTree as ET
from urllib.parse import quote


CELEX_RDF_BASE: Final[str] = "https://publications.europa.eu/resource/celex/"


def _local(tag: str) -> str:
    """Return the local name of an XML tag (strip any namespace)."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_celex_ids_from_search_xml(xml_text: str) -> list[str]:
    """Extract CELEX identifiers from a EUR-Lex SOAP search XML response, ignoring namespaces."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as err:
        raise ValueError(f"Invalid XML provided: {err}") from err

    celex_ids: list[str] = []

    for elem in root.iter():
        if _local(elem.tag) == "ID_CELEX":
            # Find its VALUE child regardless of namespace
            celex_value = None
            for child in elem:
                if _local(child.tag) == "VALUE" and child.text:
                    celex_value = child.text.strip()
                    break
            if celex_value and celex_value not in celex_ids:
                celex_ids.append(celex_value)

    return celex_ids


def build_celex_rdf_url(celex: str, language: str | None = None) -> str:
    # URL encode the CELEX ID to handle special characters like parentheses
    encoded_celex = quote(celex, safe="")
    base = f"{CELEX_RDF_BASE}{encoded_celex}"
    if language:
        return f"{base}?language={language}"
    return base


def fetch_rdf_for_celex(
    *,
    celex: str,
    accept: str = "application/rdf+xml",
    timeout_seconds: float = 60.0,
    language: str | None = None,
) -> bytes:
    url = build_celex_rdf_url(celex, language=language)
    headers = {"Accept": accept}
    # Follow 303 See Other redirects served by Publications Office for content negotiation
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise RuntimeError(
                f"Publications RDF HTTP {response.status_code} for {celex}: {response.text}"
            ) from err
        return response.content


def save_rdf_bytes(*, data: bytes, out_dir: Path, celex: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{celex}.rdf"
    out_path.write_bytes(data)
    return out_path
