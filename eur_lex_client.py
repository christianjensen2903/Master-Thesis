from __future__ import annotations

import httpx
from dataclasses import dataclass
from typing import Final


SOAP_NAMESPACE: Final[str] = "http://www.w3.org/2003/05/soap-envelope"
SEARCH_NAMESPACE: Final[str] = "http://eur-lex.europa.eu/search"
WSSE_NAMESPACE: Final[str] = (
    "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
)
WSU_NAMESPACE: Final[str] = (
    "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
)


@dataclass
class EurLexClient:
    username: str
    password: str
    timeout_seconds: float = 60.0

    def _build_soap_envelope(
        self, *, query: str, page: int, page_size: int, language: str
    ) -> str:
        """Return SOAP 1.2 envelope string with WS-Security UsernameToken and expert query CDATA."""
        envelope = f"""
<soap:Envelope xmlns:soap="{SOAP_NAMESPACE}" xmlns:sear="{SEARCH_NAMESPACE}">
  <soap:Header>
    <wsse:Security xmlns:wsse="{WSSE_NAMESPACE}" xmlns:wsu="{WSU_NAMESPACE}" soap:mustUnderstand="1">
      <wsse:UsernameToken wsu:Id="UsernameToken-1">
        <wsse:Username>{self.username}</wsse:Username>
        <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText">{self.password}</wsse:Password>
      </wsse:UsernameToken>
    </wsse:Security>
  </soap:Header>
  <soap:Body>
    <sear:searchRequest>
      <sear:expertQuery><![CDATA[{query}]]></sear:expertQuery>
      <sear:page>{page}</sear:page>
      <sear:pageSize>{page_size}</sear:pageSize>
      <sear:searchLanguage>{language}</sear:searchLanguage>
    </sear:searchRequest>
  </soap:Body>
</soap:Envelope>
""".strip()
        return envelope

    def search(
        self, *, query: str, page: int = 1, page_size: int = 10, language: str = "en"
    ) -> str:
        """Send expert query to EUR-Lex SOAP API and return raw XML response as string."""
        payload = self._build_soap_envelope(
            query=query, page=page, page_size=page_size, language=language
        )

        headers = {
            "Content-Type": 'application/soap+xml; charset=utf-8; action="https://eur-lex.europa.eu/ws/doQuery"',
            "Accept": "application/soap+xml",
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                "https://eur-lex.europa.eu/EURLexWebService",
                content=payload.encode("utf-8"),
                headers=headers,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                raise RuntimeError(
                    f"EUR-Lex HTTP {response.status_code}: {response.text}"
                ) from err
            return response.text
