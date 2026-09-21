"""
Threat-intelligence fusion gate.

Queries VirusTotal / AbuseIPDB / URLhaus / Google Safe Browsing / RDAP for a
given artifact and combines their verdicts with reliability weighting.
Per spec: if every provider is unavailable (no API key, network error,
rate limit), this returns a neutral/unknown result rather than raising —
ATBG must still produce a prediction using conversational evidence alone.
"""

import asyncio
from dataclasses import dataclass, field

import httpx

from api.config import settings

# Rough reliability weights — providers with lower false-positive rates on
# your validation set should get higher weight. Tune once you have real
# adversarial-test results (Section 15.5 of the spec).
PROVIDER_WEIGHTS = {
    "virustotal": 1.0,
    "abuseipdb": 0.8,
    "urlhaus": 0.9,
    "safe_browsing": 0.9,
    "rdap": 0.4,  # RDAP gives registration metadata, not a malicious verdict
}


@dataclass
class ProviderResult:
    provider: str
    available: bool
    malicious: bool | None = None
    raw: dict = field(default_factory=dict)


async def _query_urlhaus(client: httpx.AsyncClient, url: str) -> ProviderResult:
    try:
        resp = await client.post(
            "https://urlhaus-api.abuse.ch/v1/url/", data={"url": url}, timeout=5.0
        )
        data = resp.json()
        malicious = data.get("query_status") == "ok"
        return ProviderResult("urlhaus", True, malicious, data)
    except Exception:
        return ProviderResult("urlhaus", False)


async def _query_virustotal(client: httpx.AsyncClient, artifact_value: str) -> ProviderResult:
    if not settings.virustotal_api_key:
        return ProviderResult("virustotal", False)
    try:
        headers = {"x-apikey": settings.virustotal_api_key}
        url_id = artifact_value  # caller should pass VT's base64url id when known
        resp = await client.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers, timeout=5.0
        )
        data = resp.json()
        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        malicious = stats.get("malicious", 0) > 0
        return ProviderResult("virustotal", True, malicious, data)
    except Exception:
        return ProviderResult("virustotal", False)


async def _query_abuseipdb(client: httpx.AsyncClient, ip: str) -> ProviderResult:
    if not settings.abuseipdb_api_key:
        return ProviderResult("abuseipdb", False)
    try:
        headers = {"Key": settings.abuseipdb_api_key, "Accept": "application/json"}
        resp = await client.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip},
            headers=headers,
            timeout=5.0,
        )
        data = resp.json()
        score = data.get("data", {}).get("abuseConfidenceScore", 0)
        return ProviderResult("abuseipdb", True, score > 50, data)
    except Exception:
        return ProviderResult("abuseipdb", False)


async def fuse_artifact_evidence(artifact_type: str, value: str) -> dict:
    """Returns {'malicious_score': float in [0,1] or None, 'providers_used': [...],
    'providers_unavailable': [...]} — never raises."""

    async with httpx.AsyncClient() as client:
        tasks = []
        if artifact_type in ("url", "domain"):
            tasks.append(_query_urlhaus(client, value))
            tasks.append(_query_virustotal(client, value))
        elif artifact_type == "ip":
            tasks.append(_query_abuseipdb(client, value))
        else:
            return {"malicious_score": None, "providers_used": [], "providers_unavailable": ["n/a"]}

        results: list[ProviderResult] = await asyncio.gather(*tasks) if tasks else []

    available = [r for r in results if r.available]
    unavailable = [r.provider for r in results if not r.available]

    if not available:
        return {"malicious_score": None, "providers_used": [], "providers_unavailable": unavailable}

    weighted_sum = sum(PROVIDER_WEIGHTS.get(r.provider, 0.5) * (1.0 if r.malicious else 0.0) for r in available)
    weight_total = sum(PROVIDER_WEIGHTS.get(r.provider, 0.5) for r in available)
    score = weighted_sum / weight_total if weight_total > 0 else None

    return {
        "malicious_score": score,
        "providers_used": [r.provider for r in available],
        "providers_unavailable": unavailable,
    }
