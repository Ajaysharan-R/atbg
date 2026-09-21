"""
Extracts cybersecurity-relevant artifacts from turn text: URLs, domains, IPs,
emails, phone numbers, UPI IDs, crypto wallet addresses, file hashes.

Runs AFTER redaction is a bad idea for URLs/wallets (redaction only removes
personal PII like names/generic phone/email, not URLs/crypto addresses —
so run extraction on the ORIGINAL text, before or independent of redaction,
then only the surviving human-identifying fields go through the redactor.
"""

import re

PATTERNS = {
    "url": re.compile(r"https?://[^\s<>\"]+"),
    "domain": re.compile(r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "email": re.compile(r"\b[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+91[\-\s]?|0)?[6-9]\d{9}\b"),
    "upi": re.compile(r"\b[\w.\-]{2,256}@(?:okaxis|okhdfcbank|oksbi|okicici|ybl|upi|paytm)\b", re.IGNORECASE),
    "btc_wallet": re.compile(r"\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b"),
    "eth_wallet": re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
    "file_hash_md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "file_hash_sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
}


def extract_artifacts(text: str) -> list[dict]:
    found = []
    seen = set()

    for artifact_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            key = (artifact_type, value)
            if key in seen:
                continue
            seen.add(key)
            found.append({"artifact_type": artifact_type, "value": value})

    # A URL's netloc is redundant with a separately-matched 'domain' —
    # drop bare domain hits that are already inside a matched URL.
    urls = [f["value"] for f in found if f["artifact_type"] == "url"]
    found = [
        f
        for f in found
        if not (f["artifact_type"] == "domain" and any(f["value"] in u for u in urls))
    ]

    return found
