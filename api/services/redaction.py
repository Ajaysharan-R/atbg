"""
PII redaction.

Uses Presidio where available for general PII (names, generic phone/email),
plus hand-written regexes for identifiers Presidio doesn't cover well:
Aadhaar, PAN, UPI VPA, IFSC, and Indian-format phone numbers.

Falls back to regex-only mode if presidio isn't installed, so the rest of
the pipeline never hard-depends on it.
"""

import re
import uuid

INDIAN_PATTERNS = {
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "UPI_VPA": re.compile(r"\b[\w.\-]{2,256}@[a-zA-Z]{2,64}\b"),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "PHONE_IN": re.compile(r"\b(?:\+91[\-\s]?|0)?[6-9]\d{9}\b"),
}

GENERIC_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.\-]+@[\w\-]+\.[a-zA-Z]{2,}\b"),
    "PHONE": re.compile(r"\b\+?\d[\d\-\s]{7,14}\d\b"),
}

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _PRESIDIO_AVAILABLE = True
except Exception:  # presidio not installed, or model download unavailable offline
    _PRESIDIO_AVAILABLE = False


class Redactor:
    """Stateful per-conversation redactor: the same underlying value maps
    to the same placeholder every time it appears in one conversation,
    e.g. every mention of the same phone number becomes <PHONE_1>."""

    def __init__(self):
        self._value_to_placeholder: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def _placeholder_for(self, kind: str, value: str) -> str:
        key = f"{kind}:{value}"
        if key not in self._value_to_placeholder:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            self._value_to_placeholder[key] = f"<{kind}_{self._counters[kind]}>"
        return self._value_to_placeholder[key]

    def redact(self, text: str) -> str:
        redacted = text

        # Indian identifiers first (more specific patterns before generic ones)
        for kind, pattern in INDIAN_PATTERNS.items():
            for match in list(pattern.finditer(redacted))[::-1]:
                value = match.group(0)
                placeholder = self._placeholder_for(kind, value)
                redacted = redacted[: match.start()] + placeholder + redacted[match.end() :]

        # Presidio for names/locations/generic PII, if available
        if _PRESIDIO_AVAILABLE:
            try:
                results = _analyzer.analyze(text=redacted, language="en")
                for r in sorted(results, key=lambda x: x.start, reverse=True):
                    value = redacted[r.start : r.end]
                    placeholder = self._placeholder_for(r.entity_type, value)
                    redacted = redacted[: r.start] + placeholder + redacted[r.end :]
            except Exception:
                pass  # never let redaction failure block the pipeline

        # Generic fallback patterns (also used when presidio is unavailable)
        for kind, pattern in GENERIC_PATTERNS.items():
            for match in list(pattern.finditer(redacted))[::-1]:
                value = match.group(0)
                if value.startswith("<") and value.endswith(">"):
                    continue  # already a placeholder
                placeholder = self._placeholder_for(kind, value)
                redacted = redacted[: match.start()] + placeholder + redacted[match.end() :]

        return redacted


def redact_conversation(turns: list[str]) -> list[str]:
    """Redact a whole conversation with one Redactor instance so repeated
    identifiers get consistent placeholders across turns."""
    redactor = Redactor()
    return [redactor.redact(t) for t in turns]
