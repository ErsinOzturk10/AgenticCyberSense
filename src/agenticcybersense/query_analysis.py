"""Helpers for extracting observable-focused search terms from user queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[A-Za-z]{2,}\b")
TOKEN_RE = re.compile(r"[a-z0-9@._:/-]+")
MIN_KEYWORD_LENGTH = 2

BREACH_TERMS = (
    "breach",
    "breached",
    "credential",
    "credentials",
    "dump",
    "exposed",
    "exposure",
    "leak",
    "leakage",
    "pwned",
)

VULNERABILITY_TERMS = (
    "0day",
    "cve",
    "exploit",
    "exploited",
    "poc",
    "proof of concept",
    "rce",
    "vulnerability",
    "vulnerabilities",
    "weaponization",
    "weaponized",
    "zero day",
    "zero-day",
)

THREAT_INTEL_TERMS = (
    "ddos",
    "dos",
    "attack",
    "attacks",
    "attacked",
    "attacking",
    "botnet",
    "campaign",
    "campaigns",
    "compromise",
    "compromised",
    "defacement",
    "incident",
    "ioc",
    "iocs",
    "malware",
    "outage",
    "phishing",
    "ransomware",
    "threat actor",
)

STOPWORDS = {
    "a",
    "about",
    "address",
    "adress",
    "agent",
    "an",
    "and",
    "are",
    "can",
    "check",
    "could",
    "find",
    "for",
    "had",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "on",
    "please",
    "search",
    "see",
    "show",
    "stored",
    "strored",
    "telegram",
    "that",
    "the",
    "there",
    "this",
    "to",
    "website",
    "websites",
    "what",
    "whether",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class QueryAnalysis:
    """Normalized representation of a user query for search and matching."""

    raw_query: str
    search_query: str
    match_terms: tuple[str, ...]
    observables: tuple[str, ...]
    intent_terms: tuple[str, ...]
    keywords: tuple[str, ...]

    @property
    def is_observable_lookup(self) -> bool:
        """Whether the query centers on a specific observable."""
        return bool(self.observables)


def analyze_query(query: str) -> QueryAnalysis:
    """Extract observable-focused search and matching terms from a user query."""
    raw_query = (query or "").strip()
    lowered = raw_query.lower()

    observables = _dedupe(
        [
            *EMAIL_RE.findall(raw_query),
            *[match.upper() for match in CVE_RE.findall(raw_query)],
            *URL_RE.findall(raw_query),
            *_extract_domains(raw_query),
            *IP_RE.findall(raw_query),
        ],
    )

    intent_terms: list[str] = []
    if any(term in lowered for term in BREACH_TERMS):
        intent_terms.extend(["breach", "leak"])
    if any(term in lowered for term in VULNERABILITY_TERMS):
        intent_terms.extend(["vulnerability", "exploit"])

    keywords = _extract_keywords(raw_query, observables)

    search_terms = _dedupe([*observables, *intent_terms, *keywords[:4]])
    match_terms = _dedupe([*observables, *intent_terms, *keywords[:3]])

    if not search_terms and raw_query:
        search_terms = [raw_query]
    if not match_terms and raw_query:
        match_terms = [raw_query.lower()]

    return QueryAnalysis(
        raw_query=raw_query,
        search_query=" ".join(search_terms),
        match_terms=tuple(match_terms),
        observables=tuple(observables),
        intent_terms=tuple(_dedupe(intent_terms)),
        keywords=tuple(keywords),
    )


def query_matches_text(query: str | QueryAnalysis, text: str) -> bool:
    """Return whether text matches the refined intent/observable terms."""
    analysis = query if isinstance(query, QueryAnalysis) else analyze_query(query)
    if not analysis.match_terms:
        return True

    haystack = (text or "").lower()
    return any(term.lower() in haystack for term in analysis.match_terms)


def has_threat_intel_intent(query: str | QueryAnalysis) -> bool:
    """Whether the query asks about ongoing threat activity or attack chatter."""
    analysis = query if isinstance(query, QueryAnalysis) else analyze_query(query)
    lowered = analysis.raw_query.lower()
    return any(term in lowered for term in THREAT_INTEL_TERMS)


def _extract_domains(query: str) -> list[str]:
    scrubbed = EMAIL_RE.sub(" ", URL_RE.sub(" ", query))
    return [match for match in DOMAIN_RE.findall(scrubbed) if "." in match]


def _extract_keywords(query: str, observables: list[str]) -> list[str]:
    lowered = query.lower()
    for observable in observables:
        lowered = lowered.replace(observable.lower(), " ")

    tokens = TOKEN_RE.findall(lowered)
    keywords = [token for token in tokens if token not in STOPWORDS and len(token) > MIN_KEYWORD_LENGTH]
    return _dedupe(keywords)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered
