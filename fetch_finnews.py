"""Utilities to harvest finance news from GDELT Doc 2.0."""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode, urlparse, urlunparse

import requests
import trafilatura
import re


GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_HEADERS = {"User-Agent": "finnews-harvester/1.0 (+https://github.com/)"}
# GDELT rejects requests whose URL-encoded query parameter is too long.  The
# service does not publish the exact limit, but empirical testing shows that
# keeping the encoded query under roughly 300 characters avoids the
# "too short or too long" error responses.  Because URL encoding expands
# quotation marks and other characters, we measure against the encoded length
# rather than the raw string length when chunking domains and keywords.
MAX_QUERY_LENGTH = 300
# Reserve space for at least one keyword in each query; GDELT adds ~20 bytes of
# overhead for a short phrase, so keep some slack for safety.
KEYWORD_RESERVED_BYTES = 60

# Keep the keyword clause compact so each chunked query stays within Doc 2.0's
# practical length ceiling.  These curated phrases focus on language that shows
# up frequently in mainstream finance coverage; meme slang and duplicates were
# removed because they add noise while inflating the query length.
BULLISH_TERMS: Sequence[str] = (
    "strong buy",
    "buy signal",
    "buy recommendation",
    "accumulate",
    "bullish",
    "outperform",
    "analyst upgrade",
    "raised price target",
    "raised guidance",
    "positive catalyst",
    "earnings surprise",
    "reaffirmed outlook",
    "expanding margins",
    "institutional accumulation",
    "accelerating momentum",
    "support holding",
    "breakout",
    "uptrend",
    "positive outlook",
    "investor confidence rising",
)

BEARISH_TERMS: Sequence[str] = (
    "strong sell",
    "sell signal",
    "sell recommendation",
    "downgraded to sell",
    "reduced price target",
    "guidance cut",
    "earnings miss",
    "weak demand",
    "margin compression",
    "declining sales",
    "valuation stretched",
    "negative catalyst",
    "bearish",
    "downtrend",
    "breakdown",
    "resistance holding",
    "selling pressure",
    "risk-off",
    "flight to safety",
    "fear rising",
)

NEUTRAL_TERMS: Sequence[str] = (
    "neutral",
    "mixed outlook",
    "rangebound",
    "consolidation",
    "sideways trend",
    "stable sentiment",
    "muted reaction",
    "balanced view",
    "balanced risk/reward",
    "steady fundamentals",
    "unchanged outlook",
    "directionless",
    "holding pattern",
    "waiting for confirmation",
    "cautious positioning",
    "stability returning",
    "no strong bias",
    "steady conditions",
    "neutral stance",
    "awaiting catalyst",
)

def _quote_terms(terms: Sequence[str]) -> List[str]:
    """Return each keyword wrapped in double quotes for the query."""

    return [f'"{term}"' for term in terms]


def _build_keyword_block(terms: Sequence[str]) -> str:
    """Join the provided terms into a single OR clause."""

    if not terms:
        return ""
    return " OR ".join(_quote_terms(terms))


KEYWORD_TERMS: List[str] = (
    list(BULLISH_TERMS) + list(BEARISH_TERMS) + list(NEUTRAL_TERMS)
)


def _wrap_clause(clause: str) -> str:
    """Return a clause wrapped in parentheses only when needed."""

    clause = clause.strip()
    if not clause:
        return ""
    if " OR " in clause:
        return f"({clause})"
    return clause


def _canonicalize_url(url: str) -> str:
    """Return a best-effort canonical URL for duplicate detection."""

    if not isinstance(url, str) or not url:
        return ""
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    canonical = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(canonical)


def _extract_article_text(url: str) -> Optional[str]:
    """Fetch and extract article text using trafilatura."""

    if not url:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise RuntimeError("No content fetched")
    except Exception as exc:  # pragma: no cover - network dependent
        logging.debug("trafilatura.fetch_url failed for %s: %s", url, exc)
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            response.raise_for_status()
            downloaded = response.text
        except Exception as req_exc:  # pragma: no cover - network dependent
            logging.debug("Fallback fetch failed for %s: %s", url, req_exc)
            return None

    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            include_formatting=False,
        )
        if text:
            return text.strip()
    except Exception as exc:  # pragma: no cover - network dependent
        logging.debug("trafilatura.extract failed for %s: %s", url, exc)
    return None


_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: Optional[str]) -> str:
    """Collapse whitespace so CSV rows remain single-line."""

    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


DEFAULT_TIERS_CFG: Dict[str, Dict[str, object]] = {
    "professional": {
        "weight": 1.0,
        "domains": [
            "reuters.com",
            "wsj.com",
            "ft.com",
            "barrons.com",
        ],
    },
    "semi": {
        "weight": 0.65,
        "domains": [
            "finance.yahoo.com",
            "marketwatch.com",
            "investorplace.com",
            "fool.com",
        ],
    },
    "low": {
        "weight": 0.2,
        "domains": [
            "seekingalpha.com",
            "medium.com",
            "substack.com",
        ],
    },
}


def load_domains() -> Dict[str, Dict[str, object]]:
    """Return a copy of the built-in tier configuration."""

    logging.debug("Using built-in tier configuration")
    return {
        tier: {
            "weight": info.get("weight", 0.0),
            "domains": list(info.get("domains", [])),
        }
        for tier, info in DEFAULT_TIERS_CFG.items()
    }


def tier_for_domain(domain: str, tiers_cfg: Dict[str, Dict[str, object]]) -> Tuple[str, float]:
    """Return the tier label and weight for a given domain."""
    domain = domain.lower()
    for tier_name, tier_info in tiers_cfg.items():
        domains = tier_info.get("domains", [])
        for allowed in domains:
            allowed = allowed.lower()
            if domain == allowed or domain.endswith(f".{allowed}"):
                return tier_name, float(tier_info.get("weight", 0.0))
    return "unknown", 0.0


def _escape_company(company: str) -> str:
    """Escape double quotes for inclusion in the query string."""

    return company.replace("\"", "\\\"")


def _build_domain_clause(domains: Sequence[str]) -> str:
    """Return the OR expression covering the supplied publisher domains."""

    domain_terms = []
    for domain in domains:
        domain = domain.strip()
        if not domain:
            continue
        domain_terms.append(f"domainis:{domain}")
    if not domain_terms:
        raise ValueError("Domain list must contain at least one entry")
    return " OR ".join(domain_terms)


def _build_company_clause(company: str, ticker: str) -> str:
    """Return the company/ticker portion of the boolean query."""

    company_escaped = _escape_company(company)
    terms: List[str] = [f'"{company_escaped}"']
    ticker_clean = (ticker or "").strip()
    if ticker_clean:
        alnum_ticker = "".join(ch for ch in ticker_clean if ch.isalnum())
        if len(alnum_ticker) >= 3:
            if ticker_clean.isalnum():
                terms.append(ticker_clean)
            else:
                terms.append(f'"{ticker_clean}"')
        if len(alnum_ticker) >= 2 and ticker_clean.isalnum():
            terms.append(f"${ticker_clean}")
    # Deduplicate while preserving order
    ordered_terms = []
    seen = set()
    for term in terms:
        if term not in seen:
            ordered_terms.append(term)
            seen.add(term)
    return " OR ".join(ordered_terms)


def build_query(
    company: str,
    ticker: str,
    domains: Sequence[str],
    keywords: Sequence[str] | None = None,
) -> str:
    """Construct a boolean query for GDELT."""

    domain_clause = _build_domain_clause(domains)
    company_clause = _build_company_clause(company, ticker)
    if keywords is None:
        keywords = KEYWORD_TERMS
    keyword_clause = _build_keyword_block(keywords)
    query_parts = [
        _wrap_clause(domain_clause),
        _wrap_clause(company_clause),
    ]
    if keyword_clause:
        query_parts.append(_wrap_clause(keyword_clause))
    return " AND ".join(part for part in query_parts if part)


def _encoded_query_length(query: str) -> int:
    """Return the length of the query string once URL encoded."""

    # ``urlencode`` applies the same quoting logic that ``requests`` uses when
    # serialising query parameters.  Measuring the encoded length keeps our
    # guardrail aligned with what the GDELT endpoint receives.
    return len(urlencode({"query": query}))


def chunk_keywords(
    company: str,
    ticker: str,
    domains: Sequence[str],
    keywords: Sequence[str],
    max_query_length: int = MAX_QUERY_LENGTH,
) -> List[List[str]]:
    """Group keywords so each query's *encoded* length stays within the limit."""

    chunks: List[List[str]] = []
    current: List[str] = []
    for keyword in keywords:
        trial = current + [keyword]
        query = build_query(company, ticker, domains, keywords=trial)
        if _encoded_query_length(query) <= max_query_length:
            current = trial
            continue
        if not current:
            raise ValueError(
                "Keyword too long for query allowance; adjust MAX_QUERY_LENGTH or shorten term"
            )
        chunks.append(current)
        current = [keyword]

    if current:
        chunks.append(current)

    if not chunks:
        return [[]]

    return chunks


def chunk_domains(
    company: str,
    ticker: str,
    domains: Sequence[str],
    max_query_length: int = MAX_QUERY_LENGTH,
) -> List[List[str]]:
    """Group domains so that each resulting query stays within the max length."""

    if not domains:
        raise ValueError("Domain list must contain at least one entry")

    chunks: List[List[str]] = []
    current: List[str] = []

    budget = max(0, max_query_length - KEYWORD_RESERVED_BYTES)

    for domain in domains:
        trial = current + [domain]
        query = build_query(company, ticker, trial, keywords=())
        length = _encoded_query_length(query)
        if length <= budget:
            current = trial
            continue
        if not current:
            raise ValueError(
                "Query cannot be constructed within max length; try reducing keywords"
            )
        chunks.append(current)
        current = [domain]

    if current:
        chunks.append(current)

    if len(chunks) > 1 and len(chunks[-1]) == 1:
        tail = chunks[-1]
        prev = chunks[-2]
        redistributed = False
        for idx in range(len(prev) - 1, -1, -1):
            candidate = prev[idx]
            prev_trial = prev[:idx] + prev[idx + 1 :]
            if not prev_trial:
                continue
            tail_trial = [candidate] + tail
            if (
                _encoded_query_length(
                    build_query(company, ticker, prev_trial, keywords=())
                )
                <= budget
                and _encoded_query_length(
                    build_query(company, ticker, tail_trial, keywords=())
                )
                <= budget
            ):
                tail.insert(0, candidate)
                del prev[idx]
                redistributed = True
                break
        if not redistributed:
            # As a fallback, merge the single-domain tail back into the previous chunk.
            merged = prev + tail
            if (
                _encoded_query_length(
                    build_query(company, ticker, merged, keywords=())
                )
                <= budget
            ):
                chunks[-2] = merged
                chunks.pop()

    return chunks


def gdelt_search(query: str, maxrecords: int = 250) -> List[Dict[str, object]]:
    """Execute a query against the GDELT Doc 2.0 API."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(maxrecords),
        "sort": "DateDesc",
    }
    response = requests.get(
        GDELT_ENDPOINT, params=params, headers=REQUEST_HEADERS, timeout=30
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        snippet = response.text[:200].strip()
        raise RuntimeError(
            f"GDELT returned a non-JSON response: {snippet or '<<empty>>'}"
        ) from exc
    return payload.get("articles", [])


def harvest(
    companies: Sequence[Tuple[str, str]],
    tiers_cfg: Dict[str, Dict[str, object]],
    maxrecords: int = 250,
) -> List[Dict[str, object]]:
    """Harvest articles for a list of (ticker, company_name)."""
    all_domains = sorted(
        {domain for tier in tiers_cfg.values() for domain in tier.get("domains", [])}
    )
    rows: List[Dict[str, object]] = []
    seen_urls = set()
    content_cache: Dict[str, Optional[str]] = {}

    for ticker, company in companies:
        domain_chunks = chunk_domains(company, ticker, all_domains)
        query_plan: List[Tuple[List[str], List[str]]] = []
        for domain_chunk in domain_chunks:
            keyword_chunks = chunk_keywords(
                company,
                ticker,
                domain_chunk,
                KEYWORD_TERMS,
                max_query_length=MAX_QUERY_LENGTH,
            )
            for keyword_chunk in keyword_chunks:
                query_plan.append((domain_chunk, keyword_chunk))

        total_queries = len(query_plan)
        for idx, (domain_chunk, keyword_chunk) in enumerate(query_plan, start=1):
            query = build_query(
                company,
                ticker,
                domain_chunk,
                keywords=keyword_chunk,
            )
            logging.info(
                "Querying GDELT for %s (%d/%d)", ticker, idx, total_queries
            )
            logging.debug(
                "GDELT query for %s [%d/%d]: %s (encoded bytes: %d)",
                ticker,
                idx,
                total_queries,
                query,
                _encoded_query_length(query),
            )
            try:
                articles = gdelt_search(query, maxrecords=maxrecords)
            except Exception as exc:  # pragma: no cover - network error
                logging.error("Failed to query GDELT for %s: %s", ticker, exc)
                continue

            for article in articles:
                url = article.get("url") or article.get("documentIdentifier")
                if not url:
                    continue
                canonical_url = _canonicalize_url(url)
                dedupe_key = canonical_url or url
                domain = article.get("sourceDomain")
                if not domain:
                    parsed = urlparse(url)
                    domain = parsed.netloc
                tier_name, weight = tier_for_domain(domain, tiers_cfg)
                doc_id = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()
                published = article.get("seendate") or article.get("publishedDate")
                if (
                    isinstance(published, str)
                    and len(published) == 14
                    and published.isdigit()
                ):
                    # GDELT seendate format YYYYMMDDHHMMSS
                    published = (
                        f"{published[0:4]}-{published[4:6]}-{published[6:8]} "
                        f"{published[8:10]}:{published[10:12]}:{published[12:14]}"
                    )
                if dedupe_key in seen_urls:
                    continue
                if dedupe_key in content_cache:
                    article_text = content_cache[dedupe_key]
                else:
                    article_text = _extract_article_text(url)
                    content_cache[dedupe_key] = article_text
                seen_urls.add(dedupe_key)
                clean_text = _clean_text(article_text)
                if not clean_text:
                    clean_text = _clean_text(article.get("snippet"))
                if not clean_text:
                    logging.debug("Skipping %s due to empty content", url)
                    continue
                rows.append(
                    {
                        "doc_id": doc_id,
                        "symbol": ticker,
                        "title": article.get("title"),
                        "url": url,
                        "source_domain": domain,
                        "published_at": published,
                        "article_type": tier_name,
                        "tier_weight": weight,
                        "text": clean_text,
                    }
                )
    return rows


def save_csv(rows: Iterable[Dict[str, object]], path: str | Path) -> None:
    """Persist harvested rows to a CSV file."""
    rows = list(rows)
    fieldnames = [
        "doc_id",
        "symbol",
        "title",
        "url",
        "source_domain",
        "published_at",
        "article_type",
        "tier_weight",
        "text",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_companies(path: str | Path) -> List[Tuple[str, str]]:
    """Read (ticker, company) pairs from CSV with headers symbol,company."""
    companies: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "symbol" not in reader.fieldnames or "company" not in reader.fieldnames:
            raise ValueError("Ticker file must have 'symbol' and 'company' columns")
        for row in reader:
            symbol = row.get("symbol")
            company = row.get("company")
            if symbol and company:
                companies.append((symbol.strip(), company.strip()))
    return companies


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest finance news via GDELT Doc 2.0")
    parser.add_argument("--tickers", required=True, help="CSV with symbol,company columns")
    parser.add_argument("--out", required=True, help="Destination CSV path for harvested data")
    parser.add_argument("--maxrecords", type=int, default=250, help="Max records per ticker query")
    parser.add_argument("--log", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    tiers_cfg = load_domains()
    companies = read_companies(args.tickers)
    rows = harvest(companies, tiers_cfg, maxrecords=args.maxrecords)
    save_csv(rows, args.out)
    if rows:
        logging.info("Saved %d rows to %s", len(rows), args.out)
    else:
        logging.warning(
            "Harvest returned no rows; wrote header-only CSV to %s. "
            "Check earlier log messages for API errors or adjust your query inputs.",
            args.out,
        )


if __name__ == "__main__":
    main()
