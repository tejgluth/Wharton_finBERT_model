"""Utilities to harvest finance news from GDELT Doc 2.0."""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import requests


GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
KEYWORD_BLOCK = (
    "earnings OR \"earnings call\" OR guidance OR outlook OR "
    "\"price target\" OR \"10-Q\" OR \"10-K\" OR acquisition OR merger"
)


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


def build_query(company: str, ticker: str, domains: Sequence[str]) -> str:
    """Construct a boolean query for GDELT."""
    domain_clause = " OR ".join(f"\"{d}\"" for d in domains)
    company_clause = f"\"{company}\" OR {ticker} OR ${ticker}"
    query_parts = [f"site:({domain_clause})", f"({company_clause})", f"({KEYWORD_BLOCK})"]
    return " AND ".join(query_parts)


def gdelt_search(query: str, maxrecords: int = 250) -> List[Dict[str, object]]:
    """Execute a query against the GDELT Doc 2.0 API."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(maxrecords),
        "sort": "DateDesc",
    }
    response = requests.get(GDELT_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("articles", [])


def harvest(
    companies: Sequence[Tuple[str, str]],
    tiers_cfg: Dict[str, Dict[str, object]],
    maxrecords: int = 250,
) -> List[Dict[str, object]]:
    """Harvest articles for a list of (ticker, company_name)."""
    all_domains = sorted({domain for tier in tiers_cfg.values() for domain in tier.get("domains", [])})
    rows: List[Dict[str, object]] = []
    seen_ids = set()

    for ticker, company in companies:
        query = build_query(company, ticker, all_domains)
        logging.info("Querying GDELT for %s", ticker)
        try:
            articles = gdelt_search(query, maxrecords=maxrecords)
        except Exception as exc:  # pragma: no cover - network error
            logging.error("Failed to query GDELT for %s: %s", ticker, exc)
            continue

        for article in articles:
            url = article.get("url") or article.get("documentIdentifier")
            if not url:
                continue
            domain = article.get("sourceDomain")
            if not domain:
                parsed = urlparse(url)
                domain = parsed.netloc
            tier_name, weight = tier_for_domain(domain, tiers_cfg)
            doc_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
            published = article.get("seendate") or article.get("publishedDate")
            if isinstance(published, str) and len(published) == 14 and published.isdigit():
                # GDELT seendate format YYYYMMDDHHMMSS
                published = (
                    f"{published[0:4]}-{published[4:6]}-{published[6:8]} "
                    f"{published[8:10]}:{published[10:12]}:{published[12:14]}"
                )
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
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
                    "text": article.get("snippet"),
                }
            )
    return rows


def save_csv(rows: Iterable[Dict[str, object]], path: str | Path) -> None:
    """Persist harvested rows to a CSV file."""
    rows = list(rows)
    if not rows:
        logging.warning("No rows to save to %s", path)
        return
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
    logging.info("Saved %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
