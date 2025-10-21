"""Convenience CLI to generate snippet-level data ready for FinBERT."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

import make_snippets


def parse_cap_map(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        sym, cap = pair.split(":", 1)
        sym = sym.strip().upper()
        cap = cap.strip()
        if sym and cap:
            out[sym] = cap
    return out


def load_company_map(
    ticker_csv: str | None,
    inline_map: str | None,
    inline_cap_map: str | None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    mapping: Dict[str, str] = {}
    cap_map: Dict[str, str] = parse_cap_map(inline_cap_map)
    if ticker_csv:
        ticker_path = Path(ticker_csv)
        if not ticker_path.exists():
            raise FileNotFoundError(f"Ticker CSV not found: {ticker_path}")
        with ticker_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = [h.lower() for h in reader.fieldnames or []]
            if "symbol" not in headers or "company" not in headers:
                raise ValueError("Ticker CSV must include 'symbol' and 'company' columns")
            cap_header = None
            for candidate in ("market_cap_category", "market_cap", "cap_category"):
                if candidate in headers:
                    cap_header = candidate
                    break
            for row in reader:
                if not row:
                    continue
                symbol = (row.get("symbol") or row.get("Symbol") or "").strip().upper()
                company = (row.get("company") or row.get("Company") or "").strip()
                if symbol and company:
                    mapping[symbol] = company
                if cap_header and symbol:
                    cap_val = row.get(cap_header) or row.get(cap_header.upper())
                    if cap_val:
                        cap_map[symbol] = str(cap_val).strip()
    if inline_map:
        mapping.update(make_snippets.parse_company_map(inline_map))
    return mapping, cap_map


def parse_kinds(raw: str | None) -> Set[str]:
    if not raw:
        return {"headline", "paragraph", "sentence"}
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate snippet-level dataset from harvested articles.")
    parser.add_argument("--articles", required=True, help="Path to articles CSV (e.g., articles.csv)")
    parser.add_argument("--out", dest="output_path", required=True, help="Destination for snippets CSV")
    parser.add_argument("--tickers", help="Optional ticker CSV with columns symbol,company,(market_cap_category) to enrich matching")
    parser.add_argument("--company-map", default="", help='Inline overrides like "AAPL:Apple;MSFT:Microsoft"')
    parser.add_argument("--cap-map", default="", help='Inline market-cap overrides like "AAPL:large;TSLA:mid"')
    parser.add_argument("--max-chars", type=int, default=800, help="Maximum characters per snippet")
    parser.add_argument("--only-kinds", default="", help="Comma list limiting snippet kinds (headline,paragraph,sentence)")
    parser.add_argument("--keep-empty-text", action="store_true", help="Include rows even if text column is empty")
    args = parser.parse_args()

    articles_path = Path(args.articles)
    if not articles_path.exists():
        raise FileNotFoundError(f"Articles file not found: {articles_path}")

    company_map, cap_map = load_company_map(args.tickers, args.company_map, args.cap_map)
    kinds = parse_kinds(args.only_kinds)

    df_articles = pd.read_csv(articles_path)
    snippets = make_snippets.create_snippets(
        df_articles,
        company_map=company_map,
        max_chars=args.max_chars,
        keep_empty_text=args.keep_empty_text,
        kinds=kinds,
    )

    if cap_map:
        snippets["market_cap_category"] = snippets["symbol"].map(lambda sym: cap_map.get(str(sym).upper()))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snippets.to_csv(output_path, index=False)
    print(f"Saved {len(snippets)} snippets to {output_path}")


if __name__ == "__main__":
    main()
