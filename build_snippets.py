"""Convenience CLI to generate snippet-level data ready for FinBERT."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set

import pandas as pd

import make_snippets


def load_company_map(ticker_csv: str | None, inline_map: str | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if ticker_csv:
        ticker_path = Path(ticker_csv)
        if not ticker_path.exists():
            raise FileNotFoundError(f"Ticker CSV not found: {ticker_path}")
        with ticker_path.open("r", encoding="utf-8") as fh:
            header = fh.readline()
            if "symbol" not in header.lower():
                raise ValueError("Ticker CSV must include a 'symbol' header")
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                parts = raw.split(",", 1)
                if len(parts) != 2:
                    continue
                symbol, company = parts
                symbol = symbol.strip().upper()
                company = company.strip()
                if symbol and company:
                    mapping[symbol] = company
    if inline_map:
        mapping.update(make_snippets.parse_company_map(inline_map))
    return mapping


def parse_kinds(raw: str | None) -> Set[str]:
    if not raw:
        return {"headline", "paragraph", "sentence"}
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate snippet-level dataset from harvested articles.")
    parser.add_argument("--articles", required=True, help="Path to articles CSV (e.g., articles.csv)")
    parser.add_argument("--out", dest="output_path", required=True, help="Destination for snippets CSV")
    parser.add_argument("--tickers", help="Optional ticker CSV with columns symbol,company to enrich matching")
    parser.add_argument("--company-map", default="", help='Inline overrides like "AAPL:Apple;MSFT:Microsoft"')
    parser.add_argument("--max-chars", type=int, default=800, help="Maximum characters per snippet")
    parser.add_argument("--only-kinds", default="", help="Comma list limiting snippet kinds (headline,paragraph,sentence)")
    parser.add_argument("--keep-empty-text", action="store_true", help="Include rows even if text column is empty")
    args = parser.parse_args()

    articles_path = Path(args.articles)
    if not articles_path.exists():
        raise FileNotFoundError(f"Articles file not found: {articles_path}")

    company_map = load_company_map(args.tickers, args.company_map)
    kinds = parse_kinds(args.only_kinds)

    df_articles = pd.read_csv(articles_path)
    snippets = make_snippets.create_snippets(
        df_articles,
        company_map=company_map,
        max_chars=args.max_chars,
        keep_empty_text=args.keep_empty_text,
        kinds=kinds,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snippets.to_csv(output_path, index=False)
    print(f"Saved {len(snippets)} snippets to {output_path}")
if __name__ == "__main__":
    main()
