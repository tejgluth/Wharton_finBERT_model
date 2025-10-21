# Wharton FinBERT Sentiment Pipeline

A minimal pipeline that scores articles with **FinBERT**, applies **source/section weights**, and aggregates to a **log-scaled per-stock sentiment score (β)**. Accuracy first; efficiency can come later.

---

## Quick Start

### 1. Create & activate a virtual environment (macOS/Linux)
```bash
cd ~/Wharton
python3 -m venv Wharton_env
source Wharton_env/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install pandas numpy transformers torch pyyaml requests tqdm
```
> **Note:** The first run downloads the FinBERT model (~400–500 MB).

### 3. Run the full pipeline: Articles → Snippets → Sentiment

1. **Harvest full-text articles** (deduped, with Trafilatura extraction):
   ```bash
   python fetch_finnews.py \
     --tickers tickers.csv \
     --out articles.csv \
     --maxrecords 250
   ```
   The output file keeps only rows with non-empty article text and adds source-specific tier weights (`professional`, `semi`, `low`).

2. **Generate snippet-level training/evaluation units** that explicitly mention each ticker/company:
   ```bash
   python build_snippets.py \
     --articles articles.csv \
     --tickers tickers.csv \
     --cap-map "AAPL:large;MSFT:large;GOOGL:large" \
     --out snippets.csv
   ```
   You can optionally add inline overrides with `--company-map "AAPL:Apple Inc.;TSLA:Tesla"` or restrict snippet types with `--only-kinds headline,paragraph`. When `tickers.csv` includes a `market_cap_category` column it is copied into `snippets.csv`; otherwise use `--cap-map` to specify buckets manually. The resulting CSV contains one row per snippet with metadata so scores can be remapped to the original article.

3. **Score snippets with FinBERT and aggregate per ticker:**
   ```bash
   python run_finbert_pipeline.py \
     --input snippets.csv \
     --output snippet_scores.csv \
     --cap-map "AAPL:large;MSFT:large;GOOGL:large"
   ```
   By default FinBERT weights headlines/paragraphs/sentences (`headline=0.75`, `paragraph=0.65`, `sentence=0.5`) as well as the article tiers (`professional=1.0`, `semi=0.65`, `low=0.2`).

   **Market-cap weighting.** Each symbol is assigned a market-cap bucket (large/mid/small). Provide it via the `market_cap_category` column in `tickers.csv` (propagated into `snippets.csv`) or pass `--cap-map "SYM:large;SYM2:mid"`. Final sentiment is computed as:

   | market cap | reports weight (C) | news weight (N) | other weight (O) |
   |------------|--------------------|-----------------|------------------|
   | Large      | 0.9                | 0.1             | 0.0              |
   | Mid        | 0.7                | 0.2             | 0.1              |
   | Small      | 0.5                | 0.3             | 0.2              |

   We classify each snippet into `reports` (e.g., professional sources, filings), `news` (mid-tier outlets, headlines/paragraphs), or `other` (low-tier, social). The final score is:

   \[ S_{final} = C \cdot S_{reports} + N \cdot S_{news} + O \cdot S_{other} \]

   Use `--symbol TICKER` to focus on a single name if desired.

These three steps take you from a ticker list in `tickers.csv` to ticker-level sentiment scores suitable for backtesting or daily monitoring.

## Understanding the output

`run_finbert_pipeline.py` writes a table (and optional CSV) with columns:

| column | meaning |
| --- | --- |
| `symbol` | Ticker symbol. |
| `n_docs` | Number of snippets contributing to that ticker. |
| `market_cap_category` | Market-cap bucket used for weighting (`large`, `mid`, `small`). |
| `qrs` | Final market-cap weighted sum (positive probability minus negative probability). |
| `qrs_mean_boot` | Bootstrap mean of `qrs` (30 resamples by default). |
| `qrs_ci_lo` / `qrs_ci_hi` | 95% bootstrap confidence interval bounds for `qrs`. |
| `beta` | Signed log transform of `qrs`: `log(1+pos) − log(1+neg)` with `pos=max(qrs,0)`, `neg=max(-qrs,0)`. Stabilises scale, comparable across tickers. |
| `beta_mean_boot` | Bootstrap mean of `beta`. |
| `beta_ci_lo` / `beta_ci_hi` | 95% bootstrap confidence interval bounds for `beta`. |
| `reports_score`, `news_score`, `other_score` | Unweighted sums per snippet bucket prior to market-cap scaling. |
| `qrs_unweighted` | Sum of per-snippet weights before applying market-cap coefficients. |

### Interpreting β

- β ≈ 0 → neutral sentiment.
- 0.15–0.5 → mild bullish/bearish tilt.
- 0.5–1.0 → clear bias.
- 1.0+ → strong bias.

If the confidence interval spans 0, the model considers the direction inconclusive with the current snippets.持续 negative β indicates net bearish tone; persistent positives indicate bullish tone.
