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

### 3. Prepare your input file
Provide a CSV or Parquet file with the following columns (case-insensitive):

- `doc_id` (string) — unique id per article (or chunk)
- `symbol` (string) — stock ticker (e.g., `AAPL`)
- `text` (string) — cleaned article text
- `article_type` (string) — e.g., `reports`, `tier1_news`, `news_blogs`, `social`
- `section_key` (string, optional) — e.g., `mdna_weight`, `risk_weight`, `biz_weight`

Example (`articles.csv`):
```csv
doc_id,symbol,text,article_type,section_key
a1,AAPL,"Apple reported strong iPhone sales this quarter.",tier1_news,
a2,AAPL,"In Item 7 MD&A, Apple discussed supply chain risks.",reports,mdna_weight
a3,MSFT,"Microsoft announced new AI initiatives boosting confidence.",news_blogs,
a4,MSFT,"Microsoft filed its quarterly 10-Q with stable revenue growth.",reports,risk_weight
```

### 4. Run the pipeline
```bash
# basic run (prints a table)
python run_finbert_pipeline.py --input articles.csv

# save results to a file
python run_finbert_pipeline.py --input articles.csv --output results.csv

# focus on one ticker
python run_finbert_pipeline.py --input articles.csv --symbol AAPL --output aapl_results.csv

# use a config YAML (weights/model/bootstraps)
python run_finbert_pipeline.py --input articles.csv --config weights.yaml --output results.csv
```

## What the Script Does
1. Scores each text with FinBERT → scalar sentiment in `[-1, 1]` via `P(positive) − P(negative)`.
2. Applies `article_type` weight (and optional `section_key` weight).
3. Aggregates per stock to a raw score (QRS, sum of weighted sentiments).
4. Runs a bootstrap (default 30 resamples) to estimate confidence intervals.
5. Transforms QRS to a signed log scale (β) for stability and comparability.

## Output Explained
Typical columns:

| column        | meaning                                                                                                 |
|---------------|---------------------------------------------------------------------------------------------------------|
| `symbol`      | Stock ticker.                                                                                            |
| `n_docs`      | Number of documents included for that ticker.                                                            |
| `qrs`         | Raw sum of weighted per-doc scores (`FinBERT score × type weight × optional section weight`).            |
| `qrs_mean_boot` | Bootstrap mean of `qrs` (resampling documents with replacement).                                       |
| `qrs_ci_lo/hi`  | 95% bootstrap confidence interval for `qrs`.                                                           |
| `beta`        | Signed log transform of `qrs`: `log(1+pos) − log(1+neg)` with `pos=max(qrs,0)` and `neg=max(-qrs,0)`.    |
| `beta_mean_boot` | Bootstrap mean of `beta` (applied to bootstrap `qrs` samples).                                        |
| `beta_ci_lo/hi`  | 95% bootstrap confidence interval for `beta`.                                                         |

### Interpreting β (rule of thumb)
- ~0 → neutral
- 0.15–0.5 → mild tilt
- 0.5–1.0 → clear tilt
- 1.0+ → strong tilt

If the β confidence interval crosses 0, the direction is inconclusive with the current documents.

## Collecting Articles with GDELT Doc 2.0

Use `fetch_finnews.py` to pull recent articles for a ticker list and map each source domain to a tier weight. The script ships with sensible defaults embedded directly in the code (see `DEFAULT_TIERS_CFG` inside `fetch_finnews.py`).

```bash
python fetch_finnews.py \
  --tickers tickers.csv \
  --out data/news_harvest.csv \
  --maxrecords 250
```

Where `tickers.csv` contains:

```csv
symbol,company
AAPL,Apple Inc.
MSFT,Microsoft Corporation
```

Each harvested row includes `doc_id`, `symbol`, `title`, `url`, `source_domain`, `published_at`, `article_type`, `tier_weight`, and a placeholder `text` column (currently populated with the GDELT snippet). Update the domain tiers by editing the `DEFAULT_TIERS_CFG` dictionary if you need to change weights or add publishers.

## Optional Configuration (`weights.yaml`)
```yaml
model_name: "ProsusAI/finbert"        # or "yiyanghkust/finbert-tone" or a local fine-tuned checkpoint
max_seq_len: 256
batch_size: 16
bootstrap_replicates: 30
random_seed: 1337

type_weights:
  reports: 1.0
  tier1_news: 0.75
  news_blogs: 0.45
  social: 0.15

section_weights:
  mdna_weight: 1.2
  risk_weight: 1.1
  biz_weight: 1.0
```

Run with:
```bash
python run_finbert_pipeline.py --input articles.csv --config weights.yaml --output results.csv
```
