import argparse
import math
import os
import random
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Config (edit here or pass a YAML)
# -------------------------
DEFAULT_CFG = {
    "model_name": "ProsusAI/finbert",  # or "yiyanghkust/finbert-tone"
    "max_seq_len": 256,
    "batch_size": 16,
    "bootstrap_replicates": 30,
    "random_seed": 1337,
    # weights by article_type (strata)
    "type_weights": {
        "reports": 1.0,
        "tier1_news": 0.75,
        "news_blogs": 0.45,
        "social": 0.15,
        # support fetch_finnews tiers
        "professional": 1.0,
        "semi": 0.65,
        "low": 0.2,
        # snippet-specific kinds
        "headline": 0.75,
        "paragraph": 0.65,
        "sentence": 0.5,
    },
    # optional section weights if you chunk filings
    "section_weights": {
        "mdna_weight": 1.2,
        "risk_weight": 1.1,
        "biz_weight": 1.0
    }
}

# -------------------------
# Utils
# -------------------------
CAP_WEIGHTS: Dict[str, Dict[str, float]] = {
    "large": {"reports": 0.9, "news": 0.1, "other": 0.0},
    "mid": {"reports": 0.7, "news": 0.2, "other": 0.1},
    "small": {"reports": 0.5, "news": 0.3, "other": 0.2},
}

CAP_NORMALISER = {
    "l": "large",
    "large-cap": "large",
    "large cap": "large",
    "megacap": "large",
    "mega": "large",
    "m": "mid",
    "mid-cap": "mid",
    "mid cap": "mid",
    "medium": "mid",
    "s": "small",
    "small-cap": "small",
    "small cap": "small",
}


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalise_cap(value: Optional[str], default: str) -> str:
    if not value:
        return default
    val = value.strip().lower()
    if val in CAP_WEIGHTS:
        return val
    return CAP_NORMALISER.get(val, default)


def parse_cap_map(mapping: str, default: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not mapping:
        return out
    for pair in mapping.split(";"):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        sym, cap = pair.split(":", 1)
        sym = sym.strip().upper()
        if not sym:
            continue
        out[sym] = normalise_cap(cap, default)
    return out


def bucket_for_article(article_type: Optional[str], snippet_kind: Optional[str], tier_weight: Optional[float]) -> str:
    atype = (article_type or "").strip().lower()
    skind = (snippet_kind or "").strip().lower()
    try:
        weight = float(tier_weight) if tier_weight is not None else None
    except Exception:
        weight = None

    if atype in {"reports", "filing"}:
        return "reports"
    if atype in {"professional", "tier1_news"}:
        return "reports"
    if atype in {"semi", "news_blogs"}:
        return "news"
    if atype in {"low", "social"}:
        return "other"

    if skind in {"headline", "paragraph"}:
        return "news"
    if skind in {"sentence"}:
        return "news"

    if weight is not None:
        if weight >= 0.9:
            return "reports"
        if weight >= 0.5:
            return "news"
        return "other"
    return "other"

def log_signed(qrs: float) -> float:
    """Symmetric, stable log transform."""
    pos = max(qrs, 0.0)
    neg = max(-qrs, 0.0)
    return math.log1p(pos) - math.log1p(neg)

# -------------------------
# FinBERT scorer
# -------------------------
class FinBertScorer:
    def __init__(self, model_name: str, max_seq_len: int = 256, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_seq_len = max_seq_len
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.autocast_enabled = self.device_type == "cuda"
        self.autocast_dtype = torch.float16 if self.autocast_enabled else torch.float32
        # FinBERT (ProsusAI) label order is [positive, negative, neutral]

    @torch.no_grad()
    def score_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            ).to(self.device)
            autocast_ctx = (
                torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype)
                if self.autocast_enabled
                else nullcontext()
            )
            with autocast_ctx:
                logits = self.model(**tok).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # FinBERT label order (id2label) is {0: positive, 1: negative, 2: neutral}
            # Map to scalar in [-1,1]: P(pos) - P(neg)
            pos = probs[:, 0]
            neg = probs[:, 1]
            s = pos - neg
            out.extend(s.tolist())
        return np.array(out)

# -------------------------
# Per-doc scoring
# -------------------------
def per_doc_weighted_score(
    article_type: str,
    section_key: Optional[str],
    finbert_score: float,
    type_weights: Dict[str, float],
    section_weights: Dict[str, float],
) -> float:
    u = finbert_score  # already in [-1, 1]
    if section_key and section_key in section_weights:
        u *= float(section_weights[section_key])

    atype = (article_type or "").strip()
    w = float(type_weights.get(atype, 0.0))  # default 0 if unknown
    return w * u

# -------------------------
# Bootstrap aggregation
# -------------------------
def bootstrap_sum(values: np.ndarray, n_reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.array([0.0] * n_reps)
    # sample with replacement n values each replicate, then sum
    idx = rng.integers(0, n, size=(n_reps, n))
    sampled = values[idx]
    return sampled.sum(axis=1)

# -------------------------
# Main
# -------------------------
def main(args):
    # Load config
    cfg = DEFAULT_CFG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            user_cfg = yaml.safe_load(f)
        # deep-merge naive
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v

    set_seeds(int(cfg["random_seed"]))

    # Load data
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # Minimal schema check
    required = {"doc_id", "symbol", "text", "article_type"}
    missing = required - set(map(str.lower, df.columns))
    # Try case-insensitive align
    cols = {c.lower(): c for c in df.columns}
    if missing:
        raise ValueError(f"Missing required columns (case-insensitive): {missing}")

    # normalize column names to lower for access
    df.columns = [c.lower() for c in df.columns]

    # Optional filter by symbol (so you can run one ticker at a time)
    if args.symbol:
        df = df[df["symbol"] == args.symbol]

    if df.empty:
        print("No rows after filtering. Exiting.")
        return

    # Run FinBERT
    scorer = FinBertScorer(cfg["model_name"], cfg["max_seq_len"])
    texts = df["text"].astype(str).fillna("").tolist()
    s_fin = scorer.score_texts(texts, batch_size=cfg["batch_size"])

    # Per-doc weighted scores
    type_weights = cfg["type_weights"]
    section_weights = cfg.get("section_weights", {})
    df["_finbert"] = s_fin
    df["_weighted"] = [
        per_doc_weighted_score(
            article_type=getattr(row, "article_type"),
            section_key=getattr(row, "section_key", None),  # safe even if column missing
            finbert_score=score,
            type_weights=type_weights,
            section_weights=section_weights,
        )
        for row, score in zip(df.itertuples(index=False), s_fin)
    ]

    # Market-cap categories
    default_cap = normalise_cap(args.cap_default, "large")
    cap_map_cli = parse_cap_map(args.cap_map, default_cap)
    cap_column = None
    for candidate in ("market_cap_category", "market_cap", "cap_category"):
        if candidate in df.columns:
            cap_column = candidate
            break

    if cap_column:
        df["_cap_category"] = df[cap_column].apply(lambda val: normalise_cap(str(val) if val is not None else "", default_cap))
    else:
        df["_cap_category"] = df["symbol"].apply(lambda sym: cap_map_cli.get(str(sym).upper(), default_cap))

    # Determine article bucket (reports/news/other)
    def determine_bucket(row) -> str:
        return bucket_for_article(
            getattr(row, "article_type", None),
            getattr(row, "snippet_kind", None),
            getattr(row, "tier_weight", None),
        )

    df["_bucket"] = [determine_bucket(row) for row in df.itertuples(index=False)]

    df["_cap_coeff"] = [
        CAP_WEIGHTS.get(cap, CAP_WEIGHTS[default_cap]).get(bucket, 0.0)
        for cap, bucket in zip(df["_cap_category"], df["_bucket"])
    ]

    df["_weighted_final"] = df["_weighted"] * df["_cap_coeff"]


    # Aggregate per symbol
    out_rows = []
    for sym, g in df.groupby("symbol"):
        cap_cat = g["_cap_category"].mode().iat[0] if not g["_cap_category"].empty else default_cap
        bucket_scores = g.groupby("_bucket")["_weighted"].sum()
        vals_final = g["_weighted_final"].to_numpy(dtype=float)
        vals_raw = g["_weighted"].to_numpy(dtype=float)

        qrs = float(vals_final.sum())
        qrs_raw = float(vals_raw.sum())

        boots = bootstrap_sum(vals_final, n_reps=int(cfg["bootstrap_replicates"]), seed=int(cfg["random_seed"]))
        qrs_mean = float(boots.mean())
        qrs_lo, qrs_hi = np.percentile(boots, [2.5, 97.5])

        # Log-scaled beta
        beta = log_signed(qrs)
        beta_mean = log_signed(qrs_mean)
        beta_lo = log_signed(float(qrs_lo))
        beta_hi = log_signed(float(qrs_hi))

        out_rows.append({
            "symbol": sym,
            "n_docs": int(len(g)),
            "n_snippets": int(len(g)),
            "market_cap_category": cap_cat,
            "qrs": qrs,
            "qrs_mean_boot": qrs_mean,
            "qrs_ci_lo": float(qrs_lo),
            "qrs_ci_hi": float(qrs_hi),
            "beta": beta,
            "beta_mean_boot": beta_mean,
            "beta_ci_lo": beta_lo,
            "beta_ci_hi": beta_hi,
            "reports_score": float(bucket_scores.get("reports", 0.0)),
            "news_score": float(bucket_scores.get("news", 0.0)),
            "other_score": float(bucket_scores.get("other", 0.0)),
            "qrs_unweighted": qrs_raw,
        })

    out = pd.DataFrame(out_rows).sort_values("beta", ascending=False)

    # Save & print
    if args.output:
        if args.output.endswith(".parquet"):
            out.to_parquet(args.output, index=False)
        else:
            out.to_csv(args.output, index=False)
        print(f"Saved results → {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV or Parquet with columns: doc_id,symbol,text,article_type[,section_key]")
    p.add_argument("--symbol", default=None, help="Optional ticker filter (e.g., AAPL)")
    p.add_argument("--config", default=None, help="Optional YAML to override defaults")
    p.add_argument("--output", default=None, help="Optional CSV/Parquet for scores")
    p.add_argument("--cap-map", default="", help="Optional symbol:cap mapping (large/mid/small, e.g. 'AAPL:large;TSLA:mid')")
    p.add_argument("--cap-default", default="large", help="Default market cap category when unspecified")
    main(p.parse_args())
