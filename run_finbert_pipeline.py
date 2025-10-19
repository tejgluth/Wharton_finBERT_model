import argparse
import math
import os
import random
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
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        # Most FinBERT variants use label order: [negative, neutral, positive]

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
            logits = self.model(**tok).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Map to scalar in [-1,1]: P(pos) - P(neg)
            s = probs[:, 2] - probs[:, 0]
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


    # Aggregate per symbol
    out_rows = []
    for sym, g in df.groupby("symbol"):
        vals = g["_weighted"].to_numpy(dtype=float)
        # Raw QRS (sum)
        qrs = float(vals.sum())
        # Bootstrap for CI
        boots = bootstrap_sum(vals, n_reps=int(cfg["bootstrap_replicates"]), seed=int(cfg["random_seed"]))
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
            "qrs": qrs,
            "qrs_mean_boot": qrs_mean,
            "qrs_ci_lo": float(qrs_lo),
            "qrs_ci_hi": float(qrs_hi),
            "beta": beta,
            "beta_mean_boot": beta_mean,
            "beta_ci_lo": beta_lo,
            "beta_ci_hi": beta_hi,
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
    main(p.parse_args())
