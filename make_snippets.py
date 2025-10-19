import re, argparse, hashlib
import pandas as pd
from tqdm import tqdm

# Try to use blingfire for higher quality sentence splitting if available.
try:
    import blingfire
    HAS_BLINGFIRE = True
except Exception:
    HAS_BLINGFIRE = False

HEADLINE_KIND  = "headline"
PARAGRAPH_KIND = "paragraph"
SENTENCE_KIND  = "sentence"

# --- Utilities ---
def parse_company_map(arg: str):
    """
    Parse mapping like "AAPL:Apple;MSFT:Microsoft;GOOGL:Alphabet"
    into a dict {"AAPL":"Apple", "MSFT":"Microsoft", "GOOGL":"Alphabet"}.
    """
    m = {}
    if not arg:
        return m
    for pair in arg.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        m[k.strip().upper()] = v.strip()
    return m

# Fallback sentence splitter (if blingfire not available).
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')

def sent_tokenize(text: str) -> list[str]:
    if not text:
        return []
    if HAS_BLINGFIRE:
        return [s.strip() for s in blingfire.text_to_sentences(text).split("\n") if s.strip()]
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]

def para_split(text: str) -> list[str]:
    """
    Split into paragraphs using blank lines; if none, group by ~3 sentences.
    """
    if not text:
        return []
    # Split on blank-line boundaries
    parts = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    if len(parts) > 1:
        return parts
    # Fallback: bundle sentences into paragraphs of ~3 sentences
    sents = sent_tokenize(text)
    if not sents:
        return []
    bundle, out = [], []
    for s in sents:
        bundle.append(s)
        if len(bundle) >= 3:
            out.append(" ".join(bundle))
            bundle = []
    if bundle:
        out.append(" ".join(bundle))
    return out

def mentions_target(text: str, ticker: str, company: str) -> bool:
    """
    Return True if the text mentions the ticker (TICKER or $TICKER) or the company name.
    """
    if not text:
        return False
    ticker = (ticker or "").upper()
    company = company or ticker
    pats = []
    if ticker:
        t = re.escape(ticker)
        pats.append(rf'(?<!\w){t}(?!\w)')      # AAPL
        pats.append(rf'\${t}(?!\w)')           # $AAPL
    if company:
        c = re.escape(company)
        pats.append(rf'(?<!\w){c}(?!\w)')      # Apple / Apple Inc.
    for p in pats:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False

def stable_doc_id(symbol: str, url: str, title: str) -> str:
    """
    Make a deterministic doc id from URL + symbol; fallback to title + symbol.
    """
    base = (url or "") + "|" + (symbol or "") + "|" + (title or "")
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def make_row(doc_id, symbol, kind, text, title, meta, idx):
    return {
        "doc_id": doc_id,
        "symbol": symbol,
        "snippet_id": f"{doc_id}_{kind}_{idx}",
        "snippet_kind": kind,
        "text": text,
        "title": title,
        "published_at": meta.get("published_at",""),
        "source_domain": meta.get("source_domain",""),
        "article_type": meta.get("article_type",""),
        "tier_weight": meta.get("tier_weight", 1.0),
        "position_in_doc": idx,
        "url": meta.get("url",""),
    }


def create_snippets(
    df: pd.DataFrame,
    company_map: dict[str, str] | None = None,
    max_chars: int = 800,
    keep_empty_text: bool = False,
    kinds: set[str] | None = None,
) -> pd.DataFrame:
    kinds = kinds or {"headline", "paragraph", "sentence"}
    company_map = company_map or {}

    # Normalize column access case-insensitively
    cols = {c.lower(): c for c in df.columns}

    def col(name, default=None):
        return cols.get(name.lower(), default)

    req = {"symbol","title","text"}
    missing = [c for c in req if c not in (n.lower() for n in df.columns)]
    if missing:
        print("WARNING: missing columns (case-insensitive):", missing)

    out_rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        symbol = str(r[col("symbol","symbol")]).upper() if col("symbol","symbol") in r else ""
        company = company_map.get(symbol, symbol)
        title = str(r[col("title","title")]).strip() if col("title","title") in r else ""
        body  = str(r[col("text","text")]).strip()  if col("text","text")    in r else ""
        url   = str(r[col("url","url")]).strip()    if col("url","url")      in r else ""
        meta = {
            "published_at": r[col("published_at","published_at")] if col("published_at","published_at") in r else "",
            "source_domain": r[col("source_domain","source_domain")] if col("source_domain","source_domain") in r else "",
            "article_type": r[col("article_type","article_type")] if col("article_type","article_type") in r else "",
            "tier_weight": r[col("tier_weight","tier_weight")] if col("tier_weight","tier_weight") in r else 1.0,
            "url": url,
        }

        if not body and not keep_empty_text and not title:
            continue

        doc_id = stable_doc_id(symbol, url, title)
        idx = 0

        # 1) HEADLINE
        if "headline" in kinds and title and mentions_target(title, symbol, company):
            out_rows.append(make_row(doc_id, symbol, HEADLINE_KIND, title[:max_chars], title, meta, idx))
            idx += 1

        # 2) PARAGRAPHS
        if "paragraph" in kinds and body:
            for p in para_split(body):
                if mentions_target(p, symbol, company):
                    out_rows.append(make_row(doc_id, symbol, PARAGRAPH_KIND, p[:max_chars], title, meta, idx))
                    idx += 1

        # 3) SENTENCES
        if "sentence" in kinds and body:
            for s in sent_tokenize(body):
                if mentions_target(s, symbol, company):
                    out_rows.append(make_row(doc_id, symbol, SENTENCE_KIND, s[:max_chars], title, meta, idx))
                    idx += 1

    return pd.DataFrame(out_rows)


# --- Main logic ---
def main():
    ap = argparse.ArgumentParser(description="Create training snippets (headline/paragraph/sentence) per article.")
    ap.add_argument("--in", dest="input_path", required=True, help="CSV with columns: symbol,title,text,(url,source_domain,published_at,article_type,tier_weight)")
    ap.add_argument("--out", dest="output_path", required=True, help="Output CSV path for snippets.")
    ap.add_argument("--company-map", default="", help='Mapping like "AAPL:Apple;MSFT:Microsoft" to improve matching.')
    ap.add_argument("--max_chars", type=int, default=800, help="Max characters per snippet (truncate).")
    ap.add_argument("--keep-empty-text", action="store_true", help="Keep rows with missing text (not recommended).")
    ap.add_argument("--only-kinds", default="", help='Limit to these kinds (comma list): headline,paragraph,sentence. Empty = all.')
    args = ap.parse_args()

    kinds = set(k.strip().lower() for k in args.only_kinds.split(",") if k.strip()) or {"headline","paragraph","sentence"}
    company_map = parse_company_map(args.company_map)

    df = pd.read_csv(args.input_path)
    out = create_snippets(
        df,
        company_map=company_map,
        max_chars=args.max_chars,
        keep_empty_text=args.keep_empty_text,
        kinds=kinds,
    )
    out.to_csv(args.output_path, index=False)
    print(f"Saved {len(out)} snippets to {args.output_path}")
    if len(out) == 0:
        print("NOTE: 0 snippets produced. Check that 'symbol/title/text' columns exist and your company-map contains readable names.")

if __name__ == "__main__":
    main()
