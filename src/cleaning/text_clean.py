from pathlib import Path
import sys, re, json, html, argparse, os
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import yaml

# -----------------------------
# Constants / Defaults
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_INPUT_DIR   = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "data" / "clean_raw"

DEFAULT_MAX_SENT_LEN = 2000   # 单条 text 最大长度
DEFAULT_HARD_WRAP_LEN = 2800  # 兜底强切长度

SCALE_MAP = {
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

CURRENCY_TOKENS = {
    "$": "USD", "usd": "USD", "us$": "USD",
    "eur": "EUR", "€": "EUR",
    "cny": "CNY", "rmb": "CNY", "¥": "CNY",
    "jpy": "JPY", "¥¥": "JPY",
    "gbp": "GBP", "£": "GBP",
}

# -----------------------------
# Regexes
# -----------------------------
RE_CIK = re.compile(r"\b0\d{9}\b")
RE_DATE_ISO = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
RE_XBRL_QNAME = re.compile(r"\b[a-z][a-z0-9\-]*:[A-Za-z0-9\.]+Member\b", re.IGNORECASE)
RE_SOFT_SPLIT = re.compile(
    r"(?:(?<=\.)\s+)|"
    r"(?<=;)\s+|(?<=:)\s+|"
    r"(?<=—)\s+|(?<=-)\s+|"
    r"(?=\bus-gaap:|\baapl:|\bxbrli:|\biso4217:)|"
    r"(?=" + RE_DATE_ISO.pattern + r")",
    re.IGNORECASE
)

RE_NUMBER = re.compile(r"""
    (?P<prefix>[$€£¥])?
    (?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    \s*
    (?P<unit>billion|million|thousand|bn|mn|m|k)?
""", re.IGNORECASE | re.VERBOSE)



RE_PERCENT = re.compile(r"(?P<pct>\d+(?:\.\d+)?)\s*%")
RE_CURRENCY_WORD = re.compile(r"\b(USD|EUR|CNY|RMB|JPY|GBP)\b", re.IGNORECASE)
RE_ISO_DUR = re.compile(r"\bP(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<days>\d+)D)?\b", re.IGNORECASE)
RE_MEMBER_NEARBY = re.compile(r"\b[A-Za-z0-9:\.\-]*Member\b")
RE_MONTH_CONTEXT = re.compile(r"\b(month|months|preceding\s+\d+\s+months?)\b", re.IGNORECASE)
RE_LEGAL_RULE = re.compile(r"\b(Rule|Item)\s+\d+[a-z0-9\-]*\b", re.IGNORECASE)
RE_LEGAL_SECTION = re.compile(r"\bSection\s+\d+(?:\([a-z]\))?\b", re.IGNORECASE)
RE_LEGAL_PAR = re.compile(r"§\s*\d+(?:\.\d+)*", re.IGNORECASE)
RE_12B2 = re.compile(r"\b12b-2\b", re.IGNORECASE)
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
RE_DATE_LONG = re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}},\s*(?:19|20)\d{{2}}\b", re.IGNORECASE)
RE_HYPHEN_ID = re.compile(r"\b\d{3}-\d{5}\b")
RE_COMMISSION_FILE = re.compile(r"\bCommission\s+File\s+Number\b", re.I)
RE_ZIP = re.compile(r"^\d{5}(?:-\d{4})?$")


RE_HEADER_NOISE = re.compile(r"""
    ^\s*(table\s+of\s+contents|contents|index|page\s+\d+|exhibit\s+\d+|signature[s]?|
    united\s+states|securities\s+and\s+exchange\s+commission)\s*$
""", re.IGNORECASE | re.VERBOSE)

RE_SHORT_PUNC = re.compile(r"^[\W_]+$")  # only punctuation/whitespace

# -----------------------------
# Utilities
# -----------------------------
def _abs_from_project(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else (PROJECT_ROOT / q)

def read_config(cfg_path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            paths = raw.get("paths") or {}
            in_dir  = _abs_from_project(paths.get("processed_dir"))
            out_dir = _abs_from_project(paths.get("clean_dir"))
            if in_dir:  cfg["input_dir"]  = in_dir
            if out_dir: cfg["output_dir"] = out_dir

            tc = raw.get("text_clean") or {}
            if "pattern" in tc:            cfg["pattern"] = tc["pattern"]
            if "min_chars" in tc:          cfg["min_chars"] = int(tc["min_chars"])
            if "max_sentence_len" in tc:   cfg["max_sentence_len"] = int(tc["max_sentence_len"])
            if "hard_wrap_len" in tc:      cfg["hard_wrap_len"] = int(tc["hard_wrap_len"])
        except Exception:
            pass
    return cfg

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def to_ascii(s: str) -> str:
    repl = {"—": "-", "–": "-", "•": "*", "§": "S", "☒": "[X]", "☐": "[ ]", "\xa0": " "}
    out = [repl.get(ch, ch) for ch in s]
    s2 = "".join(out)
    s2 = re.sub(r"[ \t]+", " ", s2)
    s2 = re.sub(r"\s*\n\s*", "\n", s2)
    return s2.strip()

def unescape_and_normalize(text: str) -> str:
    if text is None:
        return ""
    s = html.unescape(text)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def is_noise_line(s: str, min_chars: int) -> bool:
    if not s:
        return True
    if len(s) < min_chars:
        if (RE_NUMBER.search(s) or RE_PERCENT.search(s) or RE_ISO_DUR.search(s) or ("☒" in s) or ("☐" in s)):
            pass
        else:
            return True
    if RE_HEADER_NOISE.search(s):
        return True
    if RE_SHORT_PUNC.match(s):
        return True
    return False

def split_sentences(s: str) -> List[str]:
    s = s.replace("。", ".").replace("；", ";").replace("！", "!").replace("？", "?")
    s = re.sub(r"([\.!?;：:])(\s|$)", r"\1\n", s)
    parts = [p.strip() for p in s.split("\n") if p.strip()]
    return parts

def chunk_long_text(s: str, max_len: int = DEFAULT_MAX_SENT_LEN, hard_wrap: int = DEFAULT_HARD_WRAP_LEN) -> List[str]:
    if len(s) <= max_len:
        return [s]
    parts: List[str] = []
    cur = s
    while len(cur) > max_len:
        cut = None
        for m in RE_SOFT_SPLIT.finditer(cur):
            if m.start() <= max_len:
                cut = m.start()
            else:
                break
        if cut is None or cut < max_len * 0.6:
            cut = min(len(cur), hard_wrap)
        parts.append(cur[:cut].strip())
        cur = cur[cut:].lstrip()
    if cur:
        parts.append(cur)
    return [p for p in parts if p]

def _to_float(num_str: str) -> Optional[float]:
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return None

def _spans_union(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]

def _inside_any(i: int, spans: List[Tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= i < b:
            return True
    return False

def parse_percent_tokens(s: str) -> Tuple[List[str], List[Optional[float]]]:
    raw, vals = [], []
    for m in RE_PERCENT.finditer(s):
        raw.append(m.group(0))
        try:
            vals.append(float(m.group("pct")) / 100.0)
        except Exception:
            vals.append(None)
    return raw, vals

def parse_currency_tokens(s: str) -> List[str]:
    tokens = set()
    for sym, code in CURRENCY_TOKENS.items():
        if sym in {"$", "€", "£", "¥", "¥¥"} and sym in s:
            tokens.add(code)
    for m in RE_CURRENCY_WORD.finditer(s):
        tokens.add(m.group(0).upper().replace("RMB", "CNY"))
    return sorted(tokens)

def parse_iso_durations(s: str) -> Tuple[List[str], List[Optional[int]]]:
    raws, totals = [], []
    for m in RE_ISO_DUR.finditer(s):
        raws.append(m.group(0))
        y = int(m.group("years") or 0)
        mo = int(m.group("months") or 0)
        d = int(m.group("days") or 0)
        totals.append(y*365 + mo*30 + d)
    return raws, totals

def _next_word_after(s: str, end_idx: int) -> str:
    m = re.search(r"\S+", s[end_idx:])
    return m.group(0) if m else ""

def parse_number_tokens(s: str) -> Tuple[List[str], List[Optional[float]]]:
    legal_spans: List[Tuple[int, int]] = []
    for rx in (RE_LEGAL_RULE, RE_LEGAL_SECTION, RE_LEGAL_PAR, RE_12B2):
        for m in rx.finditer(s):
            legal_spans.append(m.span())
    legal_spans = _spans_union(legal_spans)

    raw, vals = [], []
    for m in RE_NUMBER.finditer(s):
        start, end = m.span()
        if _inside_any(start, legal_spans):
            continue

        token = m.group(0).strip()
        num = _to_float(m.group("num"))
        unit = m.group("unit").lower() if m.group("unit") else None

        window = s[max(0, start-30):min(len(s), end+30)]

        if RE_MEMBER_NEARBY.search(window) or RE_XBRL_QNAME.search(window):
            continue
        if RE_DATE_ISO.search(window):
            continue
        if RE_CIK.search(window):
            continue
        if RE_DATE_LONG.search(window):
            continue
        if RE_HYPHEN_ID.search(window):
            continue
        if RE_DATE_LONG.search(window):
            continue
        if RE_HYPHEN_ID.search(window):
            continue


        suppress_scale = False
        if unit in {"m"} and RE_MONTH_CONTEXT.search(window):
            suppress_scale = True

        scale = SCALE_MAP.get(unit, 1.0) if (unit and not suppress_scale) else 1.0
        if num is not None:
            num *= scale

        raw.append(token)
        vals.append(num)
    return raw, vals

# -----------------------------
# Core pipeline
# -----------------------------
def process_jsonl_line(
    line: Dict[str, Any],
    *, 
    min_chars: int,
    max_sent_len: int = DEFAULT_MAX_SENT_LEN,
    hard_wrap_len: int = DEFAULT_HARD_WRAP_LEN
) -> List[Dict[str, Any]]:
    idx = line.get("idx") or line.get("idx_source")
    tag = line.get("tag")

    text_raw_in = line.get("text_raw")
    text_in = line.get("text")
    base_text = text_raw_in if text_raw_in else (text_in or "")
    text_norm = unescape_and_normalize(base_text)

    if is_noise_line(text_norm, min_chars=min_chars):
        return []

    out_rows: List[Dict[str, Any]] = []
    for sent in split_sentences(text_norm):
        if is_noise_line(sent, min_chars=min_chars):
            continue

        # 超长句切分
        segments = chunk_long_text(sent, max_len=max_sent_len, hard_wrap=hard_wrap_len)
        for seg in segments:
            if is_noise_line(seg, min_chars=min_chars):
                continue

            pct_raw, pct_vals = parse_percent_tokens(seg)
            num_raw, num_vals = parse_number_tokens(seg)
            currs = parse_currency_tokens(seg)
            dur_raw, dur_days = parse_iso_durations(seg)

            has_checked = "☒" in seg
            has_unchecked = "☐" in seg

            out_rows.append({
                "idx_source": idx,
                "tag": tag,
                "text_raw": base_text,
                "text": seg,
                "text_ascii": to_ascii(seg),
                "length": len(seg),

                "numbers_raw": num_raw,
                "numbers": num_vals,
                "percents_raw": pct_raw,
                "percents": pct_vals,
                "currencies": currs,

                "durations_raw": dur_raw,
                "durations_days": dur_days,

                "has_numbers": bool(num_raw),
                "has_percents": bool(pct_raw),
                "has_checked": has_checked,
                "has_unchecked": has_unchecked,
            })
    return out_rows

def clean_one_file(
    input_path: Path, 
    out_jsonl: Path, 
    out_parquet: Optional[Path], 
    *, 
    min_chars: int = 30,
    max_sent_len: int = DEFAULT_MAX_SENT_LEN,
    hard_wrap_len: int = DEFAULT_HARD_WRAP_LEN
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total = 0
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.extend(process_jsonl_line(
                obj, 
                min_chars=min_chars,
                max_sent_len=max_sent_len,
                hard_wrap_len=hard_wrap_len
            ))

    ensure_dir(out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for r in rows:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    parquet_written = None
    csv_fallback = None
    if out_parquet is not None:
        try:
            ensure_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            parquet_written = str(out_parquet)
        except Exception:
            csv_fallback = str(out_parquet.with_suffix(".csv"))
            df.to_csv(csv_fallback, index=False)

    return {
        "input_file": str(input_path),
        "lines_read": total,
        "sentences": len(rows),
        "out_jsonl": str(out_jsonl),
        "out_parquet": parquet_written,
        "out_csv_fallback": csv_fallback,
    }

def main():
    cfg = read_config(DEFAULT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Clean text.jsonl into sentence-level corpus with numeric/percent/duration extraction.")
    in_grp = parser.add_mutually_exclusive_group(required=False)
    in_grp.add_argument("--input_file", help="Path to a single text.jsonl")
    in_grp.add_argument("--input_dir", help="Directory to scan recursively for files")

    parser.add_argument("--output_jsonl", help="Output JSONL (only when --input_file is used)")
    parser.add_argument("--output_parquet", help="Output Parquet (only when --input_file is used; falls back to CSV)")
    parser.add_argument("--output_dir", help="Base output dir when --input_dir is used")
    parser.add_argument("--pattern", default=None, help="Filename pattern (default from config or 'text.jsonl')")
    parser.add_argument("--min_chars", type=int, default=None, help="Minimum chars unless numbers/percents/durations/checkboxes present")
    parser.add_argument("--max_sentence_len", type=int, default=None, help="Max chars per sentence/chunk")
    parser.add_argument("--hard_wrap_len", type=int, default=None, help="Emergency wrap length")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir) if args.input_dir else cfg.get("input_dir", DEFAULT_INPUT_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    pattern    = args.pattern or cfg.get("pattern", "text.jsonl")
    min_chars  = args.min_chars if args.min_chars is not None else cfg.get("min_chars", 30)
    max_sent_len = args.max_sentence_len if args.max_sentence_len is not None else cfg.get("max_sentence_len", DEFAULT_MAX_SENT_LEN)
    hard_wrap_len = args.hard_wrap_len if args.hard_wrap_len is not None else cfg.get("hard_wrap_len", DEFAULT_HARD_WRAP_LEN)

    results: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    if args.input_file:
        in_file = Path(args.input_file)
        out_jsonl = Path(args.output_jsonl) if args.output_jsonl else in_file.with_name("text_corpus.jsonl")
        out_parquet = Path(args.output_parquet) if args.output_parquet else in_file.with_name("text_corpus.parquet")
        res = clean_one_file(
            in_file, out_jsonl, out_parquet,
            min_chars=min_chars,
            max_sent_len=max_sent_len,
            hard_wrap_len=hard_wrap_len
        )
        results.append(res)
        summary = {"mode": "single_file"}
    else:
        files = list(input_dir.rglob(pattern))
        if args.verbose:
            print(f"[INFO] Found {len(files)} files under {input_dir} matching '{pattern}'")
        for f in files:
            rel = f.relative_to(input_dir)
            out_base_dir = output_dir / rel.parent
            out_jsonl = out_base_dir / "text_corpus.jsonl"
            out_parquet = out_base_dir / "text_corpus.parquet"
            res = clean_one_file(
                f, out_jsonl, out_parquet,
                min_chars=min_chars,
                max_sent_len=max_sent_len,
                hard_wrap_len=hard_wrap_len
            )
            results.append(res)
        summary = {"mode": "directory", "input_dir": str(input_dir), "output_dir": str(output_dir), "files": len(results)}

    agg = {"summary": summary, "results": results}
    print(json.dumps(agg, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
