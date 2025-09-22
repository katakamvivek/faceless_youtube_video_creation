#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, pathlib, csv, traceback
from typing import Dict, List
from textwrap import shorten
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, ValidationError, field_validator
from dataclasses import dataclass, asdict
# -------- CONFIG --------
ROOT_DIR   = os.environ.get("REVIEWS_ROOT", r"E:\Agentic_Ai\output_mirai")  # ROOT/<domain>/*.txt
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "review_summaries.csv")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_CHARS_PER_DOMAIN = int(os.environ.get("MAX_CHARS_PER_DOMAIN", "40000"))
RETRY = 3
BACKOFF = 2.0
TIMEOUT = 60

print(f"[CONFIG] ROOT_DIR={ROOT_DIR}")
print(f"[CONFIG] OUTPUT_CSV={OUTPUT_CSV}")
print(f"[CONFIG] OPENAI_MODEL={OPENAI_MODEL}")

# -------- KEYS: load once and show logs --------
def load_keys_file(path: str) -> dict:
    keys = {}
    if not path or not os.path.exists(path):
        print(f"[WARN] Keys file {path} not found.")
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip().strip('"').strip("'")
    print(f"[INFO] Loaded {len(keys)} keys from {path}")
    return keys

KEYS_FILE = os.environ.get("KEYS_FILE", "E:\Agentic_Ai\key.env")
print(f"[CONFIG] KEYS_FILE={os.path.abspath(KEYS_FILE)}")

file_keys = load_keys_file(KEYS_FILE)
# Show redacted keys in logs
print("[INFO] Keys snapshot:", {k: ("***" if "KEY" in k else v) for k, v in file_keys.items()})

# Merge into env so the rest of the script can read via os.environ.get(...)
for k, v in file_keys.items():
    os.environ[k] = v  # file overrides env; switch to setdefault if you prefer the opposite

# Validate critical values
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing. Put it in keys.env or set env var.")


# -------- RATING (EXPLICIT ONLY) --------
# How far to scan after the word "Rating" (can override via env)
RATING_SCAN_WINDOW = int(os.environ.get("RATING_SCAN_WINDOW", "60"))

# Match the label "Rating" with optional separators/verbs
RATING_LABEL = re.compile(r"\brating\b\s*(?:is|=)?\s*[:\-–—]?\s*", re.I)

# Stars next to Rating (support both ★ and ⭐; empties ☆/✩ are optional)
STAR_SYMBOLS_TIGHT = re.compile(r"([★⭐]{1,5})([☆✩]{0,4})")

# Numeric forms inside the scan window
FRACTION_ANY = re.compile(r"([0-9]{1,2}(?:\.[0-9]+)?)\s*/\s*(5|10|100)", re.I)
OUTOF_ANY    = re.compile(r"([0-9]{1,2}(?:\.[0-9]+)?)\s*(?:out of)\s*(5|10|100)", re.I)
PERCENT_ANY  = re.compile(r"(100|[1-9][0-9]?)\s*%", re.I)


def extract_explicit_rating(text: str) -> str:
    """
    Look for the word 'Rating' and scan only the next RATING_SCAN_WINDOW characters
    for stars or numeric ratings. Returns '' if nothing is found.
    Examples matched:
      - 'Rating: 3.25 / 5'
      - 'Rating – ★★★☆☆'
      - 'Rating: 80%'
      - 'Rating 7/10'
      - 'Rating is 3.5 out of 5'
    """
    if not text:
        return ""

    for label_match in RATING_LABEL.finditer(text):
        start = label_match.end()
        snippet = text[start : start + RATING_SCAN_WINDOW]

        # 1) Stars like ★★★★☆ or ⭐⭐⭐⭐
        m = STAR_SYMBOLS_TIGHT.search(snippet)
        if m:
            return f"{len(m.group(1))}/5"

        # 2) Fractions like 3.25 / 5 or 7/10 or 60/100
        m = FRACTION_ANY.search(snippet)
        if m:
            return f"{m.group(1)}/{m.group(2)}"

        # 3) "3.5 out of 5"
        m = OUTOF_ANY.search(snippet)
        if m:
            return f"{m.group(1)}/{m.group(2)}"

        # 4) Percentages like 85%
        m = PERCENT_ANY.search(snippet)
        if m:
            return f"{m.group(1)}%"

    # No rating after any 'Rating' label
    return ""



# -------- FILE IO --------
def load_domain_files(root: str) -> Dict[str, List[pathlib.Path]]:
    root_p = pathlib.Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"Root dir not found: {root_p.resolve()}")
    mapping: Dict[str, List[pathlib.Path]] = {}
    for domain_dir in sorted([p for p in root_p.iterdir() if p.is_dir()]):
        txts = sorted(domain_dir.glob("*.txt"))
        if txts:  # ← only include domains that actually have review files
            mapping[domain_dir.name] = txts
            print(f"[INFO] Found {len(txts)} txt files under {domain_dir.name}")
        else:
            print(f"[SKIP] {domain_dir.name}: no .txt files; skipping domain.")
    return mapping


def read_and_concat(paths: List[pathlib.Path], cap: int) -> str:
    chunks, n = [], 0
    for fp in paths:
        try:
            t = fp.read_text(encoding="utf-8", errors="ignore")
            print(f"[INFO] Reading {fp} ({len(t)} chars)")
        except Exception as e:
            print(f"[ERROR] Could not read {fp}: {e}")
            t = f"[READ_ERROR {fp.name}] {e}"
        if n + len(t) > cap:
            print(f"[WARN] Truncating {fp.name} to cap {cap} chars")
            t = t[: max(0, cap - n)]
        chunks.append(f"\n\n===== FILE: {fp.name} =====\n{t}")
        n += len(t)
        if n >= cap: 
            print("[WARN] Cap reached, stopping further file reads.")
            break
    return "".join(chunks).strip()

def load_keys_file(path: str) -> dict:
    keys = {}
    if not path or not os.path.exists(path):
        print(f"[WARN] Keys file {path} not found.")
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip().strip('"').strip("'")
    print(f"[INFO] Loaded {len(keys)} keys from {path}")
    return keys


# Show redacted keys in logs
print("[INFO] Keys snapshot:", {k: ("***" if "KEY" in k else v) for k, v in file_keys.items()})

# -------- LLM (summary + verdict only; NO rating inference) --------


from typing import List, Literal
# for pydantic v2
from pydantic import BaseModel, ValidationError, field_validator
# or for v1: from pydantic import BaseModel, ValidationError, validator as field_validator

class SummaryVerdict(BaseModel):
    summary: str
    final_verdict: Literal["very good", "average", "disappointed"]
    pros: List[str] = []
    cons: List[str] = []

    @field_validator("summary")
    @classmethod
    def _non_empty_summary(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("summary is empty")
        return v

    @field_validator("final_verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, v: str) -> str:
        return (v or "").strip().lower()

    # Trim items and drop empties; keep it tight (max 5)
    @field_validator("pros", "cons", mode="before")
    @classmethod
    def _normalize_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # if the model returned a single string, wrap it
            v = [v]
        if isinstance(v, list):
            out = []
            for s in v:
                if isinstance(s, str):
                    s = s.strip()
                    if s:
                        out.append(s)
            return out[:5]
        return []


ALLOWED_VERDICTS = {"very good", "average", "disappointed"}


def verdict_from_rating_or_llm(rating_str: str, llm_verdict: str) -> str:
    """
    If rating_str is present (and not NO_RATING), compute verdict from rating:
      - >= 4.0/5  -> 'very good'
      - <  2.5/5  -> 'disappointed'
      - else      -> 'average'
    If rating_str is blank or NO_RATING, return normalized llm_verdict.
    """
    rs = (rating_str or "").strip()

    # Treat NO_RATING (if you use that sentinel) or empty string as "no rating"
    if not rs or rs.upper() == "NO RATING" or rs == "NO_RATING":
        v = (llm_verdict or "").strip().lower()
        return v if v in ALLOWED_VERDICTS else ""

    # Try to parse rating formats: "x/5", "x/10", "x/100", or "x%"
    frac = re.match(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(5|10|100)\s*$", rs)
    perc = re.match(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$", rs)

    value_5 = None
    if frac:
        val = float(frac.group(1))
        den = int(frac.group(2))
        if den == 5:
            value_5 = val
        elif den == 10:
            value_5 = val / 2.0
        elif den == 100:
            value_5 = val / 20.0
    elif perc:
        val = float(perc.group(1))
        value_5 = val / 20.0

    # If parsed successfully, map to verdict by your scale/thresholds
    if value_5 is not None:
        value_5 = max(0.0, min(5.0, value_5))
        if value_5 >= 3.5:
            return "very good"        # e.g., 4/5 ≈ 8/10 ≈ 80%
        if value_5 < 2.5:
            return "disappointed"     # e.g., 2/5 ≈ 4/10 ≈ 40%
        return "average"              # around 3/5 ≈ 6/10 ≈ 60%

    # If rating couldn't be parsed, fall back to LLM verdict
    v = (llm_verdict or "").strip().lower()
    return v if v in ALLOWED_VERDICTS else ""


def llm_summarize_and_verdict(domain: str, combined_text: str) -> SummaryVerdict:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    model   = os.environ.get("OPENAI_MODEL", OPENAI_MODEL)
    print(f"[INFO] LLM config for {domain}: model={model}")

    client = OpenAI(api_key=api_key, timeout=TIMEOUT)

    sys_prompt = (
    "You are given combined movie review text from a single website domain. "
    "Your task is to summarize ONLY the movie review content.\n"
    "Do NOT include or consider box office collections, financial data, or unrelated details. "
    "Focus purely on what the reviewer is saying about the movie.\n"
    "Rules:\n"
    "- Summary: exactly 2–3 sentences, clear, no emojis.\n"
    "- The summary should reflect the website’s overall opinion so that a reader "
    "immediately understands what that site thinks about the movie.\n"
    "- Also extract concise Pros (what worked) and Cons (what didn’t) as short bullet phrases "
    "(aim for 2–5 each, no spoilers, no emojis).\n"
    "- Final verdict reflects the sentiment and recommendation in the review text "
    "(choose: very good / average / disappointed).\n"
    "- Do NOT invent or infer a numeric rating. Leave that to the caller.\n"
    "- Respond ONLY with JSON using:\n"
    "{\n"
    '  "summary": "2–3 sentences",\n'
    '  "final_verdict": "very good" | "average" | "disappointed",\n'
    '  "pros": ["short phrase", "short phrase"],\n'
    '  "cons": ["short phrase", "short phrase"]\n'
    "}\n"
    )

    user_prompt = f"Domain: {domain}\n\nCombined review text (may be truncated):\n\"\"\"{combined_text}\"\"\""

    last_err = None
    for i in range(RETRY):
        try:
            print(f"[INFO] Calling LLM for domain={domain}, attempt {i+1}")
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                # If your runtime supports it, you can enforce JSON mode:
                # response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            print(f"[DEBUG] Raw LLM response (first 120 chars) for {domain}: {raw[:120000000]}...")

            # Extract the JSON object from the raw text
            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1:
                raise ValueError("JSON braces not found in LLM output")

            data = json.loads(raw[s:e+1])

            # Normalize fields then validate with Pydantic
            
            normalized = {
                "summary": (data.get("summary") or "").strip(),
                "final_verdict": (data.get("final_verdict") or "").strip().lower(),
                "pros": [s.strip() for s in (data.get("pros") or []) if isinstance(s, str) and s.strip()],
                "cons": [s.strip() for s in (data.get("cons") or []) if isinstance(s, str) and s.strip()],
            }

            return SummaryVerdict(**normalized)

        except ValidationError as ve:
            print(f"[WARN] Pydantic validation failed for {domain}: {ve}")
            last_err = ve
        except Exception as e:
            print(f"[ERROR] LLM call/parse failed for {domain} (attempt {i+1}): {e}")
            last_err = e

        if i < RETRY - 1:
            time.sleep(BACKOFF ** i)
        else:
            traceback.print_exc()
            raise last_err



# -------- PIPELINE --------
@dataclass
class Row:
    domain: str
    summary: str
    pros: str
    cons: str
    rating: str
    final_verdict: str


def process_domain(domain: str, files: List[pathlib.Path]) -> Row:
    print(f"[INFO] Processing domain {domain} with {len(files)} files")
    combined = read_and_concat(files, MAX_CHARS_PER_DOMAIN)
    rating_str = extract_explicit_rating(combined)
    if rating_str == "NO_RATING":
        print(f"[INFO] No explicit rating found for {domain}")
    else:
        print(f"[INFO] Extracted rating for {domain}: {rating_str}")

    print(f"[CONFIG] KEYS_FILE={os.path.abspath(KEYS_FILE)}")

    file_keys = load_keys_file(KEYS_FILE)
    llm = llm_summarize_and_verdict(domain, combined)
    print(f"[DEBUG] LLM result type for {domain}: {type(llm).__name__}")
    
    summary = llm.summary.strip()
    verdict = (llm.final_verdict or "").strip().lower()
    pros_str = " | ".join(llm.pros or [])
    cons_str = " | ".join(llm.cons or [])
    final_v = verdict_from_rating_or_llm(rating_str, verdict)
    if final_v not in {"very good", "average", "disappointed"}:
        final_v = ""
    
    row = Row(
    domain=domain,
    summary=summary,
    pros=pros_str,
    cons=cons_str,
    rating=rating_str,
    final_verdict=final_v,
    )
    print(f"[DEBUG] Built Row for {domain}: {asdict(row)}")
    return row
   

def main():
    print("[INFO] Starting summarization pipeline...")
    
    domain_map = load_domain_files(ROOT_DIR)
    if not domain_map:
        print(f"[INFO] No domain folders with *.txt under {ROOT_DIR}")
        return

    rows: List[Row] = []
    for domain, files in sorted(domain_map.items(), key=lambda kv: kv[0].lower()):
        try:
            row = process_domain(domain, files)
            rows.append(row)
            print(f"[DEBUG] Appended row: {asdict(row)}")
        except Exception as e:
            print(f"[ERROR] Failed to process {domain}: {e}")
            traceback.print_exc()
            rows.append(Row(domain, f"[ERROR] {e}", "", ""))

    print("\n=== SUMMARY TABLE (one row per domain) ===")
    print(f"{'Domain':30} | {'Rating':10} | {'Final Verdict':13} | Summary")
    print("-"*120)
    for r in rows:
        summary_text = (r.summary or "").replace("\n", " ")
        pros_text    = (getattr(r, "pros", "") or "").replace("\n", " ")
        cons_text    = (getattr(r, "cons", "") or "").replace("\n", " ")

        summary_cell = shorten(summary_text, 120, placeholder="…")
        pros_cell    = shorten(pros_text,    80,  placeholder="…")
        cons_cell    = shorten(cons_text,    80,  placeholder="…")

        rating_cell  = r.rating or "-"
        verdict_cell = r.final_verdict or "-"
        print(f"[DEBUG] Row -> domain={r.domain}, rating={rating_cell!r}, verdict={verdict_cell!r}, summary_len={len(summary_text)}")
        print(f"{r.domain:30} | {rating_cell:10} | {verdict_cell:13} | {summary_cell}")
        if pros_cell:
            print(f"  Pros:    {pros_cell}")
        if cons_cell:
            print(f"  Cons:    {cons_cell}")
        print("-" * 120)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "summarize_movie_review", "pros", "cons", "rating", "final_verdict"])
        for r in rows:
            w.writerow([r.domain, r.summary, r.pros, r.cons, r.rating, r.final_verdict])

    print(f"\n[INFO] Saved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
