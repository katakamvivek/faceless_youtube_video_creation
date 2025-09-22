#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, time, glob, argparse, textwrap
from urllib.parse import urlparse
import requests, certifi

# --- third party extraction ---
from bs4 import BeautifulSoup
from readability import Document

# --- Playwright (sync API works great outside notebooks) ---
from playwright.sync_api import sync_playwright


# ==========================
# Config / Defaults
# ==========================
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0 Safari/537.36"
)

# ==========================
# Utilities
# ==========================

def load_keys_file(path: str) -> dict:
    """
    Reads a simple KEY=VALUE file and returns dict.
    Lines starting with # are ignored.
    """
    keys = {}
    if not path or not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip().strip('"').strip("'")
    return keys

def _safe_name(s: str, maxlen: int = 80) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (s or "").strip()).strip("_")[:maxlen]

def _read_domains(path: str):
    domains = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("http://", "").replace("https://", "").strip("/")
            domains.append(line)
    return domains

def _amp_candidates(url: str):
    u = url.rstrip("/")
    return [u + "/amp", u + "?amp", u + "?amp=1"]


# ==========================
# Core functions (names kept)
# ==========================
def search_serpapi_site_strict(domain: str, movie: str, api_key: str, max_results: int = 5,
                               google_domain: str = "google.com", gl: str = "us", hl: str = "en"):
    """
    Google via SerpAPI:
      q = site:<domain> "<movie>" review
    Keep only results:
      - on the domain
      - title/snippet mentions movie
      - URL path contains BOTH 'review' AND movie tokens (hyphen/space tolerant)
    """
    import requests
    from urllib.parse import urlparse
    print(domain)
    if domain.lower() == "www.idlebrain.com":
        query = f'site:idlebrain.com/movie/archive/ "{movie}"'
        print(query)
    else:
        query = f'site:{domain} "{movie}" review'

    #query = f'site:{domain} "{movie}" review'
    print (query)
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "google_domain": google_domain,
        "gl": gl,
        "hl": hl,
        "device": "desktop",
        "safe": "off",
        "api_key": api_key,
    }
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
    if r.status_code != 200:
        print("  SerpAPI error:", r.status_code, r.text[:200])
    r.raise_for_status()
    data = r.json()

    movie_l = movie.lower()
    movie_tokens = movie_l.split()
    urls = []
    for it in data.get("organic_results", []):
        link = it.get("link")
        title = (it.get("title") or "").lower()
        snippet = (it.get("snippet") or "").lower()
        if not link:
            continue

        p = urlparse(link)
        netloc = p.netloc.lower()
        path = p.path.lower()

        # domain guard
        if not netloc.endswith(domain.replace("www.", "").lower()):
            continue
        # title/snippet must mention movie
        
        if "idlebrain" in link.lower():
            if movie_l in title or movie_l in snippet:
                urls.append(link)
                
                if len(urls) >= max_results:
                    break
            continue
        if (movie_l not in title) and (movie_l not in snippet):
            continue
        # URL path must contain 'review' and all tokens of the movie title
        if "review" not in path:
            continue
        if not all(tok in path for tok in movie_tokens):
            continue
       
        urls.append(link)
        if len(urls) >= max_results:
            break
    print(urls)
    return urls


def search_serpapi_site_loose(domain: str, movie: str, api_key: str, max_results: int = 5,
                              google_domain: str = "google.com", gl: str = "us", hl: str = "en"):
    """
    Fallback query with looser filters:
      site:<domain> "<movie>" (review OR "film review" OR "movie review")
    Keep if on domain and movie in title or snippet. No in-path constraints.
    """
    import requests
    from urllib.parse import urlparse

    query = f'site:{domain} "{movie}" (review OR "film review" OR "movie review")'
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "google_domain": google_domain,
        "gl": gl,
        "hl": hl,
        "device": "desktop",
        "safe": "off",
        "api_key": api_key,
    }
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
    if r.status_code != 200:
        print("  SerpAPI fallback error:", r.status_code, r.text[:200])
    r.raise_for_status()
    data = r.json()

    movie_l = movie.lower()
    urls = []
    for it in data.get("organic_results", []):
        link = it.get("link")
        title = (it.get("title") or "").lower()
        snippet = (it.get("snippet") or "").lower()
        if not link:
            continue
        if urlparse(link).netloc.endswith(domain.replace("www.", "").lower()) and \
           (movie_l in title or movie_l in snippet):
            urls.append(link)
        if len(urls) >= max_results:
            break
    print(urls)    
    return urls


def fetch_html(url: str, timeout: int = 25, user_agent: str = DEFAULT_USER_AGENT) -> str:
    """Basic GET with requests; verify via certifi."""
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout, verify=certifi.where())
    resp.raise_for_status()
    return resp.text


def fetch_html_playwright(url: str, user_agent: str,
                          wait_selector: str = "article, .article, p",
                          timeout_ms: int = 25000) -> str:
    """Render with Playwright (sync API). Works well outside notebooks."""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="Asia/Kolkata",
        )
        page = ctx.new_page()
        page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        # give JS/cookie walls a chance to settle
        try:
            page.wait_for_selector(wait_selector, timeout=timeout_ms)
        except Exception:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            pass
        html = page.content()
        ctx.close()
        browser.close()
        return html


def extract_main_text(html: str):
    """Readability -> clean text; prefer <h1> if it looks like the article title."""
    doc = Document(html)
    short = (doc.short_title() or "").strip()
    main_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(main_html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    # Prefer a real article heading when available
    h1 = soup.find("h1")
    title = short
    if h1:
        h1t = h1.get_text(" ", strip=True)
        if "review" in h1t.lower() or (len(h1t) > len(short) + 5):
            title = h1t

    text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all(["p","li","blockquote"]))
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return title, text



def save_txt_only(title: str, text: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = _safe_name(title or "review") + ".txt"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ==========================
# Domain-specific helpers
# ==========================
def _try_amp_fetch(url: str, user_agent: str, timeout: int = 25):
    """
    Try common AMP variants first (fast, static; great for Indiaglitz and others).
    Returns (used_url, html) or (None, None).
    """
    hdrs = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Referer": "https://www.google.com/",
    }
    for amp in _amp_candidates(url):
        try:
            r = requests.get(amp, headers=hdrs, timeout=timeout, verify=certifi.where())
            if r.status_code == 200 and len(r.content) > 1500:
                return amp, r.text
        except Exception:
            pass
    return None, None


# ==========================
# Runner
# ==========================

# Precedence: CLI flag > KEYS FILE > ENV VAR

def run(movie: str, sites_file: str, serpapi_key: str,
        google_domain: str, gl: str, hl: str,
        max_results: int, min_chars: int, user_agent: str,
        allow_insecure_last_resort: bool = False):
    domains = _read_domains(sites_file)
    print(f"Loaded {len(domains)} domain(s): {domains}")

    base_out_dir = f"output_{_safe_name(movie)}"
    os.makedirs(base_out_dir, exist_ok=True)

    total_saved = 0

    for DOMAIN in domains:
        domain_out = os.path.join(base_out_dir, _safe_name(DOMAIN))
        os.makedirs(domain_out, exist_ok=True)

        # Skip if a .txt already exists for this domain
        if glob.glob(os.path.join(domain_out, "*.txt")):
            print(f"\n=== {DOMAIN} ===")
            print(f"  Found existing .txt in '{domain_out}'. Skipping.")
            continue

        print(f"\n=== {DOMAIN} → saving to: {domain_out} ===")

        # 1) strict search
        urls = []
        if serpapi_key:
            try:
                urls = search_serpapi_site_strict(
                    domain=DOMAIN, movie=movie, api_key=serpapi_key,
                    max_results=max_results, google_domain=google_domain, gl=gl, hl=hl
                )
                print(urls)
            except Exception as e:
                print("  Strict search error:", e)

        # 2) loose search fallback
        if not urls and serpapi_key:
            print("  No strict results. Trying fallback search …")
            try:
                urls = search_serpapi_site_loose(
                    domain=DOMAIN, movie=movie, api_key=serpapi_key,
                    max_results=max_results, google_domain=google_domain, gl=gl, hl=hl
                )
            except Exception as e:
                print("  Fallback search error:", e)

        if not urls:
            print("  (no candidate URLs found)")
            continue

        for i, u in enumerate(urls, 1):
            try:
                # Ensure URL is on this domain
                netloc = urlparse(u).netloc.lower()
                if not netloc.endswith(DOMAIN.replace("www.", "").lower()):
                    print(f"  [{i}/{len(urls)}] skip (off-domain): {u}")
                    continue

                # === Unified generic pipeline ===
                title, text, used = _generic_fetch_pipeline(
                    u,
                    user_agent=user_agent,
                    min_chars=min_chars,
                    allow_insecure_last_resort=allow_insecure_last_resort,
                    quality_phrase=movie 
                )

                if not text or len(text) < min_chars:
                    print(f"  [{i}/{len(urls)}] too short after all fallbacks. Skipping.")
                    continue

                path = save_txt_only(title, text, domain_out)
                print(f"  [{i}/{len(urls)}] {used} saved: {path}")
                total_saved += 1
                time.sleep(0.2)

            except Exception as e:
                print(f"  [{i}/{len(urls)}] ERROR: {e}")

    print(f"\n✅ Finished. Saved {total_saved} review file(s) under '{base_out_dir}'.")


def _generic_fetch_pipeline(u: str, user_agent: str, min_chars: int,
                            allow_insecure_last_resort: bool = False,
                            quality_phrase: str | None = None):
    """
    Universal fetch chain for ANY domain:
      1) AMP variants (fast, static)
      2) requests (certifi)  -- strict TLS
      3) Playwright render   -- real browser
      4) last resort (optional): insecure GET (verify=False)

    Returns (title, text, used_method) or (None, "", "") on failure.
    """
    def _passes_quality(title: str, text: str) -> bool:
        if not text or len(text) < min_chars:
            return False
        if not quality_phrase:
            return True
        blob = (title or "") + "\n" + (text or "")
        low = blob.lower()
        # require movie tokens to appear somewhere
        if not all(tok in low for tok in quality_phrase.lower().split()):
            return False
        # require the word "review" somewhere early (title or top of body)
        if "idlebrain.com" in u.lower():
            return True
        head = ((title or "") + " " + (text[:800] or "")).lower()
        return "review" in head

    # 1) AMP first
    amp_url, amp_html = _try_amp_fetch(u, user_agent)
    if amp_html:
        title, text = extract_main_text(amp_html)
        if _passes_quality(title, text):
            return title, text, "amp"

    # 2) requests (certifi)
    html = None
    try:
        html = fetch_html(u, user_agent=user_agent)
    except requests.exceptions.SSLError as e_ssl:
        print("    requests SSL error:", e_ssl)
    except Exception as e_req:
        print("    requests error:", e_req)

    if html:
        t, x = extract_main_text(html)
        if _passes_quality(t, x):
            return t, x, "requests"

    # 3) Playwright render
    try:
        print("    trying Playwright …")
        html2 = fetch_html_playwright(u, user_agent)
        t2, x2 = extract_main_text(html2)
        if _passes_quality(t2, x2):
            return (t2, x2, "playwright")
    except Exception as e_pw:
        print("    Playwright error:", e_pw)

    # 4) optional: insecure last resort (verify=False)
    if allow_insecure_last_resort:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            print("    trying insecure GET (verify=False) …")
            r = requests.get(u, headers={"User-Agent": user_agent}, timeout=25, verify=False)
            r.raise_for_status()
            t3, x3 = extract_main_text(r.text)
            if _passes_quality(t3, x3):
                return (t3, x3, "insecure")
        except Exception as e_insec:
            print("    insecure fetch error:", e_insec)

    return (None, "", "")



# ==========================
# CLI
# ==========================
def main():
    parser = argparse.ArgumentParser(
        description="Movie review crawler (SerpAPI + AMP/Playwright fallbacks). "
                    "Saves ONLY .txt content per domain.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--movie", required=True, help="Movie title (just the name, e.g., 'coolie').")
    parser.add_argument("--sites", required=True, help="Path to file with domains (one per line).")
    parser.add_argument("--max_results", type=int, default=5, help="Top N links per site (default: 5).")
    parser.add_argument("--google_domain", default="google.co.in", help="Google domain (e.g., google.co.in).")
    parser.add_argument("--gl", default="in", help="Geolocation/country code (default: in).")
    parser.add_argument("--hl", default="en", help="Language code (default: en).")
    parser.add_argument("--min_chars", type=int, default=500, help="Minimum extracted chars to accept/save.")
    parser.add_argument("--user_agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent.")
    parser.add_argument("--serpapi_key", default=os.getenv("SERPAPI_KEY", ""), help="SerpAPI key (or set SERPAPI_KEY env).")
    parser.add_argument("--keys_file", default="", help="Path to keys file (KEY=VALUE)")
    parser.add_argument(
        "--allow_insecure_last_resort",
        action="store_true",
        help="If all secure methods fail, do one last fetch with verify=False (use only for trusted sites)."
    )

    args = parser.parse_args()
    keys = load_keys_file(args.keys_file) if args.keys_file else {}
    print("Loaded keys dict:", keys)

    serpapi_key = (
        (args.serpapi_key or "").strip()
        or str(keys.get("SERPAPI_KEY", "")).strip()
        or os.getenv("SERPAPI_KEY", "").strip()
    )

    if not serpapi_key:
        print(
            "[WARN] SERPAPI_KEY is not set (env, --serpapi_key, or keys file).\n"
            "       The script will still run, but search will return no URLs.\n"
            "       Set it with:\n"
            "         set SERPAPI_KEY=YOUR_KEY    (Windows CMD)\n"
            '         $env:SERPAPI_KEY="YOUR_KEY" (PowerShell)\n'
        )
    else:
        print("Using SERPAPI_KEY:", serpapi_key[:6] + "…")

    run(
        movie=args.movie,
        sites_file=args.sites,
        serpapi_key=serpapi_key,  # (keep your fixed resolved key here)
        google_domain=args.google_domain,
        gl=args.gl,
        hl=args.hl,
        max_results=args.max_results,
        min_chars=args.min_chars,
        user_agent=args.user_agent,
        allow_insecure_last_resort=args.allow_insecure_last_resort,  # <-- new
    )


if __name__ == "__main__":
    main()

#python movie_reviews_crawler.py --movie mirai --sites sites.txt --keys_file key.env --allow_insecure_last_resort
