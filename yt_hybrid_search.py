#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, sys, subprocess
from typing import List, Dict, Tuple
from googleapiclient.discovery import build
from rapidfuzz import fuzz
from openai import OpenAI
import whisper

# ---------- CONFIG ----------
MAX_PER_CHANNEL = 2
UPLOADS_SCAN_PAGES = 3  # ~150 uploads fallback
OUTPUT_ROOT_PREFIX = "output_" 
MAX_RESULTS = 25

# ---------------------------

YOUTUBE = None
OPENAI_KEY = ''  # uses OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_KEY)
# ---------- utils ----------
def safe_slug(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w.\-]+", "_", (s or "").strip())
    return s[:max_len].strip("_")

def title_has_movie_and_review(title: str, movie: str) -> bool:
    t = title.casefold()
    m = movie.casefold()
    return (m in t) and ("review" in t)
# ---------------------------

def init_youtube(api_key: str):
    global YOUTUBE
    YOUTUBE = build("youtube", "v3", developerKey=api_key)

def resolve_channel_id_from_handle(handle: str) -> Tuple[str, str]:
    if not handle.startswith("@"):
        handle = "@" + handle
    resp = YOUTUBE.search().list(part="snippet", q=handle, type="channel", maxResults=5).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError(f"No channels found for handle {handle}")
    best = None
    for it in items:
        custom = it["snippet"].get("customUrl")
        if custom and custom.lower() == handle.lower():
            best = it; break
    if best is None:
        best = items[0]
    ch_id = best["snippet"]["channelId"]
    info = YOUTUBE.channels().list(part="snippet", id=ch_id, maxResults=1).execute()
    return ch_id, info["items"][0]["snippet"]["title"]

# -------- fast path: search().list(q=...) --------
def fast_search_channel(channel_id: str, movie: str, per_query_max: int = 25) -> List[Dict]:
    query_terms = [
        f"{movie} review",
        f"{movie} movie review",
        f"{movie} public review",
        f"{movie} review",  # as requested
        f"{movie} first day first show",
    ]
    seen: Dict[str, Dict] = {}
    for q in query_terms:
        resp = YOUTUBE.search().list(
            part="snippet",
            channelId=channel_id,
            q=q,
            type="video",
            order="date",
            maxResults=min(per_query_max, 50),
        ).execute()
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            sn  = it["snippet"]
            title = sn["title"]
            if not title_has_movie_and_review(title, movie):
                continue
            url = f"https://www.youtube.com/watch?v={vid}"
            base = fuzz.token_set_ratio(title.casefold(), f"{movie.casefold()} review")
            bonus = 5 if re.search(r"\brating|verdict|public talk|public review|FDFS\b", title, re.I) else 0
            score = base + bonus
            seen[vid] = {"videoId": vid, "title": title, "publishedAt": sn.get("publishedAt"), "url": url, "score": score}
    return sorted(seen.values(), key=lambda d: d["score"], reverse=True)

# -------- fallback: uploads playlist scan --------
def get_uploads_playlist_id(channel_id: str) -> str:
    resp = YOUTUBE.channels().list(part="contentDetails", id=channel_id, maxResults=1).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError(f"No contentDetails for channel {channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def list_uploads(uploads_playlist_id: str, limit: int = 20) -> List[Dict]:
    """Return up to `limit` latest uploads (newest first)."""
    results = []
    page_token = None
    while len(results) < limit:
        resp = YOUTUBE.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=min(50, limit - len(results)),  # don’t over-fetch
            pageToken=page_token
        ).execute()

        for it in resp.get("items", []):
            sn = it["snippet"]
            vid = sn["resourceId"]["videoId"]
            results.append({
                "videoId": vid,
                "title": sn["title"],
                "publishedAt": sn.get("publishedAt"),
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
            if len(results) >= limit:
                break

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    # Ensure newest first (usually already is)
    results.sort(key=lambda x: x.get("publishedAt") or "", reverse=True)
    return results


def hybrid_find_candidates(channel_id: str, movie: str, per_query_max: int) -> List[Dict]:
    # 1) Fast search first
    hits = fast_search_channel(channel_id, movie, per_query_max)
    if hits:
        return hits

    # 2) Fallback: get latest 20 uploads and filter locally
    uploads_id = get_uploads_playlist_id(channel_id)
    uploads = list_uploads(uploads_id, limit=20)

    filtered = []
    movie_norm = movie.casefold()
    for v in uploads:
        title_norm = v["title"].casefold()
        if (movie_norm not in title_norm) or ("review" not in title_norm):
            continue
        filtered.append(v)

    # Already newest first; return as-is
    return filtered


# -------- download + transcribe (no JSON meta) --------
def download_audio_as_mp3(youtube_url: str, out_dir: str, base_filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_tmpl = os.path.join(out_dir, f"{base_filename}.%(ext)s")
    cmd = [
    "yt-dlp",
    "-x", "--audio-format", "mp3", "--audio-quality", "5",
    "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",   # downsample to 16kHz mono
    "-o", out_tmpl, youtube_url
]

    subprocess.run(cmd, check=True)
    mp3_path = os.path.join(out_dir, f"{base_filename}.mp3")
    if not os.path.exists(mp3_path):
        raise RuntimeError(f"Audio not found after download: {mp3_path}")
    return mp3_path
#model = whisper.load_model("small")
#def transcribe_audio_whisper_api(audio_path: str) -> str:
    #result = model.transcribe(audio_path, task="translate")  # force English
    #return result["text"].strip()

def transcribe_audio_whisper_api(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.translations.create(
           model="whisper-1",
           file=audio_file
    )
    return transcript.text.strip()
# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True, help="YouTube Data API v3 key")
    ap.add_argument("--openai-key", required=True, help="OpenAI API key")
    ap.add_argument("--channels-file", required=True, help="channels.txt with language,@handle")
    ap.add_argument("--movie", required=True, help="Movie name")
    args = ap.parse_args()
    
    client = OpenAI(api_key=args.openai_key)
    

    init_youtube(args.api_key)

    out_root = f"{OUTPUT_ROOT_PREFIX}{safe_slug(args.movie)}"
    os.makedirs(out_root, exist_ok=True)

    # read channels
    with open(args.channels_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        try:
            lang, handle = line.split(",", 1)
            lang, handle = lang.strip(), handle.strip()
            ch_id, ch_title = resolve_channel_id_from_handle(handle)
            print(f"\n[INFO] {handle} | {ch_title} | lang={lang}")

            candidates = hybrid_find_candidates(ch_id, args.movie, MAX_RESULTS)
            chan_dir = os.path.join(out_root, safe_slug(handle))
            if os.path.isdir(chan_dir) and os.listdir(chan_dir):
                print(f"[SKIP] Channel {handle} already processed — files found in {chan_dir}")
                continue

            if not candidates:
                print(f"[WARN] No matching videos for '{args.movie}' on {handle} — skipping")
                continue

            picks = candidates[:MAX_PER_CHANNEL]
            for i, c in enumerate(picks, 1):
                print(f"  {i}. {c['title']} | {c['url']} | {c.get('publishedAt','')}")
                chan_dir = os.path.join(out_root, safe_slug(handle))
                base = f"{safe_slug(c['videoId'])}__{safe_slug(c['title'])[:80]}"

                # download audio
                try:
                    mp3_path = download_audio_as_mp3(c["url"], chan_dir, base)
                except Exception as e:
                    print(f"     [ERROR] download failed: {e}")
                    continue

                # transcribe
                try:
                    txt = transcribe_audio_whisper_api(mp3_path)
                except Exception as e:
                    print(f"     [ERROR] whisper failed: {e}")
                    continue

                # save transcript only
                txt_path = os.path.join(chan_dir, f"{base}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(txt)
                print(f"     [OK] transcript: {txt_path}")
                # remove mp3 to save space
                try:
                    os.remove(mp3_path)
                except OSError as e:
                    print(f"     [WARN] Could not delete {mp3_path}: {e}")

        except Exception as e:
            print(f"[ERROR] Failed on line '{line}': {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
