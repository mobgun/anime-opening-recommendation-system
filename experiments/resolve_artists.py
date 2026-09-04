"""Resolve the corpus's artists to MusicBrainz ids, so behavioural data can be joined.

AnimeThemes stores artists romanized ("Eir Aoi", "Nightmare"); MusicBrainz stores them
in their canonical script (藍井エイル, ナイトメア) but carries the romanizations as
aliases, and its artist search matches those. That is why resolving *artists* works
where resolving recordings did not: 217 artists instead of 621 titles, and no
cross-script title matching.

Results are cached per artist, so the script is resumable and a rerun costs nothing.
MusicBrainz is a volunteer-run service that asks for <=1 request/second and a real
User-Agent; both are honoured here, which is why this takes minutes rather than seconds.

    python -m experiments.resolve_artists
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

from src.config import Config

log = logging.getLogger(__name__)

MB_URL = "https://musicbrainz.org/ws/2/artist"
MIN_INTERVAL = 1.3  # seconds between requests; MusicBrainz asks for <= 1/s
MAX_TRIES = 5


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _matches(query: str, artist: dict) -> bool:
    """Accept only if the romanized query equals the name or one of the aliases.

    Score alone is not enough: MusicBrainz happily returns a 100 for a different
    artist whose name merely contains the query.
    """
    target = _norm(query)
    if _norm(artist.get("name", "")) == target:
        return True
    for alias in artist.get("aliases") or []:
        if _norm(alias.get("name", "")) == target:
            return True
    for alias in artist.get("aliases") or []:
        if _norm(alias.get("sort-name", "")) == target:
            return True
    return False


def resolve_one(client: httpx.Client, name: str) -> dict | None:
    for attempt in range(MAX_TRIES):
        try:
            r = client.get(MB_URL, params={"query": name, "fmt": "json", "limit": 5})
        except httpx.HTTPError as exc:
            log.debug("network error for %r: %s", name, exc)
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code == 503:
            time.sleep(2.5 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        for artist in r.json().get("artists", []):
            if _matches(name, artist):
                return {
                    "query": name,
                    "mbid": artist["id"],
                    "mb_name": artist.get("name"),
                    "country": artist.get("country"),
                    "type": artist.get("type"),
                    "score": artist.get("score"),
                }
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resolve-artists")
    parser.add_argument("--cache", type=Path, default=Path("data/raw/mb_artists.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config()
    df = pd.read_parquet(cfg.dataset_parquet)
    names: dict[str, int] = {}
    for arr in df["artist_names"]:
        for n in arr if arr is not None else []:
            names[str(n)] = names.get(str(n), 0) + 1
    ordered = sorted(names, key=lambda n: -names[n])
    if args.limit:
        ordered = ordered[: args.limit]

    cache: dict[str, dict | None] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))
    todo = [n for n in ordered if n not in cache]
    log.info("%d distinct artists, %d cached, %d to fetch", len(ordered), len(cache), len(todo))

    client = httpx.Client(
        headers={"User-Agent": cfg.user_agent}, timeout=cfg.http_timeout_seconds
    )
    last = 0.0
    try:
        for name in tqdm(todo, desc="musicbrainz", unit="artist"):
            wait = MIN_INTERVAL - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            cache[name] = resolve_one(client, name)
            if len(cache) % 25 == 0:
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    resolved = {k: v for k, v in cache.items() if v}
    themes_covered = int(
        df["artist_names"]
        .apply(lambda a: bool(a is not None and len(a) and any(str(x) in resolved for x in a)))
        .sum()
    )
    log.info(
        "resolved %d/%d artists (%.0f%%); they cover %d/%d themes (%.0f%%)",
        len(resolved), len(cache), 100 * len(resolved) / max(1, len(cache)),
        themes_covered, len(df), 100 * themes_covered / len(df),
    )
    print(f"cache: {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
