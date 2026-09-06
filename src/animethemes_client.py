from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .config import Config

log = logging.getLogger(__name__)

INCLUDE = ",".join(
    [
        "animethemes.song.artists",
        "animethemes.animethemeentries.videos.audio",
        "resources",
    ]
)


def _client(cfg: Config) -> httpx.Client:
    return httpx.Client(
        base_url=cfg.animethemes_base_url,
        headers={"User-Agent": cfg.user_agent, "Accept": "application/json"},
        timeout=cfg.http_timeout_seconds,
        follow_redirects=True,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _get_json(
    client: httpx.Client, url: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    r = client.get(url, params=params)
    if r.status_code == 429:
        raise httpx.HTTPError("rate limited")
    r.raise_for_status()
    return r.json()


def iter_anime_pages(cfg: Config, limit_pages: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield each anime object across the paginated /anime endpoint."""
    with _client(cfg) as client:
        params: dict[str, Any] = {
            "include": INCLUDE,
            "page[size]": cfg.animethemes_page_size,
        }
        url = "/anime"
        page_idx = 0
        with tqdm(desc="animethemes pages", unit="page") as bar:
            while True:
                payload = _get_json(client, url, params=params)
                yield from payload.get("anime", [])
                page_idx += 1
                bar.update(1)
                if limit_pages is not None and page_idx >= limit_pages:
                    return
                next_url = (payload.get("links") or {}).get("next")
                if not next_url:
                    return
                url = next_url
                params = None


def _extract_mal_id(anime: dict[str, Any]) -> int | None:
    for r in anime.get("resources") or []:
        if r.get("site") == "MyAnimeList":
            ext = r.get("external_id")
            if ext is not None:
                return int(ext)
    return None


def _pick_audio_for_theme(theme: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return (audio_obj, source_video_obj) for a theme, preferring nc=true and highest resolution.

    Returns None if no entry has a non-null audio.
    """
    candidates: list[tuple[bool, int, dict[str, Any], dict[str, Any]]] = []
    for entry in theme.get("animethemeentries") or []:
        for video in entry.get("videos") or []:
            audio = video.get("audio")
            if not audio:
                continue
            nc = bool(video.get("nc"))
            resolution = int(video.get("resolution") or 0)
            candidates.append((nc, resolution, audio, video))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, _, audio, video = candidates[0]
    return audio, video


def flatten_anime(anime: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one anime payload into one row per theme (OP/ED)."""
    mal_id = _extract_mal_id(anime)
    if mal_id is None:
        return []
    rows: list[dict[str, Any]] = []
    seen_audio_ids: set[int] = set()
    for theme in anime.get("animethemes") or []:
        picked = _pick_audio_for_theme(theme)
        if picked is None:
            continue
        audio, video = picked
        audio_id = int(audio["id"])
        if audio_id in seen_audio_ids:
            continue
        seen_audio_ids.add(audio_id)
        song = theme.get("song") or {}
        artists = song.get("artists") or []
        rows.append(
            {
                "theme_id": int(theme["id"]),
                "anime_slug": anime.get("slug"),
                "mal_id": mal_id,
                "theme_type": theme.get("type"),
                "sequence": theme.get("sequence"),
                "theme_slug": theme.get("slug"),
                "song_id": song.get("id"),
                "song_title": song.get("title"),
                "artist_ids": [int(a["id"]) for a in artists],
                "artist_names": [a.get("name") for a in artists],
                "audio_id": audio_id,
                "audio_link": audio.get("link"),
                "audio_basename": audio.get("basename"),
                "audio_size": audio.get("size"),
                "audio_path": audio.get("path"),
                "source_video_nc": bool(video.get("nc")),
                "source_video_resolution": video.get("resolution"),
            }
        )
    return rows


def fetch_themes(cfg: Config, limit_pages: int | None = None) -> pd.DataFrame:
    """Stage 1: fetch all themes from AnimeThemes, return one DataFrame row per theme."""
    rows: list[dict[str, Any]] = []
    for anime in iter_anime_pages(cfg, limit_pages=limit_pages):
        rows.extend(flatten_anime(anime))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["theme_id"], keep="first").reset_index(drop=True)
    return df


def fetch_anime_for_mal_id(client: httpx.Client, mal_id: int) -> dict[str, Any] | None:
    """Look up a single AnimeThemes anime by its MyAnimeList external_id."""
    payload = _get_json(
        client,
        "/anime",
        params={
            "filter[has]": "resources",
            "filter[site]": "MyAnimeList",
            "filter[external_id]": str(mal_id),
            "include": INCLUDE,
            "page[size]": 1,
        },
    )
    anime_list = payload.get("anime") or []
    if len(anime_list) > 1:
        log.warning(
            "mal_id=%d matched %d AnimeThemes records, taking the first",
            mal_id, len(anime_list),
        )
    return anime_list[0] if anime_list else None


def fetch_themes_for_mal_ids(cfg: Config, mal_ids: Iterable[int]) -> pd.DataFrame:
    """Stage 1 (top-N variant): fetch themes only for the given mal_ids."""
    mal_ids = [int(m) for m in mal_ids]
    rows: list[dict[str, Any]] = []
    not_found: list[int] = []
    with _client(cfg) as client:
        for mid in tqdm(mal_ids, desc="animethemes (by mal_id)", unit="anime"):
            anime = fetch_anime_for_mal_id(client, mid)
            if anime is None:
                not_found.append(mid)
                continue
            rows.extend(flatten_anime(anime))
    if not_found:
        log.warning(
            "AnimeThemes had no record for %d/%d mal_ids (sample: %s)",
            len(not_found), len(mal_ids), not_found[:10],
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["theme_id"], keep="first").reset_index(drop=True)
    return df


def run(
    cfg: Config,
    limit_pages: int | None = None,
    top_mal_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    cfg.ensure_dirs()
    if top_mal_ids is not None:
        df = fetch_themes_for_mal_ids(cfg, top_mal_ids)
    else:
        df = fetch_themes(cfg, limit_pages=limit_pages)
    df.to_parquet(cfg.animethemes_parquet, index=False)
    log.info("Stage 1: wrote %d themes to %s", len(df), cfg.animethemes_parquet)
    return df
