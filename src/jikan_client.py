from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

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

LIST_FIELDS = ("genres", "themes", "demographics", "studios")


class _RateLimiter:
    def __init__(self, rate_per_sec: float) -> None:
        self.min_interval = 1.0 / max(rate_per_sec, 0.001)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _get_full(client: httpx.Client, mal_id: int, limiter: _RateLimiter) -> dict[str, Any]:
    limiter.wait()
    r = client.get(f"/anime/{mal_id}/full")
    if r.status_code == 429:
        time.sleep(2.0)
        raise httpx.HTTPError("rate limited")
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json().get("data") or {}


def _names(items: Iterable[dict[str, Any]] | None) -> list[str]:
    return [i.get("name") for i in (items or []) if i.get("name")]


def _ids(items: Iterable[dict[str, Any]] | None) -> list[int]:
    return [int(i["mal_id"]) for i in (items or []) if i.get("mal_id") is not None]


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    aired = record.get("aired") or {}
    return {
        "mal_id": record.get("mal_id"),
        "title": record.get("title"),
        "title_english": record.get("title_english"),
        "title_japanese": record.get("title_japanese"),
        "type": record.get("type"),
        "episodes": record.get("episodes"),
        "status": record.get("status"),
        "aired_from": aired.get("from"),
        "aired_to": aired.get("to"),
        "season": record.get("season"),
        "year": record.get("year"),
        "score": record.get("score"),
        "scored_by": record.get("scored_by"),
        "rank": record.get("rank"),
        "popularity": record.get("popularity"),
        "members": record.get("members"),
        "favorites": record.get("favorites"),
        "source": record.get("source"),
        "rating": record.get("rating"),
        "synopsis": record.get("synopsis"),
        "genres": _names(record.get("genres")),
        "genre_ids": _ids(record.get("genres")),
        "themes": _names(record.get("themes")),
        "demographics": _names(record.get("demographics")),
        "studios": _names(record.get("studios")),
        "studio_ids": _ids(record.get("studios")),
    }


def _cache_path(cfg: Config, mal_id: int) -> Path:
    bucket = mal_id // 1000
    d = cfg.raw_dir / "jikan_cache" / f"{bucket:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mal_id}.json"


def fetch_mal(cfg: Config, mal_ids: Iterable[int]) -> pd.DataFrame:
    """Stage 2: pull MAL metadata for each unique mal_id, with on-disk JSON cache."""
    mal_ids = sorted(set(int(x) for x in mal_ids))
    rows: list[dict[str, Any]] = []
    limiter = _RateLimiter(cfg.jikan_rate_per_sec)
    with httpx.Client(
        base_url=cfg.jikan_base_url,
        headers={"User-Agent": cfg.user_agent, "Accept": "application/json"},
        timeout=cfg.http_timeout_seconds,
        follow_redirects=True,
    ) as client:
        for mal_id in tqdm(mal_ids, desc="jikan", unit="anime"):
            cache_file = _cache_path(cfg, mal_id)
            if cache_file.exists():
                try:
                    record = json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    record = _get_full(client, mal_id, limiter)
                    cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            else:
                record = _get_full(client, mal_id, limiter)
                cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            if not record:
                continue
            rows.append(_flatten(record))
    return pd.DataFrame(rows)


def fetch_top_mal_ids(cfg: Config, n: int, by: str = "popularity") -> list[int]:
    """Return top-N mal_ids via Jikan /top/anime, ordered by 'popularity' or 'score'."""
    if by not in {"popularity", "score"}:
        raise ValueError(f"unknown 'by' {by!r}; expected 'popularity' or 'score'")
    extra = {"filter": "bypopularity"} if by == "popularity" else {}
    limiter = _RateLimiter(cfg.jikan_rate_per_sec)
    per_page = 25  # Jikan /top/anime hard cap
    ids: list[int] = []
    page = 1
    with httpx.Client(
        base_url=cfg.jikan_base_url,
        headers={"User-Agent": cfg.user_agent, "Accept": "application/json"},
        timeout=cfg.http_timeout_seconds,
        follow_redirects=True,
    ) as client:
        while len(ids) < n:
            limiter.wait()
            r = client.get("/top/anime", params={"limit": per_page, "page": page, **extra})
            if r.status_code == 429:
                time.sleep(2.0)
                continue
            r.raise_for_status()
            payload = r.json()
            for item in payload.get("data") or []:
                mid = item.get("mal_id")
                if mid is not None:
                    ids.append(int(mid))
                    if len(ids) >= n:
                        break
            if not (payload.get("pagination") or {}).get("has_next_page"):
                break
            page += 1
    log.info("Jikan top-%d by %s: collected %d mal_ids", n, by, len(ids))
    return ids[:n]


def run(cfg: Config, mal_ids: Iterable[int] | None = None) -> pd.DataFrame:
    cfg.ensure_dirs()
    if mal_ids is None:
        themes = pd.read_parquet(cfg.animethemes_parquet, columns=["mal_id"])
        mal_ids = themes["mal_id"].dropna().astype(int).tolist()
    df = fetch_mal(cfg, mal_ids)
    df.to_parquet(cfg.mal_parquet, index=False)
    log.info("Stage 2: wrote %d MAL records to %s", len(df), cfg.mal_parquet)
    return df
