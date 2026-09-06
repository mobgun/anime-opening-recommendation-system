"""Pull behavioural artist similarity from ListenBrainz for the resolved artists.

This is the ground truth the metadata columns cannot give us. `artist_recall` rewards
recognising one performer's voice and production signature across their own songs —
in MIR that effect is a confound to be filtered out, not a target. Listener-derived
similarity asks a different and better question: does the audio ranking surface themes
by a *different* artist that real listeners treat as adjacent? Different singer,
different production, same audience.

The similarity is computed by ListenBrainz from listening sessions: artists that turn
up close together in the same listening sessions score high. No key required.

    python -m experiments.resolve_artists          # first
    python -m experiments.fetch_similar_artists    # then this
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

log = logging.getLogger(__name__)

LB_URL = "https://labs.api.listenbrainz.org/similar-artists/json"
ALGORITHM = (
    "session_based_days_7500_session_300_contribution_5_threshold_10_"
    "limit_100_filter_True_skip_30"
)
MIN_INTERVAL = 0.35


def fetch_one(client: httpx.Client, mbid: str, tries: int = 4) -> list[dict] | None:
    for attempt in range(tries):
        try:
            r = client.get(LB_URL, params={"artist_mbids": mbid, "algorithm": ALGORITHM})
        except httpx.HTTPError as exc:
            log.debug("network error for %s: %s", mbid, exc)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code in (429, 503):
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code != 200:
            return None
        payload = r.json()
        # The endpoint has returned both a bare list and a {..., "data": [...]} wrapper
        # depending on version; accept either rather than guessing.
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("similar_artists") or []
        out = []
        for row in payload:
            if isinstance(row, dict) and row.get("artist_mbid"):
                out.append(
                    {
                        "mbid": row["artist_mbid"],
                        "name": row.get("name"),
                        "score": row.get("score"),
                    }
                )
        return out
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fetch-similar-artists")
    parser.add_argument("--artists", type=Path, default=Path("data/raw/mb_artists.json"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/lb_similar_artists.json"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.artists.exists():
        raise SystemExit(f"{args.artists} not found — run experiments.resolve_artists first")
    resolved = {k: v for k, v in json.loads(args.artists.read_text(encoding="utf-8")).items() if v}
    log.info("%d resolved artists", len(resolved))

    cache: dict[str, list[dict] | None] = {}
    if args.out.exists():
        cache = json.loads(args.out.read_text(encoding="utf-8"))

    todo = [(n, m["mbid"]) for n, m in resolved.items() if m["mbid"] not in cache]
    log.info("%d cached, %d to fetch", len(cache), len(todo))

    client = httpx.Client(
        headers={"User-Agent": "anime-themes-recsys/0.1 (+research)"}, timeout=60.0
    )
    try:
        for _name, mbid in tqdm(todo, desc="listenbrainz", unit="artist"):
            cache[mbid] = fetch_one(client, mbid)
            time.sleep(MIN_INTERVAL)
            if len(cache) % 25 == 0:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    with_data = {k: v for k, v in cache.items() if v}
    sizes = sorted(len(v) for v in with_data.values())
    log.info(
        "%d/%d artists have similarity data (%.0f%%); neighbours per artist: median %d, max %d",
        len(with_data), len(cache), 100 * len(with_data) / max(1, len(cache)),
        sizes[len(sizes) // 2] if sizes else 0, sizes[-1] if sizes else 0,
    )

    # How much of this is usable depends on overlap: a neighbour only counts if it is
    # also an artist somewhere in our own corpus.
    ours = {m["mbid"] for m in resolved.values()}
    internal = {
        k: [n for n in v if n["mbid"] in ours and n["mbid"] != k]
        for k, v in with_data.items()
    }
    useful = {k: v for k, v in internal.items() if v}
    log.info(
        "%d artists have >=1 behaviourally similar artist that is ALSO in this corpus "
        "(that is what the metric can actually score)",
        len(useful),
    )
    print(f"cache: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
