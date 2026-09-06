from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import animethemes_client, audio_download, jikan_client
from .config import Config
from .console import enable_utf8_output
from .embed_store import read_all_embeddings

log = logging.getLogger(__name__)

STAGES = ["themes", "mal", "audio", "embed", "join", "all"]


def _excluded_theme_ids(cfg: Config) -> set[int]:
    excluded: set[int] = set()
    if cfg.download_errors_parquet.exists():
        de = pd.read_parquet(cfg.download_errors_parquet)
        if not de.empty and "audio_id" in de.columns:
            bad_audio = set(de["audio_id"].astype(int).tolist())
            themes = pd.read_parquet(cfg.animethemes_parquet, columns=["theme_id", "audio_id"])
            excluded |= set(
                themes.loc[themes["audio_id"].astype(int).isin(bad_audio), "theme_id"]
                .astype(int)
                .tolist()
            )
    if cfg.embed_errors_parquet.exists():
        ee = pd.read_parquet(cfg.embed_errors_parquet)
        if not ee.empty and "theme_id" in ee.columns:
            excluded |= set(ee["theme_id"].astype(int).tolist())
    return excluded


def build_dataset(cfg: Config) -> pd.DataFrame:
    """Stage 5: inner-join themes ⨝ mal ⨝ embeddings, drop errored themes, write dataset.parquet."""
    cfg.ensure_dirs()
    themes = pd.read_parquet(cfg.animethemes_parquet)
    mal = pd.read_parquet(cfg.mal_parquet)
    embeddings = read_all_embeddings(cfg)
    if embeddings.empty:
        raise SystemExit(
            f"no embeddings found in {cfg.embeddings_dir} or {cfg.embeddings_parquet}; "
            f"run --stage embed first."
        )

    excluded = _excluded_theme_ids(cfg)
    if excluded:
        before = len(themes)
        themes = themes[~themes["theme_id"].astype(int).isin(excluded)]
        log.info(
            "Stage 5: excluded %d errored themes (%d → %d)", len(excluded), before, len(themes)
        )

    themes_n, mal_n = len(themes), len(mal)
    df = themes.merge(mal, on="mal_id", how="inner")
    if len(df) < themes_n:
        log.warning(
            "Stage 5: %d themes lost on MAL join (themes=%d, mal=%d → %d)",
            themes_n - len(df), themes_n, mal_n, len(df),
        )
    before_embed = len(df)
    df = df.merge(
        embeddings[
            [
                "theme_id", "embedding", "embed_model", "embed_version",
                "n_chunks_used", "total_seconds_used",
            ]
        ],
        on="theme_id",
        how="inner",
    )
    if len(df) < before_embed:
        log.warning(
            "Stage 5: %d themes lost on embeddings join (%d → %d)",
            before_embed - len(df), before_embed, len(df),
        )
    df.to_parquet(cfg.dataset_parquet, index=False)
    log.info("Stage 5: wrote %d rows to %s", len(df), cfg.dataset_parquet)
    return df


def run_stage(
    cfg: Config,
    stage: str,
    *,
    limit_pages: int | None,
    force_reembed: bool,
    top_n: int | None = None,
    top_by: str = "popularity",
) -> None:
    def _embed() -> None:
        # imported here, not at module scope: torch is an optional extra, so the
        # other four stages must stay runnable without a 2.6 GB install
        try:
            from . import embed
        except ImportError as e:
            raise SystemExit(
                f"stage 'embed' needs the optional embedding dependencies ({e.name} is "
                f"missing). Install them with:\n\n"
                f"    pip install -e \".[embed]\"\n\n"
                f"See the README's GPU section first if you want a CUDA build of torch — "
                f"the default wheel is CPU-only on Windows and ~18x slower here."
            ) from e

        embed.run(cfg, force_reembed=force_reembed)

    def _themes() -> None:
        if top_n is not None:
            if limit_pages is not None:
                raise SystemExit("--top-n and --limit-pages are mutually exclusive")
            mal_ids = jikan_client.fetch_top_mal_ids(cfg, top_n, by=top_by)
            animethemes_client.run(cfg, top_mal_ids=mal_ids)
        else:
            animethemes_client.run(cfg, limit_pages=limit_pages)

    if stage == "themes":
        _themes()
    elif stage == "mal":
        jikan_client.run(cfg)
    elif stage == "audio":
        audio_download.run(cfg)
    elif stage == "embed":
        _embed()
    elif stage == "join":
        build_dataset(cfg)
    elif stage == "all":
        _themes()
        jikan_client.run(cfg)
        audio_download.run(cfg)
        _embed()
        build_dataset(cfg)
    else:
        raise ValueError(f"unknown stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(prog="anime-themes-recsys")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Limit Stage 1 to first N AnimeThemes pages (for smoke tests).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Stage 1: collect themes only for top-N anime via Jikan (skip full pagination).",
    )
    parser.add_argument(
        "--top-by",
        choices=["popularity", "score"],
        default="popularity",
        help="Sort key for --top-n (default: popularity).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep .ogg files after embedding (default: delete to save disk).",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Truncate existing embeddings.parquet on Stage 4 startup.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config()
    if args.keep_audio:
        cfg.keep_audio = True

    run_stage(
        cfg,
        args.stage,
        limit_pages=args.limit_pages,
        force_reembed=args.force_reembed,
        top_n=args.top_n,
        top_by=args.top_by,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
