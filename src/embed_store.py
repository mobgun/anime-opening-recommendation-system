"""Read and write the sharded embeddings store — no torch, no model.

Split out of `embed.py` so that everything downstream of the embeddings (the join
stage, the recommender, the evaluator, the tests) can run without importing torch.
`embed.py` pulls in a 2.6 GB CUDA wheel; producing embeddings needs that, reading
them back does not.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


def shard_paths(cfg: Config) -> list[Path]:
    if not cfg.embeddings_dir.exists():
        return []
    return sorted(cfg.embeddings_dir.glob("part_*.parquet"))


def read_all_embeddings(cfg: Config) -> pd.DataFrame:
    """Read every shard plus the legacy single-file store, if present."""
    frames: list[pd.DataFrame] = []
    for p in shard_paths(cfg):
        frames.append(pd.read_parquet(p))
    if cfg.embeddings_parquet.exists():
        frames.append(pd.read_parquet(cfg.embeddings_parquet))
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def next_shard_path(cfg: Config) -> Path:
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)
    parts = shard_paths(cfg)
    next_idx = (int(parts[-1].stem.split("_")[-1]) + 1) if parts else 1
    return cfg.embeddings_dir / f"part_{next_idx:06d}.parquet"


def write_shard(cfg: Config, new_rows: list[dict]) -> int:
    if not new_rows:
        return 0
    path = next_shard_path(cfg)
    pd.DataFrame(new_rows).to_parquet(path, index=False)
    return len(new_rows)


def check_invalidation(cfg: Config, force_reembed: bool) -> pd.DataFrame:
    """Load existing embeddings, validating model match. Wipe shards + legacy on --force-reembed."""
    existing = read_all_embeddings(cfg)
    if existing.empty:
        return existing
    bad_model = existing["embed_model"] != cfg.embed_model
    bad_version = existing["embed_version"] != cfg.embed_version
    if (bad_model | bad_version).any():
        if force_reembed:
            log.warning(
                "embeddings store contained mismatched model/version; wiping due to --force-reembed"
            )
            for p in shard_paths(cfg):
                p.unlink()
            cfg.embeddings_parquet.unlink(missing_ok=True)
            return pd.DataFrame()
        offending = existing.loc[bad_model | bad_version, ["embed_model", "embed_version"]]
        sample = offending.drop_duplicates().head(5).to_dict("records")
        raise SystemExit(
            f"embeddings store at {cfg.embeddings_dir} (and legacy {cfg.embeddings_parquet}) "
            f"contains rows with a different embed_model/embed_version than config "
            f"(config={cfg.embed_model}/{cfg.embed_version}). "
            f"Offending entries (sample): {sample}. "
            f"Delete the files or rerun with --force-reembed."
        )
    return existing
