"""The shard store and its staleness guard.

These matter because the guard is what stops a dataset from silently mixing vectors
from two different models — an error that produces plausible-looking recommendations
and no warning at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.embed_store import (
    check_invalidation,
    next_shard_path,
    read_all_embeddings,
    shard_paths,
    write_shard,
)


def _rows(cfg, theme_ids, model=None, version=None):
    return [
        {
            "theme_id": t,
            "audio_id": 100 + t,
            "embedding": [0.1, 0.2, 0.3],
            "embed_model": model or cfg.embed_model,
            "embed_version": version or cfg.embed_version,
            "n_chunks_used": 3,
            "total_seconds_used": 90.0,
        }
        for t in theme_ids
    ]


def test_reads_back_every_shard(cfg) -> None:
    write_shard(cfg, _rows(cfg, [1, 2]))
    write_shard(cfg, _rows(cfg, [3]))
    df = read_all_embeddings(cfg)
    assert sorted(df["theme_id"]) == [1, 2, 3]
    assert len(shard_paths(cfg)) == 2


def test_empty_store_reads_as_an_empty_frame(cfg) -> None:
    assert read_all_embeddings(cfg).empty


def test_writing_no_rows_creates_no_shard(cfg) -> None:
    assert write_shard(cfg, []) == 0
    assert shard_paths(cfg) == []


def test_shard_numbering_increments_and_sorts(cfg) -> None:
    for i in range(11):
        write_shard(cfg, _rows(cfg, [i + 1]))
    names = [p.name for p in shard_paths(cfg)]
    assert names == sorted(names), "zero-padding must keep lexical order == numeric order"
    assert next_shard_path(cfg).name == "part_000012.parquet"


def test_legacy_single_file_store_is_still_read(cfg) -> None:
    """Datasets built before sharding must keep working."""
    pd.DataFrame(_rows(cfg, [7])).to_parquet(cfg.embeddings_parquet, index=False)
    write_shard(cfg, _rows(cfg, [8]))
    assert sorted(read_all_embeddings(cfg)["theme_id"]) == [7, 8]


def test_mismatched_model_is_refused(cfg) -> None:
    write_shard(cfg, _rows(cfg, [1], model="some-other-model"))
    with pytest.raises(SystemExit, match="different embed_model/embed_version"):
        check_invalidation(cfg, force_reembed=False)


def test_mismatched_version_is_refused(cfg) -> None:
    write_shard(cfg, _rows(cfg, [1], version="sr16000-c10x1s10"))
    with pytest.raises(SystemExit, match="different embed_model/embed_version"):
        check_invalidation(cfg, force_reembed=False)


def test_force_reembed_wipes_shards_and_legacy(cfg) -> None:
    write_shard(cfg, _rows(cfg, [1], model="some-other-model"))
    pd.DataFrame(_rows(cfg, [2], model="some-other-model")).to_parquet(
        cfg.embeddings_parquet, index=False
    )
    assert check_invalidation(cfg, force_reembed=True).empty
    assert shard_paths(cfg) == []
    assert not cfg.embeddings_parquet.exists()


def test_matching_store_passes_through_untouched(cfg) -> None:
    write_shard(cfg, _rows(cfg, [1, 2]))
    assert sorted(check_invalidation(cfg, force_reembed=False)["theme_id"]) == [1, 2]
    assert len(shard_paths(cfg)) == 1


def test_embed_version_encodes_the_chunk_config(cfg) -> None:
    """Config drift must invalidate the cache, which is what the version string is for."""
    base = cfg.embed_version
    cfg.chunk_seconds = 20
    assert cfg.embed_version != base
    cfg.chunk_seconds = 30
    cfg.embed_layer = -1
    # -1 stays unsuffixed so pre-layer-knob datasets remain valid
    assert not cfg.embed_version.endswith("-L-1")
