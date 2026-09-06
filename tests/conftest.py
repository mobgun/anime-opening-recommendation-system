"""Synthetic corpora for the tests.

Nothing here touches the network, the 2.2 GB audio tree or a GPU. The point is to
build datasets whose right answer is known by construction — planted near-duplicate
pairs, planted artist partners — so a ranking bug shows up as a wrong rank rather
than as a metric that drifted by 0.01.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config

EMBED_MODEL = "test-model"


def make_dataset(
    cfg: Config,
    *,
    embeddings: np.ndarray,
    mal_ids: list[int],
    artist_ids: list[list[int]],
    artist_names: list[list[str]] | None = None,
    genres: list[list[str]] | None = None,
    years: list[float] | None = None,
    audio_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Write cfg.dataset_parquet with exactly the columns Recommender/Evaluator read.

    embed_model/embed_version are taken from `cfg` so the dataset always satisfies the
    staleness guard — the test that exercises the guard breaks them on purpose.
    """
    n = len(embeddings)
    assert len(mal_ids) == n and len(artist_ids) == n
    df = pd.DataFrame(
        {
            "theme_id": list(range(1, n + 1)),
            "mal_id": mal_ids,
            "audio_id": audio_ids if audio_ids is not None else list(range(101, 101 + n)),
            "title": [f"Anime {m}" for m in mal_ids],
            "theme_type": ["OP"] * n,
            "sequence": [1] * n,
            "song_title": [f"Song {i}" for i in range(1, n + 1)],
            "artist_ids": artist_ids,
            "artist_names": (
                artist_names
                if artist_names is not None
                else [[f"artist{a}" for a in ids] for ids in artist_ids]
            ),
            "year": years if years is not None else [2010.0] * n,
            "score": [7.5] * n,
            "genres": genres if genres is not None else [["Action"]] * n,
            "embedding": [e.astype(np.float32).tolist() for e in embeddings],
            "embed_model": [cfg.embed_model] * n,
            "embed_version": [cfg.embed_version] * n,
        }
    )
    cfg.dataset_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.dataset_parquet, index=False)
    return df


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A Config rooted in tmp_path, with the test model/version and whitening off.

    Whitening is off by default because most ranking tests want to assert on the
    geometry they built; the whitening tests turn it back on explicitly.
    """
    c = Config()
    c.data_dir = tmp_path / "data"
    c.embed_model = EMBED_MODEL
    c.recommend_whiten_components = 0
    c.ensure_dirs()
    return c


@pytest.fixture
def anisotropic_corpus() -> np.ndarray:
    """40 vectors sharing one dominant direction — the shape whitening exists to fix.

    This is the MERT pathology in miniature: a large common component plus a small
    per-item component, so raw cosine between any two items is near 1.
    """
    rng = np.random.default_rng(0)
    n, d = 40, 16
    common = np.ones(d, dtype=np.float64)
    idiosyncratic = rng.normal(scale=0.05, size=(n, d))
    return (common + idiosyncratic).astype(np.float32)
