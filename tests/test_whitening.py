"""Whitening is the change the README credits with +0.085 held-out MRR.

Its whole justification is a geometric claim — "mean cosine between two unrelated
themes is 0.918, so what distinguishes songs survives only in the tail" — so the
tests assert that geometry directly on a corpus built to have that pathology.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recommend import Recommender
from tests.conftest import make_dataset


def _mean_offdiag_cosine(E: np.ndarray) -> float:
    gram = E @ E.T
    return float(gram[np.triu_indices(len(E), 1)].mean())


def _build(cfg, embeddings: np.ndarray) -> Recommender:
    n = len(embeddings)
    make_dataset(
        cfg,
        embeddings=embeddings,
        mal_ids=list(range(1, n + 1)),
        artist_ids=[[] for _ in range(n)],
    )
    return Recommender(cfg)


def test_whitening_removes_the_shared_direction(cfg, anisotropic_corpus) -> None:
    cfg.recommend_whiten_components = 0
    raw = _build(cfg, anisotropic_corpus)
    assert _mean_offdiag_cosine(raw.E) > 0.9, "fixture should be anisotropic to begin with"

    cfg.recommend_whiten_components = 8
    whitened = _build(cfg, anisotropic_corpus)
    assert abs(_mean_offdiag_cosine(whitened.E)) < 0.1


def test_whitening_disabled_leaves_vectors_alone(cfg, anisotropic_corpus) -> None:
    cfg.recommend_whiten_components = 0
    rec = _build(cfg, anisotropic_corpus)
    assert rec.whiten_info == {"enabled": False}
    expected = anisotropic_corpus / np.linalg.norm(anisotropic_corpus, axis=1, keepdims=True)
    assert np.allclose(rec.E, expected, atol=1e-5)


def test_whitening_reports_variance_kept(cfg, anisotropic_corpus) -> None:
    cfg.recommend_whiten_components = 4
    rec = _build(cfg, anisotropic_corpus)
    assert rec.whiten_info["enabled"] is True
    assert rec.whiten_info["components"] == 4
    assert 0.0 < rec.whiten_info["variance_kept"] <= 1.0 + 1e-9


def test_components_are_clamped_to_the_corpus_shape(cfg, anisotropic_corpus) -> None:
    """Asking for more components than there are themes must clamp, not crash."""
    cfg.recommend_whiten_components = 10_000
    rec = _build(cfg, anisotropic_corpus)
    assert rec.whiten_info["components"] == min(anisotropic_corpus.shape)
    assert rec.E.shape[1] == min(anisotropic_corpus.shape)


def test_output_rows_are_unit_norm(cfg, anisotropic_corpus) -> None:
    """Cosine is computed as a plain dot product, so the rows must be normalized."""
    cfg.recommend_whiten_components = 8
    rec = _build(cfg, anisotropic_corpus)
    assert np.allclose(np.linalg.norm(rec.E, axis=1), 1.0, atol=1e-5)


def test_whitening_preserves_relative_neighbourhoods(cfg) -> None:
    """A planted near-duplicate pair must stay each other's nearest after whitening."""
    rng = np.random.default_rng(3)
    e = (np.ones((20, 12)) + rng.normal(scale=0.05, size=(20, 12))).astype(np.float32)
    e[1] = e[0] + rng.normal(scale=0.001, size=12)  # theme 2 is a near-duplicate of theme 1
    cfg.recommend_whiten_components = 8
    rec = _build(cfg, e)
    assert rec.recommend(seed_theme_id=1, mode="purist", k=1)["theme_id"].iloc[0] == 2
    assert rec.recommend(seed_theme_id=2, mode="purist", k=1)["theme_id"].iloc[0] == 1


def test_degenerate_corpus_is_refused_not_silently_nan(cfg) -> None:
    """Identical vectors have no variance to whiten; the guard must fire."""
    e = np.ones((6, 8), dtype=np.float32)
    cfg.recommend_whiten_components = 4
    with pytest.raises(SystemExit, match="collapsed to zero"):
        _build(cfg, e)
