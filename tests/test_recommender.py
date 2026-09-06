"""Ranking behaviour, on corpora whose correct answer is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from src.recommend import MODE_SCALE, Recommender
from tests.conftest import make_dataset


def _orthogonal_corpus(n: int, d: int = 8, gap: float = 0.1) -> np.ndarray:
    """n vectors with a planted similarity order: item 0 is nearest 1, then 2, ...

    `gap` sets how far apart consecutive items are acoustically. The default is a
    wide spread, so ranking tests are unambiguous; the mode tests shrink it, because
    a metadata bonus is only supposed to reorder things the audio calls near-ties.
    """
    e = np.zeros((n, d), dtype=np.float32)
    e[0, 0] = 1.0
    for i in range(1, n):
        e[i, 0] = 1.0 - gap * i
        e[i, 1] = np.sqrt(max(1e-6, 1.0 - e[i, 0] ** 2))
    return e


def _build(cfg, **kwargs) -> Recommender:
    make_dataset(cfg, **kwargs)
    return Recommender(cfg)


def test_ranks_by_planted_similarity_order(cfg) -> None:
    n = 6
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=list(range(1, n + 1)),
        artist_ids=[[] for _ in range(n)],
    )
    top = rec.recommend(seed_theme_id=1, mode="purist", k=5)
    assert top["theme_id"].tolist() == [2, 3, 4, 5, 6]
    assert top["l1_cos"].is_monotonic_decreasing


def test_seed_is_never_its_own_recommendation(cfg) -> None:
    n = 6
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=list(range(1, n + 1)),
        artist_ids=[[] for _ in range(n)],
    )
    for mode in MODE_SCALE:
        assert 1 not in rec.recommend(seed_theme_id=1, mode=mode, k=n)["theme_id"].tolist()


def test_self_cosine_is_one(cfg) -> None:
    n = 6
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=list(range(1, n + 1)),
        artist_ids=[[] for _ in range(n)],
    )
    for i in range(n):
        assert rec.cosine_scores(i)[i] == pytest.approx(1.0, abs=1e-5)


def test_exclude_same_anime_drops_every_sibling(cfg) -> None:
    n = 6
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=[1, 1, 1, 2, 2, 3],  # themes 1-3 share an anime
        artist_ids=[[] for _ in range(n)],
    )
    kept = rec.recommend(seed_theme_id=1, mode="purist", k=n, exclude_same_anime=True)
    assert set(kept["theme_id"]) == {4, 5, 6}


def test_dedup_audio_keeps_only_the_best_of_a_shared_recording(cfg) -> None:
    """Themes 2 and 3 are the same recording reused; only the higher-scoring one shows."""
    n = 4
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=[1, 2, 3, 4],
        artist_ids=[[] for _ in range(n)],
        audio_ids=[101, 202, 202, 404],
    )
    deduped = rec.recommend(seed_theme_id=1, mode="purist", k=n, dedup_audio=True)
    assert deduped["audio_id"].is_unique
    assert 2 in deduped["theme_id"].tolist() and 3 not in deduped["theme_id"].tolist()

    raw = rec.recommend(seed_theme_id=1, mode="purist", k=n, dedup_audio=False)
    assert {2, 3} <= set(raw["theme_id"])


def test_max_per_anime_caps_franchise_pileups(cfg) -> None:
    """The One Piece problem: one show must not be able to fill the whole page."""
    n = 9
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=[99] + [7] * 6 + [8, 9],  # six themes from one anime, all close to the seed
        artist_ids=[[] for _ in range(n)],
    )
    capped = rec.recommend(seed_theme_id=1, mode="purist", k=8, max_per_anime=2)
    assert (capped["mal_id"] == 7).sum() == 2
    uncapped = rec.recommend(seed_theme_id=1, mode="purist", k=8, max_per_anime=None)
    assert (uncapped["mal_id"] == 7).sum() == 6


def test_metadata_weights_stay_in_unit_range(cfg) -> None:
    """`final = cos + scale * alpha * w` only means something if w is normalized."""
    n = 8
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=[1, 1, 2, 2, 3, 3, 4, 4],
        artist_ids=[[10], [10], [10, 11], [12], [], [11], [13], [10]],
        genres=[["Action", "Drama"], ["Action"], [], ["Drama"], ["Action", "Drama"],
                ["Comedy"], ["Action"], ["Drama", "Comedy"]],
        years=[2005.0, 2006.0, 2011.0, np.nan, 2005.0, 2020.0, 2007.0, 2005.0],
    )
    for i in range(n):
        w = rec.metadata_weights(i)
        assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-6
        # the seed matches itself on anime, artist, era and genre -> the maximum
        assert w[i] == pytest.approx(w.max())


def test_metadata_weights_ignore_nan_years(cfg) -> None:
    """A theme with no air date must not count as 'same era' as everything."""
    n = 4
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n),
        mal_ids=[1, 2, 3, 4],
        artist_ids=[[] for _ in range(n)],
        years=[2005.0, 2005.0, np.nan, 2020.0],
    )
    w = rec.metadata_weights(0)
    assert w[1] > w[2], "the undated theme should not earn the era bonus"
    assert w[2] == pytest.approx(w[3])


def test_comfort_outranks_purist_for_same_anime(cfg) -> None:
    """The mode scale must actually move siblings up, or the modes are decoration.

    Acoustic near-ties (gap=0.005) on purpose: `comfort` adds at most
    2 * alpha_blend * w_anime / w_max ~= 0.08, so a corpus with wide cosine gaps
    would prove only that the bonus is smaller than the gaps.
    """
    n = 6
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(n, gap=0.005),
        mal_ids=[1, 2, 3, 4, 5, 1],  # theme 6 is the seed's sibling but acoustically last
        artist_ids=[[] for _ in range(n)],
    )

    def rank_of(mode: str) -> int:
        top = rec.recommend(seed_theme_id=1, mode=mode, k=n)
        return int(top.loc[top["theme_id"] == 6, "rank"].iloc[0])

    assert rank_of("comfort") < rank_of("purist")
    assert rank_of("discovery") <= rank_of("purist")


def test_unknown_mode_and_unknown_seed_are_rejected(cfg) -> None:
    rec = _build(
        cfg,
        embeddings=_orthogonal_corpus(4),
        mal_ids=[1, 2, 3, 4],
        artist_ids=[[] for _ in range(4)],
    )
    with pytest.raises(ValueError, match="unknown mode"):
        rec.recommend(seed_theme_id=1, mode="nonsense")
    with pytest.raises(SystemExit, match="not found in dataset"):
        rec.recommend(seed_theme_id=9999, mode="purist")


def test_zero_norm_embedding_is_refused_not_silently_nan(cfg) -> None:
    """A zero vector makes cosine NaN, which would sort silently and rank randomly."""
    e = _orthogonal_corpus(4)
    e[2] = 0.0
    make_dataset(
        cfg,
        embeddings=e,
        mal_ids=[1, 2, 3, 4],
        artist_ids=[[] for _ in range(4)],
    )
    with pytest.raises(SystemExit, match="zero norm"):
        Recommender(cfg)


def test_embed_metadata_guard_catches_a_stale_dataset(cfg) -> None:
    make_dataset(
        cfg,
        embeddings=_orthogonal_corpus(4),
        mal_ids=[1, 2, 3, 4],
        artist_ids=[[] for _ in range(4)],
    )
    cfg.embed_model = "some-other-model"
    with pytest.raises(SystemExit, match="disagree with config"):
        Recommender(cfg)
    Recommender(cfg, strict_embed_check=False)  # the experiments' escape hatch still works
