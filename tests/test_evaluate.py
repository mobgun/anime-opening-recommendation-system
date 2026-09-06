"""The evaluator, on corpora where the correct metric value is known in advance.

`Evaluator` scores the live `Recommender`, so these also pin the contract between
the two: if `recommend()` ever stops returning a full ranking, or starts returning
theme_ids the evaluator cannot map back, these fail rather than the metric quietly
dropping seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import Evaluator, corpus_fingerprint, paired_mrr_delta
from src.recommend import Recommender
from tests.conftest import make_dataset


def _corpus(order_is_correct: bool) -> np.ndarray:
    """8 themes. Theme 1's artist partner is theme 2 — acoustically first or last."""
    e = np.zeros((8, 4), dtype=np.float32)
    e[0] = [1.0, 0.0, 0.0, 0.0]
    partner_gap = 0.02 if order_is_correct else 0.9
    e[1, 0], e[1, 1] = 1.0 - partner_gap, np.sqrt(1 - (1 - partner_gap) ** 2)
    for i in range(2, 8):
        # the distractors sit between the two cases, so the partner is either
        # clearly first or clearly last depending on partner_gap
        e[i, 0] = 0.7 - 0.02 * i
        e[i, 1] = np.sqrt(1 - e[i, 0] ** 2)
    return e


def _evaluate(cfg, order_is_correct: bool):
    make_dataset(
        cfg,
        embeddings=_corpus(order_is_correct),
        mal_ids=[1, 2, 3, 4, 5, 6, 7, 8],  # all different, so nothing is excluded as a sibling
        artist_ids=[[10], [10], [20], [21], [22], [23], [24], [25]],
    )
    rec = Recommender(cfg)
    return rec, Evaluator(rec, mode="purist", exclude_same_anime=True).run(control_k=3)


def test_perfect_ranking_scores_rank_one(cfg) -> None:
    _, res = _evaluate(cfg, order_is_correct=True)
    assert res.artist_seed_ids == [1, 2]  # only these two have a partner elsewhere
    assert res.artist_ranks == [1, 1]
    by_name = {m.name: m for m in res.artist}
    assert by_name["artist_recall@1"].value == pytest.approx(1.0)
    assert by_name["artist_mrr"].value == pytest.approx(1.0)
    assert by_name["artist_mrr"].lift > 1.0


def test_worst_ranking_scores_last_place(cfg) -> None:
    _, res = _evaluate(cfg, order_is_correct=False)
    assert res.artist_ranks == [7, 7]  # 7 candidates once the seed is removed
    by_name = {m.name: m for m in res.artist}
    assert by_name["artist_recall@1"].value == pytest.approx(0.0)
    assert by_name["artist_mrr"].value == pytest.approx(1 / 7)


def test_seeds_without_an_artist_partner_are_not_scored(cfg) -> None:
    """Themes 3-8 each have a unique artist, so they contribute no artist seed."""
    _, res = _evaluate(cfg, order_is_correct=True)
    assert set(res.artist_seed_ids) == {1, 2}
    assert res.artist[0].n_seeds == 2


def test_baseline_is_computed_against_the_real_pool_size(cfg) -> None:
    _, res = _evaluate(cfg, order_is_correct=True)
    assert res.artist_pool_sizes == [7, 7]
    # one relevant item in a pool of 7 -> a random top-1 hits 1/7 of the time
    by_name = {m.name: m for m in res.artist}
    assert by_name["artist_recall@1"].baseline == pytest.approx(1 / 7)


def test_paired_delta_is_signed_and_reports_wins(cfg, tmp_path) -> None:
    """The comparison that every decision in this project was made with."""
    _, good = _evaluate(cfg, order_is_correct=True)
    _, bad = _evaluate(cfg, order_is_correct=False)

    improved = paired_mrr_delta(bad, good, family="artist")
    assert improved["delta"] > 0
    assert improved["seeds_better"] == 2 and improved["seeds_worse"] == 0
    assert improved["n_shared_seeds"] == 2

    regressed = paired_mrr_delta(good, bad, family="artist")
    assert regressed["delta"] == pytest.approx(-improved["delta"])
    assert regressed["seeds_better"] == 0 and regressed["seeds_worse"] == 2


def test_paired_delta_rejects_an_unknown_family(cfg) -> None:
    _, res = _evaluate(cfg, order_is_correct=True)
    with pytest.raises(ValueError, match="unknown family"):
        paired_mrr_delta(res, res, family="genre")


def test_paired_delta_is_empty_without_behavioural_ground_truth(cfg) -> None:
    """No MusicBrainz cache in a tmp dir, so `related_*` must degrade, not explode."""
    _, res = _evaluate(cfg, order_is_correct=True)
    assert res.behavioural == []
    assert paired_mrr_delta(res, res, family="behavioural") == {}


def test_corpus_fingerprint_is_order_independent_but_content_sensitive() -> None:
    """It exists so two runs can be proven to be on the same themes."""
    a = corpus_fingerprint(np.array([3, 1, 2]))
    assert a == corpus_fingerprint(np.array([1, 2, 3]))
    assert a != corpus_fingerprint(np.array([1, 2, 4]))
