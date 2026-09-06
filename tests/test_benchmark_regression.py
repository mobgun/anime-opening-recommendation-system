"""Re-derive the README's headline numbers from the real corpus.

`benchmarks/layer6-whitened.json` is the run every table in the README quotes. This
re-scores the live code against it, which is the only test here that would catch a
change that leaves every unit test green while moving `artist_mrr` by 0.05.

Skipped when data/processed/dataset.parquet is absent (CI, a fresh clone), and
skipped with a clear reason when the corpus has drifted — `--top-n 100` follows live
MAL popularity, so a rebuilt dataset is legitimately a different corpus and comparing
metrics across it would be meaningless rather than informative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.evaluate import Evaluator, corpus_fingerprint
from src.recommend import Recommender

BENCHMARK = Path("benchmarks/layer6-whitened.json")
PINNED_CORPUS = Path("benchmarks/corpus-621.txt")
TOLERANCE = 5e-3

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def frozen() -> dict:
    if not BENCHMARK.exists():
        pytest.skip(f"{BENCHMARK} not present")
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live(frozen: dict):
    cfg = Config()
    if not cfg.dataset_parquet.exists():
        pytest.skip(f"{cfg.dataset_parquet} not built; run the pipeline first")
    rec = Recommender(cfg)
    fingerprint = corpus_fingerprint(rec.df["theme_id"].to_numpy())
    if fingerprint != frozen["meta"]["corpus_sha256"]:
        pytest.skip(
            f"corpus has drifted (live {fingerprint}, frozen "
            f"{frozen['meta']['corpus_sha256']}); metrics are not comparable"
        )
    res = Evaluator(
        rec,
        mode=frozen["meta"]["mode"],
        exclude_same_anime=frozen["meta"]["exclude_same_anime"],
    ).run()
    return rec, res


def test_pinned_corpus_file_matches_the_dataset(live) -> None:
    if not PINNED_CORPUS.exists():
        pytest.skip(f"{PINNED_CORPUS} not present")
    rec, _ = live
    pinned = {int(x) for x in PINNED_CORPUS.read_text(encoding="utf-8").split()}
    assert set(rec.df["theme_id"].astype(int)) == pinned


def test_corpus_shape_matches_the_readme(live, frozen: dict) -> None:
    rec, _ = live
    assert len(rec.df) == frozen["meta"]["themes"]
    assert int(rec.df["mal_id"].nunique()) == frozen["meta"]["anime"]
    assert rec.E.shape[1] == frozen["meta"]["embed_dim"]


def test_whitening_still_removes_the_anisotropy(live, frozen: dict) -> None:
    """The README claims mean off-diagonal cosine of -0.002 after whitening."""
    rec, _ = live
    gram = rec.E @ rec.E.T
    mean_offdiag = float(gram[np.triu_indices(len(rec.df), 1)].mean())
    assert mean_offdiag == pytest.approx(frozen["meta"]["mean_offdiag_cosine"], abs=TOLERANCE)


@pytest.mark.parametrize("family", ["artist", "behavioural", "controls"])
def test_every_frozen_metric_reproduces(live, frozen: dict, family: str) -> None:
    _, res = live
    expected = {m["name"]: m for m in frozen[family]}
    if not expected:
        pytest.skip(f"no {family} metrics frozen in the benchmark")
    actual = {m.name: m for m in getattr(res, family)}
    assert set(actual) == set(expected), "the set of reported metrics changed"
    for name, exp in expected.items():
        got = actual[name]
        # first_rank is a rank, not a probability, so it needs a proportional tolerance
        tol = max(TOLERANCE, abs(exp["value"]) * 0.01) if "rank" in name else TOLERANCE
        assert got.value == pytest.approx(exp["value"], abs=tol), name
        assert got.baseline == pytest.approx(exp["baseline"], abs=tol), f"{name} baseline"
        assert got.n_seeds == exp["n_seeds"], f"{name} n_seeds"


def test_per_seed_ranks_are_identical(live, frozen: dict) -> None:
    """Aggregates can match while individual rankings churn; this catches that."""
    _, res = live
    assert res.artist_ranks == frozen["artist_ranks"]
    assert res.behavioural_ranks == frozen["behavioural_ranks"]
