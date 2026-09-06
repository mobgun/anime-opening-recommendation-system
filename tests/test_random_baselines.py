"""The random baselines are the load-bearing part of every claim in the README.

`artist_recall@1 = 0.219` means nothing on its own; `53x chance` is the claim, and
the denominator comes from these two functions. So they are checked against an
*independently derived* closed form (hypergeometric via math.comb) rather than
against a restatement of the same expression, and then against simulation.

If these are wrong, every lift in the README is wrong by the same factor and
nothing else in the test suite would notice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluate import _bootstrap_ci, _miss_probs, _random_mrr, _random_recall

# (n_candidates, n_relevant) pairs, including the shapes the real corpus produces
CASES = [(10, 1), (10, 3), (50, 1), (50, 7), (617, 1), (617, 4), (617, 60)]


def exact_recall(n: int, m: int, k: int) -> float:
    """P(at least one of m relevant lands in the top k) = 1 - C(n-m, k)/C(n, k)."""
    k = min(k, n)
    if m <= 0 or k <= 0:
        return 0.0
    if n - m < k:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def exact_mrr(n: int, m: int) -> float:
    """E[1/R] with P(R = r) = C(n-r, m-1) / C(n, m)."""
    if m <= 0:
        return 0.0
    total = math.comb(n, m)
    return sum(math.comb(n - r, m - 1) / total / r for r in range(1, n - m + 2))


@pytest.mark.parametrize("n,m", CASES)
@pytest.mark.parametrize("k", [1, 3, 5, 10, 20, 50, 100])
def test_random_recall_matches_hypergeometric(n: int, m: int, k: int) -> None:
    assert _random_recall(n, m, k) == pytest.approx(exact_recall(n, m, k), abs=1e-9)


@pytest.mark.parametrize("n,m", CASES)
def test_random_mrr_matches_closed_form(n: int, m: int) -> None:
    assert _random_mrr(n, m) == pytest.approx(exact_mrr(n, m), abs=1e-9)


def test_random_recall_matches_simulation() -> None:
    """One end-to-end sanity check that the closed form describes actual sampling."""
    n, m, k, trials = 40, 3, 5, 200_000
    rng = np.random.default_rng(0)
    relevant = np.zeros(n, dtype=bool)
    relevant[:m] = True
    hits = sum(relevant[rng.permutation(n)[:k]].any() for _ in range(trials))
    assert hits / trials == pytest.approx(_random_recall(n, m, k), abs=0.005)


def test_random_mrr_matches_simulation() -> None:
    n, m, trials = 40, 3, 200_000
    rng = np.random.default_rng(1)
    relevant = np.zeros(n, dtype=bool)
    relevant[:m] = True
    total = 0.0
    for _ in range(trials):
        order = rng.permutation(n)
        total += 1.0 / (int(np.argmax(relevant[order])) + 1)
    assert total / trials == pytest.approx(_random_mrr(n, m), abs=0.005)


def test_random_recall_degenerate_cases() -> None:
    assert _random_recall(100, 0, 10) == 0.0        # nothing relevant -> never a hit
    assert _random_recall(100, 5, 0) == 0.0         # empty top-k -> never a hit
    assert _random_recall(10, 10, 1) == 1.0         # everything relevant -> always
    # k past the end of the pool is the whole pool, not an index error
    assert _random_recall(10, 1, 50) == pytest.approx(1.0)


def test_random_recall_is_monotone_in_k() -> None:
    vals = [_random_recall(617, 4, k) for k in (1, 3, 5, 10, 20, 50, 100)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_random_mrr_is_monotone_in_n_relevant() -> None:
    vals = [_random_mrr(617, m) for m in (1, 2, 4, 8, 16)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_miss_probs_starts_at_one_and_decreases() -> None:
    p = _miss_probs(50, 5, 20)
    assert p[0] == 1.0
    assert all(a >= b for a, b in zip(p, p[1:]))
    assert all(0.0 <= v <= 1.0 for v in p)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    sample = rng.normal(loc=0.3, scale=0.1, size=200)
    lo, hi = _bootstrap_ci(sample)
    assert lo < sample.mean() < hi
    assert (lo, hi) == _bootstrap_ci(sample)  # fixed seed: reruns must not move the CI


def test_bootstrap_ci_of_empty_sample_is_nan() -> None:
    lo, hi = _bootstrap_ci(np.array([]))
    assert math.isnan(lo) and math.isnan(hi)
