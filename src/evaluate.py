"""Offline evaluation of the recommender against metadata ground truth.

The audio embeddings never see genres, artists or air dates — those columns come
from MyAnimeList. So asking how often a purely acoustic top-k lands on a theme
that shares an artist (or a genre, or an era) with the seed is a cheap sanity
check that the embeddings carry musical signal rather than noise.

Every metric is reported next to the baseline a random top-k would score on the
same candidate pool, because the pool is far from uniform: two thirds of random
theme pairs already share a genre. The ratio of the two is the number to read.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .recommend import MODE_SCALE, Recommender

log = logging.getLogger(__name__)

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


@dataclass
class Metric:
    """One evaluated quantity, its random-pick baseline and a bootstrap CI."""

    name: str
    description: str
    value: float
    baseline: float
    lift: float
    ci_low: float
    ci_high: float
    n_seeds: int

    def format_row(self) -> str:
        return (
            f"{self.name:<22} {self.value:>7.3f}  "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}]".ljust(18)
            + f"{self.baseline:>8.3f}  {self.lift:>5.2f}x  {self.n_seeds:>6d}"
        )


def _prob_any_hit(n_candidates: int, n_relevant: int, k: int) -> float:
    """P(a uniform random k-subset of the pool contains at least one relevant item)."""
    if n_relevant <= 0 or k <= 0:
        return 0.0
    if n_relevant + k > n_candidates:
        return 1.0
    miss = 1.0
    for j in range(k):
        miss *= (n_candidates - n_relevant - j) / (n_candidates - j)
    return 1.0 - miss


def _bootstrap_ci(
    per_seed: np.ndarray, per_seed_baseline: np.ndarray
) -> tuple[float, float, float]:
    """95% percentile CI for the mean, plus the lift over the paired baseline."""
    if len(per_seed) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(per_seed), size=(BOOTSTRAP_RESAMPLES, len(per_seed)))
    means = per_seed[idx].mean(axis=1)
    base = float(per_seed_baseline.mean())
    lift = float(per_seed.mean() / base) if base > 0 else float("nan")
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), lift


class Evaluator:
    """Scores the live Recommender — not a reimplementation of it — seed by seed."""

    def __init__(self, rec: Recommender, k: int, mode: str, exclude_same_anime: bool) -> None:
        self.rec = rec
        self.k = k
        self.mode = mode
        self.exclude_same_anime = exclude_same_anime

        df = rec.df
        self.n = len(df)
        self.mal_id = rec.mal_id
        self.year = rec.year
        self.artist_sets = rec.artist_sets
        self.genre_sets: list[set[str]] = [
            set(arr if arr is not None else []) for arr in df["genres"].to_numpy()
        ]
        self.theme_ids = df["theme_id"].to_numpy(dtype=np.int64)

    def _candidate_mask(self, seed_idx: int) -> np.ndarray:
        keep = np.ones(self.n, dtype=bool)
        keep[seed_idx] = False
        if self.exclude_same_anime:
            keep &= self.mal_id != self.mal_id[seed_idx]
        return keep

    def _topk_indices(self, seed_idx: int) -> list[int]:
        top = self.rec.recommend(
            seed_theme_id=int(self.theme_ids[seed_idx]),
            mode=self.mode,
            k=self.k,
            exclude_same_anime=self.exclude_same_anime,
            dedup_audio=True,
            max_per_anime=None,
        )
        return [self.rec.theme_id_to_idx[int(t)] for t in top["theme_id"]]

    def run(self) -> list[Metric]:
        genre_hit: list[float] = []
        genre_hit_base: list[float] = []
        genre_jac: list[float] = []
        genre_jac_base: list[float] = []
        era_hit: list[float] = []
        era_hit_base: list[float] = []
        artist_hit: list[float] = []
        artist_hit_base: list[float] = []
        window = self.rec.cfg.recommend_era_window_years

        for i in range(self.n):
            cand = np.flatnonzero(self._candidate_mask(i))
            if len(cand) < self.k:
                continue
            top = self._topk_indices(i)
            pos = {int(c): p for p, c in enumerate(cand)}

            # --- genre: shares at least one genre, and set overlap (Jaccard) ---
            seed_g = self.genre_sets[i]
            if seed_g:
                shares = np.fromiter(
                    (1.0 if seed_g & self.genre_sets[j] else 0.0 for j in cand),
                    dtype=np.float64,
                    count=len(cand),
                )
                jac = np.fromiter(
                    (
                        len(seed_g & self.genre_sets[j]) / len(seed_g | self.genre_sets[j])
                        if (seed_g | self.genre_sets[j])
                        else 0.0
                        for j in cand
                    ),
                    dtype=np.float64,
                    count=len(cand),
                )
                genre_hit.append(float(np.mean([shares[pos[j]] for j in top])))
                genre_hit_base.append(float(shares.mean()))
                genre_jac.append(float(np.mean([jac[pos[j]] for j in top])))
                genre_jac_base.append(float(jac.mean()))

            # --- era: aired within +/- window years of the seed ---
            if not np.isnan(self.year[i]):
                near = (np.abs(self.year[cand] - self.year[i]) <= window) & ~np.isnan(
                    self.year[cand]
                )
                era_hit.append(float(np.mean([near[pos[j]] for j in top])))
                era_hit_base.append(float(near.mean()))

            # --- artist: hard ground truth, scored only where a partner exists ---
            seed_a = self.artist_sets[i]
            if seed_a:
                partners = {int(j) for j in cand if self.artist_sets[j] & seed_a}
                if partners:
                    artist_hit.append(1.0 if any(j in partners for j in top) else 0.0)
                    artist_hit_base.append(_prob_any_hit(len(cand), len(partners), self.k))

        return [
            self._metric(
                f"artist_recall@{self.k}",
                "at least one recommendation by an artist behind the seed song, over seeds "
                "where such a theme exists in another anime",
                artist_hit,
                artist_hit_base,
            ),
            self._metric(
                f"genre_precision@{self.k}",
                "share of recommendations sharing at least one MAL genre with the seed",
                genre_hit,
                genre_hit_base,
            ),
            self._metric(
                f"genre_jaccard@{self.k}",
                "mean Jaccard overlap of MAL genre sets, seed vs recommendation",
                genre_jac,
                genre_jac_base,
            ),
            self._metric(
                f"era_precision@{self.k}",
                f"share of recommendations that aired within +/-{window} years of the seed",
                era_hit,
                era_hit_base,
            ),
        ]

    @staticmethod
    def _metric(
        name: str, description: str, values: list[float], baselines: list[float]
    ) -> Metric:
        v = np.asarray(values, dtype=np.float64)
        b = np.asarray(baselines, dtype=np.float64)
        lo, hi, lift = _bootstrap_ci(v, b)
        return Metric(
            name=name,
            description=description,
            value=float(v.mean()) if len(v) else float("nan"),
            baseline=float(b.mean()) if len(b) else float("nan"),
            lift=lift,
            ci_low=lo,
            ci_high=hi,
            n_seeds=int(len(v)),
        )


def _print_report(metrics: list[Metric], meta: dict[str, object]) -> None:
    print("\n=== evaluation ===")
    for key, val in meta.items():
        print(f"{key}: {val}")
    print()
    header = (
        f"{'metric':<22} {'value':>7}  " + "95% CI".ljust(18) + f"{'random':>8}  {'lift':>6}  {'seeds':>6}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(m.format_row())
    print()
    for m in metrics:
        print(f"{m.name}: {m.description}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animethemes-evaluate")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=list(MODE_SCALE),
        default="purist",
        help=(
            "scoring mode to evaluate (default: purist). purist is audio-only; the other "
            "modes fold genre and artist into the score, which is exactly what these "
            "metrics measure, so their numbers are circular"
        ),
    )
    parser.add_argument(
        "--include-same-anime",
        dest="exclude_same_anime",
        action="store_false",
        default=True,
        help="also score themes from the seed's own anime (they match on genre/era for free)",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config()
    rec = Recommender(cfg)
    ev = Evaluator(rec, k=args.k, mode=args.mode, exclude_same_anime=args.exclude_same_anime)
    metrics = ev.run()

    meta: dict[str, object] = {
        "dataset": str(cfg.dataset_parquet),
        "themes": len(rec.df),
        "anime": int(pd.Series(rec.mal_id).nunique()),
        "embed_model": cfg.embed_model,
        "embed_version": cfg.embed_version,
        "mode": args.mode,
        "k": args.k,
        "exclude_same_anime": args.exclude_same_anime,
    }
    _print_report(metrics, meta)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"meta": meta, "metrics": [asdict(m) for m in metrics]}, indent=2),
            encoding="utf-8",
        )
        log.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
