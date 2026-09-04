"""Offline evaluation of the recommender against metadata ground truth.

The audio embeddings never see genres, artists or air dates — those columns come
from MyAnimeList. So asking where a purely acoustic ranking puts a theme that
shares an artist (or a genre, or an era) with the seed is a cheap check that the
embeddings carry musical signal rather than noise.

Every metric is reported next to what a random ranking would score on the same
candidate pool, because the pool is far from uniform: two thirds of random theme
pairs already share a genre. The ratio of the two is the number to read.

Artist agreement is the metric with teeth — it has hard ground truth, and the
seed's own anime is excluded, so a hit is never the same song or a sibling OP.
The genre and era numbers are near-useless as a quality signal (the random
baseline is already at 0.68) and are kept as negative controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .recommend import MODE_SCALE, Recommender

log = logging.getLogger(__name__)

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
RECALL_KS: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100)
# How many of an artist's listener-derived neighbours count as related. The full
# ListenBrainz list runs to 100 and marks ~15% of this corpus as relevant, which
# leaves a metric with no headroom (a random top-10 would score 0.80). Three keeps
# the random baseline near 0.18. Chosen on target density alone, before any system
# was scored against it.
SIMILAR_ARTIST_TOP_N = 3


@dataclass
class Metric:
    """One evaluated quantity, its random-ranking baseline and a bootstrap CI."""

    name: str
    value: float
    baseline: float
    lift: float
    ci_low: float
    ci_high: float
    n_seeds: int


@dataclass
class Results:
    meta: dict[str, object]
    artist: list[Metric] = field(default_factory=list)
    controls: list[Metric] = field(default_factory=list)
    artist_ranks: list[int] = field(default_factory=list)
    artist_pool_sizes: list[int] = field(default_factory=list)
    artist_seed_ids: list[int] = field(default_factory=list)
    behavioural: list[Metric] = field(default_factory=list)
    behavioural_ranks: list[int] = field(default_factory=list)
    behavioural_seed_ids: list[int] = field(default_factory=list)


def load_behavioural_neighbours(cfg: Config) -> dict[str, set[str]] | None:
    """Listener-derived artist adjacency, keyed by MusicBrainz id.

    Returns None when the two cache files have not been built — the metric is optional
    because it needs two external services, while everything else here runs offline.
    """
    if not (cfg.mb_artists_json.exists() and cfg.lb_similar_json.exists()):
        return None
    resolved = json.loads(cfg.mb_artists_json.read_text(encoding="utf-8"))
    similar = json.loads(cfg.lb_similar_json.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for entry in resolved.values():
        if not entry:
            continue
        mbid = entry["mbid"]
        rows = similar.get(mbid) or []
        top = sorted(rows, key=lambda r: -(r.get("score") or 0))[:SIMILAR_ARTIST_TOP_N]
        out[mbid] = {r["mbid"] for r in top if r.get("mbid")}
    return out


def load_artist_mbids(cfg: Config) -> dict[str, str]:
    """Corpus artist name -> MusicBrainz id, for the artists that resolved."""
    if not cfg.mb_artists_json.exists():
        return {}
    resolved = json.loads(cfg.mb_artists_json.read_text(encoding="utf-8"))
    return {name: e["mbid"] for name, e in resolved.items() if e}


def _miss_probs(n_candidates: int, n_relevant: int, max_r: int) -> np.ndarray:
    """P(no relevant item in the first r of a random ranking), for r = 0..max_r."""
    r = np.arange(max_r + 1)
    num = n_candidates - n_relevant - r
    den = n_candidates - r
    step = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)
    step = np.clip(step, 0.0, 1.0)
    return np.concatenate([[1.0], np.cumprod(step[:-1])])


def _random_recall(n_candidates: int, n_relevant: int, k: int) -> float:
    """P(a random ranking puts a relevant item in the top k)."""
    if n_relevant <= 0 or k <= 0:
        return 0.0
    k = min(k, n_candidates)
    return float(1.0 - _miss_probs(n_candidates, n_relevant, k)[k])


def _random_mrr(n_candidates: int, n_relevant: int) -> float:
    """E[1/rank of the first relevant item] under a uniformly random ranking."""
    if n_relevant <= 0:
        return 0.0
    max_r = n_candidates - n_relevant + 1
    miss = _miss_probs(n_candidates, n_relevant, max_r)
    # P(first hit at rank r) = P(miss through r-1) - P(miss through r)
    p_at = miss[:-1] - miss[1:]
    ranks = np.arange(1, len(p_at) + 1, dtype=np.float64)
    return float(np.sum(p_at / ranks))


def _bootstrap_ci(per_seed: np.ndarray) -> tuple[float, float]:
    """95% percentile CI for the mean over seeds."""
    if len(per_seed) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(per_seed), size=(BOOTSTRAP_RESAMPLES, len(per_seed)))
    means = per_seed[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _metric(name: str, values: np.ndarray, baselines: np.ndarray) -> Metric:
    lo, hi = _bootstrap_ci(values)
    base = float(baselines.mean()) if len(baselines) else float("nan")
    val = float(values.mean()) if len(values) else float("nan")
    return Metric(
        name=name,
        value=val,
        baseline=base,
        lift=(val / base) if base > 0 else float("nan"),
        ci_low=lo,
        ci_high=hi,
        n_seeds=int(len(values)),
    )


class Evaluator:
    """Scores the live Recommender — not a reimplementation of it — seed by seed."""

    def __init__(self, rec: Recommender, mode: str, exclude_same_anime: bool) -> None:
        self.rec = rec
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

        name_to_mbid = load_artist_mbids(rec.cfg)
        self.neighbours = load_behavioural_neighbours(rec.cfg)
        self.theme_mbids: list[set[str]] = [
            {name_to_mbid[str(a)] for a in (arr if arr is not None else []) if str(a) in name_to_mbid}
            for arr in df["artist_names"].to_numpy()
        ]

    def _ranked_indices(self, seed_idx: int) -> list[int]:
        """The seed's full candidate ranking, straight out of the production path."""
        top = self.rec.recommend(
            seed_theme_id=int(self.theme_ids[seed_idx]),
            mode=self.mode,
            k=self.n,
            exclude_same_anime=self.exclude_same_anime,
            dedup_audio=True,
            max_per_anime=None,
        )
        return [self.rec.theme_id_to_idx[int(t)] for t in top["theme_id"]]

    def run(self, control_k: int = 10) -> Results:
        window = self.rec.cfg.recommend_era_window_years

        first_rank: list[int] = []          # rank of the first same-artist hit
        seed_ids: list[int] = []
        beh_rank: list[int] = []            # rank of the first behaviourally-related hit
        beh_seed_ids: list[int] = []
        beh_pool: list[int] = []
        beh_rel: list[int] = []
        pool_size: list[int] = []
        n_partners: list[int] = []
        genre_hit: list[float] = []
        genre_hit_base: list[float] = []
        genre_jac: list[float] = []
        genre_jac_base: list[float] = []
        era_hit: list[float] = []
        era_hit_base: list[float] = []

        for i in range(self.n):
            order = self._ranked_indices(i)
            if len(order) < control_k:
                continue
            head = order[:control_k]

            seed_g = self.genre_sets[i]
            if seed_g:
                shares = np.fromiter(
                    (1.0 if seed_g & self.genre_sets[j] else 0.0 for j in order),
                    dtype=np.float64,
                    count=len(order),
                )
                jac = np.fromiter(
                    (
                        len(seed_g & self.genre_sets[j]) / len(seed_g | self.genre_sets[j])
                        if (seed_g | self.genre_sets[j])
                        else 0.0
                        for j in order
                    ),
                    dtype=np.float64,
                    count=len(order),
                )
                genre_hit.append(float(shares[:control_k].mean()))
                genre_hit_base.append(float(shares.mean()))
                genre_jac.append(float(jac[:control_k].mean()))
                genre_jac_base.append(float(jac.mean()))

            if not np.isnan(self.year[i]):
                yrs = self.year[list(order)]
                near = (np.abs(yrs - self.year[i]) <= window) & ~np.isnan(yrs)
                era_hit.append(float(near[:control_k].mean()))
                era_hit_base.append(float(near.mean()))

            seed_a = self.artist_sets[i]
            if seed_a:
                partners = {j for j in order if self.artist_sets[j] & seed_a}
                if partners:
                    rank = next(
                        (p + 1 for p, j in enumerate(order) if j in partners), len(order) + 1
                    )
                    first_rank.append(rank)
                    seed_ids.append(int(self.theme_ids[i]))
                    pool_size.append(len(order))
                    n_partners.append(len(partners))

            # Behavioural relevance deliberately EXCLUDES the seed's own artists: the
            # question is whether audio finds a *different* act that listeners treat as
            # adjacent, which is the part artist agreement cannot see.
            if self.neighbours is not None and self.theme_mbids[i]:
                nbrs: set[str] = set()
                for mbid in self.theme_mbids[i]:
                    nbrs |= self.neighbours.get(mbid, set())
                nbrs -= self.theme_mbids[i]
                if nbrs:
                    related = {
                        j
                        for j in order
                        if self.theme_mbids[j]
                        and not (self.theme_mbids[j] & self.theme_mbids[i])
                        and (self.theme_mbids[j] & nbrs)
                    }
                    if related:
                        beh_rank.append(
                            next(
                                (p + 1 for p, j in enumerate(order) if j in related),
                                len(order) + 1,
                            )
                        )
                        beh_seed_ids.append(int(self.theme_ids[i]))
                        beh_pool.append(len(order))
                        beh_rel.append(len(related))

        ranks = np.asarray(first_rank, dtype=np.float64)
        pools = np.asarray(pool_size, dtype=np.int64)
        parts = np.asarray(n_partners, dtype=np.int64)

        artist: list[Metric] = []
        for k in RECALL_KS:
            hits = (ranks <= k).astype(np.float64)
            base = np.asarray(
                [_random_recall(int(m), int(p), k) for m, p in zip(pools, parts)],
                dtype=np.float64,
            )
            artist.append(_metric(f"artist_recall@{k}", hits, base))
        mrr_base = np.asarray(
            [_random_mrr(int(m), int(p)) for m, p in zip(pools, parts)], dtype=np.float64
        )
        artist.append(_metric("artist_mrr", 1.0 / ranks, mrr_base))
        # E[rank of the first of p relevant items in a random ranking of m] = (m+1)/(p+1)
        artist.append(_metric("artist_first_rank", ranks, (pools + 1) / (parts + 1)))

        controls = [
            _metric(
                f"genre_precision@{control_k}",
                np.asarray(genre_hit),
                np.asarray(genre_hit_base),
            ),
            _metric(
                f"genre_jaccard@{control_k}", np.asarray(genre_jac), np.asarray(genre_jac_base)
            ),
            _metric(f"era_precision@{control_k}", np.asarray(era_hit), np.asarray(era_hit_base)),
        ]

        behavioural: list[Metric] = []
        if beh_rank:
            br = np.asarray(beh_rank, dtype=np.float64)
            bp = np.asarray(beh_pool, dtype=np.int64)
            bn = np.asarray(beh_rel, dtype=np.int64)
            for k in (1, 5, 10, 20):
                hits = (br <= k).astype(np.float64)
                base = np.asarray(
                    [_random_recall(int(m), int(r), k) for m, r in zip(bp, bn)],
                    dtype=np.float64,
                )
                behavioural.append(_metric(f"related_recall@{k}", hits, base))
            behavioural.append(
                _metric(
                    "related_mrr",
                    1.0 / br,
                    np.asarray(
                        [_random_mrr(int(m), int(r)) for m, r in zip(bp, bn)],
                        dtype=np.float64,
                    ),
                )
            )

        return Results(
            meta={},
            artist=artist,
            controls=controls,
            behavioural=behavioural,
            behavioural_ranks=[int(r) for r in beh_rank],
            behavioural_seed_ids=beh_seed_ids,
            artist_ranks=[int(r) for r in first_rank],
            artist_pool_sizes=[int(m) for m in pool_size],
            artist_seed_ids=seed_ids,
        )


def paired_mrr_delta(a: Results, b: Results, family: str = "artist") -> dict[str, float]:
    """Per-seed MRR difference between two runs, with a bootstrap CI.

    Comparing two overlapping confidence intervals is the wrong test: every seed is
    scored by both systems, so the paired difference cancels the (large) variance of
    "some seeds are simply easier" and leaves the variance that matters.

    `family` picks the ground truth. Both are worth running: this project has a
    documented case where a change was a large, significant win on "artist" and bought
    nothing on "behavioural", which is invisible if you only ever compare one.
    """
    if family == "artist":
        ra = dict(zip(a.artist_seed_ids, a.artist_ranks))
        rb = dict(zip(b.artist_seed_ids, b.artist_ranks))
    elif family == "behavioural":
        ra = dict(zip(a.behavioural_seed_ids, a.behavioural_ranks))
        rb = dict(zip(b.behavioural_seed_ids, b.behavioural_ranks))
    else:
        raise ValueError(f"unknown family {family!r}")
    if not (ra and rb):
        return {}
    shared = sorted(set(ra) & set(rb))
    if not shared:
        raise SystemExit("the two runs share no scored seeds; are they the same corpus?")
    diff = np.array([1.0 / rb[s] - 1.0 / ra[s] for s in shared], dtype=np.float64)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    boot = diff[idx].mean(axis=1)
    wins = int(np.sum([rb[s] < ra[s] for s in shared]))
    losses = int(np.sum([rb[s] > ra[s] for s in shared]))
    return {
        "n_shared_seeds": float(len(shared)),
        "mrr_a": float(np.mean([1.0 / ra[s] for s in shared])),
        "mrr_b": float(np.mean([1.0 / rb[s] for s in shared])),
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "seeds_better": float(wins),
        "seeds_worse": float(losses),
    }


def corpus_fingerprint(theme_ids: np.ndarray) -> str:
    """Stable id for the exact set of themes scored, so runs stay comparable."""
    blob = ",".join(str(int(t)) for t in np.sort(theme_ids))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()[:16]


def _print_report(res: Results) -> None:
    print("\n=== evaluation ===")
    for key, val in res.meta.items():
        print(f"{key}: {val}")

    n = res.artist[0].n_seeds if res.artist else 0
    median_pool = int(np.median(res.artist_pool_sizes)) if res.artist_pool_sizes else 0
    print(
        f"\nartist agreement — {n} seeds whose artist also made a theme for a different "
        f"anime\n(median candidate pool {median_pool})"
    )
    print(f"  {'metric':<20} {'value':>8}  {'95% CI':<18} {'random':>8}  {'lift':>7}")
    for m in res.artist:
        if m.name == "artist_first_rank":
            # lower is better here, so the lift reads the other way round
            ci = f"[{m.ci_low:.1f}, {m.ci_high:.1f}]"
            better = (1.0 / m.lift) if m.lift else float("nan")
            print(
                f"  {m.name:<20} {m.value:>8.1f}  {ci:<18} {m.baseline:>8.1f}  {better:>6.2f}x"
            )
        else:
            ci = f"[{m.ci_low:.3f}, {m.ci_high:.3f}]"
            print(
                f"  {m.name:<20} {m.value:>8.3f}  {ci:<18} {m.baseline:>8.3f}  {m.lift:>6.2f}x"
            )
    if res.artist_ranks:
        print(f"  median rank of first hit: {int(np.median(res.artist_ranks))}")

    if res.behavioural:
        n_b = res.behavioural[0].n_seeds
        print(
            f"\nbehavioural agreement — {n_b} seeds; relevant = a theme by a DIFFERENT artist\n"
            "whom ListenBrainz listeners treat as adjacent (voice and production signature\n"
            "cannot carry this one, so it does not reward the same confound)"
        )
        print(f"  {'metric':<20} {'value':>8}  {'95% CI':<18} {'random':>8}  {'lift':>7}")
        for m in res.behavioural:
            ci = f"[{m.ci_low:.3f}, {m.ci_high:.3f}]"
            print(
                f"  {m.name:<20} {m.value:>8.3f}  {ci:<18} {m.baseline:>8.3f}  {m.lift:>6.2f}x"
            )
        if res.behavioural_ranks:
            print(f"  median rank of first hit: {int(np.median(res.behavioural_ranks))}")

    print("\nnegative controls — the random baseline is near the ceiling, so these")
    print("cannot separate a good system from a mediocre one; watch them, don't chase them")
    print(f"  {'metric':<20} {'value':>8}  {'95% CI':<18} {'random':>8}  {'lift':>7}")
    for m in res.controls:
        ci = f"[{m.ci_low:.3f}, {m.ci_high:.3f}]"
        print(f"  {m.name:<20} {m.value:>8.3f}  {ci:<18} {m.baseline:>8.3f}  {m.lift:>6.2f}x")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animethemes-evaluate")
    parser.add_argument(
        "--control-k", type=int, default=10, help="top-k used for the genre/era controls"
    )
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
    parser.add_argument(
        "--dataset", type=Path, default=None, help="score an alternative dataset parquet"
    )
    parser.add_argument(
        "--no-embed-check",
        dest="strict_embed_check",
        action="store_false",
        default=True,
        help="skip the embed_model/embed_version guard (for experimental embeddings)",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="also score this dataset and report the paired per-seed MRR difference "
        "(this run is the baseline, --dataset is the challenger)",
    )
    parser.add_argument("--label", default=None, help="name for this run, stored in the JSON")
    parser.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    parser.add_argument(
        "--dump-corpus", type=Path, default=None, help="write the scored theme_ids, one per line"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config()
    rec = Recommender(cfg, dataset_path=args.dataset, strict_embed_check=args.strict_embed_check)
    ev = Evaluator(rec, mode=args.mode, exclude_same_anime=args.exclude_same_anime)
    res = ev.run(control_k=args.control_k)

    theme_ids = rec.df["theme_id"].to_numpy()
    res.meta = {
        "label": args.label or "current",
        "dataset": str(rec.dataset_path),
        "themes": len(rec.df),
        "anime": int(pd.Series(rec.mal_id).nunique()),
        "corpus_sha256": corpus_fingerprint(theme_ids),
        "embed_model": str(rec.df["embed_model"].iloc[0]),
        "embed_version": str(rec.df["embed_version"].iloc[0]),
        "embed_dim": int(rec.E.shape[1]),
        "mode": args.mode,
        "exclude_same_anime": args.exclude_same_anime,
        "mean_offdiag_cosine": round(
            float((rec.E @ rec.E.T)[np.triu_indices(len(rec.df), 1)].mean()), 4
        ),
    }
    _print_report(res)

    if args.compare_to is not None:
        base_rec = Recommender(
            cfg, dataset_path=args.compare_to, strict_embed_check=args.strict_embed_check
        )
        base_res = Evaluator(
            base_rec, mode=args.mode, exclude_same_anime=args.exclude_same_anime
        ).run(control_k=args.control_k)
        res.meta["compared_to"] = str(args.compare_to)
        print(f"\npaired comparison vs {args.compare_to}")
        paired_all: dict[str, dict[str, float]] = {}
        for family, blurb in (
            ("artist", "same performer, another anime"),
            ("behavioural", "different performer, listener-adjacent"),
        ):
            cmp = paired_mrr_delta(base_res, res, family)
            if not cmp:
                print(f"\n  [{family}] no shared seeds — ground truth missing for one side")
                continue
            paired_all[family] = cmp
            verdict = (
                "SIGNIFICANT" if (cmp["ci_low"] > 0 or cmp["ci_high"] < 0) else "not significant"
            )
            print(f"\n  [{family}] {blurb}")
            print(f"    baseline MRR {cmp['mrr_a']:.4f}   this run {cmp['mrr_b']:.4f}")
            print(
                f"    delta        {cmp['delta']:+.4f}  "
                f"95% CI [{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}]  "
                f"n={int(cmp['n_shared_seeds'])}  -> {verdict}"
            )
            print(
                f"    per-seed     {int(cmp['seeds_better'])} better, "
                f"{int(cmp['seeds_worse'])} worse"
            )
        res.meta["paired"] = paired_all
        # The whole reason both families are reported: this project has a documented
        # change that was a large significant win on one and bought nothing on the other.
        if len(paired_all) == 2:
            signs = {f: (c["delta"] > 0) for f, c in paired_all.items()}
            if signs["artist"] != signs["behavioural"]:
                print(
                    "\n  NOTE: the two ground truths disagree in sign. Decide which "
                    "objective this change is for before shipping it."
                )

    if args.dump_corpus is not None:
        args.dump_corpus.parent.mkdir(parents=True, exist_ok=True)
        args.dump_corpus.write_text(
            "\n".join(str(int(t)) for t in np.sort(theme_ids)) + "\n", encoding="utf-8"
        )
        log.info("wrote %s", args.dump_corpus)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": res.meta,
            "artist": [asdict(m) for m in res.artist],
            "controls": [asdict(m) for m in res.controls],
            "behavioural": [asdict(m) for m in res.behavioural],
            "artist_ranks": res.artist_ranks,
            "behavioural_ranks": res.behavioural_ranks,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
