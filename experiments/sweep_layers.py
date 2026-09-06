"""Score every MERT layer through the production ranking path, honestly.

Thirteen candidates scored on 192 seeds is thirteen chances to find noise that
looks like a result, so the layer is chosen on one half of the anime and reported
on the other. The holdout number is the one to quote; the tune column is only
there to show what the selection saw.

    python -m experiments.sweep_layers
    python -m experiments.sweep_layers --family behavioural
    python -m experiments.sweep_layers --family artist

`--family` picks the ground truth the selection optimizes:

  artist       another song by one of the seed's own artists. Hard labels, ~12x lift,
               and a known confound - two songs by one performer share a voice, a band
               and a mastering chain, so this rewards recognising the singer. A plain
               MFCC vector (the classic speaker-ID feature set) reaches 75% of MERT's
               score on it, which is how we know the confound is real, not paranoia.
  behavioural  a theme by a DIFFERENT artist whom ListenBrainz listeners treat as
               adjacent. The confound cannot carry it, but it is the weaker instrument:
               ~1.7x lift, and its listener base skews away from Japanese music.
  both         score each family separately and say plainly when they disagree.

`both` is the default because it is the honest one for this project: EMBED_LAYER=6
was originally selected on `artist` alone, and the README then had to report that the
behavioural metric did not confirm the win. A sweep that can only see one family is
exactly how that happens.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np

from src.config import Config
from src.console import enable_utf8_output
from src.evaluate import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, Evaluator, Results
from src.recommend import Recommender

log = logging.getLogger(__name__)

FAMILIES = ("artist", "behavioural")


def _half(mal_id: int) -> int:
    """Stable anime-level split, so no show contributes seeds to both halves."""
    return int(hashlib.md5(str(int(mal_id)).encode()).hexdigest(), 16) % 2


def _ranks_for_family(res: Results, family: str) -> dict[int, int]:
    """{seed theme_id: rank of the first relevant hit} under one ground truth."""
    if family == "artist":
        return dict(zip(res.artist_seed_ids, res.artist_ranks))
    if family == "behavioural":
        return dict(zip(res.behavioural_seed_ids, res.behavioural_ranks))
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def _score_variant(
    cfg: Config, dataset: Path, mode: str, families: tuple[str, ...]
) -> tuple[dict[str, dict[int, int]], dict[int, int]]:
    """One evaluation pass, split into per-family rank tables.

    Both families fall out of the same `Evaluator.run()`, so asking for both costs
    nothing beyond the single pass one family already needs.
    """
    rec = Recommender(cfg, dataset_path=dataset, strict_embed_check=False)
    res = Evaluator(rec, mode=mode, exclude_same_anime=True).run()
    seed_to_mal = {int(t): int(m) for t, m in zip(rec.df["theme_id"], rec.mal_id)}
    return {f: _ranks_for_family(res, f) for f in families}, seed_to_mal


def _mrr(ranks: dict[int, int], seeds: list[int]) -> float:
    return float(np.mean([1.0 / ranks[s] for s in seeds])) if seeds else float("nan")


def _paired(a: dict[int, int], b: dict[int, int], seeds: list[int]) -> tuple[float, float, float]:
    diff = np.array([1.0 / b[s] - 1.0 / a[s] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _sweep_one_family(
    family: str,
    ranks_by_variant: dict[str, dict[int, int]],
    seed_mal: dict[int, int],
    reference: str,
) -> str | None:
    """Report one family's table and holdout test. Returns the variant it selects."""
    scored = {name: r for name, r in ranks_by_variant.items() if r}
    if not scored:
        print(f"\n[{family}] no seeds have ground truth - is the cache in benchmarks/ present?")
        return None

    shared = sorted(set.intersection(*(set(r) for r in scored.values())))
    if not shared:
        print(f"\n[{family}] the variants share no scored seeds")
        return None
    tune = [s for s in shared if _half(seed_mal[s]) == 0]
    hold = [s for s in shared if _half(seed_mal[s]) == 1]

    print(f"\n=== {family} ===")
    print(f"{len(shared)} shared seeds -> tune {len(tune)} / holdout {len(hold)} (split by anime)")
    print(f"\n{'layer':<10} {'tune MRR':>9} {'holdout MRR':>12}   (selection sees only tune)")
    for name in sorted(scored):
        marker = "  <- incumbent" if name == reference else ""
        print(
            f"{name:<10} {_mrr(scored[name], tune):>9.4f} "
            f"{_mrr(scored[name], hold):>12.4f}{marker}"
        )

    best = max(scored, key=lambda n: _mrr(scored[n], tune))
    print(f"\nbest on tune half: {best}")
    if reference not in scored:
        print(f"reference {reference} has no seeds in this family; skipping the holdout test")
        return best
    if best == reference:
        print(f"the incumbent {reference} already wins the selection; nothing to change")
        return best

    d, lo, hi = _paired(scored[reference], scored[best], hold)
    print(f"\nHOLDOUT, {best} vs {reference}, paired over {len(hold)} seeds:")
    print(f"  {reference:<10} MRR {_mrr(scored[reference], hold):.4f}")
    print(f"  {best:<10} MRR {_mrr(scored[best], hold):.4f}")
    print(f"  delta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant, keep the incumbent"
    print(f"  -> {verdict}")
    return best


def _print_verdict(winners: dict[str, str | None]) -> None:
    artist, behavioural = winners["artist"], winners["behavioural"]
    print("\n=== verdict ===")
    if artist is None or behavioural is None:
        print("only one family could be scored; this run cannot check for disagreement")
    elif artist == behavioural:
        print(f"both ground truths select {artist} - the choice is not metric-dependent")
    else:
        print(
            f"the two ground truths DISAGREE: artist selects {artist}, behavioural "
            f"selects {behavioural}."
        )
        print(
            "Decide which objective this project is for before shipping either: a layer\n"
            "chosen on 'artist' alone can be a better performer detector and no better\n"
            "recommender. This is not hypothetical - it is what happened here."
        )


def main(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    parser = argparse.ArgumentParser(prog="sweep-layers")
    parser.add_argument("--dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--mode", default="purist")
    parser.add_argument("--reference", default="layer12", help="the incumbent to beat")
    parser.add_argument(
        "--family",
        choices=[*FAMILIES, "both"],
        default="both",
        help=(
            "ground truth the selection optimizes (default: both). 'artist' is the "
            "stronger instrument but rewards recognising the performer; 'behavioural' "
            "is confound-free but noisier. See this module's docstring."
        ),
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    cfg = Config()
    variants = sorted(args.dir.glob("layer*.parquet"))
    if not variants:
        raise SystemExit(f"no layer*.parquet under {args.dir}")

    families: tuple[str, ...] = FAMILIES if args.family == "both" else (args.family,)

    by_family: dict[str, dict[str, dict[int, int]]] = {f: {} for f in families}
    seed_mal: dict[int, int] = {}
    for path in variants:
        ranks, mals = _score_variant(cfg, path, args.mode, families)
        for f in families:
            by_family[f][path.stem] = ranks[f]
        seed_mal.update(mals)
        counts = ", ".join(f"{f} {len(ranks[f])}" for f in families)
        print(f"scored {path.stem}: {counts} seeds", flush=True)

    winners = {
        f: _sweep_one_family(f, by_family[f], seed_mal, args.reference) for f in families
    }
    if len(families) == 2:
        _print_verdict(winners)
    return 0


if __name__ == "__main__":
    sys.exit(main())
