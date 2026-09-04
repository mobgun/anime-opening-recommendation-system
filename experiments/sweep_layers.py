"""Score every MERT layer through the production ranking path, honestly.

Thirteen candidates scored on 192 seeds is thirteen chances to find noise that
looks like a result, so the layer is chosen on one half of the anime and reported
on the other. The holdout number is the one to quote; the tune column is only
there to show what the selection saw.

    python -m experiments.sweep_layers
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np

from src.config import Config
from src.evaluate import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, Evaluator
from src.recommend import Recommender

log = logging.getLogger(__name__)


def _half(mal_id: int) -> int:
    """Stable anime-level split, so no show contributes seeds to both halves."""
    return int(hashlib.md5(str(int(mal_id)).encode()).hexdigest(), 16) % 2


def _ranks_by_seed(cfg: Config, dataset: Path, mode: str) -> tuple[dict[int, int], dict[int, int]]:
    """Returns {seed theme_id: rank of first same-artist hit} and {seed: mal_id}."""
    rec = Recommender(cfg, dataset_path=dataset, strict_embed_check=False)
    res = Evaluator(rec, mode=mode, exclude_same_anime=True).run()
    seed_to_mal = {
        int(t): int(m) for t, m in zip(rec.df["theme_id"], rec.mal_id)
    }
    ranks = dict(zip(res.artist_seed_ids, res.artist_ranks))
    return ranks, {s: seed_to_mal[s] for s in ranks}


def _mrr(ranks: dict[int, int], seeds: list[int]) -> float:
    return float(np.mean([1.0 / ranks[s] for s in seeds])) if seeds else float("nan")


def _paired(a: dict[int, int], b: dict[int, int], seeds: list[int]) -> tuple[float, float, float]:
    diff = np.array([1.0 / b[s] - 1.0 / a[s] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sweep-layers")
    parser.add_argument("--dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--mode", default="purist")
    parser.add_argument("--reference", default="layer12", help="the incumbent to beat")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    cfg = Config()
    variants = sorted(args.dir.glob("layer*.parquet"))
    if not variants:
        raise SystemExit(f"no layer*.parquet under {args.dir}")

    all_ranks: dict[str, dict[int, int]] = {}
    seed_mal: dict[int, int] = {}
    for path in variants:
        name = path.stem
        ranks, mals = _ranks_by_seed(cfg, path, args.mode)
        all_ranks[name] = ranks
        seed_mal.update(mals)
        print(f"scored {name}: {len(ranks)} seeds", flush=True)

    shared = sorted(set.intersection(*(set(r) for r in all_ranks.values())))
    tune = [s for s in shared if _half(seed_mal[s]) == 0]
    hold = [s for s in shared if _half(seed_mal[s]) == 1]
    print(f"\n{len(shared)} shared seeds -> tune {len(tune)} / holdout {len(hold)} (split by anime)")

    ref = args.reference
    if ref not in all_ranks:
        raise SystemExit(f"reference {ref} not among {sorted(all_ranks)}")

    print(f"\n{'layer':<10} {'tune MRR':>9} {'holdout MRR':>12}   (selection sees only tune)")
    for name in sorted(all_ranks):
        marker = "  <- incumbent" if name == ref else ""
        print(f"{name:<10} {_mrr(all_ranks[name], tune):>9.4f} {_mrr(all_ranks[name], hold):>12.4f}{marker}")

    best = max((n for n in all_ranks), key=lambda n: _mrr(all_ranks[n], tune))
    print(f"\nbest on tune half: {best}")
    if best == ref:
        print(f"the incumbent {ref} already wins the selection; nothing to change")
        return 0

    d, lo, hi = _paired(all_ranks[ref], all_ranks[best], hold)
    print(f"\nHOLDOUT, {best} vs {ref}, paired over {len(hold)} seeds:")
    print(f"  {ref:<10} MRR {_mrr(all_ranks[ref], hold):.4f}")
    print(f"  {best:<10} MRR {_mrr(all_ranks[best], hold):.4f}")
    print(f"  delta {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("  -> " + ("significant" if (lo > 0 or hi < 0) else "NOT significant, keep the incumbent"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
