from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

MODE_SCALE: dict[str, float] = {"purist": 0.0, "discovery": 1.0, "comfort": 2.0}

SMOKE_SEEDS: list[int] = [
    2279,  # Death Note OP1 — 'THE WORLD' (Nightmare)
    3434,  # Fullmetal Alchemist: Brotherhood OP1 — 'again' (YUI)
    6972,  # Boku no Hero Academia OP — 'The Day' (Porno Graffitti)
    2281,  # Death Note ED1 — 'Alumina' (slow ballad)
]


class Recommender:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        df = pd.read_parquet(cfg.dataset_parquet)
        if df.empty:
            raise SystemExit(f"dataset is empty: {cfg.dataset_parquet}")

        self._validate_embed_metadata(df)

        raw = np.stack([np.asarray(v, dtype=np.float32) for v in df["embedding"].to_numpy()])
        if raw.ndim != 2:
            raise SystemExit(f"expected (N, D) embedding stack, got shape {raw.shape}")

        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        bad = (norms <= 0).flatten()
        if bad.any():
            offenders = df.loc[bad, "theme_id"].astype(int).tolist()[:10]
            raise SystemExit(
                f"{int(bad.sum())} embeddings have zero norm; cosine would produce NaN. "
                f"Offending theme_ids (sample): {offenders}. "
                f"Re-run Stage 4 for these themes or drop them from the dataset."
            )
        self.E: np.ndarray = (raw / norms).astype(np.float32, copy=False)

        self.df = df.reset_index(drop=True)
        self.theme_id_to_idx: dict[int, int] = {
            int(tid): i for i, tid in enumerate(self.df["theme_id"].to_numpy())
        }

        self.mal_id: np.ndarray = self.df["mal_id"].to_numpy(dtype=np.int64)
        self.audio_id: np.ndarray = self.df["audio_id"].to_numpy(dtype=np.int64)
        self.year: np.ndarray = self.df["year"].to_numpy(dtype=np.float64)
        self.artist_sets: list[set[int]] = [
            set(int(a) for a in (arr if arr is not None else []))
            for arr in self.df["artist_ids"].to_numpy()
        ]

        all_genres: list[str] = []
        seen: set[str] = set()
        for arr in self.df["genres"].to_numpy():
            for g in arr if arr is not None else []:
                if g not in seen:
                    seen.add(g)
                    all_genres.append(g)
        self._genre_index: dict[str, int] = {g: i for i, g in enumerate(all_genres)}
        G = np.zeros((len(self.df), len(all_genres)), dtype=np.float32)
        for i, arr in enumerate(self.df["genres"].to_numpy()):
            for g in arr if arr is not None else []:
                G[i, self._genre_index[g]] = 1.0
        self._genre_matrix: np.ndarray = G
        self._genre_sizes: np.ndarray = G.sum(axis=1)

        log.info(
            "Recommender ready: %d themes, %d genres, embedding dim=%d",
            len(self.df),
            len(all_genres),
            self.E.shape[1],
        )

    def _validate_embed_metadata(self, df: pd.DataFrame) -> None:
        bad_model = df["embed_model"] != self.cfg.embed_model
        bad_version = df["embed_version"] != self.cfg.embed_version
        bad = bad_model | bad_version
        if bad.any():
            offenders = (
                df.loc[bad, ["theme_id", "embed_model", "embed_version"]]
                .head(5)
                .to_dict("records")
            )
            raise SystemExit(
                f"dataset embeddings disagree with config "
                f"(config={self.cfg.embed_model}/{self.cfg.embed_version}). "
                f"Offending rows (sample): {offenders}. "
                f"Re-run Stage 4 with --force-reembed and rebuild the dataset."
            )

    def cosine_scores(self, seed_idx: int) -> np.ndarray:
        return self.E @ self.E[seed_idx]

    def _w_max(self) -> float:
        """Theoretical max of metadata_weights — used to normalize w into [0, 1]."""
        cfg = self.cfg
        return float(
            cfg.recommend_w_anime
            + cfg.recommend_w_artist
            + cfg.recommend_w_era
            + cfg.recommend_alpha_genre
        )

    def metadata_weights(self, seed_idx: int) -> np.ndarray:
        cfg = self.cfg
        n = len(self.df)
        w = np.zeros(n, dtype=np.float32)

        same_anime = (self.mal_id == self.mal_id[seed_idx]).astype(np.float32)
        w += cfg.recommend_w_anime * same_anime

        seed_artists = self.artist_sets[seed_idx]
        if seed_artists:
            same_artist = np.fromiter(
                (1.0 if (s and (s & seed_artists)) else 0.0 for s in self.artist_sets),
                dtype=np.float32,
                count=n,
            )
            w += cfg.recommend_w_artist * same_artist

        seed_year = self.year[seed_idx]
        if not np.isnan(seed_year):
            era_diff = np.abs(self.year - seed_year)
            era_match = (era_diff <= cfg.recommend_era_window_years).astype(np.float32)
            era_match[np.isnan(self.year)] = 0.0
            w += cfg.recommend_w_era * era_match

        seed_genres = self._genre_matrix[seed_idx]
        seed_size = float(self._genre_sizes[seed_idx])
        if seed_size > 0:
            inter = self._genre_matrix @ seed_genres
            union = seed_size + self._genre_sizes - inter
            jaccard = np.divide(
                inter,
                union,
                out=np.zeros_like(inter),
                where=union > 0,
            )
            w += cfg.recommend_alpha_genre * jaccard.astype(np.float32)

        w_max = self._w_max()
        if w_max > 0:
            w = w / w_max
        return w

    def recommend(
        self,
        seed_theme_id: int,
        mode: str = "discovery",
        k: int = 20,
        exclude_same_anime: bool = False,
        dedup_audio: bool = True,
        max_per_anime: int | None = None,
    ) -> pd.DataFrame:
        if mode not in MODE_SCALE:
            raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODE_SCALE)}")
        if seed_theme_id not in self.theme_id_to_idx:
            raise SystemExit(
                f"seed theme_id {seed_theme_id} not found in dataset "
                f"({len(self.theme_id_to_idx)} themes loaded)"
            )
        seed_idx = self.theme_id_to_idx[seed_theme_id]

        cos = self.cosine_scores(seed_idx)
        w = self.metadata_weights(seed_idx)
        final = cos + MODE_SCALE[mode] * self.cfg.recommend_alpha_blend * w

        keep = np.ones(len(self.df), dtype=bool)
        keep[seed_idx] = False
        if exclude_same_anime:
            keep &= self.mal_id != self.mal_id[seed_idx]

        scores = pd.DataFrame(
            {
                "theme_id": self.df["theme_id"].to_numpy(),
                "mal_id": self.df["mal_id"].to_numpy(),
                "audio_id": self.df["audio_id"].to_numpy(),
                "title": self.df["title"].to_numpy(),
                "theme_type": self.df["theme_type"].to_numpy(),
                "sequence": self.df["sequence"].to_numpy(),
                "song_title": self.df["song_title"].to_numpy(),
                "artist_names": self.df["artist_names"].to_numpy(),
                "year": self.df["year"].to_numpy(),
                "anime_score": self.df["score"].to_numpy(),
                "l1_cos": cos.astype(np.float64),
                "l2_weight": w.astype(np.float64),
                "final": final.astype(np.float64),
            }
        ).loc[keep]

        if dedup_audio:
            scores = scores.sort_values("final", ascending=False).drop_duplicates(
                subset=["audio_id"], keep="first"
            )

        if max_per_anime is not None and max_per_anime > 0:
            scores = scores.sort_values("final", ascending=False).groupby(
                "mal_id", sort=False, as_index=False
            ).head(max_per_anime)

        top = scores.nlargest(k, "final").reset_index(drop=True)
        top.insert(0, "rank", np.arange(1, len(top) + 1, dtype=np.int64))
        top["theme_label"] = [
            f"{tt}{'' if pd.isna(seq) else int(seq)}"
            for tt, seq in zip(top["theme_type"], top["sequence"])
        ]
        top["artist_names"] = top["artist_names"].apply(
            lambda arr: ", ".join(arr) if arr is not None and len(arr) else ""
        )
        return top[
            [
                "rank",
                "theme_id",
                "mal_id",
                "audio_id",
                "title",
                "theme_label",
                "song_title",
                "artist_names",
                "year",
                "anime_score",
                "l1_cos",
                "l2_weight",
                "final",
            ]
        ]


def _print_results(seed_label: str, df: pd.DataFrame) -> None:
    print(f"\n=== {seed_label} ===")
    if df.empty:
        print("(no results)")
        return
    fmt = df.copy()
    fmt["title"] = fmt["title"].str.slice(0, 40)
    fmt["song_title"] = fmt["song_title"].fillna("").astype(str).str.slice(0, 30)
    fmt["artist_names"] = fmt["artist_names"].astype(str).str.slice(0, 30)
    fmt["year"] = fmt["year"].apply(lambda v: "" if pd.isna(v) else f"{int(v)}")
    fmt["anime_score"] = fmt["anime_score"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    for col in ("l1_cos", "l2_weight", "final"):
        fmt[col] = fmt[col].map(lambda v: f"{v:+.3f}")
    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.max_colwidth", 40
    ):
        print(fmt.to_string(index=False))


def _seed_label(rec: Recommender, seed_theme_id: int) -> str:
    idx = rec.theme_id_to_idx[seed_theme_id]
    row = rec.df.iloc[idx]
    seq = "" if pd.isna(row["sequence"]) else int(row["sequence"])
    artist = ", ".join(row["artist_names"]) if len(row["artist_names"]) else "?"
    return (
        f"seed theme_id={seed_theme_id} | {row['title']} {row['theme_type']}{seq} "
        f"— {row['song_title']} ({artist})"
    )


def _run_smoke(rec: Recommender) -> int:
    failures: list[str] = []
    for seed in SMOKE_SEEDS:
        if seed not in rec.theme_id_to_idx:
            failures.append(f"smoke seed {seed} not present in dataset")
            continue

        seed_idx = rec.theme_id_to_idx[seed]
        cos_self = float(rec.cosine_scores(seed_idx)[seed_idx])
        if cos_self < 0.999:
            failures.append(
                f"self-cosine for theme_id={seed} is {cos_self:.6f}, expected ≥ 0.999"
            )

        purist = rec.recommend(seed, mode="purist", k=20)
        comfort = rec.recommend(seed, mode="comfort", k=20)
        _print_results(_seed_label(rec, seed) + "  [discovery, top-10]",
                       rec.recommend(seed, mode="discovery", k=10))

        same_mal = rec.mal_id[seed_idx]
        siblings = [
            int(t) for t in rec.df["theme_id"].to_numpy()
            if rec.mal_id[rec.theme_id_to_idx[int(t)]] == same_mal and int(t) != seed
        ]
        if siblings:
            def best_rank(df: pd.DataFrame) -> int | None:
                hits = df.index[df["theme_id"].isin(siblings)].tolist()
                return int(df.loc[hits[0], "rank"]) if hits else None

            r_purist = best_rank(purist)
            r_comfort = best_rank(comfort)
            if r_comfort is None:
                failures.append(
                    f"theme_id={seed}: no sibling found in comfort top-20 "
                    f"(siblings exist: {siblings[:5]})"
                )
            elif r_purist is not None and r_comfort >= r_purist:
                failures.append(
                    f"theme_id={seed}: comfort sibling rank ({r_comfort}) is not better "
                    f"than purist ({r_purist}); L2 wiring suspect"
                )

    print("\n=== smoke results ===")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(f"OK: {len(SMOKE_SEEDS)} seeds passed self-cosine + comfort-vs-purist checks")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animethemes-recommend")
    parser.add_argument("--seed-theme-id", type=int, default=None)
    parser.add_argument("--mode", choices=list(MODE_SCALE), default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--exclude-same-anime", action="store_true")
    parser.add_argument(
        "--no-dedup-audio",
        dest="dedup_audio",
        action="store_false",
        default=True,
        help="show every theme even if it shares an audio_id with a higher-ranked theme",
    )
    parser.add_argument(
        "--max-per-anime",
        type=int,
        default=None,
        help="cap themes from the same anime (default: cfg.recommend_max_per_anime; pass 0 to disable)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config()
    rec = Recommender(cfg)

    if args.smoke:
        return _run_smoke(rec)

    if args.seed_theme_id is None:
        parser.error("--seed-theme-id is required (or use --smoke)")

    mode = args.mode or cfg.recommend_default_mode
    k = args.k if args.k is not None else cfg.recommend_default_k
    max_per_anime = args.max_per_anime if args.max_per_anime is not None else cfg.recommend_max_per_anime
    df = rec.recommend(
        seed_theme_id=args.seed_theme_id,
        mode=mode,
        k=k,
        exclude_same_anime=args.exclude_same_anime,
        dedup_audio=args.dedup_audio,
        max_per_anime=max_per_anime if max_per_anime > 0 else None,
    )
    _print_results(_seed_label(rec, args.seed_theme_id) + f"  [{mode}, top-{k}]", df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
