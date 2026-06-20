from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))

    keep_audio: bool = field(default_factory=lambda: _env_bool("KEEP_AUDIO", False))

    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "m-a-p/MERT-v1-95M"))
    target_sr: int = field(default_factory=lambda: _env_int("TARGET_SR", 24000))
    chunk_seconds: int = field(default_factory=lambda: _env_int("CHUNK_SECONDS", 30))
    n_chunks: int = field(default_factory=lambda: _env_int("N_CHUNKS", 3))
    chunk_stride: int = field(default_factory=lambda: _env_int("CHUNK_STRIDE", 30))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 16))
    embed_flush_every: int = field(default_factory=lambda: _env_int("EMBED_FLUSH_EVERY", 50))

    animethemes_base_url: str = "https://api.animethemes.moe"
    animethemes_page_size: int = field(default_factory=lambda: _env_int("ANIMETHEMES_PAGE_SIZE", 100))

    jikan_base_url: str = "https://api.jikan.moe/v4"
    jikan_rate_per_sec: float = field(default_factory=lambda: _env_float("JIKAN_RATE_PER_SEC", 3.0))

    download_workers: int = field(default_factory=lambda: _env_int("DOWNLOAD_WORKERS", 8))
    http_timeout_seconds: float = field(default_factory=lambda: _env_float("HTTP_TIMEOUT", 60.0))

    user_agent: str = "anime-themes-recsys/0.1 (+research)"

    recommend_default_mode: str = field(
        default_factory=lambda: os.getenv("RECOMMEND_DEFAULT_MODE", "discovery")
    )
    recommend_default_k: int = field(default_factory=lambda: _env_int("RECOMMEND_DEFAULT_K", 20))
    recommend_w_anime: float = field(
        default_factory=lambda: _env_float("RECOMMEND_W_ANIME", 0.1)
    )
    recommend_w_artist: float = field(
        default_factory=lambda: _env_float("RECOMMEND_W_ARTIST", 0.3)
    )
    recommend_w_era: float = field(default_factory=lambda: _env_float("RECOMMEND_W_ERA", 0.05))
    recommend_era_window_years: int = field(
        default_factory=lambda: _env_int("RECOMMEND_ERA_WINDOW_YEARS", 2)
    )
    recommend_alpha_genre: float = field(
        default_factory=lambda: _env_float("RECOMMEND_ALPHA_GENRE", 0.3)
    )
    recommend_alpha_blend: float = field(
        default_factory=lambda: _env_float("RECOMMEND_ALPHA_BLEND", 0.3)
    )
    recommend_max_per_anime: int = field(
        default_factory=lambda: _env_int("RECOMMEND_MAX_PER_ANIME", 5)
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def animethemes_parquet(self) -> Path:
        return self.raw_dir / "animethemes.parquet"

    @property
    def mal_parquet(self) -> Path:
        return self.raw_dir / "mal.parquet"

    @property
    def download_errors_parquet(self) -> Path:
        return self.raw_dir / "download_errors.parquet"

    @property
    def embed_errors_parquet(self) -> Path:
        return self.raw_dir / "embed_errors.parquet"

    @property
    def embeddings_dir(self) -> Path:
        return self.interim_dir / "embeddings"

    @property
    def embeddings_parquet(self) -> Path:
        """Legacy single-file embeddings store. New runs write shards into embeddings_dir."""
        return self.interim_dir / "embeddings.parquet"

    @property
    def embed_version(self) -> str:
        """Derived from chunk hyperparameters so config drift invalidates stale embeddings."""
        return f"sr{self.target_sr}-c{self.chunk_seconds}x{self.n_chunks}s{self.chunk_stride}"

    @property
    def dataset_parquet(self) -> Path:
        return self.processed_dir / "dataset.parquet"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.audio_dir,
            self.embeddings_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
