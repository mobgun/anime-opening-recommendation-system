from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .config import Config

log = logging.getLogger(__name__)


def _audio_path(cfg: Config, audio_basename: str) -> Path:
    return cfg.audio_dir / audio_basename


def _is_complete(path: Path, expected_size: int | None) -> bool:
    if not path.exists():
        return False
    if expected_size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == int(expected_size)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
def _download(client: httpx.Client, url: str, dest: Path) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size


def download_one(
    client: httpx.Client,
    cfg: Config,
    audio_id: int,
    audio_link: str,
    audio_basename: str,
    audio_size: int | None,
) -> dict[str, object] | None:
    """Returns an error dict on failure, None on success."""
    dest = _audio_path(cfg, audio_basename)
    if _is_complete(dest, audio_size):
        return None
    try:
        size = _download(client, audio_link, dest)
        if audio_size is not None and size != int(audio_size):
            log.warning(
                "audio_id=%s size mismatch: got %d expected %s", audio_id, size, audio_size
            )
        return None
    except Exception as e:
        return {
            "audio_id": audio_id,
            "audio_link": audio_link,
            "audio_basename": audio_basename,
            "error_class": type(e).__name__,
            "error_message": str(e)[:500],
        }


def download_all(cfg: Config, themes: pd.DataFrame) -> pd.DataFrame:
    """Stage 3: download every theme's audio file. Returns a DataFrame of errors."""
    cfg.ensure_dirs()
    targets = themes[["audio_id", "audio_link", "audio_basename", "audio_size"]].drop_duplicates(
        subset=["audio_id"]
    )
    targets = targets[targets["audio_link"].notna() & targets["audio_basename"].notna()]

    dup_basenames = targets["audio_basename"].duplicated(keep=False)
    if dup_basenames.any():
        offenders = (
            targets.loc[dup_basenames, ["audio_id", "audio_basename"]].head(10).to_dict("records")
        )
        raise SystemExit(
            f"Duplicate audio_basename values would overwrite each other on disk. "
            f"Sample: {offenders}. Fix Stage 1 output before re-running Stage 3."
        )

    errors: list[dict[str, object]] = []
    with httpx.Client(
        headers={"User-Agent": cfg.user_agent},
        timeout=cfg.http_timeout_seconds,
        follow_redirects=True,
    ) as client:
        with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
            futures = {
                pool.submit(
                    download_one,
                    client,
                    cfg,
                    int(row.audio_id),
                    str(row.audio_link),
                    str(row.audio_basename),
                    None if pd.isna(row.audio_size) else int(row.audio_size),
                ): int(row.audio_id)
                for row in targets.itertuples(index=False)
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="downloads",
                unit="file",
            ):
                err = fut.result()
                if err is not None:
                    errors.append(err)

    err_df = pd.DataFrame(errors)
    err_df.to_parquet(cfg.download_errors_parquet, index=False)
    log.info(
        "Stage 3: %d audio files attempted, %d errors written to %s",
        len(targets),
        len(err_df),
        cfg.download_errors_parquet,
    )
    return err_df


def downloaded_audio_basenames(cfg: Config) -> set[str]:
    """Audio basenames (filenames) present on disk."""
    return {p.name for p in cfg.audio_dir.glob("*.ogg")}


def run(cfg: Config) -> pd.DataFrame:
    themes = pd.read_parquet(cfg.animethemes_parquet)
    return download_all(cfg, themes)
