"""Regression test for the crash that killed the CLI on 7 of 621 real themes.

AnimeThemes titles contain characters no 8-bit codepage covers (`Coda〜Death note`,
`R★O★C★K★S`, `Tokyo Ghoul √A`). On Windows sys.stdout follows the console codepage,
so printing one raised UnicodeEncodeError and took the whole command down.
"""

from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import pytest

from src.console import enable_utf8_output
from src.recommend import Recommender, _print_results, _seed_label
from tests.conftest import make_dataset

# The exact strings that crashed, straight out of the reference corpus
HOSTILE_TITLES = [
    "Coda〜Death note",
    "R★O★C★K★S",
    "Tokyo Ghoul √A",
    "Chikatto Chika Chika♡",
]


def test_enable_utf8_output_is_idempotent_and_safe_under_capture() -> None:
    enable_utf8_output()
    enable_utf8_output()


def test_enable_utf8_output_tolerates_a_stream_without_reconfigure(monkeypatch) -> None:
    """pytest's capture object and some CI wrappers have no .reconfigure."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    enable_utf8_output()  # must not raise


@pytest.mark.parametrize("title", HOSTILE_TITLES)
def test_hostile_titles_survive_a_legacy_codepage(title: str) -> None:
    """A cp1251 stream with the project's error policy must degrade, not raise."""
    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1251", errors="replace")
    print(title, file=buf)  # this raised UnicodeEncodeError before enable_utf8_output


def test_print_results_handles_a_hostile_title(cfg, capsys) -> None:
    e = np.eye(4, 6, dtype=np.float32)
    df = make_dataset(
        cfg,
        embeddings=e,
        mal_ids=[1, 2, 3, 4],
        artist_ids=[[10], [11], [12], [13]],
    )
    df["song_title"] = HOSTILE_TITLES
    df["title"] = ["Tokyo Ghoul √A"] * 4
    df.to_parquet(cfg.dataset_parquet, index=False)

    rec = Recommender(cfg)
    top = rec.recommend(seed_theme_id=1, mode="purist", k=3)
    _print_results(_seed_label(rec, 1), top)
    assert "Coda〜Death note" in capsys.readouterr().out


def test_cli_does_not_crash_on_a_hostile_title(cfg) -> None:
    """End-to-end through the real entry point, with a legacy codepage forced on."""
    e = np.eye(4, 6, dtype=np.float32)
    df = make_dataset(cfg, embeddings=e, mal_ids=[1, 2, 3, 4], artist_ids=[[1], [2], [3], [4]])
    df["song_title"] = HOSTILE_TITLES
    df.to_parquet(cfg.dataset_parquet, index=False)

    proc = subprocess.run(
        [sys.executable, "-m", "src.recommend", "--seed-theme-id", "1", "--k", "3",
         "--log-level", "WARNING"],
        capture_output=True,
        env={**__import__("os").environ,
             "DATA_DIR": str(cfg.data_dir),
             "EMBED_MODEL": cfg.embed_model,
             "RECOMMEND_WHITEN_COMPONENTS": "0",
             "PYTHONIOENCODING": "cp1251"},
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in proc.stderr
