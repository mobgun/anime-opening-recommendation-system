from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import Config
from .embed_store import (
    check_invalidation,
    read_all_embeddings,
    shard_paths,
    write_shard,
)

log = logging.getLogger(__name__)


class MERTEmbedder:
    def __init__(self, cfg: Config) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading %s on %s", cfg.embed_model, self.device)
        self.model = AutoModel.from_pretrained(cfg.embed_model, trust_remote_code=True)
        self.model.eval().to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.embed_model, trust_remote_code=True
        )
        proc_sr = int(self.processor.sampling_rate or 0)
        if proc_sr and proc_sr != cfg.target_sr:
            raise SystemExit(
                f"target_sr mismatch: config={cfg.target_sr}, processor={proc_sr}. "
                f"Set TARGET_SR={proc_sr} (and rerun with --force-reembed) or pick a "
                f"model whose sample rate matches your config."
            )
        self.target_sr = proc_sr or cfg.target_sr
        self.hidden_size = int(self.model.config.hidden_size)

    def _split_chunks(self, audio: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
        """Returns (chunks, real_lengths). Only the first chunk is zero-padded if short."""
        sr = self.target_sr
        chunk_len = self.cfg.chunk_seconds * sr
        stride = self.cfg.chunk_stride * sr
        chunks: list[np.ndarray] = []
        real_lens: list[int] = []
        for i in range(self.cfg.n_chunks):
            start = i * stride
            end = start + chunk_len
            piece = audio[start:end]
            if len(piece) == 0:
                break
            real_len = len(piece)
            if len(piece) < chunk_len:
                if i == 0:
                    piece = np.pad(piece, (0, chunk_len - len(piece)))
                else:
                    break
            chunks.append(piece)
            real_lens.append(real_len)
        return chunks, real_lens

    def _feature_lengths(self, sample_lens: torch.Tensor, t_feat: int) -> torch.Tensor:
        """Translate sample-level lengths to feature-level lengths via the conv stack."""
        if hasattr(self.model, "_get_feat_extract_output_lengths"):
            feat = self.model._get_feat_extract_output_lengths(sample_lens)
            if not torch.is_tensor(feat):
                feat = torch.as_tensor(feat, device=sample_lens.device)
            return feat.to(sample_lens.device).clamp(min=1, max=t_feat)
        cfg_m = self.model.config
        kernels = getattr(cfg_m, "conv_kernel", None)
        strides = getattr(cfg_m, "conv_stride", None)
        if kernels and strides:
            feat = sample_lens.clone()
            for k, s in zip(kernels, strides):
                feat = (feat - k) // s + 1
            return feat.clamp(min=1, max=t_feat)
        return torch.full_like(sample_lens, t_feat)

    @torch.inference_mode()
    def embed(self, audio: np.ndarray) -> tuple[np.ndarray, int, float]:
        """Returns (vector, n_chunks_used, total_seconds_used).

        Mean-pools only over real samples.
        """
        chunks, real_lens = self._split_chunks(audio)
        if not chunks:
            raise ValueError("audio shorter than one chunk")
        inputs = self.processor(
            chunks,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        b, t_in = input_values.shape

        sample_lens = torch.tensor(real_lens, dtype=torch.long, device=self.device)
        sample_idx = torch.arange(t_in, device=self.device).unsqueeze(0)
        sample_mask = (sample_idx < sample_lens.unsqueeze(1)).long()

        layer = self.cfg.embed_layer
        out = self.model(
            input_values=input_values,
            attention_mask=sample_mask,
            output_hidden_states=layer != -1,
        )
        # every layer falls out of the same forward pass, so selecting one costs nothing
        hidden = out.last_hidden_state if layer == -1 else out.hidden_states[layer]  # (B, T, H)
        t_feat = hidden.shape[1]

        feat_lens = self._feature_lengths(sample_lens, t_feat)
        feat_idx = torch.arange(t_feat, device=self.device).unsqueeze(0)
        feat_mask = (feat_idx < feat_lens.unsqueeze(1)).float().unsqueeze(-1)
        denom = feat_mask.sum(dim=1).clamp(min=1.0)
        per_chunk = (hidden * feat_mask).sum(dim=1) / denom  # (B, H)

        vec = per_chunk.mean(dim=0).cpu().numpy().astype(np.float32)
        total_seconds = sum(real_lens) / self.target_sr
        return vec, len(chunks), total_seconds


def embed_all(cfg: Config, force_reembed: bool = False) -> pd.DataFrame:
    """Stage 4: produce 768-dim MERT embedding per theme via 3x30s mean-pooling."""
    import librosa

    cfg.ensure_dirs()
    existing = check_invalidation(cfg, force_reembed)
    done_theme_ids: set[int] = (
        set(existing["theme_id"].astype(int).tolist()) if not existing.empty else set()
    )

    themes = pd.read_parquet(cfg.animethemes_parquet)
    targets = themes[~themes["theme_id"].astype(int).isin(done_theme_ids)].reset_index(drop=True)
    if targets.empty:
        log.info("Stage 4: nothing to embed (all %d themes already done)", len(themes))
        return existing

    embedder = MERTEmbedder(cfg)

    new_rows: list[dict] = []
    error_rows: list[dict] = []
    flush_every = cfg.embed_flush_every
    written = 0

    for row in tqdm(
        targets.itertuples(index=False), total=len(targets), desc="embed", unit="theme"
    ):
        theme_id = int(row.theme_id)
        audio_id = int(row.audio_id)
        audio_basename = str(row.audio_basename)
        ogg = cfg.audio_dir / audio_basename
        try:
            if not ogg.exists():
                raise FileNotFoundError(f"missing audio at {ogg} (audio_id={audio_id})")
            audio, _ = librosa.load(str(ogg), sr=embedder.target_sr, mono=True)
            if audio.size == 0:
                raise ValueError("decoded audio is empty")
            vec, n_chunks, total_seconds = embedder.embed(audio.astype(np.float32))
            new_rows.append(
                {
                    "theme_id": theme_id,
                    "audio_id": audio_id,
                    "embedding": vec.tolist(),
                    "embed_model": cfg.embed_model,
                    "embed_version": cfg.embed_version,
                    "n_chunks_used": n_chunks,
                    "total_seconds_used": total_seconds,
                }
            )
            if not cfg.keep_audio:
                ogg.unlink(missing_ok=True)
        except Exception as e:
            error_rows.append(
                {
                    "theme_id": theme_id,
                    "audio_id": audio_id,
                    "error_class": type(e).__name__,
                    "error_message": str(e)[:500],
                }
            )

        if len(new_rows) >= flush_every:
            written += write_shard(cfg, new_rows)
            new_rows = []

    written += write_shard(cfg, new_rows)

    if error_rows:
        existing_errors = (
            pd.read_parquet(cfg.embed_errors_parquet)
            if cfg.embed_errors_parquet.exists()
            else pd.DataFrame()
        )
        all_errors = pd.concat([existing_errors, pd.DataFrame(error_rows)], ignore_index=True)
        all_errors.to_parquet(cfg.embed_errors_parquet, index=False)
        log.warning("Stage 4: %d errors logged to %s", len(error_rows), cfg.embed_errors_parquet)

    final = read_all_embeddings(cfg)
    log.info(
        "Stage 4: wrote %d new embeddings; store now has %d rows across %d shard(s)",
        written,
        len(final),
        len(shard_paths(cfg)),
    )
    return final


def run(cfg: Config, force_reembed: bool = False) -> pd.DataFrame:
    return embed_all(cfg, force_reembed=force_reembed)
