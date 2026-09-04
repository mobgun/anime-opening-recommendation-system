"""One pass over the audio that produces every embedding variant we want to compare.

Two things are being asked at once:

  * which MERT layer to pool — all of them come out of a single forward pass, so a
    13-way sweep costs exactly one embed run, not thirteen;
  * whether MERT earns its keep at all — a plain MFCC/chroma/contrast vector from
    librosa is the calibration point. "5x better than random" says little when
    random is the floor; "better than hand-built spectral features" is the claim
    that matters.

Each variant is written as a full dataset parquet: the frozen dataset's metadata with
the `embedding` column swapped out. That keeps the corpus byte-identical across
variants, so `animethemes-evaluate --dataset <variant> --no-embed-check` compares
rankings and nothing else.

    python -m experiments.extract_variants --out data/experiments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import Config
from src.embed import MERTEmbedder

log = logging.getLogger(__name__)

MFCC_VARIANT = "mfcc"


def _librosa_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mean+std over time of MFCC, chroma and spectral contrast — the classic baseline.

    chroma_stft rather than chroma_cqt: on this corpus the CQT version costs ~3x more
    per track for the same role in the vector.
    """
    import librosa

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    parts = [
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
        contrast.mean(axis=1), contrast.std(axis=1),
    ]
    return np.concatenate(parts).astype(np.float32)


class _AllLayerEmbedder(MERTEmbedder):
    """MERTEmbedder that returns the pooled vector for every layer at once."""

    @torch.inference_mode()
    def embed_all_layers(self, audio: np.ndarray) -> tuple[np.ndarray, int, float]:
        """Returns (n_layers, hidden) stacked pooled vectors, plus chunk bookkeeping."""
        chunks, real_lens = self._split_chunks(audio)
        if not chunks:
            raise ValueError("audio shorter than one chunk")
        inputs = self.processor(
            chunks, sampling_rate=self.target_sr, return_tensors="pt", padding=True
        )
        input_values = inputs["input_values"].to(self.device)
        _, t_in = input_values.shape

        sample_lens = torch.tensor(real_lens, dtype=torch.long, device=self.device)
        sample_idx = torch.arange(t_in, device=self.device).unsqueeze(0)
        sample_mask = (sample_idx < sample_lens.unsqueeze(1)).long()

        out = self.model(
            input_values=input_values, attention_mask=sample_mask, output_hidden_states=True
        )

        vecs = []
        for hidden in out.hidden_states:
            t_feat = hidden.shape[1]
            feat_lens = self._feature_lengths(sample_lens, t_feat)
            feat_idx = torch.arange(t_feat, device=self.device).unsqueeze(0)
            feat_mask = (feat_idx < feat_lens.unsqueeze(1)).float().unsqueeze(-1)
            denom = feat_mask.sum(dim=1).clamp(min=1.0)
            per_chunk = (hidden * feat_mask).sum(dim=1) / denom
            vecs.append(per_chunk.mean(dim=0))

        stacked = torch.stack(vecs).float().cpu().numpy()
        return stacked, len(chunks), sum(real_lens) / self.target_sr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="extract-variants")
    parser.add_argument("--out", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N themes (smoke run)"
    )
    parser.add_argument("--skip-mfcc", action="store_true")
    parser.add_argument(
        "--skip-mert",
        action="store_true",
        help="only build the librosa baseline — it needs no GPU, so it can run while a "
        "CUDA install is still downloading",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import librosa

    cfg = Config()
    frozen = pd.read_parquet(cfg.dataset_parquet)
    if args.limit:
        frozen = frozen.head(args.limit).copy()
    log.info("target corpus: %d themes from %s", len(frozen), cfg.dataset_parquet)

    if args.skip_mert and args.skip_mfcc:
        log.error("--skip-mert and --skip-mfcc together leave nothing to compute")
        return 1

    embedder: _AllLayerEmbedder | None = None
    target_sr = cfg.target_sr
    if not args.skip_mert:
        embedder = _AllLayerEmbedder(cfg)
        target_sr = embedder.target_sr
        log.info("device=%s hidden=%d", embedder.device, embedder.hidden_size)
    else:
        log.info("--skip-mert: building the librosa baseline only, on CPU")

    layer_vecs: dict[int, dict[int, list[float]]] = {}
    mfcc_vecs: dict[int, list[float]] = {}
    missing: list[int] = []
    failed: list[tuple[int, str]] = []

    for row in tqdm(frozen.itertuples(index=False), total=len(frozen), desc="variants"):
        theme_id = int(row.theme_id)
        ogg = cfg.audio_dir / str(row.audio_basename)
        if not ogg.exists():
            missing.append(theme_id)
            continue
        try:
            audio, _ = librosa.load(str(ogg), sr=target_sr, mono=True)
            if audio.size == 0:
                raise ValueError("decoded audio is empty")
            audio = audio.astype(np.float32)
            if embedder is not None:
                stacked, _, _ = embedder.embed_all_layers(audio)
                for li in range(stacked.shape[0]):
                    layer_vecs.setdefault(li, {})[theme_id] = stacked[li].tolist()
            if not args.skip_mfcc:
                mfcc_vecs[theme_id] = _librosa_features(audio, target_sr).tolist()
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the sweep
            failed.append((theme_id, f"{type(exc).__name__}: {exc}"))

    if missing:
        log.error(
            "%d/%d themes have no audio on disk (e.g. %s). Re-run the audio stage with "
            "KEEP_AUDIO=true before this script.",
            len(missing), len(frozen), missing[:5],
        )
    for theme_id, err in failed[:10]:
        log.warning("theme %d failed: %s", theme_id, err)

    # Only themes that produced every variant, so all datasets share one corpus.
    complete = set(frozen["theme_id"].astype(int))
    for per_theme in layer_vecs.values():
        complete &= set(per_theme)
    if not args.skip_mfcc:
        complete &= set(mfcc_vecs)
    if not complete:
        log.error("no theme produced a complete set of variants; nothing written")
        return 1
    if len(complete) < len(frozen):
        log.warning(
            "corpus shrank from %d to %d themes; variants are mutually comparable but not "
            "comparable to benchmarks/*.json",
            len(frozen), len(complete),
        )

    args.out.mkdir(parents=True, exist_ok=True)
    base = frozen[frozen["theme_id"].astype(int).isin(complete)].reset_index(drop=True)

    written: list[str] = []
    variants: list[tuple[str, dict[int, list[float]], str]] = [
        (f"layer{li:02d}", vecs, f"{cfg.embed_model}#layer{li}") for li, vecs in sorted(layer_vecs.items())
    ]
    if not args.skip_mfcc:
        variants.append((MFCC_VARIANT, mfcc_vecs, "librosa-mfcc-chroma-contrast"))

    for name, vecs, model_tag in variants:
        out = base.copy()
        out["embedding"] = [vecs[int(t)] for t in out["theme_id"]]
        out["embed_model"] = model_tag
        out["embed_version"] = f"{cfg.embed_version}-{name}"
        path = args.out / f"{name}.parquet"
        out.to_parquet(path, index=False)
        written.append(str(path))

    log.info("wrote %d variant datasets to %s (%d themes each)", len(written), args.out, len(base))
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
