# Anime OP/ED Recommendation System

A dataset pipeline and content-based recommender for anime opening/ending themes.
It joins **MyAnimeList** metadata (via [Jikan](https://jikan.moe/)) with
**[AnimeThemes](https://animethemes.moe/)** audio, computes
[MERT](https://huggingface.co/m-a-p/MERT-v1-95M) audio embeddings, and recommends
songs that *sound* similar to a seed theme — with tunable bias toward the same
artist/era and control over how adventurous the results are.

It also measures whether that works, against two independent ground truths the audio
never sees, with random baselines and paired significance tests
([Evaluation](#evaluation)). On the reference corpus, audio-only ranking puts a song by
one of the seed's own artists at **rank 1 for 22% of seeds — 53x what chance gives** —
and the two ground truths disagree about one of the changes that produced it, which the
README says out loud rather than rounding off.

## How it works

The pipeline runs in five stages (see [`src/pipeline.py`](src/pipeline.py)):

| Stage    | What it does                                                        | Output                          |
|----------|--------------------------------------------------------------------|---------------------------------|
| `themes` | Fetch OP/ED themes from the AnimeThemes API                        | `data/raw/animethemes.parquet`  |
| `mal`    | Fetch MyAnimeList metadata via Jikan                               | `data/raw/mal.parquet`          |
| `audio`  | Download `.ogg` audio for each theme                               | `data/audio/*.ogg`              |
| `embed`  | Compute MERT embeddings per theme (layer `EMBED_LAYER`, default 6) | `data/interim/embeddings/*.parquet` |
| `join`   | Inner-join themes ⨝ MAL ⨝ embeddings, drop errored themes         | `data/processed/dataset.parquet` |

The recommender ([`src/recommend.py`](src/recommend.py)) loads `dataset.parquet`,
whitens the embedding space, ranks by cosine similarity, and blends in metadata signals
(artist, era, genre). Both the whitening and the choice of MERT layer came out of
measurements, not intuition — [`src/evaluate.py`](src/evaluate.py) scores the ranking
against metadata the embeddings never saw, and is how you would check any further change.

See [Example output](#example-output) for what it returns and [Evaluation](#evaluation)
for how well it does.

## Installation

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -e .
```

Requires Python 3.10+.

**Torch is an optional extra.** The core install has no `torch`, because only the
`embed` stage needs one — the recommender, the evaluator and the other four stages
run on numpy and pandas. So `pip install -e .` is enough to rank and score a prebuilt
`dataset.parquet` without pulling a ~2.6 GB wheel. To build embeddings yourself:

```bash
pip install -e ".[embed]"   # transformers, torch, torchaudio, librosa, soundfile
pip install -e ".[dev]"     # pytest, ruff
```

Running `--stage embed` without the extra fails with a message telling you this,
rather than a `ModuleNotFoundError` traceback.

**GPU:** `pip install torch` resolves to a **CPU-only** wheel on Windows, so the command
above gives you a CPU build even on a machine with a working CUDA card — on the hardware
this project was built on that turned a 5-minute embed stage into a 100-minute one.
For CUDA, install torch from PyTorch's own index instead:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Pick the `cuXXX` matching your driver (`nvidia-smi`); cu126 and cu130 both cover Ada
(RTX 40-series). The check must print `True` — if it prints a version ending in `+cpu`,
the CUDA wheel did not install. Should pip stall on the ~2.6 GB download (it did here,
repeatedly, on an otherwise healthy connection), fetch the wheel directly and install
from the file:

```bash
curl -L -C - --retry 10 --speed-time 60 --speed-limit 10000 \
  -o torch-cu126.whl \
  "https://download.pytorch.org/whl/cu126/torch-2.14.0%2Bcu126-cp312-cp312-win_amd64.whl"
pip install --no-deps --force-reinstall torch-cu126.whl
```

The `embed` stage runs on CPU without complaint; it is just ~18x slower.

## Usage

### Build the dataset

```bash
# Full run (all stages)
animethemes-pipeline run --stage all

# Or run a single stage
animethemes-pipeline run --stage themes

# Quick smoke run: only themes for the top-N most popular anime
animethemes-pipeline run --stage all --top-n 100
```

Useful flags: `--limit-pages N` (limit AnimeThemes pagination for tests),
`--top-n N` / `--top-by {popularity,score}`, `--keep-audio` (keep `.ogg` files
after embedding), `--force-reembed`.

### Get recommendations

```bash
animethemes-recommend --seed-theme-id 2279 --mode discovery --k 20
```

Modes (`src/recommend.py`) control how far from the seed the results may roam:

- `purist` — closest matches, minimal metadata influence
- `discovery` — balanced (default)
- `comfort` — strongly biased toward the same artist/era

Other flags: `--exclude-same-anime`, `--max-per-anime N`, `--smoke` (run against
a few built-in seed themes).

## Example output

Real output from the reference corpus described under [Evaluation](#evaluation).

**Death Note OP2 — "What's up people?" (Maximum the Hormone)**, `theme_id=2280`:

```bash
animethemes-recommend --seed-theme-id 2280 --mode purist --k 5 --exclude-same-anime
```

| # | Anime | Theme | Song | Artist | Year | `cos` |
|---|-------|-------|------|--------|------|-------|
| 1 | Chainsaw Man | ED3 | Hawatari 2-Oku Centi | **Maximum the Hormone** | 2022 | 0.117 |
| 2 | Naruto: Shippuuden | ED20 | By My Side | — | 2007 | 0.112 |
| 3 | Naruto: Shippuuden | OP17 | Kaze | Yamazaru | 2007 | 0.106 |
| 4 | Fairy Tail | OP14 | Fairy Tail: Yakusoku no Hi | — | 2009 | 0.106 |
| 5 | Fairy Tail | ED6 | Be As One | — | 2009 | 0.098 |

Rank 1 is the same band, sixteen years and one unrelated series later. The ranking
never saw the artist column — `purist` scores audio only, and that audio was at most
90 seconds per theme. This is exactly what [`artist_recall@1`](#evaluation) counts,
and it happens for 22% of the seeds that have such a partner.

A second seed, same show, different sound. **Death Note OP1 — "THE WORLD"
(Nightmare)**, `theme_id=2279`:

| # | Anime | Theme | Song | Artist | Year | `cos` |
|---|-------|-------|------|--------|------|-------|
| 1 | Bleach | ED26 | Song for... | ROOKiEZ is PUNK'D | 2004 | 0.163 |
| 2 | One Piece | OP12 | Kaze wo Sagashite | — | 1999 | 0.148 |
| 3 | Fairy Tail | OP8 | The Rock City Boy | — | 2009 | 0.119 |
| 4 | One Piece | ED8 | Shining ray | — | 1999 | 0.114 |
| 5 | Ao no Exorcist | OP2 | IN MY WORLD | ROOKiEZ is PUNK'D | 2011 | 0.113 |

Five guitar-driven rock tracks with a male lead, and ranks 1 and 5 are the same band
seven years apart — picked out of 617 candidates without ever consulting a credit.

The default `discovery` mode folds artist/genre/era back into the score, pulling
results toward the seed's own neighbourhood; cap franchise pile-ups with
`--max-per-anime 1`.

Notes on reading the tables: the CLI also prints `theme_id`, `mal_id`, `audio_id`,
`anime_score`, `l2_weight` and `final`, dropped here for width. `—` means AnimeThemes
has no artist linked to that song. `cos` is cosine in the whitened embedding space,
where two unrelated themes score ~0.00 — so unlike raw MERT cosine (which starts at
0.92 for *everything*) the number itself carries information.

## Evaluation

`animethemes-evaluate` scores the recommender against metadata it never saw: the
embeddings come from audio only, while genre, artist and air date come from
MyAnimeList. Every metric is printed next to what a *random* ranking scores on the
same candidate pool, because that pool is lopsided — two thirds of random theme pairs
already share a genre — so the lift is the part that carries information.

```bash
animethemes-evaluate --json benchmarks/layer6-whitened.json
```

**Artist agreement** is the metric with teeth: does the ranking surface another song
by one of the seed's own artists? Ground truth is hard, and the seed's own anime is
excluded, so a hit is never the same song or a sibling OP. Scored over the 192 seeds
where such a theme exists, median candidate pool 617.

| metric | value | 95% CI | random | lift |
|--------|-------|--------|--------|------|
| `artist_recall@1` | **0.219** | [0.167, 0.276] | 0.004 | **53.0x** |
| `artist_recall@3` | 0.344 | [0.281, 0.417] | 0.012 | 27.9x |
| `artist_recall@5` | 0.406 | [0.344, 0.479] | 0.020 | 19.9x |
| `artist_recall@10` | 0.479 | [0.411, 0.552] | 0.040 | 11.9x |
| `artist_recall@20` | 0.615 | [0.547, 0.682] | 0.079 | 7.8x |
| `artist_recall@100` | 0.802 | [0.740, 0.859] | 0.328 | 2.5x |
| `artist_mrr` | **0.311** | [0.259, 0.371] | 0.023 | 13.8x |
| median rank of first hit | **12** | — | 217 | — |

### How it got there

Two changes, neither of them to the model, the audio or the data — both to how the
embeddings are *read*. Read the [second ground truth](#what-the-two-ground-truths-disagree-about)
below before taking this table as a quality improvement: one of the two changes moves
this metric and not the other one. Every frozen run is in [`benchmarks/`](benchmarks/), all on the
same 621 themes (`corpus_sha256 c6d1d706ed8b9390`).

| configuration | MRR | recall@1 | median rank |
|---------------|-----|----------|-------------|
| MERT last layer, raw cosine — as originally written | 0.128 | 0.078 | 62 |
| librosa MFCC/chroma/contrast baseline, whitened | 0.172 | 0.115 | 46 |
| MERT last layer + whitening | 0.229 | 0.146 | 23 |
| **MERT layer 6 + whitening** | **0.311** | **0.219** | **12** |

**Whitening.** Raw MERT vectors are strongly anisotropic — mean cosine between two
unrelated themes is 0.918, so almost all of the vector length points in directions
every song shares and what distinguishes songs survives only in the tail. Project onto
the top 256 principal components, divide by their singular values, renormalize; mean
off-diagonal cosine drops to −0.002. `RECOMMEND_WHITEN_COMPONENTS` in
[`src/config.py`](src/config.py), 0 to disable.

**Layer choice.** The pipeline originally pooled MERT's final hidden state, which a
sweep of all 13 layers found to be the **worst** of them: the top of the stack
specializes toward the pre-training objective while musical style sits in the middle.
Layers 4-10 all score 0.30-0.32 on held-out seeds against 0.21 for the last layer — a
plateau, not a spike. `EMBED_LAYER=6` is the default; every layer falls out of one
forward pass, so the sweep cost one embed run, not thirteen
([`experiments/sweep_layers.py`](experiments/sweep_layers.py)).

**How both were validated.** Each change was selected on half the seeds and scored on
the other half, split by *anime* so no show straddles the split, then compared **paired
per seed** — comparing two overlapping confidence intervals is the wrong test when both
systems score the same seeds.

| change | held-out paired ΔMRR | 95% CI |
|--------|---------------------|--------|
| whitening | +0.085 | [+0.034, +0.137] |
| layer 12 → 6 | +0.105 | [+0.036, +0.173] |
| MERT vs librosa baseline (both whitened) | +0.057 | [+0.010, +0.106] |

**A second ground truth, and why it matters.** Artist agreement has a known weakness:
two songs by one performer share a voice, a band and a mastering chain, so a system can
score well by recognising the *singer* rather than the music. In MIR that effect is
normally treated as a confound to filter out — we made it the target. The evidence that
this is not paranoia: a plain MFCC vector (the classic **speaker-identification** feature
set) reaches 75% of MERT's artist score.

So the corpus's artists were resolved to MusicBrainz ids and joined with ListenBrainz's
listener-derived artist similarity — which artists actually turn up together in real
listening sessions. `related_*` asks whether the audio ranking surfaces a theme by a
**different** artist that listeners treat as adjacent. Different singer, different
production, so the confound cannot carry it.

| metric | value | 95% CI | random | lift |
|--------|-------|--------|--------|------|
| `related_recall@10` | 0.283 | [0.231, 0.339] | 0.164 | 1.7x |
| `related_recall@20` | 0.450 | [0.394, 0.508] | 0.296 | 1.5x |
| `related_mrr` | 0.121 | [0.098, 0.146] | 0.072 | 1.7x |

Scored over 307 seeds — more than the 192 artist agreement reaches. Relevance is the top
3 listener-derived neighbours per artist; the full list of 100 marks 15% of the corpus as
relevant, which leaves a random top-10 scoring 0.80 and no headroom to measure anything.
The ground truth is pinned in [`benchmarks/`](benchmarks/) so the metric reproduces
without re-querying either service.

### What the two ground truths disagree about

This is the uncomfortable part, and it is the reason both are now reported.

| configuration | `artist_mrr` (n=192) | `related_mrr` (n=307) |
|---------------|----------------------|------------------------|
| layer 12, raw — as originally written | 0.128 | 0.123 |
| layer 12 + whitening | 0.229 | **0.093** |
| **layer 6 + whitening — current** | **0.311** | 0.121 |
| layer 6, raw | 0.228 | 0.128 |

Paired against the original system on the behavioural metric: layer 12 + whitening is
**significantly worse** (−0.030, CI [−0.055, −0.008]); the current configuration is
**−0.002, CI [−0.031, +0.028] — not significant**. In other words the 2.4x reported above
belongs to the metric that rewards recognising a performer. On listener-derived
relatedness, none of it is demonstrably better than where the pipeline started.

The mechanism is measurable: whitening more than doubles how many same-artist tracks sit
in a top-10 (0.129 → 0.242 per seed, 0.291 at layer 6), and those crowd out the
different-artist neighbours the behavioural metric counts.

**Why whitening is still on.** At layer 6 specifically, the direct paired test says
whitening gains **+0.084 (CI [+0.036, +0.132], significant)** on artist agreement and
costs **−0.007 (CI [−0.034, +0.022], not significant)** on relatedness. The significant
harm was at layer 12; the better layer largely dissolves it. Trading a measured gain for
an undetectable one would be the wrong call — but note that "not significant" here means
a real cost up to ~0.034 could hide inside that interval. `RECOMMEND_WHITEN_COMPONENTS=0`
turns it off, and layer 6 raw is the configuration to use if relatedness matters more
than performer identity for your use.

**What the behavioural metric says when it gets to choose.** The layer was originally
selected on artist agreement alone, which is precisely the objection this section
raises. So the sweep now runs the selection separately under each ground truth
(`--family both`), and the answer is reassuring for the layer if not for the effect
size: given its own tune half and no knowledge of the artist metric, the behavioural
family **also selects layer 6** (tune MRR 0.126, the best of all 13).

| ground truth | selects | holdout ΔMRR vs layer 12 | 95% CI | verdict |
|--------------|---------|--------------------------|--------|---------|
| artist (n=105 holdout) | layer06 | +0.105 | [+0.036, +0.173] | significant |
| behavioural (n=163 holdout) | layer06 | +0.017 | [−0.018, +0.053] | not significant |

So the two ground truths agree on *which* layer, and disagree only on whether the
improvement is demonstrable — the behavioural metric points the same way but is too
noisy to prove it. That is a materially weaker claim than "significant on both", and a
materially stronger one than "chosen against a metric that measures the confound".
Whitening remains the change where the disagreement is real.

**The general lesson.** Two of the three changes in this project were chosen against a
single metric, and the second ground truth shows that one of them bought nothing outside
it. That is why both families are printed on every run: not because the behavioural one is
truer — it is noisier, its lift is 1.7x against artist agreement's 12x, and ListenBrainz's
listener base skews away from Japanese music — but because a single number is exactly how
a recommender drifts into being a voice detector without anyone noticing.

**Negative controls.** Genre and era agreement barely move (0.708 vs 0.684 random;
0.187 vs 0.157) and that is expected rather than broken — MERT hears timbre, not
whether a show is shelved under Action. Their baselines sit near the ceiling, so they
cannot discriminate; they are printed to catch regressions, not to be optimized.

**Reference corpus.** `animethemes-pipeline run --stage all --top-n 100` → 621 themes
across 98 anime, `m-a-p/MERT-v1-95M` layer 6, up to 3x30 s chunks @ 24 kHz. Evaluation
is leave-one-out over every theme with the seed's own anime excluded. Random baselines
are closed-form per seed (hypergeometric for recall, the exact first-hit distribution
for MRR), not sampled; intervals are 1000-resample bootstraps over seeds. Because
`--top-n 100` ranks by *current* MAL popularity it is a moving target, so the exact
theme ids are pinned in [`benchmarks/corpus-621.txt`](benchmarks/corpus-621.txt).

**Why `purist` is the default here.** `discovery` and `comfort` add artist and genre
terms *into the score*, so scoring them on artist and genre is circular —
`--mode discovery` reports `artist_recall@10 = 0.99`, which measures the weighting, not
the audio.

**What this still does not establish.** Aggregates move; individual seeds do not follow.
The layer change improved 95 seeds and hurt 75, and one recommendation this README used
to showcase (YUI's "again" surfacing her "Rolling Star") fell from rank 4 to rank 24 —
better on average is not better everywhere. 621 themes from 98 anime is also small and
popularity-skewed, one franchise (One Piece, 67 themes) being a tenth of it. And
"sounds similar" remains a listening judgement that no metadata proxy settles: artist
agreement is a proxy for musical identity, not for whether a human would enjoy the
result.

## Experiments

Every number in [Evaluation](#evaluation) came from a script in
[`experiments/`](experiments/), kept in the repo so the claims can be re-derived rather
than believed. They are one-off analyses, not part of the pipeline.

| script | what it answers |
|--------|-----------------|
| [`extract_variants.py`](experiments/extract_variants.py) | Builds every candidate embedding in one pass over the audio: all 13 MERT layers (they fall out of a single forward pass) plus a librosa MFCC/chroma/contrast baseline. `--skip-mert` builds only the baseline, on CPU. |
| [`sweep_layers.py`](experiments/sweep_layers.py) | Which layer to pool. Selects on half the anime, reports on the other half, compares paired per seed. `--family {artist,behavioural,both}` picks which ground truth the *selection* optimizes; `both` is the default and states plainly when they disagree. |
| [`resolve_artists.py`](experiments/resolve_artists.py) | Resolves the corpus's 217 artists to MusicBrainz ids (95% hit rate). Romanized names match through MusicBrainz aliases — which is why this works at artist level and fails at track level. Cached and resumable; honours the 1 req/s limit. |
| [`fetch_similar_artists.py`](experiments/fetch_similar_artists.py) | Pulls listener-derived artist similarity from ListenBrainz, the ground truth behind `related_*`. |

```bash
python -m experiments.extract_variants --skip-mfcc     # needs audio on disk (KEEP_AUDIO=true)
python -m experiments.sweep_layers                     # both ground truths
python -m experiments.sweep_layers --family behavioural
python -m experiments.resolve_artists && python -m experiments.fetch_similar_artists
```

To compare any two configurations the way this project's decisions were made — paired
per seed, on **both** ground truths at once:

```bash
animethemes-evaluate --compare-to data/experiments/layer12.parquet --no-embed-check
```

It prints a warning when the two ground truths disagree in sign, which is the case this
project ran into and the reason the comparison covers both.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

116 tests, no network, no GPU, no audio — everything runs on synthetic corpora built
so the right answer is known in advance (planted near-duplicate pairs, planted artist
partners), plus one suite that re-derives the real numbers.

What they actually defend:

| suite | what would otherwise break silently |
|-------|--------------------------------------|
| [`test_random_baselines.py`](tests/test_random_baselines.py) | Every "53x chance" in this README is a ratio, and these functions are its denominator. They are checked against an **independently derived** closed form (hypergeometric via `math.comb`, and `P(R=r) = C(n-r, m-1)/C(n, m)` for MRR) rather than a restatement of the same expression, then against 200k-trial simulation. A bug here would scale every lift in the README by the same factor and nothing else would notice. |
| [`test_whitening.py`](tests/test_whitening.py) | The geometric claim whitening rests on — that mean off-diagonal cosine collapses from >0.9 to ~0 — asserted directly on a corpus built with the MERT pathology. |
| [`test_recommender.py`](tests/test_recommender.py) | Ranking order, dedup, `--max-per-anime`, mode scaling, and that `metadata_weights` stays normalized into [0, 1] (otherwise `final = cos + scale * alpha * w` means nothing). |
| [`test_evaluate.py`](tests/test_evaluate.py) | Corpora with a known correct metric value: a planted partner ranked first must score `artist_mrr = 1.0`, ranked last must score `1/7`. Also the sign and pairing of `paired_mrr_delta`. |
| [`test_embed_store.py`](tests/test_embed_store.py) | The staleness guard — the thing that stops a dataset silently mixing vectors from two different models, an error that produces plausible recommendations and no warning. |
| [`test_console_output.py`](tests/test_console_output.py) | Regression test for a real crash: AnimeThemes titles contain `〜`, `★`, `√`, `♡`, and on a Windows console codepage printing one raised `UnicodeEncodeError` and killed the CLI on 7 of the 621 themes. |
| [`test_benchmark_regression.py`](tests/test_benchmark_regression.py) | Re-scores the live code against [`benchmarks/layer6-whitened.json`](benchmarks/) and asserts **every frozen metric and every per-seed rank**. This is the one that would catch a change leaving all the unit tests green while moving `artist_mrr` by 0.05. Self-skips without `data/processed/dataset.parquet`, and skips with a reason if `corpus_sha256` has drifted, because metrics across different corpora are not comparable. |

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint and tests on
Linux and Windows, Python 3.10 and 3.12. Windows is in the matrix deliberately: the
console bug above is invisible on Linux.

## Configuration

All settings live in [`src/config.py`](src/config.py) and can be overridden via
environment variables (optionally through a `.env` file — see
[`.env.example`](.env.example)). **No secrets are required**: every service this project
touches — AnimeThemes, Jikan, MusicBrainz, ListenBrainz — is public and needs no key.

The knobs that change results the most, both measured in [Evaluation](#evaluation):

| variable | default | effect |
|----------|---------|--------|
| `EMBED_LAYER` | `6` | Which MERT layer to pool. `-1` (the final layer) measured worst of all 13. Changing it invalidates the embedding cache — rerun `embed --force-reembed`. |
| `RECOMMEND_WHITEN_COMPONENTS` | `256` | Whitening; `0` ranks on raw vectors. Buys performer recognition, costs a little relatedness — see [the disagreement](#what-the-two-ground-truths-disagree-about). Applied at load, so no re-embed needed. |

The rest — `EMBED_MODEL`, `TARGET_SR`, `CHUNK_SECONDS`, `JIKAN_RATE_PER_SEC`, the
`RECOMMEND_*` blend weights — are documented in `.env.example`.

## Data

Everything under `data/` is **regenerable from the public APIs** and is therefore
git-ignored. Run the pipeline to populate it locally.

[`benchmarks/`](benchmarks/) is the exception and *is* committed, because evaluation
results are worthless if the thing they were measured on has drifted: it holds the frozen
metric runs, the pinned list of 621 theme ids (`--top-n 100` tracks live MAL popularity,
so the corpus itself moves), and the artist ground truth resolved from MusicBrainz and
ListenBrainz — 186 KB, so `related_*` reproduces without re-querying either service.

Data sources:

- [AnimeThemes API](https://api.animethemes.moe) — theme audio & metadata
- [Jikan](https://api.jikan.moe/v4) — MyAnimeList metadata
- [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M) — audio embeddings
- [MusicBrainz](https://musicbrainz.org) — artist identity (romanized names resolve through its aliases)
- [ListenBrainz](https://listenbrainz.org) — listener-derived artist similarity

The two music-metadata services are only needed to rebuild the behavioural ground truth;
both are volunteer-run, and the scripts honour their rate limits.

## Limitations

Stated once, in one place, so they are not scattered through the numbers:

- **The corpus is small and skewed.** 621 themes from 98 anime, chosen by MAL popularity; one franchise (One Piece) is 67 of them. These are retrieval statistics on a fixed corpus, not a generalization estimate.
- **Nobody has listened.** Every number here is a metadata proxy. No human has judged a single recommendation, and the two ground truths already disagree with each other.
- **The default mode is unmeasured.** Evaluation runs in `purist`; the shipped default is `discovery`, which folds artist and genre into the score — scoring it on artist and genre would be circular, so it is simply not measured.
- **Whitening is fitted on the corpus being ranked.** It is unsupervised and sees no labels, but it is refitted at every load, so there is no persisted transform to project a new theme into an existing space. Adding one theme means refitting on everything.
- **Artist coverage caps the metrics.** AnimeThemes links no artist for 38% of themes, so `artist_*` scores 192 seeds and `related_*` scores 307, out of 621.
- **`related_*` is the weaker instrument.** Its lift over chance is ~1.7x against artist agreement's ~12x, and ListenBrainz's listener base skews away from Japanese music.

## License

Released under the [MIT License](LICENSE).
