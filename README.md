# Anime OP/ED Recommendation System

A dataset pipeline and content-based recommender for anime opening/ending themes.
It joins **MyAnimeList** metadata (via [Jikan](https://jikan.moe/)) with
**[AnimeThemes](https://animethemes.moe/)** audio, computes
[MERT](https://huggingface.co/m-a-p/MERT-v1-95M) audio embeddings, and recommends
songs that *sound* similar to a seed theme — with tunable bias toward the same
artist/era and control over how adventurous the results are.

## How it works

The pipeline runs in five stages (see [`src/pipeline.py`](src/pipeline.py)):

| Stage    | What it does                                                        | Output                          |
|----------|--------------------------------------------------------------------|---------------------------------|
| `themes` | Fetch OP/ED themes from the AnimeThemes API                        | `data/raw/animethemes.parquet`  |
| `mal`    | Fetch MyAnimeList metadata via Jikan                               | `data/raw/mal.parquet`          |
| `audio`  | Download `.ogg` audio for each theme                               | `data/audio/*.ogg`              |
| `embed`  | Compute MERT embeddings per theme                                  | `data/interim/embeddings/*.parquet` |
| `join`   | Inner-join themes ⨝ MAL ⨝ embeddings, drop errored themes         | `data/processed/dataset.parquet` |

The recommender (`src/recommend.py`) loads `dataset.parquet`, ranks by cosine
similarity over the embeddings, and blends in metadata signals (artist, era, genre).
See [Example output](#example-output) for what it returns and
[Evaluation](#evaluation) for how well it does — `src/evaluate.py` scores the
ranking against metadata the embeddings never saw.

## Installation

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
# or, to install the console scripts:
pip install -e .
```

Requires Python 3.10+. A GPU is recommended for the `embed` stage but not required.

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

Real output from the reference dataset described under [Evaluation](#evaluation).
Seed: **Death Note OP1 — "THE WORLD" (Nightmare)**, `theme_id=2279`.

```bash
animethemes-recommend --seed-theme-id 2279 --mode purist --k 4 --exclude-same-anime
```

| # | Anime | Theme | Song | Artist | Year | `cos` |
|---|-------|-------|------|--------|------|-------|
| 1 | Nanatsu no Taizai | OP1 | Netsujou no Spectrum | Ikimonogakari | 2014 | 0.966 |
| 2 | Fullmetal Alchemist: Brotherhood | ED1 | Uso | SID | 2009 | 0.966 |
| 3 | Bleach | ED26 | Song for... | ROOKiEZ is PUNK'D | 2004 | 0.966 |
| 4 | Naruto: Shippuuden | OP12 | Moshimo | — | 2007 | 0.964 |

Four band-driven J-rock tracks with a male lead and a wall-of-guitar chorus —
which is what the seed is. Nothing in the ranking knows that: `purist` uses the
embedding alone, and that embedding was built from at most 90 seconds of audio
(60 s for this seed — see `total_seconds_used` in the dataset).

Same seed in `discovery`, the default mode, which folds artist/genre/era into the
score:

```bash
animethemes-recommend --seed-theme-id 2279 --k 4 --exclude-same-anime
```

| # | Anime | Theme | Song | Artist | Year | `cos` | `final` |
|---|-------|-------|------|--------|------|-------|---------|
| 1 | Mirai Nikki (TV) | OP2 | Dead END | Faylan | 2011 | 0.939 | 1.019 |
| 2 | Mirai Nikki (TV) | OP1 | Kuusou Mesorogiwi | Yousei Teikoku | 2011 | 0.937 | 1.017 |
| 3 | Bleach | ED26 | Song for... | ROOKiEZ is PUNK'D | 2004 | 0.966 | 1.016 |
| 4 | Mirai Nikki (TV) | ED1 | Blood Teller | Faylan | 2011 | 0.929 | 1.009 |

The metadata term pulls in the Mirai Nikki cluster — one genre-neighbourhood over
from Death Note (Suspense, Supernatural) and a nearby era — while rank 3 holds its
place on audio alone. Cap the franchise pile-up with `--max-per-anime 1`.

Swap the seed and the neighbourhood swaps with it: `--seed-theme-id 3434`
(FMA:B OP1 — "again", YUI) returns Akame ga Kill! ED2 (Sora Amamiya), Shigatsu wa
Kimi no Uso OP2 (Coala Mode.), Tate no Yuusha no Nariagari ED3 (Chiai Fujikawa)
and Nanatsu no Taizai OP1 (Ikimonogakari) — four female-vocal tracks for a
female-vocal seed.

Notes on reading the tables: the CLI also prints `theme_id`, `mal_id`, `audio_id`,
`anime_score` and `l2_weight`, dropped here for width. `—` means AnimeThemes has
no artist linked to that song. And `cos` is raw cosine over MERT embeddings, which
are anisotropic — two *random* themes in this dataset already score 0.918 on
average — so read the ordering, not the absolute value.

## Evaluation

`animethemes-evaluate` scores the recommender against metadata it never saw: the
embeddings come from audio only, while genre, artist and air date come from
MyAnimeList. Every metric is printed next to what a *random* top-k scores on the
same candidate pool, because that pool is lopsided — two thirds of random theme
pairs already share a genre — so the lift is the part that carries information.

**Headline: in audio-only ranking, 22% of seeds pull a song by one of their own
artists into the top 10 — 5.4× the 4.0% a random pick gets.**

```bash
animethemes-evaluate --k 10           # add --json data/processed/eval.json to save
```

| metric | value | 95% CI | random | lift | seeds |
|--------|-------|--------|--------|------|-------|
| `artist_recall@10` | **0.219** | [0.161, 0.276] | 0.040 | **5.43×** | 192 |
| `genre_precision@10` | 0.721 | [0.699, 0.743] | 0.684 | 1.05× | 621 |
| `genre_jaccard@10` | 0.337 | [0.322, 0.352] | 0.304 | 1.11× | 621 |
| `era_precision@10` | 0.176 | [0.165, 0.189] | 0.157 | 1.12× | 614 |

- `artist_recall@10` — at least one of the 10 results is by an artist behind the
  seed song. Scored only over the 192 seeds where such a theme exists in *another*
  anime, so a hit is never the same song or a sibling OP.
- `genre_precision@10` / `genre_jaccard@10` — share of results sharing ≥1 MAL
  genre with the seed, and mean Jaccard overlap of the full genre sets.
- `era_precision@10` — share of results that aired within ±2 years of the seed.

Reference dataset: `animethemes-pipeline run --stage all --top-n 100` → **621
themes across 98 anime**, `m-a-p/MERT-v1-95M`, up to 3×30 s chunks @ 24 kHz.
Evaluation is leave-one-out over every theme, with the seed's own anime excluded
from the candidate pool. Random baselines are closed-form per seed (hypergeometric for
artist recall, pool means for the rest), not sampled; intervals are 1000-resample
bootstraps over seeds.

**Why `purist` is the default here.** `discovery` and `comfort` add artist and
genre terms *into the score*, so scoring them on artist and genre is circular —
`--mode discovery` reports `artist_recall@10 = 0.99`, which measures the weighting,
not the audio. Pass `--mode` to see it for yourself.

**What the numbers say.** Artist recall is the real result: it has hard ground
truth, and audio similarity finds a same-artist track 5.4× more often than chance.
Genre and era barely move — which is the expected answer, not a bug. MERT hears
timbre and instrumentation, not whether a show is shelved under Action; and with
68% of random pairs already sharing a genre, there is little headroom left to win.

**What they don't say.** 621 themes from 98 anime is a small, popularity-skewed
pool, and one franchise (One Piece, 67 themes) is a tenth of it. These are
retrieval statistics on a fixed corpus, not a held-out generalization estimate.
Whether two openings actually *sound* alike stays a listening judgement that no
metadata proxy settles — the metrics say the embedding is not noise, not that the
recommendations are good.

## Configuration

All settings live in [`src/config.py`](src/config.py) and can be overridden via
environment variables (optionally through a `.env` file — see
[`.env.example`](.env.example)). **No secrets are required**; the public
AnimeThemes and Jikan APIs need no API key. Common knobs include `EMBED_MODEL`,
`TARGET_SR`, `CHUNK_SECONDS`, `JIKAN_RATE_PER_SEC`, and the `RECOMMEND_*` weights.

## Data

Everything under `data/` is **regenerable from the public APIs** and is therefore
git-ignored — only the code lives in this repository. Run the pipeline to populate
it locally. Data sources:

- [AnimeThemes API](https://api.animethemes.moe) — theme audio & metadata
- [Jikan](https://api.jikan.moe/v4) — MyAnimeList metadata
- [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M) — audio embeddings

## License

Released under the [MIT License](LICENSE).
