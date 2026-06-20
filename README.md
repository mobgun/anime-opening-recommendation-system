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
