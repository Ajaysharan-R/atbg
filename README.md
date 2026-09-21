# ATBG — Adaptive Temporal Behaviour Graph

Full codebase for the social-engineering conversation risk system: ingestion
pipeline, ATBG model (BGE-M3 → Time2Vec → AVQ-BS → HGT → GRU → TD-TO →
survival/risk heads), FastAPI backend, dataset tooling, training scripts,
evaluation, and a lightweight dashboard.

## What's real vs. what needs you to run it

Every file here is real, runnable code — nothing is a placeholder function
that just returns fake data. What it can't do by itself:

- **No datasets are downloaded.** This was built in a sandbox with no
  internet access. Run `dataset_tools/download_public_datasets.py` on your
  own machine.
- **No model has been trained.** The API runs fine with an *untrained*
  checkpoint (random weights) so you can verify the pipeline end-to-end —
  its `model_version` field will say `"untrained"` and its predictions
  will be meaningless until you actually train it (see below).
- **Dependencies aren't installed or verified to import cleanly** — this
  environment couldn't `pip install` anything either. Double-check
  `requirements.txt` installs cleanly on your laptop before relying on it.

## Quick start (dev loop, no training yet)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env

docker compose up -d postgres     # starts Postgres with the schema auto-applied
uvicorn api.main:app --reload     # starts the API on http://localhost:8000
```

Then open `dashboard/index.html` directly in a browser (no build step —
it's a single static file that calls `http://localhost:8000`).

Check `http://localhost:8000/health` → `{"status": "ok"}` and you're wired up.

## Full path from here to a trained system

### 1. Get data (you + your friend)
```bash
python -m dataset_tools.download_public_datasets --all
```
This pulls the HuggingFace sources (BothBosu/*, DailyDialog, MultiWOZ).
Sources without a clean HF release (Nazario, SpamAssassin, UCI SMS, 419
corpora, COVA) print manual-download instructions — follow those.

**Before trusting any of this**, inspect one real downloaded row per source
and fix the field-mapping in `dataset_tools/unify_schema.py` —
`convert_scam_dialogue()` and `convert_daily_dialog()` are implemented as a
template; the exact field names in each dataset need verifying against
what actually downloads (dataset cards sometimes drift from what's live).
Add a `convert_*` function per new source you bring in.

### 2. Unify → weak-label → dedupe/split
```bash
python -m dataset_tools.unify_schema
python -m dataset_tools.weak_labeler
python -m dataset_tools.dedupe_and_split
```
Produces `dataset/splits/{train,validation,test}.jsonl` in the schema every
other part of the codebase expects.

### 3. Train (needs a GPU — Kaggle free T4 or your own)
```bash
python -m training.train_stage_a --data dataset/splits/train.jsonl --epochs 5
python -m training.train_stage_b --data dataset/splits/train.jsonl --init checkpoints/stage_a.pt --epochs 10
python -m training.train_stage_c --data dataset/splits/train.jsonl --init checkpoints/stage_b.pt --epochs 5
```
Stage C's output (`checkpoints/atbg_latest.pt`) is what the API loads by
default. On Kaggle: upload this repo as a Dataset, run these three scripts
in a GPU notebook, download the checkpoints back into `checkpoints/`.

### 4. Evaluate
```bash
python -m evaluation.run_eval --checkpoint checkpoints/atbg_latest.pt --data dataset/splits/test.jsonl
```

### 5. Real data (parallel/later track)
Drop cleaned real conversations (same JSON schema) into `dataset/real/`.
When ready, re-run steps 2-4 against real validation/test data instead of
the public-only dev split — the spec requires real (not synthetic)
validation/test for any numbers you report.

## What's intentionally simplified vs. the full spec

Being upfront about the corners cut to keep this buildable solo, on a
laptop, without extra infra:

- **No Neo4j.** The heterogeneous conversation graph is built on-the-fly in
  Python (`atbg_model/hgt.py: build_conversation_graph`) rather than
  persisted in a graph database — Postgres is the single source of truth,
  graphs are cheap to rebuild per-conversation. Add Neo4j later if you
  need cross-conversation graph queries; nothing here blocks that.
- **HGT is a custom lightweight implementation**, not built on
  `torch_geometric` — avoids a notoriously fiddly dependency to install.
  It implements the same idea (multi-head attention with per-edge-type
  learned bias) but hasn't been benchmarked against a PyG reference.
- **PGExplainer (4th explainability method) isn't implemented** — trajectory
  explanation, Integrated Gradients, and counterfactual search are.
  PGExplainer needs a trained-edge-importance model of its own; add it once
  the core pipeline is proven out.
- **ASR speaker diarization is a silence-gap heuristic**, not real
  diarization (WhisperX + pyannote). Good enough to unblock the pipeline;
  upgrade if voice-call accuracy matters for your final numbers.
- **Only 2 of the spec's 8 baselines are implemented** (Keyword/Regex,
  TF-IDF+LR) — the fast ones. XLM-R/BiLSTM/Transformer/HMM/Static-GNN/LLM
  baselines each need their own training loop; add them before your final
  evaluation write-up, not before you have a working system.

## Repo layout
```
api/                 FastAPI app, DB models, redaction/artifact/threat-intel services
atbg_model/           The ATBG model itself — every component in its own file
ingestion/            OCR, ASR, and the pipeline that ties redaction+extraction+embedding together
training/              Stage A/B/C training scripts + the Dataset class
evaluation/            Metrics + baselines + eval runner
dataset_tools/         Download → unify → weak-label → dedupe/split
dashboard/              Single-file HTML dashboard, no build step
dataset/                raw/ (downloaded), processed/, splits/, real/ (your friend's track)
scripts/init_db.sql     Postgres schema, auto-applied by docker-compose
```
