# MultiLexNorm 2026 — ByT5 Final Submission

Lexical normalization for the [MultiLexNorm 2026 shared task](https://noisy-text.github.io/2026/multi-lexnorm.html):
mapping non-standard tokens (misspellings, abbreviations, informal forms) in noisy
social-media text to their standard forms across **17 languages**.

**Approach.** A single [`google/byt5-base`](https://huggingface.co/google/byt5-base)
model fine-tuned on all 17 languages jointly (no per-language modules), followed by a
deterministic **copy-fallback** post-processing layer that vetoes risky edits on
high-LAI languages.

**Best dev-phase score (CodaBench):** **51.90** weighted ERR (Error Reduction Rate).

---

## Repository contents

| File | Description |
|------|-------------|
| `final.ipynb` | **Main entry point.** Training, inference, copy-fallback, and submission-zip generation (5 cells, run top to bottom). |
| `requirements.txt` | Python dependencies. |
| `outputs/submission_dev_byt5_final_fb.zip` | Final dev-pub test predictions (`predictions.json`). **Tracked with Git LFS** — see note below. |
| `utils.py` | Legacy MFR-baseline helpers. **Not used by `final.ipynb`** (kept for reference). |
| `.gitattributes` | Git LFS rules for binary artifacts (`*.zip`, model formats, etc.). |

---

## Environment setup

Developed and verified on **Python 3.12**. A CUDA or Apple-Silicon (MPS) GPU is
recommended for training; the code auto-detects CUDA → MPS → CPU.

```bash
# Python 3.12 recommended
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> The notebook (`.ipynb`) is run with Jupyter / VS Code. If you only have the bare
> environment above, also install a kernel: `pip install notebook` (or `ipykernel`).

### Hugging Face authentication (required)

The dataset `weerayut/multilexnorm2026-dev-pub` is **gated**. First, open the dataset page
while signed in and **accept the access terms**. Then authenticate locally (any one):

```bash
export HF_TOKEN=hf_xxxxxxxx     # works with every huggingface_hub version (recommended)
# or, interactively:
hf auth login                   # huggingface_hub >= 1.0
# (older huggingface_hub used `huggingface-cli login`, now deprecated)
```

---

## Data

```python
from datasets import load_dataset

data = load_dataset("weerayut/multilexnorm2026-dev-pub")
# splits: train (39,178) / validation (8,408) / test (5,972)
# columns: raw (list[str]), norm (list[str]), lang (str)  — 17 languages
```

---

## How to run

Open `final.ipynb` and run the cells in order:

1. **Cell 1 — Training utilities.** Defines `train_byt5(...)`: flattens sentences to
   word pairs, tokenizes as `"{lang}: {raw}" -> norm`, and fine-tunes ByT5 with
   `Seq2SeqTrainer`.
2. **Cell 2 — Final-phase training.** Fine-tunes `google/byt5-base` on **train + validation
   combined** (UFAL 2021 recipe), `epochs=3`, `batch_size=64`, `lr=5e-4`,
   `max_input_len = max_target_len = 64`. Saves to `./checkpoints/byt5-base-final-3ep/final/`.
   *(Skipped automatically if the checkpoint already exists.)*
3. **Cell 3 — Inference.** Loads the final checkpoint and defines `byt5_predict_words(...)`
   for batched word-level prediction.
4. **Cell 4 — Copy-fallback.** Builds a per-language `raw → {norm: count}` table from
   train + val and defines `apply_fallback(...)`. For high-LAI languages
   (`ko, ja, th, sr, en, hr, sl, vi`) with tuned thresholds, a predicted edit is reverted
   to the original token when the original is normalized to itself frequently enough.
5. **Cell 5 — Submission.** Predicts on the dev-pub test split, applies the fallback,
   and writes `outputs/submission_dev_byt5_final_fb.zip` (containing `predictions.json`).

### Submission format

`predictions.json` is a list of records with `raw`, `pred`, `norm`, `lang`
(JSON `orient="records"`). The zip is what you upload to the
[CodaBench leaderboard](https://www.codabench.org/competitions/14162/).

---

## Note on Git LFS

`outputs/*.zip` is stored via **Git LFS**. A plain `git clone` (or GitHub "Download ZIP")
without Git LFS fetches only a ~131-byte pointer, **not** the real predictions file. To
retrieve the actual artifact:

```bash
git lfs install
git lfs pull
```

Alternatively, just re-run `final.ipynb` Cell 5 to regenerate the submission zip.

---

## Reproducibility notes

- Pin Python to **3.12** (older `numpy`/`pandas`/`torch` wheels may be unavailable on 3.13+).
- `requirements.txt` uses lower-bound (`>=`) versions; for an exact-match environment,
  freeze the installed versions with `pip freeze`.
- Training is seeded (`seed=42`); GPU non-determinism may still cause small variation.
