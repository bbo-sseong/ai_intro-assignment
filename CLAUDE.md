# CLAUDE.md

Project context for Claude Code. Read this before working on the codebase.

## What this is

A student submission for the **MultiLexNorm 2026** shared task (Introduction to
Artificial Intelligence, Spring 2026, SKKU): word-level lexical normalization
of noisy text across 17 languages, evaluated by ERR (Error Reduction Rate).

- Official task: https://noisy-text.github.io/2026/multi-lexnorm.html
- Leaderboard:   https://www.codabench.org/competitions/14162/
- Dataset (gated): `weerayut/multilexnorm2026-dev-pub`
- Submission deadline: 2026-05-20 23:59 (already passed; we are in the analysis/report phase)

The graded artifact is the **report**, not just the leaderboard score — the
guideline explicitly says cross-validation will be used and methodology matters
as much as the public score.

## Repository layout

```
demo.ipynb          # The whole pipeline lives here: MFR baseline, ByT5 training, hybrid inference
utils.py            # zip_files_flat() + evaluate() helpers
requirements.txt    # See "Environment gotchas" below — pandas<2.0 is broken on py3.12
README.md           # Upstream baseline README; do not overwrite without asking
checkpoints/        # gitignored — model checkpoints (2.2 GB each)
outputs/            # gitignored except a few submission zips
```

## demo.ipynb cell map

The notebook layers three approaches; we are currently using approach (3).

| Cells   | What                                                            |
|---------|-----------------------------------------------------------------|
| 0-8     | MFR (Most-Frequent-Replacement) baseline + submission           |
| 10-16   | ByT5-small section (older one-model approach)                   |
| 17-22   | **ByT5 base hybrid** — multilingual (15 langs) + ko, ja monolingual |
| 23-27   | ByT5-large for ko/ja (attempted, abandoned — see "History")     |

Active path is cells `11 (train_byt5 def) -> 18-20 (train base x3) -> 21-22 (hybrid inference + zip)`.

`train_byt5()` in cell 11 is the single training entry point. Supports:
- `target_langs=[...]` to filter the dataset
- `oversample_weights={lang: w, ...}` for per-language balancing
- `gradient_checkpointing=True` for OOM-safe large-model training
- `early_stop_patience` (set to 999 to effectively disable)

After saving the best checkpoint to `final/`, it removes intermediate
`checkpoint-*` dirs to free disk — important on small disks.

## Environment gotchas (vast.ai instance specifics)

- Python: **`/venv/main/bin/python`** is the interpreter the notebook kernel
  uses. System `/usr/bin/python3` exists but has nothing installed. Always
  install/run with the venv path.
- `requirements.txt` pins `pandas<2.0`. Pandas 1.x has **no Python 3.12 wheels**
  and the sdist build fails on `pkg_resources`. Drop the pin or use pandas 2.x
  (the notebook code is compatible).
- Dataset is **gated**. Set HF auth with `huggingface_hub.login(token=...)` —
  token persists at `$HF_HOME/token`. `HF_HOME` is `/workspace/.hf_home` on
  this instance.
- GPU is RTX 3090 (24 GB). ByT5-base fits at batch=64. ByT5-large needs
  `batch=8, grad_accum=8, gradient_checkpointing=True` to avoid OOM.
- Disk is 61 GB total. Each ByT5-base checkpoint is ~2.2 GB; large is ~14 GB.
  The auto-cleanup in `train_byt5()` is load-bearing here.

## Background training pattern

`tmux + nbclient` over a *subset* of notebook cells. The runner lives at
`/tmp/run_subset.py` (intentionally outside the project — it's glue, not source):

```bash
tmux new-session -d -s nb_train -c /root/ai_intro-assignment \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   /venv/main/bin/python -u /tmp/run_subset.py 2>&1 | tee logs/nb_train.log"

# monitor
tail -f /root/ai_intro-assignment/logs/nb_train.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
ls /root/ai_intro-assignment/checkpoints/
```

`run_subset.py` picks specific cell indices from `demo.ipynb`. Edit `TARGETS`
inside it if you change the cell layout.

`nbclient` buffers cell stdout until the cell finishes, so a multi-hour
training cell will look silent in `tail -f`. Use `nvidia-smi` and the
checkpoints directory as liveness signals.

## History (so we don't repeat ourselves)

- Tried scaling ko/ja to **ByT5-large**. First attempt OOM'd at eval-step
  generation (batch=16, grad_accum=4 too tight for 24 GB). Second attempt
  with batch=8/grad_accum=8/grad_checkpointing trained fine but hit
  **disk full** (each checkpoint dir was 14 GB and `load_best_model_at_end`
  keeps best + latest). Added auto-cleanup hook to `train_byt5()`.
- **Pivoted away from byt5-large** entirely. The assignment grades methodology,
  so the better use of time is per-language error analysis on the existing
  byt5-base hybrid: find *why* ko/ja are weak (data volume? jamo/syllable
  mismatch? specific error categories?), then propose targeted fixes
  (oversampling, data augmentation, post-processing rules). The byt5-large
  cells (23-27) remain in the notebook as evidence of the attempt.

## Current focus

Phase 1 of the analysis plan:
1. Per-language ERR breakdown on validation set with the byt5-base hybrid.
2. Categorize ko/ja errors: `no_change`, `over_change`, `wrong_change`,
   `length_mismatch`, `hallucination`.
3. Distribution analysis of raw tokens that fail (length, jamo vs syllable
   for ko, repeated chars like `ㅋㅋㅋ`, etc.).

Output goes into the notebook (new section), and feeds the report's
Experiments / Results sections.

## Conventions

- **Do not** add a Claude co-author line to commits in this repo — user
  preference. Match the existing author identity (`bbo-sseong
  <qhtjd7341@gmail.com>`) for new commits.
- The MFR baseline and ByT5-small sections are kept for reference / report
  comparison; don't delete them.
- Submission zips in `outputs/` are git-tracked exceptions to the
  `outputs/` ignore — preserve that.
