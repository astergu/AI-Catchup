# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Nature

AI-Catchup is a **personal study and reference repository** — not a deployed application. It mixes:

- **`ML-Implementations/`** — from-scratch implementations of ML/DL models (the bulk of the executable code)
- **`Roadmaps/`** — markdown curricula (AI Engineer, DL Engineer, Ads, System Design) plus a vanilla-JS progress tracker web app
- **`Books/`, `Papers/`, `ML-Courses/`** — read-only reference material (PDFs, notes). Don't modify these unless explicitly asked.

There is no monorepo build system, no top-level package manifest, no CI, and no lint/format config. Each subproject is self-contained.

## Active Subprojects & Common Commands

### `ML-Implementations/ads/` — CTR/CVR Models (TensorFlow/Keras)

This is the most active and most production-quality code in the repo. All ad-prediction models (WDL, ESMM, DCN, DIN, DIEN, DeepFM, MMoE, PLE, AutoInt, xDeepFM, NFM) live here as one model per file.

```bash
# Install (TF 2.10+, sklearn, pandas, numpy)
pip install -r ML-Implementations/ads/requirements.txt

# Run the full benchmark — trains every model on synthetic data and writes results.csv
cd ML-Implementations/ads && python test_models.py
```

After running `test_models.py`, the AUC/Accuracy/LogLoss table in `ML-Implementations/ads/README.md` is the source of truth for current benchmark numbers (and `results.csv` is the machine-readable form). When you add or modify a model, **rerun the benchmark and update both** — the README table is consulted as documentation.

Two test harnesses coexist, do not conflate them:
- **`test_models.py`** (top-level in `ads/`) — the actual benchmark script that produces `results.csv`. This is what gets run.
- **`test/` subdirectory** (`test_ctr.py`, `test_cvr.py`, `test_data.py`, `tf_adapter.py`) — a richer dict-of-arrays evaluation framework with a different model interface (features as `Dict[str, np.ndarray]`). Currently only used via `test/example_usage.py`; the main models in `ads/` do **not** plug into this framework. Use it only when explicitly asked.

### `Roadmaps/tracker/` — Progress Tracker Web App

Vanilla JS (no build step), reads JSON from `data/`, persists progress via `localStorage`.

```bash
cd Roadmaps/tracker && python3 -m http.server 8000
# open http://localhost:8000
```

Roadmap content is edited by changing JSON files in `Roadmaps/tracker/data/` (`ai-engineer.json`, `dl-engineer.json`, `system-design.json`), not by editing the markdown roadmaps elsewhere. The two are not auto-synced.

### `ML-Implementations/transformers/`, `optimizers/`, `basics/`, `losses/`, `metrics/`, `utils/`

PyTorch-based scratch implementations (note the framework split: **`ads/` is TensorFlow, the rest of `ML-Implementations/` leans PyTorch**). These are mostly didactic standalone scripts/notebooks; many are stubs or works-in-progress (e.g. `basics/linear_regression.py` is currently a skeleton class). There is no shared trainer beyond `utils/trainer.py`, which is a minkpt-style loop and is not wired up to most files.

## Conventions Worth Knowing

- **Model interface in `ads/`** — every model class exposes `__init__(...)`, `fit(...)`, and `predict(...)`. Multi-task models (ESMM, MMoE, PLE) return a `dict` from `predict` keyed by task (e.g. `{"p_ctr": ..., "p_cvr": ..., "p_ctcvr": ...}`). When adding a new model, follow this contract so it slots into `test_models.py`.
- **Synthetic data is the default** — benchmarks use seeded NumPy generators (see `generate_dataset` / `generate_esmm_dataset` in `test_models.py`). Numbers are reproducible but only meaningful relative to other models on the same generator. Don't claim absolute performance.
- **Two `test_models.py` test datasets coexist**: a stacked-vector dataset (`generate_dataset` for WDL/DCN) and a categorical/sequence dataset (for DIN/DIEN/DeepFM/AutoInt/xDeepFM/NFM). The README benchmark table tags which dataset each row uses.
- **Generated `.html` files** next to roadmap `README.md`s (e.g. `Roadmaps/AI-Engineer-Roadmap/README.html`) are exported renders. Don't hand-edit them; regenerate from the markdown source.

## What NOT to Do

- Don't introduce a top-level package, monorepo tooling, or shared dependency manifest — each subproject stays independent on purpose.
- Don't modify PDFs in `Books/` or `Papers/`.
- Don't translate the Chinese sections in roadmap docs into English (they're intentionally bilingual; the tracker handles language toggling at the UI level).
