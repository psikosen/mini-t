# mini-t

## Overview
Mini-t packages a working checkout of [nanochat](nanochat/) together with thin
project management scaffolding so you can run, monitor, and extend the "$100
ChatGPT" training pipeline end-to-end. The upstream nanochat tree already
contains the reference implementation for tokenizer training, base and
instruction-tuned model stages, LoRA adaptation, evaluation harnesses, and web
serving utilities. This README focuses on how to operate those pieces in this
environment and highlights the observability hooks that were added during the
mini-t customization effort.

## Repository layout
- [`nanochat/`](nanochat/) – upstream project source (Python modules, scripts,
  and assets).
- [`nanochat/scripts/`](nanochat/scripts/) – entry points for each training and
  evaluation stage (`tok_train`, `base_train`, `chat_sft`, `seal_fewshot_train`,
  etc.).
- [`nanochat/tasks/`](nanochat/tasks/) – dataset adapters (ARC, GSM8K, SEAL ARC
  few-shot curriculum, …) and sample JSON fixtures.
- [`nanochat/tests/`](nanochat/tests/) – pytest coverage for tokenizer Rust
  bindings, ternary kernels, and SEAL integration.
- [`docs/`](docs/) – high-level design notes (e.g. the SEAL few-shot plan).
- [`task.md`](task.md) – rolling checklist of work streams completed in this
  environment.

## Environment setup
1. Install [uv](https://github.com/astral-sh/uv) if it is not already present.
2. Create and activate the project virtual environment, then sync dependencies:
   ```bash
   cd nanochat
   uv venv
   source .venv/bin/activate
   uv sync
   ```
3. If you plan to execute GPU workloads, provision an 8×H100 (or comparable)
   box. Set `NANOCHAT_BASE_DIR` to a volume with at least 200 GB free for
   datasets, checkpoints, and reports. All scripts default to
   `~/.cache/nanochat` when the variable is not provided.
4. (Optional) Authenticate with Weights & Biases before running longer jobs:
   ```bash
   wandb login
   ```

## Quick start: one-command speedrun
[`speedrun.sh`](nanochat/speedrun.sh) reproduces the reference 4-hour training
journey: tokenizer → base pretraining → mid-training → supervised finetuning →
(optional) RL → reporting. Launch it from the `nanochat/` directory:
```bash
bash speedrun.sh
```
Use a `screen`/`tmux` session and set `WANDB_RUN` to label the experiment when
you want persistent wandb tracking.

## Manual pipeline breakdown
When you need fine-grained control, invoke the scripts individually. Every
script accepts overrides through `nanochat/configurator.py` so you can inject
flags via `-- --key=value` arguments.

### Tokenizer stage
- Download FineWeb-Edu shards: `python -m nanochat.dataset -n 8` (blocking) and
  `python -m nanochat.dataset -n 240 &` (background prefetch for pretraining).
- Train the tokenizer: `python -m scripts.tok_train --max_chars=2000000000`.
- Evaluate compression: `python -m scripts.tok_eval`.

### Base pretraining
- Ensure `eval_bundle/` exists (the speedrun script downloads it automatically).
- Launch distributed training: `torchrun --standalone --nproc_per_node=8 -m
  scripts.base_train -- --run=$WANDB_RUN` (adjust depth, device batch size,
  ternary flags, etc.).
- Inspect loss curves with `torchrun -m scripts.base_loss` and CORE metrics via
  `torchrun -m scripts.base_eval`.

### Mid-training, SFT, and RL
- Mid-training (introduce special tokens & multiple choice tasks):
  `torchrun -m scripts.mid_train -- --run=$WANDB_RUN` followed by
  `torchrun -m scripts.chat_eval -- -i mid`.
- Supervised finetuning: `torchrun -m scripts.chat_sft -- --run=$WANDB_RUN`
  then `torchrun -m scripts.chat_eval -- -i sft`.
- Reinforcement learning on GSM8K (optional):
  `torchrun -m scripts.chat_rl -- --run=$WANDB_RUN` and
  `torchrun -m scripts.chat_eval -- -i rl -a GSM8K`.

### SEAL ARC few-shot LoRA
- Prepare a SEAL-style JSON curriculum (see the sample in
  `nanochat/tasks/data/seal_arc_sample.json`).
- Train adapters: `torchrun -m scripts.seal_fewshot_train -- --dataset_path=tasks/data/seal_arc_sample.json --run=$WANDB_RUN`.
- Evaluate alongside CORE with `torchrun -m scripts.base_eval -- --seal-arc-json
  tasks/data/seal_arc_sample.json`.

## Configuration levers
- Most scripts expose `--ternary_weights` and `--ternary_threshold` to enable the
  ternary kernel in `nanochat/gpt.py`.
- LoRA adapters are injected through `scripts/seal_fewshot_train.py` using the
  `nanochat/lora.py` helpers (rank, alpha, dropout, etc.).
- Global defaults live in `nanochat/configurator.py`; override them by passing
  `-- --key=value` arguments to any script invoked with `python -m` or
  `torchrun -m`.

## Data management
- All long-running jobs read and write under `NANOCHAT_BASE_DIR`. Review
  `nanochat/common.py` for helper routines like `get_base_dir()`.
- Pretraining shards are fetched from the FineWeb-Edu dataset. Use `-n` to cap
  downloads when testing locally.
- Task adapters inside `nanochat/tasks/` expose deterministic sampling and
  evaluation helpers for ARC, GSM8K, HumanEval, MMLU, and the SEAL curriculum.

## Evaluation and reporting
- CORE benchmark aggregation lives in `scripts/base_eval.py`. Pass
  `--max-per-task` for quick smoke tests or `--seal-arc-json` to fold in SEAL ARC
  accuracy.
- `scripts/chat_eval.py` scores conversational checkpoints (mid, SFT, RL).
- Reports are assembled via `python -m nanochat.report generate`, which also
  embeds system metadata and dataset statistics for compliance tracking.

## Inference and serving
- Command-line chat: `python -m scripts.chat_cli -p "Why is the sky blue?"`.
- Web UI (FastAPI + Uvicorn): `python -m scripts.chat_web` and then visit the
  printed host/port.
- For batched completions or offline sampling, use
  `python -m scripts.chat_eval -- --sample-only` style invocations.

## Monitoring & observability
- Set `WANDB_RUN=<name>` to stream metrics to a wandb project (`nanochat` by
  default).
- Structured logs follow the Sherlock Protocol schema via
  `nanochat/logging_utils.py`; scripts emit JSON records plus the required
  human-readable checklist.
- Intermediate CSVs, prompt captures, and report sections are persisted under
  `$NANOCHAT_BASE_DIR` so you can diff runs post hoc.

## Deployment considerations
- The web UI is stateless—proxy it behind HTTPS and configure authentication at
  the reverse proxy layer if exposing publicly.
- For inference-only workloads, export checkpoints with
  `python -m nanochat.checkpoint_manager` utilities and load them through
  `scripts/chat_web.py` or a custom FastAPI wrapper.

## Troubleshooting
- **CUDA OOM**: drop `--device_batch_size` or switch to gradient accumulation
  (scripts adjust automatically when batch size shrinks).
- **Slow dataset downloads**: reduce `-n` when invoking `nanochat.dataset` for
  quick tests or mirror the parquet shards locally.
- **Tokenizer build issues**: reinstall Rust (`rustup`) and rerun `uv run maturin
  develop --release`.
- **wandb authentication errors**: confirm `WANDB_API_KEY` and that `WANDB_RUN`
  is not `dummy` when you expect logging.

## Resource planning
- The canonical d20 "speedrun" consumes ~24 GB of compressed training text,
  produces ~200 GB of checkpoints/artifacts, and completes in ~4 hours on an
  8×H100 node at ~$24/hour GPU cost.
- LoRA few-shot runs are lightweight (millions of parameters and a few GPU
  hours); see `docs/seal_few_shot_plan.md` for detailed FLOP estimates.

## Testing & linting
- Run the tokenizer tests (covers Rust BPE bindings):
  ```bash
  uv run pytest tests/test_rustbpe.py -v -s
  ```
- Additional suites: `tests/test_ternary.py` (ternary projections) and
  `tests/test_seal_fewshot.py` (LoRA + dataset sanity checks).

