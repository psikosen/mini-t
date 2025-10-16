# SEAL Few-Shot Integration Plan

## Feasibility math
- The base nanochat configuration (sequence length 1024, 12 layers, 768 hidden width) contains roughly 162M trainable parameters, while a rank-16 LoRA adapter across attention and MLP projections adds about 2.7M parameters (~1.6% overhead). 【F:nanochat/scripts/seal_fewshot_math.py†L24-L30】【F:nanochat/scripts/seal_fewshot_math.py†L92-L120】【fbefc6†L1-L6】
- Formatting the first 12 ARC tasks from SEAL's filtered training split into textual prompts with the `==TRAIN==`/`==TEST==` template yields a mean context of ~183 estimated tokens (max 529), leaving ample headroom inside the 1024-token window even when inserting 11 few-shot exemplars. 【F:nanochat/scripts/seal_fewshot_math.py†L77-L161】【fbefc6†L6-L13】
- Iteration-1 self-editing (12 tasks × 15 edits) would consume ~32.9k prompt tokens, implying ~3.2×10¹³ forward FLOPs when training a LoRA-adapted model—well within a few GPU-hours on the existing 8×H100 setup. 【F:nanochat/scripts/seal_fewshot_math.py†L164-L214】【fbefc6†L13-L20】

## Integration roadmap
1. **Dataset ingestion** – Build a lightweight loader that maps SEAL ARC JSON files into nanochat `Task` objects, reusing the grid rendering helpers already present in `core_eval` so that few-shot prompts remain consistent. 【F:nanochat/nanochat/core_eval.py†L172-L214】
2. **Few-shot sampling** – Extend the evaluation sampler to accept pinned exemplars from the SEAL curriculum instead of purely random draws; the existing `evaluate_example` hook already supports injecting a `fewshot_examples` list. 【F:nanochat/nanochat/core_eval.py†L177-L206】
3. **LoRA adaptation stage** – Add a dedicated training script (mirroring `scripts.mid_train`) that freezes the base weights, instantiates LoRA projections, and iterates over the curated few-shot dataset according to the compute budget above. 【F:nanochat/scripts/mid_train.py†L1-L60】【fbefc6†L13-L20】
4. **Evaluation harness** – Wire the SEAL ARC task into the `scripts.base_eval` mixture so we can track improvements alongside ARC-Easy/Challenge and GSM8K. 【F:nanochat/scripts/base_eval.py†L40-L76】
5. **Reporting & observability** – Emit structured logs (matching the Sherlock protocol schema) from the new training/evaluation paths so CI can surface unexpected regressions when few-shot adapters are toggled. 【F:nanochat/scripts/seal_fewshot_math.py†L33-L70】
