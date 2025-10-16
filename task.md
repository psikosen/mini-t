# Task Plan

- [x] Clone nanochat repository into workspace
- [x] Introduce ternary kernel and linear layer implementation
- [x] Integrate ternary option into GPT architecture/configuration
- [x] Add documentation and tests for ternary support
- [x] Run project tests/lints and document results
- [x] Identify unsupported binary assets in the repository
- [x] Remove unsupported binary files and document their locations
- [x] Expose ternary configuration flags in training checkpoints and scripts
- [x] Expand ternary documentation and tests to cover CLI usage
- [x] Integrate SEAL few-shot curriculum into nanochat pipelines

## Current Initiative: SEAL Few-Shot Integration
- [x] Review SEAL repository structure and identify reusable few-shot assets
- [x] Quantify compute/token budgets for SEAL-style few-shot adaptation on nanochat
- [x] Implement ARC few-shot dataset loader compatible with nanochat `Task`
- [x] Add LoRA-based few-shot training stage and configuration knobs
- [x] Extend evaluation/reporting to cover SEAL ARC benchmarks
- [x] Instrument CORE/SEAL evaluation scripts with Sherlock-compliant structured logging
