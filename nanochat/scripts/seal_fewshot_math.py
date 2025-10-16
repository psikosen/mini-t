"""Utilities for quantifying the compute required to adapt nanochat with SEAL few-shot data.

This helper can be used offline to sanity-check whether a particular few-shot
configuration fits within the model's context window and to approximate the
parameter/memory footprint of LoRA adapters before we wire them into the
training loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from nanochat.arc_prompt import format_arc_prompt, format_grid
from nanochat.logging_utils import LogRecord, log


@dataclass
class ModelConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def count_model_params(config: ModelConfig) -> int:
    head_dim = config.n_embd // config.n_head
    kv_out = config.n_kv_head * head_dim
    token_embedding = config.vocab_size * config.n_embd
    block_params = 0
    block_params += config.n_embd * config.n_embd  # q projection
    block_params += config.n_embd * kv_out        # k projection
    block_params += config.n_embd * kv_out        # v projection
    block_params += config.n_embd * config.n_embd  # output projection
    block_params += config.n_embd * (4 * config.n_embd)  # mlp up
    block_params += (4 * config.n_embd) * config.n_embd  # mlp down
    block_total = block_params * config.n_layer
    lm_head = config.n_embd * config.vocab_size
    return token_embedding + block_total + lm_head


def lora_parameter_count(config: ModelConfig, rank: int) -> int:
    head_dim = config.n_embd // config.n_head
    kv_out = config.n_kv_head * head_dim
    layer_shapes = [
        (config.n_embd, config.n_embd),
        (config.n_embd, kv_out),
        (config.n_embd, kv_out),
        (config.n_embd, config.n_embd),
        (config.n_embd, 4 * config.n_embd),
        (4 * config.n_embd, config.n_embd),
    ]
    per_layer = sum(rank * (fan_in + fan_out) for fan_in, fan_out in layer_shapes)
    return per_layer * config.n_layer


def approximate_tokens_from_chars(characters: int) -> int:
    return math.ceil(characters / 4)


def estimate_prompt_tokens(
    train: List[Dict[str, List[List[int]]]],
    test: Dict[str, List[List[int]]],
) -> int:
    prompt = format_arc_prompt(train, test)
    return approximate_tokens_from_chars(len(prompt))


def analyze_dataset(
    challenge_path: Path,
    limit_tasks: Optional[int],
) -> Tuple[int, int, float, int, int]:
    data = json.loads(challenge_path.read_text())
    task_items = list(data.items())
    if limit_tasks is not None:
        task_items = task_items[:limit_tasks]
    token_counts = []
    total_train_examples = 0
    max_context = 0
    total_cells = 0
    for _, payload in task_items:
        train_examples = payload["train"]
        test_example = payload["test"][0]
        total_train_examples += len(train_examples)
        token_count = estimate_prompt_tokens(train_examples, test_example)
        token_counts.append(token_count)
        max_context = max(max_context, token_count)
        for example in train_examples + [test_example]:
            for grid in (example["input"], example.get("output")):
                if grid is None:
                    continue
                cells = sum(len(row) for row in grid)
                total_cells += cells
    mean_tokens = statistics.mean(token_counts) if token_counts else 0.0
    return len(task_items), total_train_examples, mean_tokens, max_context, total_cells


def approximate_training_flops(param_count: int, token_budget: int, sequence_len: int) -> float:
    tokens_per_forward = sequence_len
    steps = math.ceil(token_budget / tokens_per_forward)
    flops_per_token = 6 * param_count
    return flops_per_token * token_budget, steps


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify SEAL few-shot integration costs.")
    parser.add_argument("--challenge-file", type=Path, help="Path to the ARC challenge JSON from SEAL.")
    parser.add_argument("--num-tasks", type=int, default=12, help="Number of ARC tasks to sample.")
    parser.add_argument("--self-edits", type=int, default=15, help="Self-edits per task in SEAL iteration 1.")
    parser.add_argument("--lora-rank", type=int, default=16, help="Assumed LoRA rank for adaptation.")
    parser.add_argument("--sequence-len", type=int, default=1024, help="Context length to compare against.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    config = ModelConfig(sequence_len=args.sequence_len)
    param_count = count_model_params(config)
    lora_params = lora_parameter_count(config, args.lora_rank)
    log(LogRecord(
        filename=os.path.basename(__file__),
        classname="seal_fewshot_math",
        function="main",
        system_section="model_stats",
        line_num=164,
        message=f"base_parameters={param_count}, lora_rank={args.lora_rank}, lora_parameters={lora_params}",
    ))
    if not args.challenge_file.exists():
        log(LogRecord(
            filename=os.path.basename(__file__),
            classname="seal_fewshot_math",
            function="main",
            system_section="dataset",
            line_num=175,
            message=f"challenge_file_missing={args.challenge_file}",
            error="file_not_found",
        ))
        return
    num_tasks, total_train_examples, mean_tokens, max_context, total_cells = analyze_dataset(
        args.challenge_file,
        args.num_tasks,
    )
    token_budget = int(mean_tokens * args.self_edits * num_tasks)
    total_flops, optim_steps = approximate_training_flops(param_count + lora_params, token_budget, args.sequence_len)
    log(LogRecord(
        filename=os.path.basename(__file__),
        classname="seal_fewshot_math",
        function="main",
        system_section="dataset_stats",
        line_num=191,
        message=(
            f"tasks={num_tasks}, train_examples={total_train_examples}, mean_prompt_tokens={mean_tokens:.2f}, "
            f"max_prompt_tokens={max_context}, token_budget={token_budget}, total_cells={total_cells}"
        ),
    ))
    log(LogRecord(
        filename=os.path.basename(__file__),
        classname="seal_fewshot_math",
        function="main",
        system_section="compute_budget",
        line_num=201,
        message=(
            f"approx_training_flops={total_flops:.2e}, estimated_steps={optim_steps}, sequence_len={args.sequence_len}"
        ),
    ))


if __name__ == "__main__":
    main()
