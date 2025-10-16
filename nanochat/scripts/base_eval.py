"""Evaluate the CORE metric and optional SEAL ARC benchmark for a given model."""

import argparse
import os
import time
import json
import random
import yaml

import pandas as pd
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task
from tasks.seal_arc import SealArcFewShot
from nanochat.logging_utils import LogRecord, log

# -----------------------------------------------------------------------------
# nanoChat specific function dealing with I/O etc.

def evaluate_model(model, tokenizer, device, max_per_task=-1, *, enable_logging=True):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    TODO: clean up this function, delete the need for all the files, for pandas dependency, etc.
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']
    eval_metadata = pd.read_csv(eval_meta_data)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the abillity to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        row = eval_metadata[eval_metadata["Eval Task"] == label]
        random_baseline = row["Random baseline"].values[0]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        if enable_logging:
            log(
                LogRecord(
                    filename=__file__,
                    classname="base_eval",
                    function="evaluate_model",
                    system_section="evaluation",
                    line_num=0,
                    message=(
                        f"task={label} accuracy={accuracy:.4f} centered={centered_result:.4f} "
                        f"examples={len(data)}"
                    ),
                )
            )
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
    if enable_logging:
        log(
            LogRecord(
                filename=__file__,
                classname="base_eval",
                function="evaluate_model",
                system_section="summary",
                line_num=0,
                message=f"core_metric={core_metric:.4f} tasks={len(results)}",
            )
        )
    return out


def evaluate_seal_arc(model, tokenizer, device, args, *, enable_logging=True):
    if not args.seal_arc_json:
        return {}
    task = SealArcFewShot(args.seal_arc_json, split=args.seal_arc_split, limit=args.seal_arc_limit)
    data = task.build_eval_items()
    if not data:
        return {}
    task_meta = {
        'task_type': 'language_modeling',
        'num_fewshot': len(data[0]['fewshot_examples']),
        'continuation_delimiter': '\n'
    }
    print0(f"Evaluating {args.seal_arc_label} ({task_meta['num_fewshot']}-shot)... ", end='')
    accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
    print0(f"accuracy: {accuracy:.4f} | examples: {len(data)}")
    if enable_logging:
        log(
            LogRecord(
                filename=__file__,
                classname="base_eval",
                function="evaluate_seal_arc",
                system_section="evaluation",
                line_num=0,
                message=(
                    f"label={args.seal_arc_label} accuracy={accuracy:.4f} "
                    f"examples={len(data)} fewshot={task_meta['num_fewshot']}"
                ),
            )
        )
    return {args.seal_arc_label: {'accuracy': accuracy, 'num_fewshot': task_meta['num_fewshot'], 'num_examples': len(data)}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate CORE and optional SEAL ARC benchmarks.")
    parser.add_argument("hf_path", nargs="?", help="Optional HuggingFace model path to evaluate.")
    parser.add_argument("--seal-arc-json", dest="seal_arc_json", help="Path to a SEAL ARC curriculum JSON file.")
    parser.add_argument("--seal-arc-split", dest="seal_arc_split", default="train", help="Split name to filter within the SEAL ARC JSON.")
    parser.add_argument("--seal-arc-limit", dest="seal_arc_limit", type=int, default=None, help="Optional limit on the number of SEAL ARC tasks.")
    parser.add_argument("--seal-arc-label", dest="seal_arc_label", default="SEAL-ARC", help="Label used when reporting SEAL ARC results.")
    parser.add_argument("--max-per-task", dest="max_per_task", type=int, default=-1, help="Maximum examples per CORE task (for debugging).")
    return parser.parse_args(argv)

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # Load the tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)

    # distributed / precision setup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Load model and tokenizer from command line or from file system
    if args.hf_path:
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path
        model_slug = hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval")
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(
            model,
            tokenizer,
            device,
            max_per_task=args.max_per_task,
            enable_logging=master_process,
        )
        seal_results = evaluate_seal_arc(
            model,
            tokenizer,
            device,
            args,
            enable_logging=master_process,
        )

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    if master_process:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                centered = centered_results.get(label)
                centered_str = f"{centered:<10.6f}" if centered is not None else ""
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_str}\n")
            for label, metrics in seal_results.items():
                f.write(f"{label:<35}, {metrics['accuracy']:<10.6f}, {'':<10}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r') as f:
            print0(f.read())
        log(
            LogRecord(
                filename=__file__,
                classname="base_eval",
                function="main",
                system_section="reporting",
                line_num=0,
                message=(
                    f"wrote_csv={output_csv_path} core_metric={core_metric:.4f} "
                    f"seal_tasks={len(seal_results)}"
                ),
            )
        )

    # Log to report
    from nanochat.report import get_report
    report_payload = [
        {
            "Model": model_name,
            "CORE metric": core_metric,
        },
        centered_results,
    ]
    if seal_results:
        report_payload.append({k: v['accuracy'] for k, v in seal_results.items()})
    get_report().log(section="Base model evaluation", data=report_payload)
    if master_process:
        log(
            LogRecord(
                filename=__file__,
                classname="base_eval",
                function="main",
                system_section="reporting",
                line_num=0,
                message="report_section=Base model evaluation",
            )
        )

    compute_cleanup()

if __name__ == "__main__":
    main()
