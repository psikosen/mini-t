"""LoRA adaptation stage for the SEAL ARC few-shot curriculum."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import random
from typing import List

import torch
import wandb

from nanochat.common import compute_init, compute_cleanup, DummyWandb, get_base_dir, print0
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task
from nanochat.logging_utils import LogRecord, log
from nanochat.lora import LoRAConfig, inject_lora, lora_parameters, lora_state_dict
from tasks.seal_arc import SealArcFewShot


# -----------------------------------------------------------------------------
# Configuration (override via configurator)
run = "dummy"
source = "mid"
model_tag = None
step = None
dtype = "bfloat16"
device_batch_size = 2
grad_accum_steps = 1
num_epochs = 3
max_steps = -1
learning_rate = 5e-4
weight_decay = 0.0
grad_clip = 1.0
lora_rank = 16
lora_alpha = 16
lora_dropout = 0.05
dataset_path = os.path.join("tasks", "data", "seal_arc_sample.json")
dataset_split = "train"
dataset_limit = None
log_every = 5
eval_every = 20
output_subdir = "seal_lora"

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith('_') and (isinstance(v, (int, float, bool, str)) or v is None)
]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Environment and logging setup
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-seal-fewshot", name=run, config=user_config)

# -----------------------------------------------------------------------------
# Load base model and tokenizer, then inject LoRA adapters
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)

lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
applied = inject_lora(model, lora_config)
log(LogRecord(
    filename=__file__,
    classname="seal_fewshot_train",
    function="inject_lora",
    system_section="initialization",
    line_num=0,
    message=f"Applied LoRA adapters to: {applied}",
))

trainable_params = lora_parameters(model)
num_trainable = sum(p.numel() for p in trainable_params)
log(LogRecord(
    filename=__file__,
    classname="seal_fewshot_train",
    function="inject_lora",
    system_section="initialization",
    line_num=0,
    message=f"LoRA parameters={num_trainable}",
))

# -----------------------------------------------------------------------------
# Dataset and data loader
dataset = SealArcFewShot(dataset_path, split=dataset_split, limit=dataset_limit)
eval_items = dataset.build_eval_items()
continuation_delimiter = "\n"
print0(f"Loaded {len(dataset)} SEAL ARC examples from {dataset_path}")


def collate(batch: List[tuple[List[int], List[int]]]):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    nrows = len(batch)
    ncols = max(len(ids) for ids, _ in batch) - 1
    inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
    targets = torch.full((nrows, ncols), -1, dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
        inputs[i, : len(ids) - 1] = ids_tensor[:-1]
        row_targets = ids_tensor[1:]
        row_targets[mask_tensor == 0] = -1
        targets[i, : len(ids) - 1] = row_targets
    return inputs.to(device), targets.to(device)


def data_iterator():
    rng = random.Random(1337 + ddp_rank)
    while True:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        local = [idx for idx in indices if idx % ddp_world_size == ddp_rank]
        batch: List[tuple[List[int], List[int]]] = []
        for idx in local:
            conversation = dataset[idx]
            ids, mask = tokenizer.render_conversation(conversation)
            batch.append((ids, mask))
            if len(batch) == device_batch_size:
                yield collate(batch)
                batch = []
        if batch:
            yield collate(batch)


loader = data_iterator()
steps_per_epoch = max(1, len(dataset) // (device_batch_size * max(ddp_world_size, 1)))
total_steps = num_epochs * steps_per_epoch if max_steps < 0 else max_steps

# -----------------------------------------------------------------------------
# Optimizer
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

# -----------------------------------------------------------------------------
# Training loop
model.train()
optimizer.zero_grad(set_to_none=True)
global_step = 0
loss_ema = None
start_time = time.time()

while True:
    if max_steps > 0 and global_step >= max_steps:
        break
    for _ in range(grad_accum_steps):
        inputs, targets = next(loader)
        with autocast_ctx:
            loss = model(inputs, targets=targets, loss_reduction='mean') / grad_accum_steps
        loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    global_step += 1
    loss_value = loss.detach().item() * grad_accum_steps
    loss_ema = loss_value if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_value
    if master_process and global_step % log_every == 0:
        wandb_run.log({"step": global_step, "train/loss": loss_value, "train/loss_ema": loss_ema})
        log(LogRecord(
            filename=__file__,
            classname="seal_fewshot_train",
            function="train_loop",
            system_section="optimization",
            line_num=0,
            message=f"step={global_step} loss={loss_value:.4f}",
        ))
    if global_step % eval_every == 0 or (max_steps > 0 and global_step >= max_steps):
        model.eval()
        task_meta = {
            "task_type": "language_modeling",
            "num_fewshot": len(eval_items[0]["fewshot_examples"]),
            "continuation_delimiter": continuation_delimiter,
        }
        with autocast_ctx:
            accuracy = evaluate_task(model, tokenizer, eval_items, device, task_meta)
        if master_process:
            wandb_run.log({"step": global_step, "eval/accuracy": accuracy})
            log(LogRecord(
                filename=__file__,
                classname="seal_fewshot_train",
                function="evaluation",
                system_section="evaluation",
                line_num=0,
                message=f"accuracy={accuracy:.4f}",
            ))
        model.train()
    if max_steps > 0 and global_step >= max_steps:
        break
    if global_step >= total_steps:
        break

training_time = time.time() - start_time

# Final evaluation if the loop exited before hitting eval cadence
if global_step > 0:
    model.eval()
    task_meta = {
        "task_type": "language_modeling",
        "num_fewshot": len(eval_items[0]["fewshot_examples"]),
        "continuation_delimiter": continuation_delimiter,
    }
    with autocast_ctx:
        final_accuracy = evaluate_task(model, tokenizer, eval_items, device, task_meta)
    if master_process:
        wandb_run.log({"step": global_step, "eval/accuracy_final": final_accuracy})
        log(LogRecord(
            filename=__file__,
            classname="seal_fewshot_train",
            function="evaluation",
            system_section="evaluation",
            line_num=0,
            message=f"final_accuracy={final_accuracy:.4f}",
        ))
    model.train()

# -----------------------------------------------------------------------------
# Save LoRA adapters
if master_process:
    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"lora_step_{global_step:05d}.pt")
    torch.save({
        "lora_state_dict": lora_state_dict(model),
        "config": user_config,
        "steps": global_step,
        "training_time_sec": training_time,
    }, ckpt_path)
    log(LogRecord(
        filename=__file__,
        classname="seal_fewshot_train",
        function="checkpoint",
        system_section="checkpointing",
        line_num=0,
        message=f"Saved LoRA checkpoint to {ckpt_path}",
    ))

compute_cleanup()

