from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nanochat.gpt import GPT, GPTConfig
from nanochat.lora import LoRAConfig, LoRALinear, inject_lora, lora_parameters, lora_state_dict
from nanochat.core_eval import render_prompts_lm
from tasks.seal_arc import SealArcFewShot


def get_sample_path() -> Path:
    return Path(__file__).resolve().parents[1] / "tasks" / "data" / "seal_arc_sample.json"


def test_seal_arc_loader_shapes():
    task = SealArcFewShot(get_sample_path(), split="train", shuffle_seed=None)
    assert len(task) == 1
    example = task.get_example(0)
    user = example["messages"][0]["content"]
    assistant = example["messages"][1]["content"]
    assert "==TEST==" in user
    assert assistant.strip() == "0 3 3\n3 0 3"

    eval_items = task.build_eval_items()
    assert len(eval_items) == 1
    item = eval_items[0]
    assert item["fewshot_examples"]
    prompts = render_prompts_lm(item, "\n", item["fewshot_examples"])
    assert prompts[0].endswith("Test output:")
    assert prompts[1].split("Test output:\n")[-1].strip() == assistant.strip()


def test_lora_injection_replaces_modules():
    config = GPTConfig(n_layer=1, n_head=1, n_kv_head=1, n_embd=32, sequence_len=8, vocab_size=32)
    model = GPT(config)
    lora_cfg = LoRAConfig(rank=4, alpha=4, dropout=0.0)
    injected = inject_lora(model, lora_cfg)
    assert injected, "LoRA injection should wrap target modules"
    targets = {name for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    assert targets, "LoRA modules should be present"
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert all(not p.requires_grad for p in module.base.parameters())
            assert all(p.requires_grad for p in module.lora_a.parameters())
            assert all(p.requires_grad for p in module.lora_b.parameters())

    params = lora_parameters(model)
    assert params, "LoRA parameters should be returned"

    sample = torch.randint(0, config.vocab_size, (1, 4))
    output = model(sample)
    assert output.shape[-1] == config.vocab_size

    state = lora_state_dict(model)
    assert state, "LoRA state dict should not be empty"
