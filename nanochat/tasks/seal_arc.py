"""SEAL ARC few-shot dataset loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from nanochat.arc_prompt import format_arc_prompt, format_grid
from tasks.common import Task


@dataclass
class SealArcExample:
    """In-memory representation of a SEAL ARC instance."""

    task_id: str
    prompt: str
    completion: str
    eval_context: str
    fewshot_examples: List[Dict[str, str]]


def _build_fewshot_examples(train_examples: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert SEAL train examples into nanochat few-shot templates."""

    formatted: List[Dict[str, str]] = []
    for idx, example in enumerate(train_examples):
        context_lines: List[str] = []
        if idx == 0:
            context_lines.append("Solve the ARC task by predicting the correct output grid.")
            context_lines.append("")
            context_lines.append("==TRAIN==")
        context_lines.append(f"Example {idx + 1} input:")
        context_lines.append(format_grid(example["input"]))
        context_lines.append("Example output:")
        context = "\n".join(context_lines)
        formatted.append({
            "context": context,
            "continuation": format_grid(example["output"]),
        })
    return formatted


def _build_eval_context(test_example: Dict[str, Any]) -> str:
    context_lines = [
        "==TEST==",
        "Test input:",
        format_grid(test_example["input"]),
        "Test output:",
    ]
    return "\n".join(context_lines)


class SealArcFewShot(Task):
    """Task wrapper for SEAL ARC few-shot curriculum."""

    def __init__(
        self,
        json_path: str | Path,
        split: Optional[str] = "train",
        limit: Optional[int] = None,
        shuffle_seed: Optional[int] = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"SEAL ARC dataset not found: {path}")
        raw = json.loads(path.read_text())
        records: List[SealArcExample] = []
        for task_id, payload in raw.items():
            metadata = payload.get("metadata", {})
            payload_split = metadata.get("split", "train")
            if split is not None and payload_split != split:
                continue
            train_examples = payload.get("train", [])
            test_examples = payload.get("test", [])
            if not train_examples or not test_examples:
                continue
            fewshot_examples = _build_fewshot_examples(train_examples)
            for idx, test_example in enumerate(test_examples):
                prompt = format_arc_prompt(train_examples, test_example)
                completion = format_grid(test_example["output"])
                records.append(
                    SealArcExample(
                        task_id=f"{task_id}:{idx}",
                        prompt=prompt,
                        completion=completion,
                        eval_context=_build_eval_context(test_example),
                        fewshot_examples=fewshot_examples,
                    )
                )

        if shuffle_seed is not None:
            import random

            rng = random.Random(shuffle_seed)
            rng.shuffle(records)

        if limit is not None:
            records = records[:limit]

        if not records:
            raise ValueError("No SEAL ARC examples found after filtering.")

        self._examples = records

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return len(self._examples)

    def get_example(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        messages = [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": example.completion},
        ]
        return {"messages": messages, "task_id": example.task_id}

    def evaluate(self, conversation: Dict[str, Any], assistant_response: str) -> bool:
        expected = conversation["messages"][-1]["content"].strip()
        predicted = assistant_response.strip()
        return expected == predicted

    def build_eval_items(self) -> List[Dict[str, Any]]:
        """Return CORE-eval compatible dictionaries."""

        items: List[Dict[str, Any]] = []
        for example in self._examples:
            items.append(
                {
                    "context": example.eval_context,
                    "continuation": example.completion,
                    "fewshot_examples": example.fewshot_examples,
                    "task_id": example.task_id,
                }
            )
        return items

