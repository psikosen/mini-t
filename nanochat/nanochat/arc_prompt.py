"""Utilities for formatting ARC style grid prompts used in SEAL few-shot data."""

from __future__ import annotations

from typing import Dict, Iterable, List


def format_grid(grid: Iterable[Iterable[int]]) -> str:
    """Render a 2D grid of integers into a whitespace separated string."""

    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def format_arc_prompt(train: List[Dict[str, List[List[int]]]], test: Dict[str, List[List[int]]]) -> str:
    """Format a SEAL ARC task into the prompt template expected by nanochat."""

    lines = ["Solve the ARC task by predicting the correct output grid.", "", "==TRAIN=="]
    for idx, example in enumerate(train):
        lines.append(f"Example {idx + 1} input:")
        lines.append(format_grid(example["input"]))
        lines.append("Example output:")
        lines.append(format_grid(example["output"]))
        lines.append("")
    lines.append("==TEST==")
    lines.append("Test input:")
    lines.append(format_grid(test["input"]))
    lines.append("Test output:")
    return "\n".join(lines)

