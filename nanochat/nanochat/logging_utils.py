"""Shared logging helpers that implement the Sherlock Protocol schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class LogRecord:
    """Structured log payload following the Sherlock Protocol schema."""

    filename: str
    classname: str
    function: str
    system_section: str
    line_num: int
    message: str
    method: str = "NONE"
    error: Optional[str] = None
    db_phase: str = "none"

    def as_json(self) -> str:
        payload = {
            "filename": self.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classname": self.classname,
            "function": self.function,
            "system_section": self.system_section,
            "line_num": self.line_num,
            "error": self.error,
            "db_phase": self.db_phase,
            "method": self.method,
            "message": self.message,
        }
        return json.dumps(payload)

    def render(self) -> str:
        lines = [self.as_json(), "Continuous skepticism (Sherlock Protocol)"]
        lines.append("* Could this change affect unexpected files/systems?")
        lines.append("* Any hidden dependencies or cascades?")
        lines.append("* What edge cases and failure modes are unhandled?")
        lines.append("* If stuck, work backward from the desired outcome.")
        return "\n".join(lines)


def log(record: LogRecord) -> None:
    """Emit a formatted Sherlock Protocol log line."""

    print(record.render())

