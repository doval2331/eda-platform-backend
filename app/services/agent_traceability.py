from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


@dataclass
class TraceCollector:
    run_id: str
    records: list[dict[str, object]] = field(default_factory=list)

    def record(
        self,
        *,
        agent_name: str,
        decision_type: str,
        prompt: str,
        response: str,
        model_name: str,
        variables_used: list[str],
        input_artifacts: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> str:
        trace_id = str(uuid.uuid4())
        self.records.append(
            {
                "trace_id": trace_id,
                "run_id": self.run_id,
                "agent_name": agent_name,
                "decision_type": decision_type,
                "model_name": model_name,
                "parameters": to_json(parameters or {}),
                "variables_used": to_json(variables_used),
                "input_artifacts": to_json(input_artifacts),
                "prompt": prompt,
                "response": response,
                "created_at": utc_now_iso(),
            }
        )
        return trace_id

    def to_frame(self) -> pd.DataFrame:
        columns = [
            "trace_id",
            "run_id",
            "agent_name",
            "decision_type",
            "model_name",
            "parameters",
            "variables_used",
            "input_artifacts",
            "prompt",
            "response",
            "created_at",
        ]
        if not self.records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.records, columns=columns)
