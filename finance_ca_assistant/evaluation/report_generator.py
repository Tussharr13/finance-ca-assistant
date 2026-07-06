"""Markdown and JSON report generation."""

from __future__ import annotations

import json
from typing import Dict, List


def generate_markdown_report(metrics: Dict[str, float], failures: List[Dict[str, object]]) -> str:
    """Generate a concise markdown evaluation report."""

    lines = ["# CA RAG Evaluation Report", "", "## Metrics"]
    for name, value in sorted(metrics.items()):
        lines.append(f"- {name}: {value:.3f}")
    lines.extend(["", "## Failures"])
    if failures:
        for failure in failures:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def generate_json_report(metrics: Dict[str, float], failures: List[Dict[str, object]]) -> str:
    """Generate a machine-readable JSON evaluation report."""

    return json.dumps({"metrics": metrics, "failures": failures}, indent=2, sort_keys=True)

