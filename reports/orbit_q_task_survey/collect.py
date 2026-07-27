"""Collect immutable ORBIT-Q source facts into one reviewed survey dataset."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


FRAMEWORK_KEYS = (
    "tensorcircuit",
    "pennylane",
    "torchquantum",
    "mindquantum",
)
EVIDENCE_CLASSES = {"official", "literature", "survey_inference"}
EXPECTED_TASK_IDS = [f"{number:02d}" for number in range(1, 13)]


def _section(text: str, title: str) -> str:
    match = re.search(
        rf"^## {re.escape(title)}\s*$\n(.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _compact_markdown(text: str) -> str:
    paragraphs = [
        " ".join(line.strip() for line in paragraph.splitlines())
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    return "\n\n".join(paragraphs)


def _parse_config(section: str) -> dict[str, Any]:
    match = re.search(r"```python\s*(\{.*?\})\s*```", section, re.DOTALL)
    if not match:
        return {}
    source = re.sub(
        r"<NumPy real array with shape \(3,\)>",
        "None",
        match.group(1),
    )
    parsed = ast.literal_eval(source)
    if not isinstance(parsed, dict):
        raise ValueError("fixed problem configuration must be a dictionary")
    return parsed


def collect_tasks(orbit_root: Path) -> list[dict[str, Any]]:
    """Parse official titles, goals, configurations, and passing criteria."""
    rows: list[dict[str, Any]] = []
    task_root = orbit_root / "tasks"
    for path in sorted(task_root.glob("challenge-*/instruction.md")):
        task_id = path.parent.name.removeprefix("challenge-")
        text = path.read_text(encoding="utf-8")
        title_match = re.search(
            r"^# (?:Problem|Challenge)\s+\d+:\s+(.+?)\s*$",
            text,
            re.MULTILINE,
        )
        if not title_match:
            raise ValueError(f"missing task title in {path}")
        goal = _compact_markdown(_section(text, "Goal"))
        config = _parse_config(_section(text, "Fixed Problem Configuration"))
        passing = _section(text, "Passing Criteria")
        bullets = [
            match.group(1).strip()
            for match in re.finditer(r"^\s*-\s+(.+)$", passing, re.MULTILINE)
        ]
        if not bullets and passing:
            bullets = [_compact_markdown(passing).split("\n\n", 1)[0]]
        rows.append(
            {
                "id": task_id,
                "title": title_match.group(1).strip(),
                "goal": goal,
                "config": config,
                "passing_criteria": bullets,
                "source_path": path.relative_to(orbit_root).as_posix(),
            }
        )
    return rows


_RESULT_ROW = re.compile(
    r"""\{\s*task:\s*"(?P<task>\d{2})",\s*
    pass:\s*(?P<pass>true|false),\s*
    runtime:\s*"(?P<runtime>[^"]*)",\s*
    ref:\s*"(?P<ref>[^"]*)",\s*
    wall:\s*"(?P<wall>[^"]*)",\s*
    tokens:\s*"(?P<tokens>[^"]*)",\s*
    cost:\s*"(?P<cost>[^"]*)",\s*
    notes:\s*"(?P<notes>[^"]*)"\s*\}""",
    re.VERBOSE,
)


def collect_framework_results(orbit_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse the four official framework result arrays from ``docs/app.js``."""
    text = (orbit_root / "docs" / "app.js").read_text(encoding="utf-8")
    results: dict[str, list[dict[str, Any]]] = {}
    for index, key in enumerate(FRAMEWORK_KEYS):
        start_match = re.search(rf"\b{key}:\s*\[", text)
        if not start_match:
            raise ValueError(f"missing framework result array: {key}")
        start = start_match.end()
        next_positions = []
        for next_key in FRAMEWORK_KEYS[index + 1 :]:
            next_match = re.search(rf"\b{next_key}:\s*\[", text[start:])
            if next_match:
                next_positions.append(start + next_match.start())
                break
        agent_match = re.search(r'"codex":\s*\[', text[start:])
        if agent_match:
            next_positions.append(start + agent_match.start())
        end = min(next_positions) if next_positions else len(text)
        rows = []
        for match in _RESULT_ROW.finditer(text[start:end]):
            row = match.groupdict()
            row["pass"] = row["pass"] == "true"
            rows.append(row)
        results[key] = rows
    return results


def _git_revision(orbit_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=orbit_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _duration_seconds(value: str) -> float:
    match = re.fullmatch(r"(?:(\d+)m )?([\d.]+)s", value)
    if not match:
        raise ValueError(f"unsupported published duration: {value}")
    return int(match.group(1) or 0) * 60 + float(match.group(2))


def _runtime_gap(
    framework_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for framework, rows in framework_results.items():
        for row in rows:
            if not row["pass"]:
                continue
            pairs.append(
                {
                    "framework": framework,
                    "task": row["task"],
                    "ratio": _duration_seconds(row["runtime"])
                    / _duration_seconds(row["ref"]),
                }
            )
    if not pairs:
        raise ValueError("no valid published runtime/reference pairs")
    fastest = min(pairs, key=lambda pair: pair["ratio"])
    slowest = max(pairs, key=lambda pair: pair["ratio"])
    return {
        "pair_count": len(pairs),
        "min_ratio": round(fastest["ratio"], 2),
        "min_framework": fastest["framework"],
        "min_task": fastest["task"],
        "max_ratio": round(slowest["ratio"], 2),
        "max_framework": slowest["framework"],
        "max_task": slowest["task"],
        "claim_class": "survey_inference",
    }


def collect_survey(
    orbit_root: Path, annotations_path: Path
) -> dict[str, Any]:
    """Merge immutable official records with reviewed survey annotations."""
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    revision = _git_revision(orbit_root)
    expected = annotations["expected_source_commit"]
    if not revision.startswith(expected):
        raise ValueError(
            f"source revision {revision} does not match approved {expected}"
        )

    task_annotations = {
        row["id"]: row for row in annotations.pop("task_annotations")
    }
    tasks = collect_tasks(orbit_root)
    source_base = (
        f"https://github.com/sxzgroup/ORBIT-Q/blob/{revision}/"
    )
    for task in tasks:
        task_id = task["id"]
        reviewed = task_annotations.get(task_id)
        if reviewed is None:
            raise ValueError(f"missing reviewed annotation for task {task_id}")
        config_override = reviewed.pop("config_override", {})
        task["config"].update(config_override)
        task.update(reviewed)
        task["source_url"] = source_base + task["source_path"]

    framework_results = collect_framework_results(orbit_root)
    data = {
        "metadata": {
            "title": "ORBIT-Q Task Atlas",
            "generated_from": "ORBIT-Q repository root",
            "source_commit": revision,
            "task_count": len(tasks),
            "evidence_classes": sorted(EVIDENCE_CLASSES),
        },
        "tasks": tasks,
        "framework_results": framework_results,
        "runtime_gap": _runtime_gap(framework_results),
        **annotations,
    }
    data["framework_summary"] = [
        {
            "id": key,
            "name": data["framework_names"][key],
            "passes": sum(
                1 for row in data["framework_results"][key] if row["pass"]
            ),
            "total": len(data["framework_results"][key]),
        }
        for key in FRAMEWORK_KEYS
    ]
    errors = validate_survey(data)
    if errors:
        raise ValueError("invalid survey: " + "; ".join(errors))
    return data


def validate_survey(data: dict[str, Any]) -> list[str]:
    """Return invariant violations without mutating the dataset."""
    errors: list[str] = []
    task_ids = [task.get("id") for task in data.get("tasks", [])]
    if len(task_ids) != len(set(task_ids)):
        errors.append("task ids must be unique")
    if task_ids != EXPECTED_TASK_IDS:
        errors.append("expected task ids 01 through 12")

    cluster_ids: list[str] = []
    for cluster in data.get("clusters", []):
        cluster_ids.extend(cluster.get("task_ids", []))
        if cluster.get("claim_class") != "survey_inference":
            errors.append("clusters must be labeled survey_inference")
    if task_ids == EXPECTED_TASK_IDS and sorted(cluster_ids) != EXPECTED_TASK_IDS:
        errors.append("each task must appear in exactly one cluster")

    framework_results = data.get("framework_results", {})
    if task_ids == EXPECTED_TASK_IDS:
        if set(framework_results) != set(FRAMEWORK_KEYS):
            errors.append("expected exactly four framework result sets")
        for key, rows in framework_results.items():
            row_ids = [row.get("task") for row in rows]
            if row_ids != EXPECTED_TASK_IDS:
                errors.append(f"{key} must contain tasks 01 through 12 once")
            if any(not isinstance(row.get("pass"), bool) for row in rows):
                errors.append(f"{key} pass values must be boolean")

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"claim_class", "evidence_class"}:
                    if child not in EVIDENCE_CLASSES:
                        errors.append(f"unknown evidence class: {child}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    for source in data.get("sources", []):
        if not source.get("url"):
            errors.append("source URLs are required")
    return list(dict.fromkeys(errors))


def write_survey(data: dict[str, Any], output: Path) -> None:
    """Write deterministic, human-readable UTF-8 JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbit-q-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect_survey(args.orbit_q_root, args.annotations)
    write_survey(data, args.output)
    cell_count = sum(map(len, data["framework_results"].values()))
    print(f"{len(data['tasks'])} tasks, {cell_count} framework cells")


if __name__ == "__main__":
    main()
