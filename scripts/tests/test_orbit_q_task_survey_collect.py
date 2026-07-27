import json
from pathlib import Path

from reports.orbit_q_task_survey.collect import (
    collect_framework_results,
    collect_tasks,
    validate_survey,
)


def test_collect_tasks_reads_title_goal_config_and_criteria(tmp_path: Path):
    task = tmp_path / "tasks" / "challenge-01"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text(
        "# Problem 1: Test workflow\n\n"
        "## Goal\n\nUse an MPS input.\n\n"
        "## Fixed Problem Configuration\n\n"
        "```python\n{\"n_qubits\": 32, \"max_steps\": 500}\n```\n\n"
        "## Passing Criteria\n\n- fidelity >= 0.85\n",
        encoding="utf-8",
    )
    assert collect_tasks(tmp_path) == [
        {
            "id": "01",
            "title": "Test workflow",
            "goal": "Use an MPS input.",
            "config": {"n_qubits": 32, "max_steps": 500},
            "passing_criteria": ["fidelity >= 0.85"],
            "source_path": "tasks/challenge-01/instruction.md",
        }
    ]


def test_collect_framework_results_parses_first_four_framework_arrays(
    tmp_path: Path,
):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "app.js").write_text(
        'const detailsData = {tensorcircuit: [{ task: "01", pass: true, '
        'runtime: "1.0s", ref: "0.5s", wall: "2s", tokens: "10", '
        'cost: "$0.01", notes: "" }], '
        'pennylane: [{ task: "01", pass: false, runtime: "n/a", '
        'ref: "0.5s", wall: "2s", tokens: "10", cost: "$0.01", '
        'notes: "missing primitive" }], '
        'torchquantum: [], mindquantum: [], "codex": []};',
        encoding="utf-8",
    )
    rows = collect_framework_results(tmp_path)
    assert rows["tensorcircuit"][0]["pass"] is True
    assert rows["pennylane"][0]["notes"] == "missing primitive"


def test_validate_survey_rejects_duplicate_or_unclassified_tasks():
    data = {
        "tasks": [{"id": "01"}, {"id": "01"}],
        "clusters": [{"id": "a", "task_ids": ["01"]}],
        "framework_results": {},
        "sources": [],
    }
    errors = validate_survey(data)
    assert "task ids must be unique" in errors
    assert "expected task ids 01 through 12" in errors


def test_committed_snapshot_has_exactly_12_tasks_and_48_framework_cells():
    root = Path(__file__).resolve().parents[2]
    data = json.loads(
        (root / "reports/orbit_q_task_survey/survey.json").read_text(
            encoding="utf-8"
        )
    )
    assert [task["id"] for task in data["tasks"]] == [
        f"{number:02d}" for number in range(1, 13)
    ]
    assert sum(map(len, data["framework_results"].values())) == 48
    assert data["metadata"]["generated_from"] == "ORBIT-Q repository root"
    assert data["runtime_gap"] == {
        "claim_class": "survey_inference",
        "max_framework": "mindquantum",
        "max_ratio": 21.81,
        "max_task": "03",
        "min_framework": "tensorcircuit",
        "min_ratio": 1.13,
        "min_task": "05",
        "pair_count": 26,
    }
    assert validate_survey(data) == []
