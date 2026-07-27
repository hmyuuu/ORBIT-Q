"""Translate ``survey.json`` into the generic offline report document."""

from __future__ import annotations

import json
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent


def _evidence_label(value: str) -> str:
    return {
        "official": "Official",
        "literature": "Literature",
        "survey_inference": "Survey inference",
    }[value]


def _pass_pattern(data: dict, task_id: str) -> str:
    symbols = []
    for summary in data["framework_summary"]:
        result = next(
            row
            for row in data["framework_results"][summary["id"]]
            if row["task"] == task_id
        )
        symbols.append("✓" if result["pass"] else "×")
    return " ".join(symbols)


def build_report_document(data: dict) -> dict:
    """Build the six-section renderer document from one survey dataset."""
    task_rows = [
        [
            f"{task['id']} · {task['title']}",
            task["system_size"],
            task["representation"],
            task["primary_success_criterion"],
            _pass_pattern(data, task["id"]),
            task["source_url"],
        ]
        for task in data["tasks"]
    ]
    cluster_rows = [
        [
            cluster["name"],
            ", ".join(cluster["task_ids"]),
            cluster["summary"],
            _evidence_label(cluster["claim_class"]),
        ]
        for cluster in data["clusters"]
    ]
    failure_rows = [
        [
            item["name"],
            item["description"],
            _evidence_label(item["claim_class"]),
        ]
        for item in data["failure_taxonomy"]
    ]
    cross_rows = [
        [
            next(
                cluster["name"]
                for cluster in data["clusters"]
                if cluster["id"] == row["cluster_id"]
            ),
            "; ".join(row["methods"]),
            ", ".join(row["reference_ids"]),
            _evidence_label(row["claim_class"]),
        ]
        for row in data["cross_method_map"]
    ]
    benchmark_rows = [
        [
            row["name"],
            row["scope"],
            row["contrast"],
            _evidence_label(row["claim_class"]),
        ]
        for row in data["related_benchmarks"]
    ]
    pass_summary = " · ".join(
        f"{row['name']} {row['passes']}/{row['total']}"
        for row in data["framework_summary"]
    )
    runtime_gap = data["runtime_gap"]
    runtime_summary = (
        f"{runtime_gap['min_ratio']:.2f}×–{runtime_gap['max_ratio']:.2f}× "
        f"the expert reference across {runtime_gap['pair_count']} valid "
        "published timing pairs [Survey inference]"
    )

    return {
        "title": "ORBIT-Q Task Atlas",
        "eyebrow": "Evidence-backed survey · 12 research workflows",
        "url": "https://github.com/sxzgroup/ORBIT-Q",
        "lede": (
            "What each benchmark task actually tests, which representation "
            "makes it tractable, and where agent-framework workflows fail."
        ),
        "sections": [
            {
                "title": "Executive finding",
                "blocks": [
                    {
                        "kind": "verdict",
                        "status": "good",
                        "label": "Agent-framework co-performance",
                        "why": data["executive_finding"]["text"],
                    },
                    {
                        "kind": "kv",
                        "pairs": [
                            ["Source snapshot", data["metadata"]["source_commit"]],
                            ["Task coverage", "12 / 12"],
                            ["Framework validity", pass_summary],
                            ["Submitted / expert runtime", runtime_summary],
                            [
                                "Evidence",
                                _evidence_label(
                                    data["executive_finding"]["claim_class"]
                                ),
                            ],
                        ],
                    },
                    {
                        "kind": "note",
                        "label": "Interpretation boundary",
                        "text": data["comparison_qualification"]["text"],
                    },
                    {
                        "kind": "note",
                        "label": "Competing-interest context",
                        "text": data["author_affiliation_caveat"]["text"],
                        "style": "info",
                    },
                ],
            },
            {
                "title": "Task landscape",
                "blocks": [
                    {
                        "kind": "table",
                        "columns": [
                            "Task",
                            "System",
                            "Required representation",
                            "Primary evaluator contract",
                            "TC · PL · TQ · MQ",
                            "Source",
                        ],
                        "rows": task_rows,
                        "widths": ["17%", "9%", "20%", "32%", "8%", "14%"],
                    },
                    {
                        "kind": "note",
                        "label": "Reading the pattern",
                        "text": (
                            "Symbols follow TensorCircuit-NG, PennyLane, "
                            "TorchQuantum, and MindQuantum. ✓ and × redundantly "
                            "encode pass/fail without relying on color."
                        ),
                    },
                ],
            },
            {
                "title": "Interpreted task clusters",
                "blocks": [
                    {
                        "kind": "verdict",
                        "status": "warn",
                        "label": "Survey taxonomy",
                        "why": (
                            "ORBIT-Q defines 12 workflows; these five clusters "
                            "are an analytical organization, not an official taxonomy."
                        ),
                    },
                    {
                        "kind": "table",
                        "columns": ["Cluster", "Tasks", "Interpretation", "Evidence"],
                        "rows": cluster_rows,
                    },
                ],
            },
            {
                "title": "Failure anatomy",
                "blocks": [
                    {
                        "kind": "figures",
                        "items": [
                            {
                                "src": "framework_heatmap.png",
                                "caption": (
                                    "Published framework-axis validity for the "
                                    "fixed Codex agent. Blue ✓ cells passed the "
                                    "full validity pipeline; orange × cells did "
                                    "not. This is agent-mediated evidence, not "
                                    "a proof of absolute framework impossibility."
                                ),
                            }
                        ],
                    },
                    {
                        "kind": "table",
                        "columns": ["Failure class", "Operational meaning", "Evidence"],
                        "rows": failure_rows,
                    },
                    {
                        "kind": "note",
                        "label": "Qualification",
                        "text": data["comparison_qualification"]["text"],
                    },
                ],
            },
            {
                "title": "Cross-method map",
                "blocks": [
                    {
                        "kind": "note",
                        "label": "Policy distinction",
                        "text": (
                            "A method can solve the physics without satisfying "
                            "ORBIT-Q's framework-native policy. External solvers "
                            "are alternatives or references, not automatically valid submissions."
                        ),
                    },
                    {
                        "kind": "table",
                        "columns": [
                            "Survey cluster",
                            "Credible alternative methods",
                            "References",
                            "Evidence",
                        ],
                        "rows": cross_rows,
                    },
                ],
            },
            {
                "title": "Benchmark position and roadmap",
                "blocks": [
                    {
                        "kind": "table",
                        "columns": ["Benchmark", "Scope", "Contrast with ORBIT-Q", "Evidence"],
                        "rows": benchmark_rows,
                    },
                    {
                        "kind": "list",
                        "title": "Prioritized roadmap · Survey inference",
                        "items": [
                            f"{row['priority']}. {row['text']}"
                            for row in data["roadmap"]
                        ],
                    },
                    {
                        "kind": "card",
                        "title": f"Citations and provenance · {len(data['references'])} references",
                        "blocks": [
                            {
                                "kind": "list",
                                "items": [
                                    f"[{_evidence_label(ref['evidence_class'])}] "
                                    f"{ref['title']} — {ref['url']}"
                                    for ref in data["references"]
                                ],
                            }
                        ],
                    },
                ],
            },
        ],
    }


def main() -> None:
    data = json.loads((REPORT_DIR / "survey.json").read_text(encoding="utf-8"))
    document = build_report_document(data)
    output = REPORT_DIR / "report.json"
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
