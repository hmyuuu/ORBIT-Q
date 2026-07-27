import json
from pathlib import Path

from reports.orbit_q_task_survey.build_report import build_report_document
from reports.orbit_q_task_survey.figures import render_heatmap


ROOT = Path(__file__).resolve().parents[2]


def load_survey():
    return json.loads(
        (ROOT / "reports/orbit_q_task_survey/survey.json").read_text(
            encoding="utf-8"
        )
    )


def test_report_has_six_sections_and_all_tasks():
    doc = build_report_document(load_survey())
    assert [section["title"] for section in doc["sections"]] == [
        "Executive finding",
        "Task landscape",
        "Interpreted task clusters",
        "Failure anatomy",
        "Cross-method map",
        "Benchmark position and roadmap",
    ]
    task_table = doc["sections"][1]["blocks"][0]
    assert task_table["kind"] == "table"
    assert len(task_table["rows"]) == 12


def test_report_marks_inference_and_cites_heatmap():
    encoded = json.dumps(build_report_document(load_survey()), ensure_ascii=False)
    assert "Survey inference" in encoded
    assert "framework_heatmap.png" in encoded
    assert "agent-mediated" in encoded.lower()
    assert "Submitted / expert runtime" in encoded
    assert "1.13×–21.81×" in encoded
    assert "tasks/challenge-01/instruction.md" in encoded


def test_heatmap_writes_raster_and_vector_outputs(tmp_path: Path):
    png, pdf = render_heatmap(load_survey(), tmp_path)
    assert png.exists() and png.stat().st_size > 1_000
    assert pdf.exists() and pdf.stat().st_size > 1_000
