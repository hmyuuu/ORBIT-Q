"""Publication-quality figures derived from the committed survey snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


REPORT_DIR = Path(__file__).resolve().parent


def render_heatmap(
    data: dict, output_dir: Path
) -> tuple[Path, Path]:
    """Render a colorblind-safe task-by-framework pass/fail heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frameworks = [row["id"] for row in data["framework_summary"]]
    names = [row["name"] for row in data["framework_summary"]]
    task_ids = [task["id"] for task in data["tasks"]]
    matrix = np.array(
        [
            [1 if row["pass"] else 0 for row in data["framework_results"][key]]
            for key in frameworks
        ],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(11.4, 4.3), constrained_layout=True)
    fig.patch.set_facecolor("#F7F3EA")
    ax.set_facecolor("#F7F3EA")
    # Okabe-Ito blue and vermillion: colorblind-safe, plus redundant symbols.
    cmap = ListedColormap(["#D55E00", "#0072B2"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            passed = bool(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                "✓" if passed else "×",
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold",
            )

    ax.set_xticks(range(len(task_ids)), task_ids)
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(axis="both", length=0, labelsize=10, pad=8)
    ax.set_xlabel("ORBIT-Q task", fontsize=10, labelpad=10)
    ax.set_title(
        "Agent-mediated framework validity across 12 ORBIT-Q tasks",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#12233F",
        pad=18,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(task_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="#F7F3EA", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.99,
        0.01,
        "✓ valid submission   × failed validity pipeline",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4A5568",
    )

    png_path = output_dir / "framework_heatmap.png"
    pdf_path = output_dir / "framework_heatmap.pdf"
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    data = json.loads((REPORT_DIR / "survey.json").read_text(encoding="utf-8"))
    png, pdf = render_heatmap(data, REPORT_DIR)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
