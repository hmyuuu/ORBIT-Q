import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import dataJson from "../../survey.json";
import { SurveyAtlas } from "./SurveyAtlas";
import type { SurveyData } from "../lib/survey";


const data = dataJson as unknown as SurveyData;


describe("SurveyAtlas", () => {
  it("filters to scalability tasks and opens task 09", () => {
    render(<SurveyAtlas data={data} />);

    fireEvent.click(
      screen.getByRole("button", { name: /scalability & representation/i }),
    );
    const explorer = screen.getByLabelText("Task explorer");
    expect(within(explorer).getByRole("button", { name: /task 08/i })).toBeVisible();
    expect(within(explorer).getByRole("button", { name: /task 09/i })).toBeVisible();
    expect(within(explorer).getByRole("button", { name: /task 10/i })).toBeVisible();
    expect(within(explorer).queryByRole("button", { name: /task 01/i })).toBeNull();

    fireEvent.click(within(explorer).getByRole("button", { name: /task 09/i }));
    const detail = screen.getByLabelText("Selected task details");
    expect(within(detail).getByText("512 qubits")).toBeVisible();
    expect(within(detail).getByText("Official")).toBeVisible();
  });

  it("uses survey framework names and supports representation filtering", () => {
    const customData = {
      ...data,
      framework_summary: data.framework_summary.map((framework, index) =>
        index === 0 ? { ...framework, name: "Dataset Framework Name" } : framework,
      ),
    };
    render(<SurveyAtlas data={customData} />);

    expect(screen.getAllByText("Dataset Framework Name").length).toBeGreaterThan(0);
    fireEvent.change(screen.getByLabelText("Representation contains"), {
      target: { value: "MPS" },
    });
    const explorer = screen.getByLabelText("Task explorer");
    expect(within(explorer).getByRole("button", { name: /task 01/i })).toBeVisible();
    expect(within(explorer).getByRole("button", { name: /task 12/i })).toBeVisible();
    expect(within(explorer).queryByRole("button", { name: /task 02/i })).toBeNull();

    fireEvent.change(screen.getByLabelText("Representation contains"), {
      target: { value: "no-such-representation" },
    });
    expect(screen.getByText("No tasks match all active filters.")).toBeVisible();
    expect(screen.queryByLabelText("Selected task details")).toBeNull();
  });
});
