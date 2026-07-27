import { describe, expect, it } from "vitest";
import dataJson from "../../survey.json";
import {
  filterTasks,
  frameworkPassCounts,
  getTask,
  type SurveyData,
} from "./survey";


const data = dataJson as unknown as SurveyData;


describe("survey helpers", () => {
  it("filters the three scalability tasks", () => {
    expect(filterTasks(data.tasks, "scalability")).toHaveLength(3);
    expect(filterTasks(data.tasks, "scalability").map((task) => task.id)).toEqual([
      "08",
      "09",
      "10",
    ]);
  });

  it("narrows by representation, workflow, and scaling pressure", () => {
    expect(
      filterTasks(data.tasks, "all", { representation: "mps" }).map(
        (task) => task.id,
      ),
    ).toEqual(["01", "12"]);
    expect(
      filterTasks(data.tasks, "all", { workflow: "cooling" }).map(
        (task) => task.id,
      ),
    ).toEqual(["03", "05"]);
    expect(
      filterTasks(data.tasks, "all", { scaling: "Extreme" }).map(
        (task) => task.id,
      ),
    ).toEqual(["08", "09", "10"]);
  });

  it("locates the 512-qubit light-cone task", () => {
    expect(getTask(data.tasks, "09")?.title).toContain("Light-Cone");
    expect(getTask(data.tasks, "09")?.system_size).toBe("512 qubits");
  });

  it("computes the published framework pass totals", () => {
    expect(frameworkPassCounts(data)).toEqual({
      tensorcircuit: 10,
      pennylane: 8,
      torchquantum: 4,
      mindquantum: 4,
    });
  });
});
