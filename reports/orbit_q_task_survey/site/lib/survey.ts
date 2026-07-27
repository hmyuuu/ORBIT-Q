import dataJson from "../../survey.json";


export type EvidenceClass = "official" | "literature" | "survey_inference";

export type CrossMethod = {
  method: string;
  reference_ids: string[];
  claim_class: EvidenceClass;
};

export type SurveyTask = {
  id: string;
  title: string;
  goal: string;
  config: Record<string, unknown>;
  passing_criteria: string[];
  source_path: string;
  source_url: string;
  cluster_id: string;
  workflow: string;
  system_size: string;
  dominant_workload: string;
  representation: string;
  differentiability_pressure: string;
  scaling_pressure: string;
  primary_success_criterion: string;
  cross_methods: CrossMethod[];
  claim_class: EvidenceClass;
};

export type Cluster = {
  id: string;
  name: string;
  task_ids: string[];
  summary: string;
  claim_class: EvidenceClass;
};

export type FrameworkResult = {
  task: string;
  pass: boolean;
  runtime: string;
  ref: string;
  notes: string;
};

export type SurveyData = {
  metadata: {
    source_commit: string;
    task_count: number;
  };
  tasks: SurveyTask[];
  clusters: Cluster[];
  framework_names: Record<string, string>;
  framework_results: Record<string, FrameworkResult[]>;
  framework_summary: Array<{
    id: string;
    name: string;
    passes: number;
    total: number;
  }>;
  runtime_gap: {
    pair_count: number;
    min_ratio: number;
    min_framework: string;
    min_task: string;
    max_ratio: number;
    max_framework: string;
    max_task: string;
    claim_class: EvidenceClass;
  };
  executive_finding: { text: string; claim_class: EvidenceClass };
  comparison_qualification: { text: string; claim_class: EvidenceClass };
  author_affiliation_caveat: { text: string; claim_class: EvidenceClass };
  failure_taxonomy: Array<{
    id: string;
    name: string;
    description: string;
    claim_class: EvidenceClass;
  }>;
  cross_method_map: Array<{
    cluster_id: string;
    methods: string[];
    reference_ids: string[];
    claim_class: EvidenceClass;
  }>;
  related_benchmarks: Array<{
    name: string;
    scope: string;
    contrast: string;
    reference_id: string;
    claim_class: EvidenceClass;
  }>;
  roadmap: Array<{
    priority: number;
    text: string;
    claim_class: EvidenceClass;
  }>;
  references: Array<{
    id: string;
    title: string;
    url: string;
    arxiv?: string;
    doi?: string;
    evidence_class: EvidenceClass;
  }>;
  sources: Array<{
    id: string;
    title: string;
    url: string;
    evidence_class: EvidenceClass;
  }>;
};

export type TaskFilters = {
  representation?: string;
  workflow?: string;
  scaling?: string;
};

const EXPECTED_TASK_IDS = Array.from(
  { length: 12 },
  (_, index) => String(index + 1).padStart(2, "0"),
);

export const surveyData = dataJson as unknown as SurveyData;
const sourceIds = surveyData.tasks.map((task) => task.id);
if (
  sourceIds.length !== EXPECTED_TASK_IDS.length ||
  sourceIds.some((id, index) => id !== EXPECTED_TASK_IDS[index])
) {
  throw new Error("ORBIT-Q survey must contain task IDs 01 through 12 once");
}

export function filterTasks(
  tasks: SurveyTask[],
  clusterId: string,
  filters: TaskFilters = {},
): SurveyTask[] {
  const representation = filters.representation?.trim().toLowerCase() ?? "";
  const workflow = filters.workflow?.trim().toLowerCase() ?? "";
  const scaling = filters.scaling?.trim().toLowerCase() ?? "";
  return tasks.filter((task) => {
    const inCluster = clusterId === "all" || task.cluster_id === clusterId;
    return (
      inCluster &&
      (!representation || task.representation.toLowerCase().includes(representation)) &&
      (!workflow || task.workflow.toLowerCase().includes(workflow)) &&
      (!scaling || task.scaling_pressure.toLowerCase().startsWith(scaling))
    );
  });
}

export function getTask(
  tasks: SurveyTask[],
  id: string,
): SurveyTask | undefined {
  return tasks.find((task) => task.id === id);
}

export function frameworkPassCounts(data: SurveyData): Record<string, number> {
  return Object.fromEntries(
    Object.entries(data.framework_results).map(([framework, rows]) => [
      framework,
      rows.filter((row) => row.pass).length,
    ]),
  );
}

export function evidenceLabel(value: EvidenceClass): string {
  return {
    official: "Official",
    literature: "Literature",
    survey_inference: "Survey inference",
  }[value];
}
