"use client";

import Image from "next/image";
import { useMemo, useState } from "react";
import {
  evidenceLabel,
  filterTasks,
  type TaskFilters,
  type SurveyData,
} from "../lib/survey";


type Props = { data: SurveyData };


function EvidenceBadge({ value }: { value: "official" | "literature" | "survey_inference" }) {
  return <span className={`evidence evidence-${value}`}>{evidenceLabel(value)}</span>;
}


export function SurveyAtlas({ data }: Props) {
  const [clusterId, setClusterId] = useState("all");
  const [selectedTaskId, setSelectedTaskId] = useState("01");
  const [representationQuery, setRepresentationQuery] = useState("");
  const [workflowQuery, setWorkflowQuery] = useState("");
  const [scalingFilter, setScalingFilter] = useState("");
  const visibleTasks = useMemo(
    () =>
      filterTasks(data.tasks, clusterId, {
        representation: representationQuery,
        workflow: workflowQuery,
        scaling: scalingFilter,
      }),
    [clusterId, data.tasks, representationQuery, workflowQuery, scalingFilter],
  );
  const scalingOptions = useMemo(
    () =>
      Array.from(
        new Set(data.tasks.map((task) => task.scaling_pressure.split(/[ —]/, 1)[0])),
      ).sort(),
    [data.tasks],
  );
  const selectedTask =
    visibleTasks.find((task) => task.id === selectedTaskId) ?? visibleTasks[0];

  function chooseCluster(nextCluster: string) {
    setClusterId(nextCluster);
    const first = filterTasks(data.tasks, nextCluster, {
      representation: representationQuery,
      workflow: workflowQuery,
      scaling: scalingFilter,
    })[0];
    if (first) setSelectedTaskId(first.id);
  }

  function chooseFilters(next: TaskFilters) {
    const merged = {
      representation: next.representation ?? representationQuery,
      workflow: next.workflow ?? workflowQuery,
      scaling: next.scaling ?? scalingFilter,
    };
    if (next.representation !== undefined) setRepresentationQuery(next.representation);
    if (next.workflow !== undefined) setWorkflowQuery(next.workflow);
    if (next.scaling !== undefined) setScalingFilter(next.scaling);
    const first = filterTasks(data.tasks, clusterId, merged)[0];
    if (first) setSelectedTaskId(first.id);
  }

  return (
    <main>
      <nav className="topbar" aria-label="Report sections">
        <a className="wordmark" href="#overview" aria-label="ORBIT-Q Task Atlas home">
          <span className="wordmark-mark">OQ</span>
          <span>Task Atlas</span>
        </a>
        <div className="navlinks">
          <a href="#tasks">Tasks</a>
          <a href="#coverage">Coverage</a>
          <a href="#methods">Methods</a>
          <a href="#evidence">Evidence</a>
        </div>
      </nav>

      <header className="hero" id="overview">
        <div className="hero-copy">
          <div className="eyebrow"><span /> ORBIT-Q · source survey</div>
          <h1>Twelve workflows.<br /><em>One representation problem.</em></h1>
          <p className="hero-lede">{data.executive_finding.text}</p>
          <div className="hero-actions">
            <a className="primary-action" href="#tasks">Explore the tasks <span>↘</span></a>
            <a className="text-action" href="#evidence">Audit the evidence</a>
          </div>
        </div>
        <aside className="hero-panel" aria-label="Published framework pass totals">
          <div className="panel-kicker">Framework axis · fixed agent</div>
          <div className="score-grid">
            {data.framework_summary.map((framework) => (
              <div className="score" key={framework.id}>
                <strong>{framework.passes}<small>/12</small></strong>
                <span>{framework.name}</span>
              </div>
            ))}
          </div>
          <div className="runtime-gap">
            <div>
              <strong>{data.runtime_gap.min_ratio.toFixed(2)}×–{data.runtime_gap.max_ratio.toFixed(2)}×</strong>
              <span>submitted / expert runtime</span>
            </div>
            <p>Across {data.runtime_gap.pair_count} valid published timing pairs.</p>
            <EvidenceBadge value="survey_inference" />
          </div>
          <div className="panel-foot">
            <EvidenceBadge value="official" />
            <span>Published validity totals</span>
          </div>
        </aside>
        <div className="hero-note">
          <span>Interpretation boundary</span>
          <p>{data.comparison_qualification.text}</p>
        </div>
      </header>

      <section className="section section-paper" id="tasks" aria-labelledby="tasks-title">
        <div className="section-heading">
          <div><span className="section-index">01</span><p className="section-kicker">Task explorer</p></div>
          <div>
            <h2 id="tasks-title">A benchmark of interfaces,<br />not just algorithms.</h2>
            <p>Five survey-inferred clusters organize the 12 official workflows. Select a task to inspect its evaluator contract and computational pressure.</p>
          </div>
        </div>

        <div className="filter-strip" aria-label="Task cluster filters">
          <button type="button" aria-pressed={clusterId === "all"} onClick={() => chooseCluster("all")}>All tasks <span>12</span></button>
          {data.clusters.map((cluster) => (
            <button key={cluster.id} type="button" aria-pressed={clusterId === cluster.id} onClick={() => chooseCluster(cluster.id)}>
              {cluster.name.replace("Tensor-network / circuit interfaces", "Tensor-network interfaces").replace("Constrained variational state engineering", "Variational engineering").replace("Non-unitary, cooling & open-system workflows", "Non-unitary workflows").replace("Scalability & representation stress tests", "Scalability & representation").replace("Many-body topology & physical diagnostics", "Many-body topology")}
              <span>{cluster.task_ids.length}</span>
            </button>
          ))}
        </div>
        <div className="facet-filters" aria-label="Task attribute filters">
          <label>
            <span>Representation contains</span>
            <input
              type="search"
              value={representationQuery}
              placeholder="e.g. MPS"
              onChange={(event) => chooseFilters({ representation: event.target.value })}
            />
          </label>
          <label>
            <span>Workflow contains</span>
            <input
              type="search"
              value={workflowQuery}
              placeholder="e.g. cooling"
              onChange={(event) => chooseFilters({ workflow: event.target.value })}
            />
          </label>
          <label>
            <span>Scaling pressure</span>
            <select
              value={scalingFilter}
              onChange={(event) => chooseFilters({ scaling: event.target.value })}
            >
              <option value="">All levels</option>
              {scalingOptions.map((value) => <option value={value} key={value}>{value}</option>)}
            </select>
          </label>
        </div>

        <div className="explorer-layout">
          <div className="task-grid" aria-label="Task explorer">
            {visibleTasks.map((task) => (
              <button
                className="task-card"
                data-active={selectedTask?.id === task.id}
                key={task.id}
                type="button"
                aria-label={`Task ${task.id}: ${task.title}`}
                aria-pressed={selectedTask?.id === task.id}
                onClick={() => setSelectedTaskId(task.id)}
              >
                <span className="task-number">{task.id}</span>
                <span className="task-title">{task.title}</span>
                <span className="task-size">{task.system_size}</span>
              </button>
            ))}
            {visibleTasks.length === 0 && (
              <p className="empty-tasks">No tasks match all active filters.</p>
            )}
          </div>

          {selectedTask && (
          <article className="task-detail" aria-label="Selected task details" aria-live="polite">
            <div className="detail-head">
              <div><span>Task {selectedTask.id}</span><h3>{selectedTask.title}</h3></div>
              <EvidenceBadge value="official" />
            </div>
            <p className="detail-goal">{selectedTask.goal}</p>
            <dl className="detail-grid">
              <div><dt>System</dt><dd>{selectedTask.system_size}</dd></div>
              <div><dt>Dominant workload</dt><dd>{selectedTask.dominant_workload}</dd></div>
              <div><dt>Required representation</dt><dd>{selectedTask.representation}</dd></div>
              <div><dt>Differentiability</dt><dd>{selectedTask.differentiability_pressure}</dd></div>
              <div><dt>Scaling pressure</dt><dd>{selectedTask.scaling_pressure}</dd></div>
              <div><dt>Evaluator contract</dt><dd>{selectedTask.primary_success_criterion}</dd></div>
            </dl>
            <div className="task-results" aria-label={`Framework results for task ${selectedTask.id}`}>
              {data.framework_summary.map((framework) => {
                const result = data.framework_results[framework.id].find((row) => row.task === selectedTask.id)!;
                return (
                  <span className={result.pass ? "result-pass" : "result-fail"} key={framework.id}>
                    <b>{result.pass ? "✓" : "×"} {framework.name}</b>
                    <small>{result.runtime} submitted · {result.ref} expert</small>
                  </span>
                );
              })}
            </div>
            <a className="source-link" href={selectedTask.source_url} target="_blank" rel="noreferrer">Read the immutable task source <span>↗</span></a>
          </article>
          )}
        </div>
      </section>

      <section className="section section-ink" id="coverage" aria-labelledby="coverage-title">
        <div className="section-heading light">
          <div><span className="section-index">02</span><p className="section-kicker">Framework coverage</p></div>
          <div><h2 id="coverage-title">Failure is a pipeline outcome,<br />not an impossibility proof.</h2><p>The published map combines functional evaluation, static policy, semantic audit, and human adjudication.</p></div>
        </div>
        <div className="coverage-layout">
          <figure className="heatmap-card">
            <Image
              src="/framework-heatmap.png"
              alt="Pass and fail heatmap for four frameworks across ORBIT-Q tasks 01 through 12"
              width={3420}
              height={1290}
              sizes="(max-width: 1050px) 86vw, 56vw"
            />
            <figcaption><EvidenceBadge value="official" /> Agent-mediated validity. Blue ✓ means a valid submitted artifact; orange × means the full validity pipeline was not passed.</figcaption>
          </figure>
          <aside className="failure-list">
            <p className="mini-heading">Failure anatomy · <span>Survey inference</span></p>
            {data.failure_taxonomy.map((item, index) => (
              <div className="failure-item" key={item.id}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <div><h3>{item.name}</h3><p>{item.description}</p></div>
              </div>
            ))}
          </aside>
        </div>
      </section>

      <section className="section section-paper" id="methods" aria-labelledby="methods-title">
        <div className="section-heading">
          <div><span className="section-index">03</span><p className="section-kicker">Cross-method map</p></div>
          <div><h2 id="methods-title">The physics method and the<br />benchmark-valid method diverge.</h2><p>An external solver can be scientifically credible without satisfying ORBIT-Q’s framework-native policy.</p></div>
        </div>
        <div className="method-grid">
          {data.cross_method_map.map((item) => {
            const cluster = data.clusters.find((candidate) => candidate.id === item.cluster_id)!;
            return (
              <article className="method-card" key={item.cluster_id}>
                <div className="method-top"><span>{cluster.task_ids.join(" · ")}</span><EvidenceBadge value="survey_inference" /></div>
                <h3>{cluster.name}</h3>
                <p>{cluster.summary}</p>
                <ul>{item.methods.map((method) => <li key={method}>{method}</li>)}</ul>
                <div className="reference-ids">Refs · {item.reference_ids.join(" · ")}</div>
              </article>
            );
          })}
        </div>
      </section>

      <section className="section section-sand" id="position" aria-labelledby="position-title">
        <div className="section-heading">
          <div><span className="section-index">04</span><p className="section-kicker">Benchmark position</p></div>
          <div><h2 id="position-title">From code completion to<br />research workflow fidelity.</h2><p>Neighboring benchmarks illuminate different slices of quantum and scientific coding.</p></div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Benchmark</th><th>Primary scope</th><th>What ORBIT-Q adds</th><th>Evidence</th></tr></thead>
            <tbody>{data.related_benchmarks.map((item) => (
              <tr key={item.name}><th>{item.name}</th><td>{item.scope}</td><td>{item.contrast}</td><td><EvidenceBadge value="literature" /></td></tr>
            ))}</tbody>
          </table>
        </div>
        <div className="roadmap">
          <div className="roadmap-title"><span>Priority</span><h3>Five extensions that make the benchmark harder to game—and easier to interpret.</h3><EvidenceBadge value="survey_inference" /></div>
          <ol>{data.roadmap.map((item) => <li key={item.priority}><span>0{item.priority}</span><p>{item.text}</p></li>)}</ol>
        </div>
      </section>

      <section className="section section-evidence" id="evidence" aria-labelledby="evidence-title">
        <div className="section-heading light">
          <div><span className="section-index">05</span><p className="section-kicker">Evidence &amp; provenance</p></div>
          <div><h2 id="evidence-title">Every claim has a lane.</h2><p>Official facts, external literature, and survey inference remain visibly distinct.</p></div>
        </div>
        <div className="provenance-grid">
          <article><EvidenceBadge value="official" /><h3>Repository &amp; paper</h3><p>Task statements, evaluator contracts, published result arrays, and the ORBIT-Q paper.</p></article>
          <article><EvidenceBadge value="literature" /><h3>Primary literature</h3><p>ArXiv records, publisher pages, and official software papers for benchmark and method comparisons.</p></article>
          <article><EvidenceBadge value="survey_inference" /><h3>Analytical synthesis</h3><p>The five clusters, failure taxonomy, cross-method mapping, and roadmap.</p></article>
        </div>
        <div className="snapshot">
          <div><span>Actual source commit</span><code>{data.metadata.source_commit}</code></div>
          <p>{data.author_affiliation_caveat.text}</p>
        </div>
        <div className="citations">
          <div className="citations-head"><h3>Citations</h3><span>{data.references.length} verified primary records</span></div>
          <ol>{data.references.map((reference) => (
            <li key={reference.id}><span>{reference.evidence_class === "official" ? "O" : "L"}</span><a href={reference.url} target="_blank" rel="noreferrer">{reference.title}</a><small>{reference.arxiv ? `arXiv:${reference.arxiv}` : reference.doi}</small></li>
          ))}</ol>
        </div>
      </section>

      <footer>
        <div className="wordmark"><span className="wordmark-mark">OQ</span><span>Task Atlas</span></div>
        <p>One structured dataset · 12 tasks · 48 framework cells</p>
        <a href="#overview">Back to top ↑</a>
      </footer>
    </main>
  );
}
