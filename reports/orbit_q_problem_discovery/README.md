# ORBIT-Q problem-discovery research loop

This package contains the first source-grounded design matrix and a gated,
resumable workflow for future TensorCircuit benchmark problems. It expands 50
scientific families across four resource scales and five operational
situations, producing 1,000 deterministic candidate contracts. These are 1,000
auditable regimes, not 1,000 independently sourced problem statements. The
pipeline then selects ten candidates for human design review.

The shortlist is **not** a claim that GPT-5.6-sol cannot solve these problems.
The frozen design rows retain `empirical_status=untested` as an
**at-discovery-time** field. Current expert and model runs live separately in
`empirical_evidence.json` and `empirical_evidence.md`, so new evidence does not
silently rewrite the 1,000-contract source snapshot. A model-hardness claim is
allowed only after expert feasibility, human approval, repeated frozen-model
trials, and human failure classification.

## What was sampled

The research pass used five complementary search strategies:

1. landscape mapping across algorithms, control, metrology, QEC, simulation,
   open systems, measurement, chemistry, dynamics, and verification;
2. cross-vocabulary search for the same computational obstruction under
   different subfield names;
3. cross-method search around TensorCircuit representations, AD, channels,
   dynamic circuits, and tensor-network contraction;
4. negative-result search for infeasible scaling, barren plateaus,
   ill-conditioning, and weak statistical certificates;
5. benchmark/dataset search for task structures and documented agent gaps.

Each candidate is scored from eight perspectives: expected agent difficulty,
TensorCircuit path confidence, scientific value, verifier feasibility, runtime
fit, determinism, novelty, and shortcut resistance. These are transparent
design priors recorded in `candidates.jsonl`; they are never mixed with model
trial results.

`research_protocol.json` defines the repeatable next-cycle stages: query
planning, immutable raw-result ingest, claim-level normalization, exact and
semantic deduplication, coverage sampling, scoring, and human source review.
The current snapshot used agent-assisted primary-source research; this package
does not silently perform network searches. A search adapter must preserve its
query log and raw response hashes under `.artifacts/problem-discovery/` before
new source records can be curated into `catalog.json`.

## Human-gated workflow

```mermaid
flowchart LR
    A["Primary sources and current-suite gaps"] --> B["50 normalized problem families"]
    B --> C["4 scales × 5 situations"]
    C --> D["1,000 scored candidate contracts"]
    D --> E["Diversity shortlist: 10"]
    E --> F{"Concept review: 3 roles"}
    F -->|approve| G["Public-API canary + gold + independent oracle"]
    F -->|reject| R["Revise or reject"]
    G --> H{"20-run verifier and resource gates"}
    H -->|pass| I{"Pilot review: 2 roles"}
    H -->|fail| R
    I -->|approve| J["Hash-bound dry-run manifest"]
    I -->|reject| R
    J --> K["5+ precommitted pilot trials"]
    K --> L{"Two-auditor failure review"}
    L -->|zero passes; substantive failures| N["Pilot hardness signal"]
    N --> O["Fresh 20+ trial confirmation"]
    O --> P{"Two-auditor confirmation review"}
    P -->|zero passes; all substantive| M["Protocol-scoped model-hard evidence"]
    L -->|infra/spec/API failure| R
```

The CLI requires distinct declared identities for the gate roles:

- `quantum_scientist`: physical meaning, conventions, and scientific value;
- `tensorcircuit_expert`: public pinned API and framework-native centrality;
- `verifier_engineer`: independent oracle, hidden tests, tolerances, and anti-cheat;
- `benchmark_owner`: protocol, resource envelope, and contamination controls;
- `independent_reviewer`: blind review of the frozen pilot package.

Prototype admission requires all of the following:

- public-API canary passes in the pinned image;
- expert baseline and independent oracle are separately hashed;
- verifier, prompt, image digest, and source commit are frozen;
- expert p95 runtime is at most 180 seconds;
- independent oracle runtime is at most 240 seconds;
- gold solution is at most 160 effective Python lines;
- verifier-only succeeds in at least 20 reproductions;
- at least 25 distinct seeds pass finished expert-only Harbor verifier runs;
- every expert run freezes the task, prompt, image, CPU/memory, and oracle mode;
- private expert verification disables the LLM audit and binds a zero provider-call budget;
- raw rewards and audit details prove that the private LLM audit was skipped;
- the evaluator emits a hash-bound admission record under one fixed threshold policy;
- measured peak memory fits the observed runner envelope.

Infrastructure, authentication, policy refusal, ambiguous specification, and
framework impossibility are excluded from model-hardness evidence. Runtime
alone is not a failure under the ORBIT-Q reward definition, although a missing
or excessive runtime blocks candidate admission here.

## Commands

Rebuild the deterministic corpus and report from the normalized catalog:

```bash
python3 scripts/run_problem_discovery.py build
python3 scripts/run_problem_discovery.py status
```

Run the resumable offline source-discovery stages with a reviewed query plan and
operator-provided raw-hit manifests:

```bash
python3 scripts/run_autoresearch_discovery.py plan \
  --plan .artifacts/problem-discovery/query-plan.json
python3 scripts/run_autoresearch_discovery.py run \
  --plan .artifacts/problem-discovery/query-plan.json \
  --raw-manifest .artifacts/problem-discovery/provider-results.json
```

This command performs deterministic ingest, normalization, exact/near
deduplication, and coverage sampling. It deliberately performs no network
search and never treats an operator-provided result as independently attested.
See `AUTORESEARCH_DISCOVERY.md` for the schemas, custody boundary, and
stage-by-stage resume commands.

Verify the sanitized empirical snapshot from its committed, file-by-file source
manifest. This mode works in a clean clone and does not need private job logs:

```bash
python3 scripts/build_problem_discovery_evidence.py --check
```

Operators with the ignored raw jobs can separately reconstruct and compare the
snapshot, including task files, rollouts, configs, and oracle transcripts:

```bash
python3 scripts/build_problem_discovery_evidence.py --check-raw
```

That ledger labels passing target-model pilots as hardness-blocking and keeps a
single substantive failure at the weaker `pilot_hardness_signal` level. It
does not convert exploratory runs into protocol-qualified hardness evidence.

Record concept approval one role at a time:

```bash
python3 scripts/run_problem_discovery.py review \
  --candidate CANDIDATE_ID \
  --gate concept \
  --role quantum_scientist \
  --decision approve \
  --reviewer REVIEWER \
  --note "reason and evidence"
```

`record-prototype` accepts one JSON evidence bundle under
`.artifacts/problem-discovery/`. It recomputes artifact and log hashes, p95
runtime, peak memory, and reproduction counts rather than accepting summary
flags. The bundle must contain 20 verifier runs, three independent-oracle runs,
two accepted valid implementation styles, and six rejected verifier mutations.
After all three concept reviews and the prototype gate pass, two pilot reviews
are recorded with the same `review` command using `--gate pilot`.

Authorization is deliberately a dry action. It writes a manifest and never
launches Harbor or a model:

```bash
python3 scripts/run_problem_discovery.py authorize \
  --candidate CANDIDATE_ID \
  --protocol-config .artifacts/problem-discovery/protocol.json \
  --stage pilot \
  --seeds 101,102,103,104,105 \
  --output .artifacts/problem-discovery/run-manifest.json
```

The manifest binds the reviewed candidate hash to the model, prompt hash,
container digest, source commit, evaluator, baseline, oracle, and reviewer
decisions. The protocol config additionally freezes provider/model identity,
reasoning effort, budgets, solver and Harbor versions, audit model, tool/network
policy, hardware class, exact task package, and a one-use seed schedule. Outputs
outside `.artifacts/problem-discovery/` or overwrites of an existing manifest
are rejected. Any reviewed content or gate change revokes the authorization.

After an approved Harbor run, `record-trial` imports a normalized result JSON
under `.artifacts/problem-discovery/`, verifies the raw Harbor job, config,
trial lock, job lock, solver transcript, and functional-output hashes, and
derives scores, execution status, resource bindings, and full case identity
from those machine artifacts. A separately typed score or supplemental operator
attestation cannot override the raw result. Local hashes provide tamper evidence,
not third-party identity proof; the two human auditors remain part of the trust
boundary. The importer does not infer why a run failed. A named reviewer must
then use `audit-trial` to classify that outcome. `hardness-status` counts only
unique, substantive failures on which two distinct failure auditors agree toward
the precommitted seed schedule; excluded, missing, extra, or disputed failures
invalidate the schedule. Any raw compound pass blocks a model-hard conclusion,
even before human audit. Five failures yield only a pilot signal. A fresh
confirmation manifest needs at least 20 valid trials before the stronger label.

```bash
python3 scripts/run_problem_discovery.py hardness-status \
  --candidate CANDIDATE_ID \
  --manifest-hash MANIFEST_HASH
```

Materialized prototypes use the separate candidate-only runner. It accepts
exactly one explicit staged task, always requests one Harbor trial, and is a
dry run unless `--execute` is supplied. Expert prequalification is explicitly
excluded from model-hardness evidence. The following public-canary preview is
non-executing and cannot become private prototype evidence; materialization
embeds the reviewed candidate contract into the exact task directory:

```bash
python3 scripts/materialize_candidate_task.py \
  --blueprint-dir reports/orbit_q_problem_discovery/blueprints/mixed_sld_qfim \
  --discovery-workspace reports/orbit_q_problem_discovery \
  --output-root .artifacts/problem-discovery/candidate-tasks
```

Post-snapshot reserve designs are tracked separately in
`post_snapshot_reserve_registry.json`. The registry self-hashes each entry and
the complete payload, pins the untouched discovery snapshot, and binds the
exact blueprint files while enforcing unique candidate, slug, and numeric
identities. Registered `design_only` reserves may be materialized
for provenance inspection, but the candidate runner rejects both expert and
model paths (including dry runs). Registration is not review, promotion, or
execution authorization, and it never appends or remaps a frozen candidate.

Preview one public development seed after resolving the image digest out of
band. The absence of `--execute` is intentional. Private expert runs must come
from a separately hashed sealed plan and exact human approval, not by copying
this command and adding an execution flag:

```bash
python3 scripts/run_harbor_candidate.py \
  --task-dir .artifacts/problem-discovery/candidate-tasks/candidate-mixed-sld-qfim \
  --workspace reports/orbit_q_problem_discovery \
  --expert-only \
  --candidate-seed 101021 \
  --docker-image challenge-benchmark-quantum-tensorcircuit:py311 \
  --container-image-digest sha256:REPLACE_WITH_64_LOWERCASE_HEX \
  --job-name preview-mixed-sld-qfim-public-seed-101021 \
  --expert-evidence-dir .artifacts/problem-discovery/expert-prequalification
```

An actual `--expert-only --execute` call additionally requires
`--expert-execution-approval ABSOLUTE_JSON`. The current-owner mode-0600 human
batch approval binds the sealed mode-0600 plan, owner-only reviewed driver,
candidate/source/task/evaluator/expert hashes, digest-addressed image, Harbor
binary path/hash/version, resources, disabled-audit policy, zero provider-call
budget, the exact runner-Python launcher/resolved target/target hash/environment
prefix/`pyvenv.cfg` hash, private output roots, and every exact ordinal/seed/job.
Both output
roots must already be nonsymlink mode-0700 directories. The reviewed driver
atomically creates its bound mode-0600 terminal start reservation before the
first case. The runner then consumes one mode-0600 per-job receipt with
`O_EXCL`, enforces strict ordinal order, requires successful bound evidence for
every predecessor, rejects every current/later artifact, and runs Harbor under
umask 077. A failed or crashed claimed case permanently blocks later cases.
Dry-run previews need no approval and never create a receipt.

The sealed driver must import and use
`build_expert_subprocess_env()` from `scripts/run_harbor_candidate.py` as the
environment for its driver-to-runner subprocess and must invoke the exact
hash-bound Harbor virtual-environment Python launcher, not the Python used to
prepare the plan. The runner recomputes the launcher path, resolved interpreter
path/bytes, environment prefix, and `pyvenv.cfg` bytes before a receipt can be
claimed. It uses the same environment mapping for `harbor --version` and the
Harbor execution. The helper copies only
`PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `TMPDIR`, `TMP`, `TEMP`, `LANG`,
`LC_ALL`, `LC_CTYPE`, `TZ`, `TERM`, `NO_COLOR`, `PYTHONUNBUFFERED`,
`XDG_RUNTIME_DIR`, and `__CF_USER_TEXT_ENCODING` when present, then forces the
repository `PYTHONPATH` and `PYTHONDONTWRITEBYTECODE=1`. Everything else is
dropped, including AWS/Bedrock, OpenAI, Anthropic, Google, Azure, Hugging Face,
proxy, and unrelated host variables. This exact exported helper is the
provider-free driver/runner contract; a blacklist or a copy of the allowlist is
not equivalent.

This is local unsigned custody: after a valid human approval and start
reservation exist, an exact same-owner invocation is indistinguishable from the
reviewed driver invoking the runner. The driver hash records reviewed
orchestration; it is not a cryptographic caller identity. The approval must
carry the exact acknowledgement
`I_ACCEPT_LOCAL_OPERATOR_CUSTODY_NOT_HOST_CONFIDENTIAL` and binds the policy
`local_operator_custody_private`. Private seeds traverse the runner/Harbor
argument vector and resolved verifier config, and therefore remain visible to
same-host process observers and the trusted operator. This workflow is not
host-adversary confidential and may run only on a dedicated, single-user,
trusted-operator host.

The approval hashes the reviewed runner, driver, materialized task, verifier
harness, Harbor entry-point bytes/version, and declared runtime bindings. It is
not complete code attestation for every transitive Python, Harbor, Docker, or
operating-system dependency; those local/runtime dependencies remain inside
the trusted-operator boundary.

Candidate 101 (`mixed-sld-qfim`) now declares the exact numerical admission
record required by this producer, but it still has no human approval or private
prequalification authorization. Candidate 109 (`robust-leakage-grape`) also
implements the record shape, but remains `HOLD`: its historical v1 expert screen
stopped at 22/23 passes, while the current v6 expert has only a 12/12 preserved
public feasibility result. The v6 task still derives hidden cases from
solver-visible seed state and lacks hostile-process isolation, so neither
revision is eligible for private prequalification or a solver trial. Candidate
115 (`noisy-ppt3-replica`) implements the convention but is a post-shortlist
reserve and cannot produce reviewed prototype evidence without explicit
promotion. Other candidates remain ineligible until their own metrics and
thresholds are reviewed and implemented; preflight rejects missing or HOLD
declarations rather than inventing a generic margin.

For frozen repeated trials, `--candidate-seed INTEGER` is passed only to the
verifier as `ORBIT_Q_CANDIDATE_SEED`; it is never added to the solver
environment. Candidate evaluators must explicitly consume that variable, and
the precommitted seed schedule must be expert-prequalified before a real model
run. A seed that violates a numerical admission margin is rejected rather than
counted as model failure. Prototype evidence schema v2 records at least 25
finished, unique expert-only Harbor jobs, not self-declared case JSON. For every
job the pipeline rehashes and reparses the raw Harbor result, resolved config,
trial lock, job lock, functional stdout, evaluator admission record, and audit
details. The reward carries numeric `llm_audit_skipped_score=1.0`, while the
audit-details record carries the exact boolean `llm_audit_skipped=true`. It
derives the job/pass/runtime, nonnegative `protocol_seed`, full lowercase
64-hex `case_digest`, numerical margins, and minimum margin; declared summaries
must match. The case identity must be the first nonempty functional-output line,
and exactly one final `Overall: PASS` must close the output. All jobs must share
one hash-derived threshold policy and must bind the frozen candidate, expert,
evaluator, task, prompt, image, and resource envelope. Authorization can select
only a subset of those records; their complete payloads are hashed into the
manifest and all seven raw files are rehashed and reparsed at authorization,
execution/reservation, trial import, and hardness summary.

The expert producer also freezes the verifier runtime surface. Before Harbor
starts and again while evidence is collected, it requires the staged
`score_submission.py`, `test.sh`, `static_policy.py`, and `audit_codex.py` to
match the repository templates byte-for-byte with their expected modes. It
does the same for `solution/solve.sh`, exact-recomputes `task.toml`, the
instruction, metadata, IDs, evaluator, expert, and admission marker from the
reviewed blueprint, and rejects every unexpected file or directory.
`PYTHONDONTWRITEBYTECODE=1` keeps the strict file allowlist stable. The canonical
verifier-harness manifest hash is carried through verifier environment, the
evaluator admission record, the raw result hash words, and recorded prototype
bindings. The collector and later prototype gates also recompute Harbor's local
task package digest and legacy `Task.checksum` from that same directory; a
well-formed but unrelated lock digest or result checksum is rejected.

A private expert verifier never constructs or installs the Codex runtime, loads
provider credentials, or sends source, seeds, digests, or output to an audit
model. A separate public/source-only policy review is required before later
promotion.

The evaluator emits only observed numerical values and fixed thresholds. The
shared verifier reads the full captured evaluator stdout, checks the case
identity, recomputes every strict pass margin and the threshold-policy hash, and
writes `expert-admission.json`. Its SHA-256 is encoded as eight integer reward
fields, so the ordinary raw Harbor trial result binds the exact record without
trusting a CLI or operator summary. The collector additionally verifies that
Harbor used its built-in Oracle agent, copied the exact reviewed expert file,
and locked the same materialized task directory whose evaluator and solution
hashes were reviewed.

These local evidence files and Harbor locks are hash-bound but not signed by a
remote attestation service. An operator with write access could fabricate a
self-consistent bundle. Human review must therefore verify provenance and job
acquisition independently; the pipeline guarantees internal consistency and
detects later mutation, not operator identity, wall-clock ordering, image-registry
custody, or hardware-backed origin. The reward-word binding is a deterministic
content link, not a cryptographic signature by Harbor or the container runtime.

All review-ledger mutations use one cross-process lock around the complete
read-modify-write transaction. Concurrent reviews, authorizations/revocations,
seed reservations, trial imports/pass vetoes, and audits therefore cannot
silently overwrite one another.

A model run additionally requires an active manifest. All model, image,
resource, prompt, timeout, tool, network, and audit settings are derived from
that manifest rather than accepted as command-line overrides:

```bash
python3 scripts/run_harbor_candidate.py \
  --task-dir .artifacts/problem-discovery/candidate-tasks/candidate-SLUG \
  --manifest .artifacts/problem-discovery/run-manifest.json \
  --candidate-seed 101 \
  --job-name pilot-CANDIDATE-seed-101
```

Each evaluator emits exactly one `orbit_q_case_identity` JSON record containing
the authorized base seed and full 64-hex public-configuration digest. Seed reuse,
digest drift, retries, parallel trials, extra mounts/hosts/tools, timeout
overrides, and unapproved environment fields fail closed during evidence import.
The evaluator emits and flushes that identity before submitted code runs, then
removes verifier seed variables before importing the solution. Static policy
also rejects environment, stack-frame, `/proc`, and test-path introspection.

The present implementation intentionally stops short of a final model-hardness
claim. Harbor/Codex does not expose a hard runtime token cap or immutable runtime
build IDs, and submitted code still runs in the evaluator process rather than a
separately sandboxed solution process. Pilot signals may guide research, but
`hardness-status` reports these as admission limitations and fails closed until
all three controls are machine-observed.

For candidates 101 and 109, the current expert-feasibility generators also
derive public configuration and hidden holdouts from one deterministic base
seed. NumPy PCG64 is reproducible, not a secrecy primitive. Trusted expert-only
prequalification may use that contract, but a solver pilot must first bind an
independent verifier-only hidden seed/key or an isolated precomputed holdout
artifact. A failure under the current shared-seed, same-process contract cannot
support a model-hardness conclusion.

## Repository boundary

Discovery records stay under `reports/orbit_q_problem_discovery/`. Raw search
downloads, private holdouts, generated staging tasks, and model job logs belong
under gitignored `.artifacts/problem-discovery/` or a temporary directory.
Nothing in this workflow writes to `tasks/challenge-01` through
`tasks/challenge-12`; promotion into the active Harbor suite is a later,
explicit human decision.

## Artifacts

- `catalog.json`: normalized sources, sampling axes, 50 families, and design priors;
- `candidates.jsonl`: all 1,000 candidate contracts and screening decisions;
- `shortlist.json`: machine-readable ten-candidate review set;
- `shortlist.md`: human review report with scientific rationale and sources;
- `NEXT_CYCLE_REVIEW.md`: empirical-lesson-adjusted priorities for the next
  human review without mutating the frozen shortlist;
- `post_snapshot_reserve_registry.json`: separately hashed, non-executable
  design bindings for post-snapshot reserve blueprints;
- `post_snapshot_reserve_registry.schema.json`: machine-readable registry
  shape and fail-closed `design_only` audit assertions;
- `reserve_registry.py`: deterministic registry, frozen-snapshot, containment,
  and exact-artifact verifier;
- `summary.json`: deterministic counts, score weights, and snapshot hash;
- `review_state.json`: empty-by-default human decisions and evidence ledger;
- `empirical_evidence.json`: deterministic, sanitized expert/model run ledger;
- `empirical_evidence.md`: human-readable empirical status and trust boundary;
- `empirical_evidence.schema.json`: machine-readable ledger contract;
- `empirical_evidence.sources.json`: committed file-by-file evidence bindings;
- `pipeline.py`: generation, validation, review gates, and manifest authorization;
- `research_protocol.json`: staged, resumable source-search and ingestion contract;
- `autoresearch_discovery.py`: executable offline ingest, normalization, dedupe,
  and coverage-sampling stages;
- `AUTORESEARCH_DISCOVERY.md`: discovery-stage schemas, commands, and evidence
  boundary;
- `autoresearch_*.schema.json`: query-plan, raw-hit, normalized-record, and
  content-addressed artifact schemas;
- `prototype_evidence.schema.json`: required file-backed prototype evidence;
- `trial_result.schema.json`: normalized, manifest-bound Harbor result format;
- `protocol_config.example.json`: frozen solver/verifier protocol template.

The next autoresearch cycle should append or replace normalized families only
after source/license review and near-duplicate analysis, then rebuild the
matrix. The first batch remains a reproducible snapshot rather than silently
changing under new search results.
