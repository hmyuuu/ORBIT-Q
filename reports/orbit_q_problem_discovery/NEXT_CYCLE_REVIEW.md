# Next-cycle human review packet

This packet is a post-snapshot research update. It does not replace the frozen
1,000-contract corpus or its ten-candidate shortlist, and it does not promote a
reserve into `review_state.json`. Its purpose is to focus the next human review
on designs that incorporate what the exploratory GPT-5.6-sol/high runs actually
taught us.

## Evidence boundary

- Confirmed protocol-scoped model-hard problems: **0**.
- Exploratory expert-pass/model-pass controls: **105, 106, 107, 108, 111**.
- Exploratory expert-pass/model-fail signal: **109**, one run only, paired by
  seed and task checksum but not by a full case digest.
- Candidate 109 historical v1 private expert screen: **22 passes in 23
  attempts**; the screen stopped at its first scientific-admission failure and
  the final two precommitted cases were not run.
- Candidate 109 current v6 public trusted-expert feasibility: **4/4 tuning and
  8/8 untouched validation passes**, with an immutable pre-run plan, preserved
  raw local evidence, and independent read-only reconstruction.
- Neither candidate 109 screen used Harbor or a solver/model. No solver run was
  launched while preparing this packet.
- A real pilot still requires the repository's concept, prototype, pilot,
  manifest, prequalified-seed, and two-auditor gates.

### Candidate 109 historical v1 and current v6 expert results

Candidate 109's historical v1 expert passed the first 22 cases in its
precommitted 25-case direct pinned screen, then failed case ordinal 23. The
screen stopped immediately as required; ordinals 24 and 25 remain unrun and
must not be used as replacement attempts. The failing case completed in
`34.299573` seconds. Its held-out p95 infidelity was
`0.000734152692426715`, above the unchanged `0.0007` gate. Worst infidelity
(`0.0010902181534833133` against `0.00125`), worst leakage
(`6.147127383582252e-05` against `0.00015`), maximum drive
(`2.3140415923829685` against `3.15`), maximum slew
(`1.0342867345004598` against `1.15`), and the zero edge amplitude all passed.

The sanitized aggregate is bound to the sealed plan, exact task/evaluator/
expert/image identities, and all 46 attempted-case stdout/stderr file hashes.
Its raw-log-set SHA-256 is
`ce5ad7e1409697ec2634170929fc3c203e6c8753c00d3ea54396efcdb66fb994`;
the sanitized summary file SHA-256 is
`8683da16e5df176eccdbcc006d204bc8d448cb683e7446076ffe54875090bc0d`.
The ignored aggregate contains no private seed or case digest. These are direct
expert-only results, not Harbor records, model evidence, or the 25 passing
schema-v2 seed/digest records required for canonical expert prequalification.
The current v6 expert is a different frozen artifact. Its immutable public plan
passed all four tuning-role canaries and all eight untouched validation
canaries with no reruns, substitutions, or byte changes. Across validation,
runtime was `41.696585`–`47.453723` seconds; maximum p95 infidelity was
`0.0006779807950682837`, worst infidelity `0.0008354746181724604`, leakage
`0.00007580683367447438`, drive `2.4102313353427793`, slew
`1.1193594678209193`, and edge amplitude `0.0`. The pre-run plan SHA-256 is
`1a6cf4930bfcedf8d1bbe23e9922c9e9d01440580308c29fa80ecab835e62254`,
the raw-log-set SHA-256 is
`e471d8e5bb0b838cf604980246fa15aab1480ddcad4c63de3a4c178ea2c7949b`,
and the branch-portable sanitized record payload SHA-256 is
`18c86fda728039597aca8073fa3379ce0f704eba2dc78a64baf8fa7138b48998`.
An independent read-only audit reconstructed all 12 local raw attempts. This is
durable public trusted-expert feasibility, not private prequalification or
Harbor evidence.

Candidate 109 remains **HOLD**. Its hidden interior ensemble is derived from
the same seed state that produces solver-visible configuration, and submitted
code still shares the evaluator process with mutable transitive globals. A
solver trial requires independent verifier-only hidden entropy, process/oracle
isolation, a new exact task review, and fresh private prequalification. The
historical v1 cases remain terminal and cannot be rerun or substituted.

### Candidate 115 pinned feasibility result

Candidate 115 subsequently passed functional and static verification on all
four frozen expert canaries in the digest-addressed TensorCircuit image
`sha256:8e627b582a5cdccca2ef40fb07f81671f5f5345ae1a07f40cef65537aa218bc7`.
TensorCircuit-to-independent-oracle errors stayed at or below `9.825e-15`, and sampled
peak memory stayed near `1.19 GiB`.

| Seed | Full case digest | Runtime | Sampled peak | Expert target |
|---:|---|---:|---:|---|
| 1152026 | `d465c54d13711c43620a0a186e3a30846df90e73d3f4026cccccb149a7d007b8` | 197.708s | 1.190 GiB | fail |
| 1152037 | `1fd934fbb07573a6b8b6a7cda224ac409a32403fba33890bd28f31bdadbcaf44` | 177.194s | 1.192 GiB | pass |
| 1152115 | `511d86b7d6003f2698e1f22c89c06ea0edb959af0ac80b70ecde4a6a74f5923c` | 207.955s | 1.194 GiB | fail |
| 9876543 | `029a058b6ac6c1ad742d9351fe458c58aa228aeb8b1625b3a839e2be8608e56b` | 216.619s | 1.190 GiB | fail |

The p50 was `202.832` seconds and protocol-style nearest-rank p95 was `216.619`
seconds. Three of four therefore exceed the frozen 180-second expert target.
The outcome is an honest **runtime admission failure**, even though every run
finished inside the 300-second submission limit. A separate 12-case dense
canary detected both tested semantic mutations in 12/12 cases. These are direct
verifier canaries without human promotion, a full Harbor record, or LLM policy
audit; they are not protocol-qualified prequalification or model evidence.

### Candidate 116 static audit result

Candidate 116 is a new post-snapshot reserve rather than a member of the frozen
1,000-contract corpus. Its NumPy transfer oracle agrees with exhaustive
three-by-three enumeration within `2.082e-17`, four development seeds pass the
case invariants, and the TensorCircuit expert is 149 effective lines. Independent
review confirmed the toric syndrome, plaquette, seam-homology, redundancy, and
recovery-relative logical-sector conventions. It also caught fail-open output
coercions; the evaluator now requires exact binary integer lists, finite real
fidelities, and exact integer costs, with focused regression coverage.

The current form remains on hold. Two stabilizer-related artifacts in each
homology class have identical logical fidelity, so eight listed candidates
reduce to four quality-distinct choices per syndrome and the final portfolio
enumeration is small. More importantly, the reserve intentionally has no entry
in the frozen discovery snapshot. It is now exact-byte-bound in the separate
post-snapshot reserve registry as `design_only`, so the safe runner rejects both
expert and model paths. No remapping or append-only mutation of the frozen
corpus is proposed. The next QEC iteration should make physical recovery
implementation affect performance, or require a genuinely synthesized
recovery set, before consuming pinned or model-test resources.

### Candidate 101 repaired expert result

The original mixed-state SLD-QFIM expert was rejected because it overfit the
four supplied training points and missed one hidden robust-gain gate. The
replacement uses a deterministic design ensemble spanning the disclosed
sensing/noise distribution and hard robust-score selection. The exact tracked
expert passed all three frozen public evaluator seeds in the digest-addressed
TensorCircuit image. Held-out score gains were `1.214079`, `1.122106`, and
`0.827440` against a `0.32` requirement; minimum QFIM eigenvalues were
`1.144806`, `1.370845`, and `2.047831` against `0.025`.

Prior unpreserved local scientific audits checked JAX derivatives against the
spectral SLD formula and a separately vectorized SLD equation. They reported a
maximum JAX-to-spectral difference of `6.656e-9`, a
spectral/vectorized-SLD difference of `4.441e-15`, and minimum support
`2.105e-3` over 420 hidden states. These are corroborating observations, not
raw-backed evidence for the current bundle; the durable canary below directly
records the evaluator's SLD-identity and support metrics.

The evaluator now expands one verifier-only base seed into the exact ordered
three-instance schedule, emits and flushes a full configuration digest before
untrusted import, scrubs both the seed environment and evaluator arguments,
and emits fixed expert-admission metrics. Hoisting otherwise unchanged JAX
kernels to module scope lets the three same-shape instances reuse compilation;
all 160 Adam steps, starts, QFIM formulas, objectives, and thresholds are
unchanged. The evaluator also precomputes all hidden ensembles before untrusted
import, prints no private schedule afterward, and invokes no seeded RNG between
solution calls. It also binds the trusted numerical oracle chain and clock
before solution import. The final exact materialized bundle finished in
`103.8569` seconds, leaving `76.1431` seconds below the strict 180-second gate.
The minimum held-out gain was `0.8274396`, minimum QFIM eigenvalue `1.1448062`,
minimum support `3.0388e-3`, and maximum SLD identity error `4.885e-15`; every
direct metric passed. The tracked canary record has payload SHA-256
`9da061c40745ebd51923de778dbf20a5fcfdfb75f6e062c55acda907250552cc`
and binds preserved raw stdout SHA-256
`1fafb35e0764c8e3324831c07c315f8463e34322389f069a642269e81fb48d58`.
Earlier direct observations used predecessor evaluator/task bytes and are
retained only as scientific/runtime corroboration.

A fresh private-v5 expert plan is sealed locally against these exact final
bytes and the byte-frozen no-provider runner. It contains 25 unique private
base seeds and case previews; all 25 single-candidate Oracle command previews
passed without executing Docker, Harbor, a solver, or an audit model. Its plan
SHA-256 is
`489f3c03cf8b1e714b922f1050745293c310a9d069de8233825070a4ec9f811f`
and payload SHA-256 is
`003aae1df402b0efd313e17674b08893774e2403c7907b197cf9aa41cea6f8f1`.
The reviewed driver SHA-256 is
`7c6ede9cf9e45a6eb88f22b1507cea135314e98e363747dbd74f9d2928521c13`,
and the byte-frozen runner SHA-256 is
`d8d213a259e96fbaec9a1b2ad91638aef8019f987feddc982ba5b8f8479f6067`.
The ignored plan, reviewed driver, and approval templates are mode `0600`, and
their dedicated output roots are mode `0700`. No human execution approval,
terminal reservation, Harbor completion, or model/provider call exists. The
Harbor-only execution path fails closed without a separately reviewed approval
that acknowledges local-operator custody and same-host process visibility.

This establishes direct expert feasibility, not protocol admission. One exact-
bundle runtime observation does not establish p95, and candidate 101 still lacks 25 private
prequalified Harbor records, authenticated policy evidence, human approval,
and a GPT-5.6-sol trial on the exact bundle.

### Candidates 117 and 118 local design results

Candidate 117 strengthens the held replica-moment task into a scored reusable
program: two 64-qubit purifications, 48 p2/p3/p4 queries per case, a generic
replica-permutation compiler, the exact canonical shared-prefix trie, and
operation/live-memory certificates. Across four development seeds the
canonical program used `23.63%` to `29.88%` of disjoint-path transfers. An
independent audit compared 192 dense partial-transpose, explicit-replica, and
transfer calculations with maximum error `2.22e-15`; 200 generated cases kept
exact intermediate rank at most four. The expert is 139 effective lines.

Candidate 117 nevertheless remains **HOLD**. No pinned TensorCircuit run has
validated rank-eight contractions, exact nontruncation, runtime, or RSS.
Moreover, functional output proves that the submitted graph is canonical, but
not that it was used to compute the moments; a solver could contract queries
separately and fabricate the graph. Source audit plus execution telemetry must
close that adherence gap before a pilot can be scientifically interpreted.

Candidate 118 combines retained-memory logical instruments with five-round QEC
intervention-policy selection. Independent spatial and temporal canaries agree
within `1.11e-16` and `2.78e-16`, four development seeds pass, and the expert is
146 effective lines. It is also **HOLD**: the prompt exposes the temporal
operators, the finite `5^5` enumeration, and the coupled budget program, making
GPT-5.6 failure insufficiently plausible; pinned TensorCircuit behavior near
`1e-39` unnormalized history probabilities is unverified.

Both 117 and 118 are post-snapshot design iterations, not frozen-shortlist
members. They are exact-byte-bound as `design_only`; that registry state is
provenance, not review, promotion, or execution authorization.

### Candidate 119 local design result

Candidate 119 jointly synthesizes a seven-fragment observable-lightcone cut
plan and one shared nine-stratum shot allocation for a 76-qubit circuit. The
independent NumPy cone oracle agrees with a dense eight-qubit canary within
`1.554e-15`; four development seeds pass and ten semantic/type mutations are
rejected per seed. The 158-line expert passes static policy and a Circuit API
shim, but has not executed in the pinned TensorCircuit image.

Independent review found and repaired two fail-open design defects: generation
now checks uniform shots over all 2,415 feasible layouts, and the evaluator no
longer accepts coercible strings or tuple matrices. It also disproved the
original minimax wording by finding a one-shot exchange that improves the
reference allocation by `1.866%`. The task now claims only threshold-feasible
synthesis. The product of independent single-RPP QPD factors remains a declared
conservative proxy, and the finite public search may still be recipe-exposed.
Candidate 119 is therefore exact-byte-bound as `design_only`; both expert and
model runner paths reject it before authorization.

The solved controls rule out several weak notions of difficulty. GPT-5.6-sol/high
successfully implemented a named fermionic-Gaussian inverse, exact CSS algebra,
a reusable tensor-network contraction plan, conditioned QKSD, and a truncated
MPS gradient. Random labels, tight line limits, specialized APIs, or a long
formula are therefore not sufficient by themselves. The useful signal from 109
is different: the submission had to synthesize a feasible artifact that
generalized over a hidden coupled distribution, while satisfying competing
fidelity, leakage, amplitude, slew, and edge constraints. Historical v1 still
failed its precommitted private screen at 22/23. Current v6 clears 12 preserved
public cases, but that cannot replace private evidence and does not resolve
shared-seed or same-process model isolation. The earlier exploratory model
failure therefore remains only a signal.

## Recommended review order

| Priority | Direction | Why it remains plausible | Current gate | Recommended human decision |
|---:|---|---|---|---|
| 1 | 101: robust mixed-state SLD-QFIM probe synthesis | The constructive expert clears every robust-gain, support, QFIM, identity, static, and strict runtime gate; the output is a continuous artifact scored on hidden nuisance ensembles | Exact schedule-scrubbed, oracle-prebound protocol and hash-bound direct expert pass exist with 76.14s runtime margin; no private Harbor records, human approval, or model evidence | Review the scientific/framework contract and sealed 25-case expert plan; require all expert records and p95 runtime before considering a solver pilot |
| 2 | 109: robust leakage-aware GRAPE | Only present exploratory expert-pass/model-fail signal; the task still demands hidden-distribution artifact synthesis under coupled physical constraints | Historical v1 stopped at 22/23 private cases; current v6 passed 4+8 preserved public canaries, but shares seed state with public config and lacks hostile-process isolation | Keep HOLD. Require independent hidden entropy, isolated oracle execution, fresh exact review, and private expert prequalification before any pilot |
| 3 | 117: replica-transfer program synthesis | Many p2/p3/p4 queries force generic replica semantics and exact shared-program certificates; local oracle independence is strong | Local canaries pass, but pinned TensorCircuit feasibility and graph-execution adherence are unproved | Approve only a pinned expert/adherence study; no model pilot until telemetry or audit proves the returned graph computed the moments |
| 4 | 116: coherent toric recovery portfolio v1 | A 98-qubit representation barrier and phase-sensitive 2^48-term coset sums are scientifically sound | Static oracle/expert exist and design-only provenance is bound, but eight artifacts collapse to four quality-distinct homology classes | Hold v1; add implementation-dependent recovery performance or a genuinely synthesized recovery set before pinned feasibility and any pilot |
| 5 | 115: noisy p3-PPT replica MPS | Dense fallback is impossible and opposite three-cycles plus traced Kraus environments create real representation pressure | Independent physics review and 4/4 pinned expert canaries pass, but 3/4 exceed the 180-second target and the 95-line expert transcribes exposed replica maps | Keep as a feasibility control; do not spend a solver pilot on the frozen form |
| 6 | 118: retained-memory QEC intervention policy | Temporal back-action and a shared resource budget are scientifically meaningful | Local spatial/temporal canaries pass, but the finite schedule/DP recipe is exposed and pinned TC is unverified | Preserve as a design iteration; redesign around causal prefix sharing or a reusable process-tensor program before feasibility work |
| 7 | Fault-tolerant CSS preparation synthesis v2 | Automated CSS preparation is grounded, but the solved CSS verifier shows direct algebra is too easy | Frozen f03 lead needs redesign | Require circuit/recovery synthesis across hidden propagated-fault sets rather than verification of a supplied construction |
| 8 | 119: robust lightcone cut planning | Joint partition/allocation and 54 framework pilots are more meaningful than fixed-cut reconstruction | Local oracle and strict artifact checks pass; pinned TensorCircuit is untested, allocation is threshold-feasible rather than optimal, and the QPD product is a conservative proxy | Keep `design_only`; review the proxy and recipe exposure before spending pinned feasibility resources |
| 9 | Non-Abelian geometric tensor and Wilson loop | Hidden rotations of a degenerate subspace defeat per-eigenvector phases; only gauge-covariant links are stable | Source-grounded lead; no prototype | Prototype only if production scale blocks dense eigenspaces and the oracle scores basis-invariant outputs |
| 10 | Implicit tensor-network fixed-point response | Requires a gauge-projected adjoint solve rather than differentiating a visible finite sweep recipe | Source-grounded lead; no prototype | Build only after demonstrating a stable transfer gap, independent finite-difference oracle, and substantial advantage over unrolling |

## Fresh autoresearch sample

The offline discovery stages were exercised on 13 manually curated primary
source leads. The deterministic coverage sampler selected 10 records spanning
QEC, quantum information, open systems, many-body response, and topology. It
covered every requested domain, situation, archetype, and evidence role.

- Search-plan SHA-256:
  `98e5e5858185c27565a6bfb3a6e9341455fbb20441016e828e4d420d9253b7ad`
- Coverage-sample SHA-256:
  `683c40b7b4c00e71ae70d02d00f7e6e87fd8df0f1b6fc4375046049b9be7ea9c`
- Custody claim: operator-provided metadata with `search_execution_claim=not_claimed`.
  The hashes establish deterministic local transformation, not independent
  search execution or source-claim verification.

Key primary sources behind the current priorities include:

- coherent/non-Pauli QEC tensor networks and logical channels:
  [Darmawan and Poulin](https://arxiv.org/abs/1607.06460),
  [Bravyi et al.](https://arxiv.org/abs/1710.02270), and
  [Huang, Doherty, and Flammia](https://arxiv.org/abs/1805.08227);
- process-tensor QEC decoding: [Kobayashi et al.](https://arxiv.org/abs/2412.13739);
- automated fault-tolerant CSS preparation: [Peham et al.](https://arxiv.org/abs/2408.11894) and [Weilandt et al.](https://arxiv.org/abs/2601.13313);
- partial-transpose moment certification: [Elben et al.](https://arxiv.org/abs/2007.06305) and [Yu, Imai, and Gühne](https://arxiv.org/abs/2103.06897);
- differentiable tensor-network fixed points: [Liao et al.](https://arxiv.org/abs/1903.09650);
- non-Abelian quantum geometry: [Ma et al.](https://arxiv.org/abs/1003.4040) and [Ding et al.](https://arxiv.org/abs/2201.01086);
- magic-state protocol variants: [Heußen](https://arxiv.org/abs/2504.17509).

## Human review questions

For each direction, reviewers should answer the following before prototype or
pilot approval:

1. Is the output a scientifically meaningful synthesized object, rather than a
   transcription of formulas supplied in the instruction?
2. Is every convention needed for correctness explicit without exposing the
   expert's construction or naming an unnecessary shortcut family?
3. Does a production instance make raw dense NumPy/JAX simulation infeasible
   while a documented TensorCircuit representation remains practical?
4. Is there an independent oracle or canary that would catch the most likely
   wrong-but-plausible implementations?
5. Are hidden cases prequalified for expert margin, identifiability,
   determinism, peak memory, and runtime before solver access?
6. Does the expert have a reproducible constructive advantage that is
   scientific rather than secret-data access or optimizer luck?
7. Would one GPT-5.6-sol/high pass veto the frozen design, and are all failures
   audited as substantive rather than infrastructure, API, or specification
   failures?

## Proposed immediate decision

Candidate 101 is the next viable human-review target. Its exact upgraded
protocol and direct expert now pass with substantial runtime margin, so the
review decision is whether to approve its sealed 25-case expert-only plan. That
approval would authorize feasibility evidence only, never a solver trial.

Candidate 109 remains HOLD. Historical v1 is terminal at 22/23 passes, its last
two cases remain unrun, and they must not be substituted or used to dilute the
failure. Current v6 has a fully frozen 12/12 public trusted-expert result, but
only an independent-hidden-seed and isolated-oracle revision followed by fresh
human review can produce new private evidence. Candidates 115–119 remain HOLD
controls or design iterations for the reasons above. No solver test should run
until human decisions and every candidate-specific admission gate are recorded.
