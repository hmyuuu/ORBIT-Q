# Ideas Session — 2026-08-10 05:25

## Phase 1 — Find a harder QEC reserve

The requested direction is a post-shortlist TensorCircuit QEC benchmark that is
plausibly harder than problem 113, emphasizes degeneracy/coherent decoding or
fault-tolerant CSS synthesis, rules out dense physical-state simulation, has an
independent non-TensorCircuit oracle, and remains honest about the absence of a
pinned-image run or model trial.

The frozen corpus already contains three nearby ideas. Frozen family f03 asks
for fault-tolerant CSS state-preparation synthesis and exhaustive fault
propagation. Frozen family f11 asks for stochastic correlated surface-code
coset likelihoods through a tensor-network decoder. Reserve 113 already asks
for coherent syndrome-conditioned logical process matrices, but its two
scrambled `[[14,2,3]]` CSS instances are still small enough for the evaluator's
dense statevectors. A new candidate should therefore add a genuine
representation barrier rather than merely more branches or looser tolerances.

Primary-source search found a coherent path that is distinct from all three:

- Bravyi, Suchara, and Vargo, *Efficient Algorithms for Maximum Likelihood
  Decoding in the Surface Code* (arXiv:1405.4883), map degenerate surface-code
  coset sums to two-dimensional tensor networks and contract them with MPS.
- Bravyi, Englbrecht, Koenig, and Peard, *Correcting coherent errors with
  surface codes* (arXiv:1710.02270), show that coherent surface-code QEC can be
  simulated without a dense physical state by exploiting special structure.
- Beale et al., *Quantum Error Correction Decoheres Noise* (PRL 121, 190501),
  establish why the syndrome-conditioned logical channel and per-syndrome
  recovery remain the right objects under coherent noise.
- Venn, Behrends, and Béri, *Coherent-Error Threshold for Surface Codes from
  Majorana Delocalization* (PRL 131, 060603), explicitly map coherent surface
  code errors to a complex-coupling Ising model.
- Chubb and Flammia, *Statistical mechanical models for quantum codes with
  correlated noise* (arXiv:1809.10704), extend code-to-partition-function
  mappings and tensor-network decoding to locally correlated noise.

The resulting design direction is an exact complex partition-function decoder
for a distance-seven toric CSS code. Ninety-eight physical edge qubits and 49
plaquette variables make dense statevectors and stabilizer-coset enumeration
infeasible, while the local lattice admits an exact row transfer. Four toric
homology sectors become four coherent logical-Pauli amplitudes; a classical
mixture of coherent branches produces a normalized syndrome-conditioned
logical process matrix. The expert can evaluate the periodic transfer trace as
a 14-virtual-qubit TensorCircuit MPS circuit of nonunitary one- and two-site
transfer gates. The evaluator can independently form and multiply NumPy row
transfer matrices, with a tiny-lattice exhaustive canary.

Risk noted before implementation: TensorCircuit's documented `MPSCircuit.apply`
and `apply_single_gate` accept general gates, but the target image must still
canary nonunitary gates, nonadjacent periodic couplings, amplitude ordering, and
exact bond growth. Until that happens, the bundle must be marked static/oracle
validated and TensorCircuit-runtime unverified.

## Phase 2 — Select a synthesis contract

Failure-mining evidence arrived while the first design was being specified.
GPT-5.6 had already solved direct or recipe-exposed tensor-network contraction,
CSS verification, QKSD, and MPS differentiation tasks; its meaningful failure
was a constrained synthesis problem with a structured expert warm start. That
changed the benchmark target. A pure request for four coherent logical-sector
amplitudes would add a representation barrier but would still be another
reconstruction contract.

The selected reserve therefore keeps the exact coherent toric coset primitive
but makes it subordinate to a recovery-design decision. Each seeded case has
five syndromes, eight physical recovery candidates per syndrome, three
two-branch coherent-noise scenarios, integer hardware costs, and one shared
budget. A submission must return the full selected recovery masks and their
opaque homology-frame declarations. It maximizes the minimum, across scenarios,
of the syndrome-weighted conditional identity fidelity. The generator admits a
case only when the cheapest portfolio, the budget-constrained optimum, and the
unconstrained optimum are separated by positive margins.

This pivot also removed the speculative 14-virtual-qubit MPS construction from
the expert. The final expert uses a smaller and more auditable TensorCircuit
transfer circuit on seven virtual qubits, with nonunitary local gates and an
operator trace. The evaluator uses independently assembled NumPy row-transfer
matrices and a separate exhaustive three-by-three plaquette canary. The public
instruction defines the mathematical coherent sums but does not disclose the
row-transfer implementation recipe.

## Phase 3 — Prototype result and claim boundary

The resulting four-file reserve is
`reports/orbit_q_problem_discovery/blueprints/coherent-toric-recovery-portfolio/`
for proposed problem 116. Four non-binding host seeds passed the NumPy oracle,
case invariants, exact portfolio search, and active fidelity-cost-conflict
checks. Across eight generated cases, the smallest cheapest-to-selected margin
was 0.011800, the smallest selected-to-unconstrained margin was 0.010553, the
smallest syndrome probability was 4.143e-17, and the smallest maximum
conditioned coherence was 0.413812. Direct plaquette enumeration and row
transfer agreed to 2.082e-17 on the independent tiny canary.

The 149-effective-line expert matched the evaluator exactly under a local API
semantic proxy, which tests the intended gate ordering and transfer algebra but
is not a TensorCircuit execution. Ruff, Python syntax, JSON parsing, and the
default-seed proxy evaluation pass. No Docker, Harbor, canonical task,
solver-model trial, commit, or push was used.

The design is scientifically coherent and plausibly harder than reserve 113
because it replaces a dense-feasible 14-qubit reconstruction with a 98-qubit,
2^48-degenerate robust recovery synthesis. It is not yet an admitted benchmark.
The very small unnormalized syndrome probabilities make pinned complex128
parity especially important. Human review, pinned-image expert execution,
runtime/memory measurement, exact private prequalification records, and policy
audits remain mandatory before any solver trial or hardness statement.

## Phase 4 — Independent audit and hold decision

Independent review confirmed the toric syndrome, plaquette, homology,
redundancy-factor, and recovery-relative process conventions, including a
separate tiny-lattice sector-shift check. It also narrowed the design claim.
The two stabilizer-related artifacts in each homology class have identical
logical fidelity, so eight listed artifacts reduce to four quality-distinct
choices per syndrome; the small final enumeration is portfolio selection, not
a difficult synthesis search. Candidate 116 is therefore held from promotion
and model trials even if its TensorCircuit transfer later proves feasible.

The audit also found fail-open output coercions: fractional or mod-256 mask
values could be cast to binary, a NaN fidelity could evade the error maximum,
and fractional costs could be truncated. The evaluator now validates raw list
shape and exact binary integers before casting, requires a finite real
fidelity, and requires an exact non-boolean integer cost. Focused regressions
cover these cases.

Finally, the reserve deliberately remains outside the frozen 50-family,
1,000-contract discovery snapshot. Before the provenance registry existed, the
safe runner rejected it because there was no discovery binding. Its later
exact-byte `design_only` binding remains non-executable; only a distinct
human-reviewed executable-promotion contract could authorize a run. It must
not be appended to or falsely remapped into the frozen corpus. A stronger QEC
successor should make physical recovery implementation affect performance or
require synthesis of the recovery set itself before consuming pinned or model
trial resources.

## Phase 5 — Process-tensor QEC policy reserve

The next source-grounded direction was the process-tensor QEC lead. The
primary references were checked directly: Kobayashi et al., *Tensor-network
decoders for process tensor descriptions of non-Markovian noise*
(arXiv:2412.13739), explicitly combine maximum-likelihood QEC decoding with a
process-tensor representation of spatiotemporal correlations; Link, Tu, and
Strunz, *Open Quantum System Dynamics from Infinite Tensor Network
Contraction* (arXiv:2307.01802), support a bounded-memory tensor-network view
of non-Markovian dynamics. The coherent surface-code sources used for reserve
116 remain the spatial foundation.

The resulting problem 118 design uses two distance-five toric memories, six
five-round syndrome histories per case, three alternative retained-memory
process scenarios, and five intervention primitives. A round-sector amplitude
coherently sums 2^24 stabilizer representatives. Those amplitudes become
logical-plus-memory process instruments; intervention-dependent memory
back-actions make earlier actions affect later rounds. Each history has 5^5
possible schedules, and all histories share one active cost budget. The
expert uses a five-virtual-qubit TensorCircuit row transfer, a cheapest-policy
warm start with deterministic coordinate sweeps, exact schedule frontiers,
and a coupled cost dynamic program. The evaluator independently uses NumPy
row transfers, a direct L=3 plaquette enumeration, and explicit dense Choi
propagation.

Four non-binding host seeds passed. Spatial transfer and enumeration agreed to
3.123e-17; compact and dense-Choi temporal fidelities agreed to 5.552e-17.
Across eight cases, the minimum cheapest-to-selected objective margin was
0.102610 and the minimum selected-to-unconstrained margin was 0.114493. Five
schema, finite-value, identity, and valid-but-suboptimal mutations were
rejected per seed. An in-memory conventional circuit-composition proxy ran the
129-effective-line expert end to end and matched the oracle to 1.61e-15, but
this was deliberately not represented as a real TensorCircuit execution.

Independent review confirmed the toric, temporal, and fidelity mathematics,
then recommended **HOLD**. The prompt necessarily states the 8-by-8 temporal
operators, finite 5^5 search, and coupled budget program, leaving only the
spatial row transfer to derive. That is genuine finite synthesis, but it is
probably too recipe-exposed to justify a GPT-5.6 pilot. The warm-start sweeps
also do not reduce the subsequent exhaustive search. Actual pinned
TensorCircuit behavior, complex128 accuracy near 1e-39 conditional
probabilities, runtime below 180 seconds, and memory remain unproved.

The review additionally found that a local `--seed` value remained visible in
`sys.argv` even after environment scrubbing. The evaluator now removes both
seed environment variables and all evaluator arguments before importing the
submission; an explicit untrusted-module probe verified the scrub. The
probability diagnostic was also renamed and made to select the actual
minimum-cost action rather than shuffled action index zero.

Problem 118 is therefore preserved as an honest four-file design iteration,
not promoted or scheduled. The recommended successor is an online causal
prefix-sharing policy over syndrome trees, or a reusable contraction-program
synthesis problem whose success cannot be reduced to transcribing the stated
temporal equations and enumerating a small finite schedule set.

## Phase 6 — Strengthen the held replica-moment reserve

The next requested direction was a materially harder successor to held reserve
115. The design constraint was unusually specific: many shared p2/p3/p4
mixed-state replica queries, no published expert einsum maps, a reusable
contraction graph in the answer, an independently derived TensorCircuit-free
oracle, and strict operation and memory certificates. The important risk was
that a larger list of direct moment queries would only repeat 115's weakness:
the model could transcribe each exposed formula independently.

The selected design makes the reusable program part of the scored artifact.
Each seeded trial contains two 32-system/32-environment Stinespring
purifications. Each case carries 48 overlapping queries—16 at each of orders
two, three, and four—whose A/B/identity layouts branch over five shared tail
segments. The prompt defines only the mathematical rule that a bra replica
index equals a forward-cycled ket index on A, a reverse-cycled index on B, and
the same index elsewhere. It does not publish an einsum expression. The expert
must derive one order-generic permutation compiler.

Every query becomes a selector string over the interleaved physical MPS. The
returned program is the unique order-separated prefix trie of those strings,
numbered in canonical depth-first preorder. The evaluator checks every node,
parent, selector, terminal, operation count, conservative live-environment
capacity, and SHA-256 program digest exactly. Across four local seeds the
canonical programs used 726–918 transfers per case, only 23.63%–29.88% of the
3,072 transfers required by 48 disjoint 64-site paths. A per-query path graph
therefore cannot pass. The remaining loophole—computing independently and
fabricating the correct compact graph afterward—cannot be closed by functional
values alone, so the claim boundary explicitly requires source audit, runtime
comparison, and process memory telemetry.

The expert uses an exact four-bond TensorCircuit MPS, derives its einsum labels
from the permutation exponent, and executes the trie recursively while
deleting each child environment after its subtree. The four-bond capacity is
not an approximation target: 200 generated cases across seeds 1172000–1172099
had maximum intermediate numerical Schmidt rank four. The live-memory
certificate conservatively charges every nonterminal bond at full capacity,
giving 4,128,770 complex elements for an order-four root-to-leaf path even when
realized tensors are smaller.

The independent evaluator uses a NumPy TEBD MPS and the integer-subscript form
of `einsum`, rather than the expert's string compiler. Its dense canary traces
the environment, forms the requested reduced density matrix and partial
transpose, and compares matrix powers against both the transfer and an
explicit permutation of k dense purification replicas for k=2,3,4. The maximum
canary error was 8.882e-16. Four local seeds passed the full expert-source API
shim with zero metric error, all five semantic mutations were rejected, Ruff
and static policy passed, and the expert used 135 effective lines. An explicit
untrusted-module probe also observed neither the seed environment variable nor
parsed `--seed`/`--solution` CLI arguments at import or execution time.

The resulting four-file reserve is proposed problem 117 under
`reports/orbit_q_problem_discovery/blueprints/replica-transfer-program-synthesis/`.
It remains a HOLD: no real TensorCircuit image execution, runtime or RSS
measurement, Harbor verifier, human review, private prequalification, or solver
trial has occurred. The graph-level reuse requirement and independent oracle
are established locally; TensorCircuit feasibility and model hardness are not.

### Post-format checkpoint note

Repository-wide Ruff formatting changed only layout, not the scientific
algorithms. Final static-policy counts are 139 effective lines for problem 117
and 146 for problem 118, both below the frozen 160-line limit with score 1.0.
The design-only registry was rehashed from these final bytes.

### Reserve-lifecycle correction

The Phase 4 statement about creating a post-snapshot binding described the
state before the provenance registry existed. Problems 116, 117, and 118 are
now exact-byte-bound there as `design_only`. The runner rejects that lifecycle
before both expert and model paths; the registry is not execution authority.
Only a future separately hashed, human-reviewed executable-promotion contract
could authorize a run, without appending or remapping the frozen snapshot.

## Phase 7 — Robust circuit-cut partition and shot-allocation reserve

The next design request combined two decisions that problem 104 had fixed in
advance: where a large circuit should be cut and how a finite execution budget
should be shared. The failure-mining lesson from the earlier GPT-5.6 runs was
kept explicit. A larger reconstruction formula alone would likely remain a
transcription task, while candidate 109's only exploratory failure signal came
from genuinely constrained robust synthesis. The new direction therefore
scores a partition/allocation artifact rather than merely asking for another
expectation value.

Primary-source search grounded five parts of the design:

- Tang et al., *CutQC: Using Small Quantum Computers for Large Quantum Circuit
  Evaluations* (arXiv:2012.02333), motivate automated capacity-constrained
  circuit partitioning.
- Brandhofer, Polian, and Krsulich, *Optimal Partitioning of Quantum Circuits
  using Gate Cuts and Wire Cuts* (arXiv:2308.09567), treat cut selection and
  execution overhead as one resource problem.
- Schmitt, Piveteau, and Sutter, *Cutting circuits with multiple two-qubit
  unitaries* (arXiv:2312.11638), establish angle- and gate-dependent QPD costs
  and, importantly, warn that cutting gates jointly can beat products of
  independent single-gate decompositions.
- Chen et al., *Enhanced Quantum Circuit Cutting Framework for Sampling
  Overhead Reduction* (arXiv:2412.17704), explicitly optimize shot
  distribution using subexperiment variance contributions.
- Eddins, Tran, and Rall, *Lightcone shading for classically accelerated
  quantum error mitigation* (arXiv:2409.04401), provide a neighboring example
  of observable-specific causal influence reducing quasiprobability overhead.

Several stronger-looking formulations were rejected before implementation. An
exact expansion over every cut configuration would make the number of terms
exponential and either violate the runtime envelope or force the generator to
use so few cuts that the partition problem becomes trivial. A per-cut shot
allocation was also rejected as physically misleading because standard QPD
sampling allocates across subexperiments, not independent edges. The retained
contract is deliberately a pre-execution robust variance proxy with shared
execution strata. It does not claim to reconstruct real samples or equal
hardware MSE.

Problem 119 uses one seeded 76-qubit circuit with three matching layers and 113
RPP interactions. The hardware layout must be divided into seven contiguous
fragments of eight to twelve qubits, giving 2,415 feasible interval
compositions. Eighteen local Pauli observables and three coherent calibration
scenarios create 54 targets. Their exact backward cones contain at most eight
qubits and seven interactions. The expert must use TensorCircuit for all pilot
expectations, identify which partition cuts lie in each observable cone, apply
the angle-dependent single-RPP factor `gamma = 1 + 2|sin(theta)|`, and assign
90,000 integer shots across nine shared strata below a worst-target variance
cutoff. Opaque fragment, gate, observable, scenario, and stratum identities,
plus exact cut/lightcone and continuous certificates, are all scored.

The evaluator never imports TensorCircuit. It constructs the cone circuits
directly from complex128 NumPy RPP matrices, exhaustively searches all feasible
layout compositions, and recomputes every artifact and variance. A separate
eight-qubit dense circuit agrees with the cone oracle to at most 1.554e-15.
Four non-binding host seeds passed; their reference robust variances ranged
from 6.996e-5 to 1.272e-4. The selected plans cut 10 to 16 interactions. Even
the balanced-layout reference allocation was at least 255% worse than the
selected plan, so the partition objective is active rather than decorative. The final
generator also exhaustively checks that uniform shots fail across all 2,415
partitions. Ten strict type, contiguity, identity, finite-value, allocation,
certificate, coercible-string/tuple, and valid-but-suboptimal-plan mutations
were rejected on every seed.

The 158-effective-line expert passes the repository static policy with 101
logical statements and no raw-simulator hint. A conventional in-memory Circuit
shim matched the independent oracle end to end: maximum pilot error 2.165e-15,
robust-variance error 8.132e-20, and default-seed proxy runtime 2.953 seconds.
This is semantic evidence about the intended gate order and basis changes, not
a TensorCircuit execution or runtime estimate.

An independent Phase 7 audit reproduced the lightcone, RPP, seed-scrub, and
static checks, but found that the initial generator only compared uniform shots
on the balanced layout, the validator coerced numeric strings and tuple
matrices, and the shared 96-step allocation heuristic was not an integer
minimax certificate. The first two defects were repaired locally: generation
now compares the best uniform-shot score over every feasible partition, and
the validator requires nested lists and finite real non-boolean numbers. The
audit improved the default heuristic allocation from 6.995601480311e-5 to
6.867465835582e-5 by one-shot exchanges. The benchmark therefore asks only for
a threshold-feasible joint design, accepts better plans, and makes no global
allocation-optimality claim.

The verdict is **HOLD**. The independent product of single-RPP QPD factors is a
declared conservative benchmark proxy, not the optimal joint multi-gate
overhead described in the literature. Real TensorCircuit behavior for the XX
and YY basis conjugations, `expectation_ps` convention, complex128 parity, and
54 repeated cone circuits remains untested. More strategically, the public
objective and 2,415-layout search may still be reproducible by GPT-5.6 once the
quantum pilots are correct; local feasibility cannot establish likely
hardness. Problem 119 should be exact-byte registered only as `design_only`,
then receive human design review and a separately hashed executable-promotion
decision before pinned expert feasibility or prequalification can begin. Only
after those gates pass may humans consider a separately authorized
one-candidate model pilot.

## Phase 8 — Evidence-qualified checkpoint and human decision boundary

The final checkpoint separates promising expert feasibility from model
hardness. Candidate 101's schedule-scrubbed, oracle-prebound mixed-state QFIM
evaluator and module-scope-JIT expert were frozen into one exact materialized
task. A network-disabled pinned-image direct run passed all functional,
scientific, static, and strict admission metrics in 103.8569 seconds, leaving
76.1431 seconds below the 180-second expert gate. The branch-portable record
binds the exact evaluator, expert, blueprint, task bundle, runtime versions,
and preserved local raw hashes. A fresh private-v5 plan now commits 25 unique
cases and 25 safe one-candidate Oracle command previews against a byte-frozen
provider-free runner, but it remains unexecuted and has no human approval or
terminal reservation. Its only permitted execution path is Harbor Oracle on a
trusted single-user host after a separate exact approval that acknowledges
same-host process visibility. This establishes a constructive solution and a
credible next review target, not Harbor admission or model hardness.

Candidate 109 now has two intentionally separate histories. The historical v1
private screen remains terminal at 22 passes in 23 attempts, with the final two
cases unrun and no rerun or substitution permitted. The current v6 expert was
instead tested on an immutable public plan: four tuning-role and eight untouched
validation canaries all passed, with raw local evidence preserved and an
independent read-only reconstruction. Its tracked record is durable sanitized
public trusted-expert feasibility. It cannot replace private prequalification,
because the hidden stream is derived from solver-visible seed state and the
evaluator does not isolate hostile submitted code from mutable oracle process
state.

Problem 119's lifecycle was also made explicit. Exact-byte `design_only`
registration is provenance only. The allowed order is human design review,
then a separately hashed executable-promotion contract, then pinned expert
feasibility and prequalification. Direct Docker commands that bypass that
lifecycle were removed; model work would require still-later human gates.

The honest conclusion remains **zero confirmed protocol-scoped GPT-5.6-sol/high
hard problems**. Problems 105, 106, 107, 108, and 111 are solved controls;
candidate 109 has only one historical exploratory failure signal. The next
scientific action is therefore human review of candidate 101's exact contract
and sealed expert-only plan. No solver trial is authorized by this checkpoint,
and any eventual benchmark run must execute only the selected candidate task.

## Phase 9 — Synthesis-first candidate selection

The next discovery cycle compared three constructive directions rather than
adding another direct diagnostic: a flagged CSS gadget and recovery synthesizer,
a verifier-executed batched-amplitude contraction compiler, and a
nuisance-projected fermionic-Gaussian experiment portfolio. All three can make
the returned artifact operationally relevant, but they have different current
risks. The contraction compiler offers the strongest mechanical adherence
check because the verifier itself executes constant-free bytecode, yet its
64-qubit TensorCircuit leaf stability, slicing, and resource margins are
unmeasured. The fermionic portfolio is scientifically distinct from QFIM and
GRAPE, but its proposed 8,208 Gaussian evolutions, integer repair, and isolated
hidden-design protocol add several feasibility dependencies at once.

Candidate 120, robust flagged CSS verification-and-recovery gadget synthesis,
was selected for the primary prototype. It extends the frozen corpus's
top-ranked fault-tolerant CSS synthesis gap and directly addresses the weakness
of solved control 106: the solver must return actual measurement-gadget gate
orders, flag placements, calibrated entangler variants, and a complete
outcome-conditioned recovery table. The independent verifier can replay the
submitted circuit under every declared Pauli fault and evaluate coherent-noise
scenarios, so no solver-declared diagnostic or fidelity is trusted. The output
artifact therefore changes both fault tolerance and robust fidelity.

The scientific mechanism is source-grounded. Peham et al. describe automated
fault-tolerant CSS state-preparation and verification synthesis
(https://arxiv.org/abs/2408.11894). Schmid et al. treat deterministic
verification/correction synthesis and hook-fault flags as a global design
problem (https://arxiv.org/abs/2501.05527). Chamberland and Beverland establish
the flag-circuit principle for arbitrary-distance stabilizer codes
(https://arxiv.org/abs/1708.02246). These sources motivate the contract; they do
not establish GPT-5.6 hardness.

The prototype remains `design_only` and **HOLD**. Its largest uncertainties are
whether a genuinely TensorCircuit-central expert can couple GF(2) fault
partitioning with coherent circuit evaluation in at most 160 effective lines,
whether targeted one-edit gadget mutants have sufficient numerical separation,
and whether pinned expert runtime stays below 180 seconds. The contraction
compiler and fermionic portfolio remain parallel design-only backups. No
Docker, Harbor, solver, audit model, private seed, or canonical task is
authorized by this phase.

The parallel fermionic-Gaussian portfolio prototype became Problem 122. Its
independent NumPy Bogoliubov-isometry evaluator, dense Fock canary, finite-
difference check, and semantic FGSSimulator shim passed on four selected public
development seeds. The 138-effective-line expert passed the static policy, and
ten complete scientific or schema mutants were rejected on each selected seed.
Four additional explored seeds failed either the expert threshold or mutant
separation, so this is not an admission-stable generator. Independent review
found no scientific defect but confirmed that real pinned TensorCircuit parity,
the proposed 40-mode scale, hidden-artifact isolation, and model hardness are
all unverified. Problem 122 is therefore exact-byte registered only as
`design_only`; the safe runner refuses both expert and model execution.

The first Problem 120 implementation also exposed a useful negative result.
Its full target-state coset oracle, dual X/Z Steane orientation, exhaustive
single-fault branches, and target-state flag-safety rule are internally sound,
but an independently regenerated sorted gadget with deterministic calibrated
variants still cleared the initial coherent-fidelity floor without using
TensorCircuit. Rejecting a changed gadget with its stale decoder was not valid
anti-shortcut evidence. The design remains unregistered while calibration is
reworked against complete regenerated artifacts; if that shortcut cannot be
made to fail with honest margin, the prototype will stay HOLD and the
verifier-executed contraction compiler will become the primary direction.

That calibration was subsequently repaired without weakening the full SHA-256
case identity, 160-line expert limit, or fault semantics. The final generator makes
the public nominal model one member of the hidden minimum, and each admission
baseline is now a complete artifact with a freshly synthesized 256-row decoder
and fully reoptimized allowed variants. Cheap, first-candidate, sorted,
reversed, and the red-team deterministic recipe all miss the case-specific
floors by at least 0.001, while the 157-effective-line expert clears both
floors. The default self-test independently re-executes 99 fault branches per
case, finds zero target-coset/syndrome discrepancies, exercises a functional
dangerous hook and wrong flag window in each case, and rejects fourteen
complete or schema mutations. Two nearby development seeds fail the expert
floor, so admission is not seed-stable and future promotion must use an honest
positive-instance rule plus at least 25 fresh records. Independent review gave
Problem 120 a `design_only` GO and an execution HOLD; its exact four-file
bundle is now provenance-registered, while the safe runner still refuses both
expert and model execution.
