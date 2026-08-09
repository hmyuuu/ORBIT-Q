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
