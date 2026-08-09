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
1,000-contract discovery snapshot. The safe runner must reject it until human
review creates a separately hashed post-snapshot reserve binding; it must not
be appended to or falsely remapped into the frozen corpus. A stronger QEC
successor should make physical recovery implementation affect performance or
require synthesis of the recovery set itself before consuming pinned or model
trial resources.
