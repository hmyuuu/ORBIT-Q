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
- No solver run was launched while preparing this packet.
- A real pilot still requires the repository's concept, prototype, pilot,
  manifest, prequalified-seed, and two-auditor gates.

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
in the frozen discovery snapshot, so the safe runner rejects it until human
review creates a separately hashed post-snapshot binding. No remapping or
append-only mutation of the frozen corpus is proposed. The next QEC iteration
should make physical recovery implementation affect performance, or require a
genuinely synthesized recovery set, before consuming pinned or model-test
resources.

The solved controls rule out several weak notions of difficulty. GPT-5.6-sol/high
successfully implemented a named fermionic-Gaussian inverse, exact CSS algebra,
a reusable tensor-network contraction plan, conditioned QKSD, and a truncated
MPS gradient. Random labels, tight line limits, specialized APIs, or a long
formula are therefore not sufficient by themselves. The useful signal from 109
is different: the submission had to synthesize a feasible artifact that
generalized over a hidden coupled distribution, while satisfying competing
fidelity, leakage, amplitude, slew, and edge constraints. The expert also had a
physics-informed construction before deterministic refinement.

## Recommended review order

| Priority | Direction | Why it remains plausible | Current gate | Recommended human decision |
|---:|---|---|---|---|
| 1 | 109: robust leakage-aware GRAPE | Only present expert-pass/model-fail signal; failure was held-out optimization quality rather than static policy or source audit | Public expert and one exploratory pilot exist; no protocol-qualified prototype | Review now for a fresh 25-seed expert prequalification campaign and then a five-seed pilot |
| 2 | 116: coherent toric recovery portfolio v1 | A 98-qubit representation barrier and phase-sensitive 2^48-term coset sums are scientifically sound | Static oracle/expert exist, but the reserve has no human-approved post-snapshot runner binding, no pinned TensorCircuit run, and eight artifacts collapse to four quality-distinct homology classes | Hold v1; preserve it as a reviewed design iteration, then add implementation-dependent recovery performance or a genuinely synthesized recovery set before pinned feasibility and any pilot |
| 3 | 115: noisy p3-PPT replica MPS | Dense fallback is impossible and opposite three-cycles plus traced Kraus environments create real representation pressure | Independent physics review passes and 4/4 pinned expert canaries pass, but 3/4 exceed the 180-second expert target and the 95-line expert largely transcribes replica maps exposed by the prompt | Hold the frozen form from solver pilots; redesign around reusable multi-order/multi-cut contraction-program synthesis with a shared memory/runtime certificate |
| 4 | Process-tensor QEC decoder | Temporal maximum-likelihood decoding combines interventions, syndrome branches, and a non-Markovian tensor network | Source-grounded lead; no production prototype | Approve a bounded-memory synthesis formulation, not a fully specified transcription task |
| 5 | Fault-tolerant CSS preparation synthesis v2 | Automated CSS preparation is scientifically grounded, but the solved CSS verifier shows direct algebra is too easy | Frozen f03 lead needs redesign | Require circuit/recovery synthesis across hidden propagated-fault sets rather than verification of a supplied construction |
| 6 | Robust mixed-state QFIM probe synthesis | A physical probe must trade information across nuisance-channel ensembles while respecting support/rank conditions | Earlier direct prototype was rejected on robustness | Reframe around robust experiment design with a constructive expert warm start and fixed-support oracle |
| 7 | Worst-case circuit-cut and shot-allocation synthesis | Correct subcircuits are insufficient; cuts, signed recombination, and variance allocation must jointly generalize to hidden observables | Direct reconstruction prototype exists; no model evidence | Add hidden graph/observable families and score a synthesized allocation rather than one declared cut |
| 8 | Non-Abelian geometric tensor and Wilson loop | Hidden rotations of a degenerate subspace defeat per-eigenvector phases; only gauge-covariant links are stable | Source-grounded lead; no prototype | Prototype only if the production scale blocks dense eigenspaces and the oracle scores basis-invariant outputs |
| 9 | Implicit tensor-network fixed-point response | Requires a gauge-projected adjoint solve rather than differentiating a visible finite sweep recipe | Source-grounded lead; no prototype | Build only after demonstrating a stable transfer gap, independent finite-difference oracle, and substantial advantage over unrolling |
| 10 | Noisy magic-state protocol synthesis | Postselection, coherent/correlated noise, yield, and logical infidelity can create competing objectives | Frozen f15 lead; no prototype | Prefer synthesis or portfolio selection across hidden noise models over direct 15-to-1 simulation |

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

The highest-value human action is to review 109 for protocol-qualified expert
prequalification. Independent review places candidate 115 on hold for hardness:
its science is sound, but the prompt supplies the routing, damping dilation, and
replica-index maps needed by a 95-line expert. Its pinned run should therefore
remain a TensorCircuit feasibility/regression control. A successor should ask
for one reusable multi-order/multi-cut contraction program over many queries,
with operation-count and peak-memory certificates, without publishing the
expert's einsum construction. Candidate 116 is a useful first QEC design
iteration, but independent review shows that its finite library still reduces
to four logical-quality choices per syndrome; it should be strengthened before
pinned feasibility rather than treated as a likely-hard candidate. No solver
test should run until those decisions are recorded.
