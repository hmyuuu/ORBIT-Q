# ORBIT-Q future-problem shortlist

> **Claim boundary:** This is a source-grounded design screen. The scores are
> priors, not evidence that GPT-5.6-sol failed. Real model tests remain blocked
> until the human and expert-baseline gates pass.

## Screening result

The deterministic matrix contains **1,000 candidates**
from **50 families**, four scale regimes, and five
situations. Ten candidates were selected for design review; none has been
empirically tested.

| Rank | Candidate | Domain | Design prior | TC path | Verifier |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `f03_five_qubit_coherent_logical_channel--benchmark--robust_ensemble` — Fault-tolerant flagged CSS state preparation | qec | 98.0/100 | 5/5 | 5/5 |
| 2 | `f01_fermionic_gaussian_conditioned_quench--representation_stress--inverse` — Conditioned fermionic-Gaussian Kitaev quench | fermions | 89.6/100 | 4/5 | 4/5 |
| 3 | `f02_mixed_state_sld_qfim_probe--benchmark--robust_ensemble` — Mixed-state multiparameter SLD-QFIM probe design | metrology | 92.8/100 | 4/5 | 5/5 |
| 4 | `f09_qsp_phase_synthesis--benchmark--inverse` — Quantum signal-processing phase synthesis | algorithms | 95.6/100 | 5/5 | 4/5 |
| 5 | `f45_semantics_preserving_routing--benchmark--robust_ensemble` — Dynamic-circuit branch equivalence certificate | verification | 94.8/100 | 4/5 | 5/5 |
| 6 | `f28_quantum_lanczos_excited_states--benchmark--robust_ensemble` — Quantum Lanczos excited-state spectrum | many_body | 92.8/100 | 4/5 | 5/5 |
| 7 | `f04_nonmarkovian_memory_identification--benchmark--inverse` — Non-Markovian memory-channel system identification | open_systems | 91.6/100 | 4/5 | 4/5 |
| 8 | `f43_circuit_cutting_reconstruction--benchmark--robust_ensemble` — Variance-aware circuit cutting reconstruction | distributed_quantum | 94.4/100 | 5/5 | 5/5 |
| 9 | `f16_robust_leakage_grape--benchmark--robust_ensemble` — Leakage-aware robust GRAPE control | control | 88.0/100 | 4/5 | 4/5 |
| 10 | `f50_hardware_metric_cross_benchmark--representation_stress--forward` — Memory-constrained contraction plan and selected amplitudes | simulation | 91.6/100 | 5/5 | 5/5 |

## 1. Fault-tolerant flagged CSS state preparation

- Candidate: `f03_five_qubit_coherent_logical_channel--benchmark--robust_ensemble` (`3d153e8d8a63cc99`)
- Contract: Synthesize a flagged CSS logical-state preparation circuit that preserves the target stabilizers and detects every declared low-weight propagated fault. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Construct and execute the candidate encoder, flag checks, and injected Pauli faults as TensorCircuit circuits; use tableau logic only as an independent verifier.
- Oracle: Polynomial stabilizer/isometry checks plus exhaustive single-fault propagation and exact small-code state verification.
- Why plausibly hard: Requires constructive stabilizer reasoning, flag placement, propagation analysis, and exact fault-tolerance semantics; this is supported by direct agent-benchmark evidence rather than size alone.
- Falsification check: Reject if the held-out CSS family lacks a short expert construction, if any fault class is underspecified, or if the TensorCircuit execution is only a wrapper around a classical tableau answer.
- Current-suite relation: Directly fills the QEC/fault-tolerant-synthesis roadmap gap; StabilizerBench reports that advanced fault-tolerant synthesis remains largely unsuccessful for frontier agents.
- Human review questions: Which held-out CSS codes have short expert circuits?; Are all injected fault locations and acceptance rules explicit?; Does exhaustive fault enumeration remain below the verifier budget?
- Sources: [StabilizerBench: Evaluating LLM Agents on Quantum Code Synthesis](https://arxiv.org/abs/2604.21287), [Fault-tolerant flagged CSS state preparation](https://arxiv.org/abs/2408.11894), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/)

## 2. Conditioned fermionic-Gaussian Kitaev quench

- Candidate: `f01_fermionic_gaussian_conditioned_quench--representation_stress--inverse` (`bf2c67e20e90990a`)
- Contract: Evolve a Kitaev-chain Gaussian state, condition on occupation outcomes, and recover parity correlators or hidden quench parameters. Scale past naive dense simulation so the documented TensorCircuit representation is essential. Infer hidden parameters from seeded observations and validate on held-out interventions.
- TensorCircuit path: Use TensorCircuit's FGS covariance engine for Bogoliubov evolution, measurement conditioning, and parity observables; forbid dense Jordan-Wigner statevectors at production scale.
- Oracle: Exact Jordan-Wigner simulation on tiny instances plus an independent covariance-matrix implementation on production instances.
- Why plausibly hard: Requires translating BdG/covariance conventions into an unfamiliar specialized API, then composing measurement conditioning with inverse validation.
- Falsification check: Reject if the pinned FGS API cannot reproduce exact small-system correlators and gradients with fixed ordering conventions.
- Current-suite relation: Adds specialized fermionic simulation and conditional Gaussian measurement, absent from tasks 01-12.
- Human review questions: Is the FGS API stable in the pinned image?; Can parity and occupation ordering be specified without ambiguity?; Does the dense oracle remain independent?
- Sources: [Fermionic Gaussian states: an introduction to numerical approaches](https://arxiv.org/abs/2111.08343), [TensorCircuit fermionic Gaussian state API](https://tensorcircuit.readthedocs.io/en/latest/api/fgs.html), [TensorCircuit: a Quantum Software Framework for the NISQ Era](https://arxiv.org/abs/2205.10091)

## 3. Mixed-state multiparameter SLD-QFIM probe design

- Candidate: `f02_mixed_state_sld_qfim_probe--benchmark--robust_ensemble` (`5b52d68978b11122`)
- Contract: Optimize a noisy probe circuit using the symmetric-logarithmic-derivative quantum Fisher information matrix under explicit rank and conditioning bounds. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Use DMCircuit for parameterized noisy channels and backend Jacobians; compute the stable spectral SLD formula as support code.
- Oracle: Finite-difference density derivatives and an independent eigenbasis SLD implementation with hidden parameter points.
- Why plausibly hard: Combines noisy circuit AD, complex-valued Jacobian conventions, rank-deficient spectral formulas, and a non-scalar design objective.
- Falsification check: Require a lower eigenvalue bound and agreement between AD and finite-difference QFIMs before any solver trial.
- Current-suite relation: Adds multiparameter information geometry and rank-aware mixed-state optimization, unlike current energy/fidelity objectives.
- Human review questions: Is the state rank safely bounded?; Are SLD and parameter-order conventions explicit?; Is the target reachable without optimizer luck?
- Sources: [Quantum Fisher information matrix and multiparameter estimation](https://arxiv.org/abs/1907.08037), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/), [TensorCircuit: a Quantum Software Framework for the NISQ Era](https://arxiv.org/abs/2205.10091)

## 4. Quantum signal-processing phase synthesis

- Candidate: `f09_qsp_phase_synthesis--benchmark--inverse` (`859d844ccc255b7b`)
- Contract: Recover parity-constrained phase factors for a hidden target polynomial and verify the resulting TensorCircuit QSP response on an independent grid. Use the intended production regime with a target expert p95 runtime below 180 seconds. Infer hidden parameters from seeded observations and validate on held-out interventions.
- TensorCircuit path: Implement the QSP SU(2) sequence as a parameterized TensorCircuit and use backend AD to fit phase factors under fixed gauge conventions.
- Oracle: Independent polynomial evaluation, unitarity/parity identities, and dense response checks on hidden grid points.
- Why plausibly hard: Requires mapping polynomial parity and gauge constraints to a numerically stable phase parameterization, not merely composing familiar gates.
- Falsification check: Reject targets whose phase solution is unreachable, ill-conditioned, or dominated by an external classical package.
- Current-suite relation: Adds constructive fault-tolerant-era algorithm synthesis instead of NISQ variational physics.
- Human review questions: Does TensorCircuit remain central rather than a final wrapper?; Is the phase gauge fixed?; Do hidden grid checks prevent polynomial hard-coding?
- Sources: [Finding Angles for Quantum Signal Processing with Machine Precision](https://arxiv.org/abs/2003.02831), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/), [QCircuitBench: A Large-Scale Dataset for Benchmarking Quantum Algorithm Design](https://arxiv.org/abs/2410.07961)

## 5. Dynamic-circuit branch equivalence certificate

- Candidate: `f45_semantics_preserving_routing--benchmark--robust_ensemble` (`d737bd9bcfe13d2a`)
- Contract: Given two measurement-conditioned circuits with resets and feed-forward, construct a branch-sensitive certificate or counterexample showing whether their induced quantum instruments agree. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Execute every declared measurement branch as a TensorCircuit instrument and compare conditional output maps; support code may perform independent Choi linear algebra.
- Oracle: Independent branchwise Choi matrices, trace-preservation identities, and planted inequivalent mutations with exact witnesses.
- Why plausibly hard: Ordinary unitary equivalence is insufficient: the solver must preserve measurement probabilities, normalized conditional states, reset semantics, and later feed-forward across all reachable branches.
- Falsification check: Reject if the pinned TensorCircuit API cannot express every branch directly or if the verifier silently averages away branch-specific errors.
- Current-suite relation: Adds semantic verification of adaptive circuits rather than another feedback-control objective.
- Human review questions: Are zero-probability branches handled explicitly?; Can mutations isolate each dynamic semantic failure?; Is TensorCircuit doing substantive instrument execution?
- Sources: [Equivalence checking of dynamic quantum circuits](https://arxiv.org/abs/2106.01658), [Characterizing and Benchmarking Dynamic Quantum Circuits](https://arxiv.org/abs/2604.03360), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/)

## 6. Quantum Lanczos excited-state spectrum

- Candidate: `f28_quantum_lanczos_excited_states--benchmark--robust_ensemble` (`e4d41b9a070e90b2`)
- Contract: Generate imaginary-time Krylov states and solve a stabilized generalized eigenproblem for low excitations. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Use TensorCircuit state preparation/overlaps and framework AD for variational imaginary time.
- Oracle: Exact eigenspectrum and subspace residual checks.
- Why plausibly hard: A correct solution must build overlap and Hamiltonian matrices from quantum states, diagnose near-linear dependence, regularize without deleting the physical subspace, and report residual-certified excited energies.
- Falsification check: Plant a known spectral gap and reject instances whose target eigenpairs are not stable across a declared regularization interval.
- Current-suite relation: Adds Krylov subspace conditioning and excited states.
- Human review questions: Is the conditioning range difficult but numerically identifiable?; Are basis gauge and truncation rules explicit?; Do residual checks prevent simply returning exact reference energies?
- Sources: [Noise and ill-conditioning in quantum Krylov subspace diagonalization](https://arxiv.org/abs/2407.14431), [Determining eigenstates and thermal states on a quantum computer using quantum imaginary time evolution](https://arxiv.org/abs/1901.07653), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/)

## 7. Non-Markovian memory-channel system identification

- Candidate: `f04_nonmarkovian_memory_identification--benchmark--inverse` (`93e955e196e1bca6`)
- Contract: Identify a recurrent system-memory interaction from intervention sequences and predict held-out multitime observables. Use the intended production regime with a target expert p95 runtime below 180 seconds. Infer hidden parameters from seeded observations and validate on held-out interventions.
- TensorCircuit path: Represent the system plus compact memory as a recurrent TensorCircuit/DMCircuit and differentiate held-out intervention losses.
- Oracle: Generate data from a sealed parameter set and score held-out process predictions, with small Choi/process-tensor cross-checks.
- Why plausibly hard: Requires recognizing a non-Markovian latent memory model, building recurrent interventions, and avoiding parameter-gauge traps.
- Falsification check: Require Fisher/identifiability rank and score predictions rather than raw parameters to eliminate gauge ambiguity.
- Current-suite relation: Adds temporal memory and predictive process identification beyond the Markovian channel calibration in task 04.
- Human review questions: Is the hidden model identifiable up to a declared gauge?; Are interventions informationally complete?; Can held-out predictions be verified deterministically?
- Sources: [Non-Markovian Quantum Process Tomography](https://arxiv.org/abs/2106.11722), [OQuPy: A Python package to efficiently simulate non-Markovian open quantum systems with process tensors](https://arxiv.org/abs/2406.16650), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/)

## 8. Variance-aware circuit cutting reconstruction

- Candidate: `f43_circuit_cutting_reconstruction--benchmark--robust_ensemble` (`de0543455ef7a503`)
- Contract: Cut a structured circuit at fixed wires, execute TensorCircuit subcircuits, and reconstruct selected observables with a bounded sampling overhead. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Use TensorCircuit for every subcircuit and measurement basis; NumPy may perform declared classical recombination.
- Oracle: Uncut small circuits and exact quasiprobability reconstruction identities.
- Why plausibly hard: The task couples quantum subcircuit construction to signed quasiprobability recombination and variance allocation; local correctness does not guarantee an unbiased global estimate.
- Falsification check: Require an exact reconstruction identity before enabling finite-shot scoring and reject cuts whose sampling overhead exceeds the resource envelope.
- Current-suite relation: Adds distributed/cut execution and classical recombination.
- Human review questions: Are basis conventions at each cut fully specified?; Is the shot budget statistically powerful?; Can hidden observables expose sign or normalization mistakes?
- Sources: [Approximate Quantum Circuit Cutting](https://arxiv.org/abs/2212.01270), [Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/)

## 9. Leakage-aware robust GRAPE control

- Candidate: `f16_robust_leakage_grape--benchmark--robust_ensemble` (`998a859f87067348`)
- Contract: Optimize a qutrit-aware control pulse across detuning and amplitude-error ensembles while penalizing leakage. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Use TensorCircuit qudit/custom-gate primitives with Diffrax/JAX evolution and automatic differentiation.
- Oracle: Independent matrix-exponential propagation and gradient finite differences on hidden ensemble members.
- Why plausibly hard: The solver must reconcile qutrit leakage, differentiable time evolution, control constraints, and worst-case ensemble performance; a pulse that works only at the nominal Hamiltonian is insufficient.
- Falsification check: Reject if the pinned image lacks a stable differentiable propagator, the expert cannot meet the runtime gate, or random restarts dominate the outcome.
- Current-suite relation: Extends task 06 to robust pulse-level control and leakage rather than variational analog blocks.
- Human review questions: Are amplitude, bandwidth, and leakage conventions complete?; Can hidden ensemble members test robustness without optimizer lottery?; Does the independent propagator agree at every gate boundary?
- Sources: [Optimal control of large quantum systems: assessing memory and runtime performance of GRAPE](https://arxiv.org/abs/2304.06200), [TensorCircuit documentation](https://tensorcircuit.readthedocs.io/en/latest/)

## 10. Memory-constrained contraction plan and selected amplitudes

- Candidate: `f50_hardware_metric_cross_benchmark--representation_stress--forward` (`3d13f511047791d9`)
- Contract: Plan and execute a tensor-network contraction for a structured nonlocal circuit under an explicit peak-memory budget, returning selected amplitudes and contraction-cost evidence. Scale past naive dense simulation so the documented TensorCircuit representation is essential. Compute a physically meaningful observable or state diagnostic from fixed inputs.
- TensorCircuit path: Build the circuit and execute the accepted contraction plan through TensorCircuit/TensorNetwork; dense statevector materialization is infeasible at the production scale.
- Oracle: Exact amplitudes on reduced circuits, independently contracted production amplitudes, and measured peak-memory/cost checks for the submitted plan.
- Why plausibly hard: The solver must translate circuit structure into an executable contraction strategy, respect an operational memory cap, and obtain correct amplitudes without an exponential intermediate.
- Falsification check: Reject instances whose graph structure makes every contraction exponential under the cap or whose reference plan relies on an unpinned private optimizer.
- Current-suite relation: Turns scalable simulation into a constructive planning problem with a checkable scientific output, complementing representation-only tasks 08-10.
- Human review questions: Is the accepted plan feasible on the actual Docker memory limit?; Can peak memory be measured reproducibly?; Do hidden amplitudes prevent hard-coded outputs while remaining independently checkable?
- Sources: [Tensor-network contraction path planning for quantum circuits](https://arxiv.org/abs/2209.02895), [Simulating quantum computation by contracting tensor networks](https://arxiv.org/abs/quant-ph/0511069), [TensorCircuit: a Quantum Software Framework for the NISQ Era](https://arxiv.org/abs/2205.10091), [SupermarQ: A Scalable Quantum Benchmark Suite](https://arxiv.org/abs/2202.11045)

## Human-gated path to a real test

1. Three concept roles approve: quantum scientist, TensorCircuit expert, and verifier engineer.
2. An expert implementation and independent evaluator are built outside `tasks/`.
3. Public-API canary and 20 verifier-only reproductions pass in the pinned image.
4. Gold code is at most 160 effective lines; expert p95 is at most 180 seconds; the independent oracle is at most 240 seconds.
5. Benchmark owner and independent reviewer approve the frozen pilot package.
6. The CLI emits a hash-bound run manifest. It never launches Harbor itself.
7. Run at least five independent GPT-5.6-sol trials under one frozen prompt/image/protocol.
8. Human audit excludes infrastructure, auth, policy refusal, ambiguous spec, and framework impossibility failures.

Only failures that remain after step 8 count toward evidence that a task is model-hard.
