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
| 1 | `f03_fault_tolerant_css_state_preparation--benchmark--robust_ensemble` — Fault-tolerant CSS state-preparation synthesis | qec | 98.0/100 | 5/5 | 5/5 |
| 2 | `f01_fermionic_gaussian_conditioned_quench--representation_stress--inverse` — Conditioned fermionic-Gaussian Kitaev quench | fermions | 89.6/100 | 4/5 | 4/5 |
| 3 | `f02_mixed_state_sld_qfim_probe--benchmark--robust_ensemble` — Mixed-state multiparameter SLD-QFIM probe design | metrology | 92.8/100 | 4/5 | 5/5 |
| 4 | `f09_qsp_phase_synthesis--benchmark--inverse` — Quantum signal-processing phase synthesis | algorithms | 95.6/100 | 5/5 | 4/5 |
| 5 | `f45_dynamic_circuit_branch_equivalence--benchmark--adaptive` — Dynamic-circuit branch equivalence certificate | verification | 91.6/100 | 4/5 | 4/5 |
| 6 | `f28_quantum_lanczos_excited_states--benchmark--robust_ensemble` — Quantum Lanczos excited-state spectrum | many_body | 92.8/100 | 4/5 | 5/5 |
| 7 | `f04_nonmarkovian_memory_identification--benchmark--inverse` — Non-Markovian memory-channel system identification | open_systems | 91.6/100 | 4/5 | 4/5 |
| 8 | `f43_circuit_cutting_reconstruction--benchmark--finite_shot_noise` — Variance-aware circuit cutting reconstruction | distributed_quantum | 91.6/100 | 5/5 | 4/5 |
| 9 | `f16_robust_leakage_grape--benchmark--robust_ensemble` — Leakage-aware robust GRAPE control | control | 88.0/100 | 4/5 | 4/5 |
| 10 | `f50_memory_constrained_contraction--representation_stress--forward` — Memory-constrained contraction plan and selected amplitudes | simulation | 91.6/100 | 5/5 | 5/5 |

## 1. Fault-tolerant CSS state-preparation synthesis

- Candidate: `f03_fault_tolerant_css_state_preparation--benchmark--robust_ensemble` (`b681a83d4e4f4cee9200b9cde0716ad36617fd54d83dd80f7860210af8d66897`)
- Contract: Synthesize a CSS logical-state preparation and verification circuit that preserves the target stabilizers and detects every declared low-weight propagated fault. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Construct and execute the candidate encoder, verification checks, and injected Pauli faults as TensorCircuit circuits; use tableau logic only as an independent verifier.
- Oracle: Polynomial stabilizer/isometry checks plus exhaustive single-fault propagation and exact small-code state verification.
- Why plausibly hard: Requires constructive stabilizer reasoning, verification-circuit synthesis, propagation analysis, and exact fault-tolerance semantics; this is supported by direct agent-benchmark evidence rather than size alone.
- Falsification check: Reject if the held-out CSS family lacks a short expert construction, if any fault class is underspecified, or if the TensorCircuit execution is only a wrapper around a classical tableau answer.
- Current-suite relation: Directly fills the QEC/fault-tolerant-synthesis roadmap gap; StabilizerBench reports substantial headroom across three frontier agents, none of which was GPT-5.6-sol.
- Human review questions: Which held-out CSS codes have short expert circuits?; Are all injected fault locations and acceptance rules explicit?; Does exhaustive fault enumeration remain below the verifier budget?
- Evidence roles: scientific_basis=Automated Synthesis of Fault-Tolerant State Preparation Circuits for Quantum Error Correction Codes; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation; agent_evidence=StabilizerBench: A Benchmark for AI-Assisted Quantum Error Correction Circuit Synthesis
- Sources: [StabilizerBench: A Benchmark for AI-Assisted Quantum Error Correction Circuit Synthesis](https://arxiv.org/abs/2604.21287), [Automated Synthesis of Fault-Tolerant State Preparation Circuits for Quantum Error Correction Codes](https://arxiv.org/abs/2408.11894), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/)

## 2. Conditioned fermionic-Gaussian Kitaev quench

- Candidate: `f01_fermionic_gaussian_conditioned_quench--representation_stress--inverse` (`40456821bcf45981180886097788fcf6867750de17ee90840ace3b7785bcbf06`)
- Contract: Evolve a Kitaev-chain Gaussian state, condition on occupation outcomes, and recover parity correlators or hidden quench parameters. Scale past naive dense simulation so the documented TensorCircuit representation is essential. Infer hidden parameters from seeded observations and validate on held-out interventions.
- TensorCircuit path: Use TensorCircuit's FGS covariance engine for Bogoliubov evolution, measurement conditioning, and parity observables; forbid dense Jordan-Wigner statevectors at production scale.
- Oracle: Exact Jordan-Wigner simulation on tiny instances plus an independent covariance-matrix implementation on production instances.
- Why plausibly hard: Requires translating BdG/covariance conventions into an unfamiliar specialized API, then composing measurement conditioning with inverse validation.
- Falsification check: Reject if the pinned FGS API cannot reproduce exact small-system correlators and gradients with fixed ordering conventions.
- Current-suite relation: Adds specialized fermionic simulation and conditional Gaussian measurement, absent from tasks 01-12.
- Human review questions: Is the FGS API stable in the pinned image?; Can parity and occupation ordering be specified without ambiguity?; Does the dense oracle remain independent?
- Evidence roles: scientific_basis=Fermionic Gaussian states: an introduction to numerical approaches; framework_evidence=TensorCircuit fermionic Gaussian state API, TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation
- Sources: [Fermionic Gaussian states: an introduction to numerical approaches](https://arxiv.org/abs/2111.08343), [TensorCircuit fermionic Gaussian state API](https://tensorcircuit.readthedocs.io/en/latest/api/fgs.html), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167)

## 3. Mixed-state multiparameter SLD-QFIM probe design

- Candidate: `f02_mixed_state_sld_qfim_probe--benchmark--robust_ensemble` (`d0947debfeb60374c68778fc577b94e81ea54baabf8e4348ecaa687d0e85101e`)
- Contract: Optimize a noisy probe circuit using the symmetric-logarithmic-derivative quantum Fisher information matrix on a fixed support whose nonzero eigenvalues obey an explicit lower bound. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Use DMCircuit for parameterized noisy channels and backend Jacobians; compute the stable spectral SLD formula as support code.
- Oracle: Finite-difference density derivatives and an independent eigenbasis SLD implementation with hidden parameter points.
- Why plausibly hard: Combines noisy circuit AD, complex-valued Jacobian conventions, support-restricted spectral formulas, and a non-scalar design objective.
- Falsification check: Require a fixed nullspace, a lower bound on every nonzero support eigenvalue, and agreement between AD and finite-difference QFIMs before any solver trial.
- Current-suite relation: Adds multiparameter information geometry and rank-aware mixed-state optimization, unlike current energy/fidelity objectives.
- Human review questions: Is the density-matrix support fixed over the full parameter domain?; Are SLD and parameter-order conventions explicit?; Is the target reachable without optimizer luck?
- Evidence roles: scientific_basis=Quantum Fisher information matrix and multiparameter estimation, Discontinuities of the quantum Fisher information and the Bures metric, On the discontinuity of the quantum Fisher information for quantum statistical models with parameter dependent rank; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation
- Sources: [Quantum Fisher information matrix and multiparameter estimation](https://arxiv.org/abs/1907.08037), [Discontinuities of the quantum Fisher information and the Bures metric](https://arxiv.org/abs/1612.04581), [On the discontinuity of the quantum Fisher information for quantum statistical models with parameter dependent rank](https://arxiv.org/abs/1906.06185), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/)

## 4. Quantum signal-processing phase synthesis

- Candidate: `f09_qsp_phase_synthesis--benchmark--inverse` (`c364fda8c6b81045f5b6c3c4ea47c111202630e6cf0a7861f1ae13b4f4a2bfe6`)
- Contract: Recover parity-constrained phase factors for a hidden target polynomial and verify the resulting TensorCircuit QSP response on an independent grid. Use the intended production regime with a target expert p95 runtime below 180 seconds. Infer hidden parameters from seeded observations and validate on held-out interventions.
- TensorCircuit path: Implement the QSP SU(2) sequence as a parameterized TensorCircuit and use backend AD to fit phase factors under fixed gauge conventions.
- Oracle: Independent polynomial evaluation, unitarity/parity identities, and dense response checks on hidden grid points.
- Why plausibly hard: Requires mapping polynomial parity and gauge constraints to a numerically stable phase parameterization, not merely composing familiar gates.
- Falsification check: Reject targets whose phase solution is unreachable, ill-conditioned, or dominated by an external classical package.
- Current-suite relation: Adds constructive fault-tolerant-era algorithm synthesis instead of NISQ variational physics.
- Human review questions: Does TensorCircuit remain central rather than a final wrapper?; Is the phase gauge fixed?; Do hidden grid checks prevent polynomial hard-coding?
- Evidence roles: scientific_basis=Finding Angles for Quantum Signal Processing with Machine Precision, Efficient phase-factor evaluation in quantum signal processing; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation; benchmark_context=QCircuitBench: A Large-Scale Dataset for Benchmarking Quantum Algorithm Design
- Sources: [Finding Angles for Quantum Signal Processing with Machine Precision](https://arxiv.org/abs/2003.02831), [Efficient phase-factor evaluation in quantum signal processing](https://arxiv.org/abs/2002.11649), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/), [QCircuitBench: A Large-Scale Dataset for Benchmarking Quantum Algorithm Design](https://arxiv.org/abs/2410.07961)

## 5. Dynamic-circuit branch equivalence certificate

- Candidate: `f45_dynamic_circuit_branch_equivalence--benchmark--adaptive` (`33fec298a816426595a6b980025906b1dfb9616757a3ea2b87204c5ef187adf7`)
- Contract: Given two measurement-conditioned circuits with resets and feed-forward, construct a branch-sensitive certificate or counterexample showing whether their induced quantum instruments agree. Use the intended production regime with a target expert p95 runtime below 180 seconds. Use intermediate results to choose later circuit operations or experiments.
- TensorCircuit path: Explicitly enumerate postselected/Kraus branches in TensorCircuit and compare their conditional output maps; support code may perform independent Choi linear algebra.
- Oracle: Independent branchwise Choi matrices, trace-preservation identities, and planted inequivalent mutations with exact witnesses.
- Why plausibly hard: Ordinary unitary equivalence is insufficient: the solver must preserve measurement probabilities, normalized conditional states, reset semantics, and later feed-forward across all reachable branches.
- Falsification check: Reject unless pinned-image canaries cover reset, zero-probability branches, normalization, and classical feed-forward, or if the verifier averages away branch-specific errors.
- Current-suite relation: Adds semantic verification of adaptive circuits rather than another feedback-control objective.
- Human review questions: Are zero-probability branches handled explicitly?; Can mutations isolate reset, normalization, and feed-forward failures?; Is TensorCircuit doing substantive branch execution?
- Evidence roles: scientific_basis=Equivalence checking of dynamic quantum circuits, Characterizing and Benchmarking Dynamic Quantum Circuits; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation
- Sources: [Equivalence checking of dynamic quantum circuits](https://arxiv.org/abs/2106.01658), [Characterizing and Benchmarking Dynamic Quantum Circuits](https://arxiv.org/abs/2604.03360), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/)

## 6. Quantum Lanczos excited-state spectrum

- Candidate: `f28_quantum_lanczos_excited_states--benchmark--robust_ensemble` (`684ddb0b1711485be21b185a5e1ffcd467833ba8faac273d2d165e4424864f72`)
- Contract: Generate imaginary-time Krylov states and solve a stabilized generalized eigenproblem for low excitations. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Use TensorCircuit state preparation/overlaps and framework AD for variational imaginary time.
- Oracle: Exact eigenspectrum and subspace residual checks.
- Why plausibly hard: A correct solution must build overlap and Hamiltonian matrices from quantum states, diagnose near-linear dependence, regularize without deleting the physical subspace, and report residual-certified excited energies.
- Falsification check: Plant a known spectral gap and reject instances whose target eigenpairs are not stable across a declared regularization interval.
- Current-suite relation: Adds Krylov subspace conditioning and excited states.
- Human review questions: Is the conditioning range difficult but numerically identifiable?; Are basis gauge and truncation rules explicit?; Do residual checks prevent simply returning exact reference energies?
- Evidence roles: scientific_basis=Sampling Error Analysis in Quantum Krylov Subspace Diagonalization, Variational ansatz-based quantum simulation of imaginary time evolution, Determining eigenstates and thermal states on a quantum computer using quantum imaginary time evolution; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation
- Sources: [Sampling Error Analysis in Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2307.16279), [Variational ansatz-based quantum simulation of imaginary time evolution](https://arxiv.org/abs/1804.03023), [Determining eigenstates and thermal states on a quantum computer using quantum imaginary time evolution](https://arxiv.org/abs/1901.07653), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/)

## 7. Non-Markovian memory-channel system identification

- Candidate: `f04_nonmarkovian_memory_identification--benchmark--inverse` (`e314b459f2436fe5725b191875c5be552a44c3c3e05a59a567080379bb587840`)
- Contract: Identify a recurrent system-memory interaction from intervention sequences and predict held-out multitime observables. Use the intended production regime with a target expert p95 runtime below 180 seconds. Infer hidden parameters from seeded observations and validate on held-out interventions.
- TensorCircuit path: Represent the system plus compact memory as a recurrent TensorCircuit/DMCircuit and differentiate held-out intervention losses.
- Oracle: Generate data from a sealed parameter set and score held-out process predictions, with small Choi/process-tensor cross-checks.
- Why plausibly hard: Requires recognizing a non-Markovian latent memory model, building recurrent interventions, and avoiding parameter-gauge traps.
- Falsification check: Require Fisher/identifiability rank and score predictions rather than raw parameters to eliminate gauge ambiguity.
- Current-suite relation: Adds temporal memory and predictive process identification beyond the Markovian channel calibration in task 04.
- Human review questions: Is the hidden model identifiable up to a declared gauge?; Are interventions informationally complete?; Can held-out predictions be verified deterministically?
- Evidence roles: scientific_basis=Non-Markovian Quantum Process Tomography, Quantum Process Identification: A Method for Characterizing Non-Markovian Quantum Dynamics, OQuPy: A Python package to efficiently simulate non-Markovian open quantum systems with process tensors; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation
- Sources: [Non-Markovian Quantum Process Tomography](https://arxiv.org/abs/2106.11722), [Quantum Process Identification: A Method for Characterizing Non-Markovian Quantum Dynamics](https://arxiv.org/abs/1803.02438), [OQuPy: A Python package to efficiently simulate non-Markovian open quantum systems with process tensors](https://arxiv.org/abs/2406.16650), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/)

## 8. Variance-aware circuit cutting reconstruction

- Candidate: `f43_circuit_cutting_reconstruction--benchmark--finite_shot_noise` (`d6eddc62baedc740d27db395f09dec808161f4adc087f96f3e5e9f8151ae4952`)
- Contract: Cut a structured circuit at fixed wires, execute TensorCircuit subcircuits, and reconstruct selected observables with a bounded sampling overhead. Use the intended production regime with a target expert p95 runtime below 180 seconds. Operate under an explicit seeded noise and shot budget with a precomputed variance margin.
- TensorCircuit path: Use TensorCircuit for every subcircuit and measurement basis; NumPy may perform declared classical recombination.
- Oracle: Uncut small circuits and exact quasiprobability reconstruction identities.
- Why plausibly hard: The task couples quantum subcircuit construction to signed quasiprobability recombination and variance allocation; local correctness does not guarantee an unbiased global estimate.
- Falsification check: Require an exact reconstruction identity before enabling finite-shot scoring and reject cuts whose sampling overhead exceeds the resource envelope.
- Current-suite relation: Adds distributed/cut execution and classical recombination.
- Human review questions: Are basis conventions at each cut fully specified?; Is the shot budget statistically powerful?; Can hidden observables expose sign or normalization mistakes?
- Evidence roles: scientific_basis=Simulating Large Quantum Circuits on a Small Quantum Computer, Optimal Partitioning of Quantum Circuits using Gate Cuts and Wire Cuts, Enhanced Quantum Circuit Cutting Framework for Sampling Overhead Reduction; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation
- Sources: [Simulating Large Quantum Circuits on a Small Quantum Computer](https://arxiv.org/abs/1904.00102), [Optimal Partitioning of Quantum Circuits using Gate Cuts and Wire Cuts](https://arxiv.org/abs/2308.09567), [Enhanced Quantum Circuit Cutting Framework for Sampling Overhead Reduction](https://arxiv.org/abs/2412.17704), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/)

## 9. Leakage-aware robust GRAPE control

- Candidate: `f16_robust_leakage_grape--benchmark--robust_ensemble` (`fb1880db3d041ae7414b7ccdb97dee9546ff080b1dedef05ce6c53050ceabdf7`)
- Contract: Optimize a qutrit-aware control pulse across detuning and amplitude-error ensembles while penalizing leakage. Use the intended production regime with a target expert p95 runtime below 180 seconds. Satisfy one frozen objective across a seeded ensemble of perturbations or hidden instances.
- TensorCircuit path: Use TensorCircuit qudit/custom-gate primitives with Diffrax/JAX evolution and automatic differentiation.
- Oracle: Independent matrix-exponential propagation and gradient finite differences on hidden ensemble members.
- Why plausibly hard: The solver must reconcile qutrit leakage, differentiable time evolution, control constraints, and worst-case ensemble performance; a pulse that works only at the nominal Hamiltonian is insufficient.
- Falsification check: Reject if the pinned image lacks a stable differentiable propagator, the expert cannot meet the runtime gate, or random restarts dominate the outcome.
- Current-suite relation: Extends task 06 to robust pulse-level control and leakage rather than variational analog blocks.
- Human review questions: Are amplitude, bandwidth, and leakage conventions complete?; Can hidden ensemble members test robustness without optimizer lottery?; Does the independent propagator agree at every gate boundary?
- Evidence roles: scientific_basis=Optimal control of large quantum systems: assessing memory and runtime performance of GRAPE, Optimal control of a leaking qubit, Risk-sensitive Optimization for Robust Quantum Controls; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation, TensorCircuit-NG documentation
- Sources: [Optimal control of large quantum systems: assessing memory and runtime performance of GRAPE](https://arxiv.org/abs/2304.06200), [Optimal control of a leaking qubit](https://arxiv.org/abs/0808.2680), [Risk-sensitive Optimization for Robust Quantum Controls](https://arxiv.org/abs/2104.01323), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [TensorCircuit-NG documentation](https://tensorcircuit-ng.readthedocs.io/en/latest/)

## 10. Memory-constrained contraction plan and selected amplitudes

- Candidate: `f50_memory_constrained_contraction--representation_stress--forward` (`09d873ad318aa815cd242ef317f0574e4b0060383d670b475119ce1c063b2610`)
- Contract: Plan and execute a tensor-network contraction for a structured nonlocal circuit under an explicit peak-memory budget, returning selected amplitudes and contraction-cost evidence. Scale past naive dense simulation so the documented TensorCircuit representation is essential. Compute a physically meaningful observable or state diagnostic from fixed inputs.
- TensorCircuit path: Build the circuit and execute the accepted contraction plan through TensorCircuit/TensorNetwork; dense statevector materialization is infeasible at the production scale.
- Oracle: Exact amplitudes on reduced circuits, independently contracted production amplitudes, and measured peak-memory/cost checks for the submitted plan.
- Why plausibly hard: The solver must translate circuit structure into an executable contraction strategy, respect an operational memory cap, and obtain correct amplitudes without an exponential intermediate.
- Falsification check: Reject instances whose graph structure makes every contraction exponential under the cap or whose reference plan relies on an unpinned private optimizer.
- Current-suite relation: Turns scalable simulation into a constructive planning problem with a checkable scientific output, complementing representation-only tasks 08-10.
- Human review questions: Is the accepted plan feasible on the actual Docker memory limit?; Can peak memory be measured reproducibly?; Do hidden amplitudes prevent hard-coded outputs while remaining independently checkable?
- Evidence roles: scientific_basis=Constructing Optimal Contraction Trees for Tensor Network Quantum Circuit Simulation, Simulating quantum computation by contracting tensor networks; framework_evidence=TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation; benchmark_context=SupermarQ: A Scalable Quantum Benchmark Suite
- Sources: [Constructing Optimal Contraction Trees for Tensor Network Quantum Circuit Simulation](https://arxiv.org/abs/2209.02895), [Simulating quantum computation by contracting tensor networks](https://arxiv.org/abs/quant-ph/0511069), [TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation](https://arxiv.org/abs/2602.14167), [SupermarQ: A Scalable Quantum Benchmark Suite](https://arxiv.org/abs/2202.11045)

## Human-gated path to a real test

1. Three concept roles approve: quantum scientist, TensorCircuit expert, and verifier engineer.
2. An expert implementation and independent evaluator are built outside `tasks/`.
3. Public-API canary and 20 verifier-only reproductions pass in the pinned image.
4. Gold code is at most 160 effective lines; expert p95 is at most 180 seconds; the independent oracle is at most 240 seconds.
5. Benchmark owner and independent reviewer approve the frozen pilot package.
6. The CLI emits a hash-bound run manifest. It never launches Harbor itself.
7. Run a precommitted five-trial GPT-5.6-sol pilot under one frozen protocol.
8. Two independent auditors classify every failure and exclude infrastructure, auth, policy refusal, ambiguous spec, and framework impossibility.
9. A model-hard evidence label requires a fresh precommitted confirmation with at least 20 valid trials and zero passes.

Five failures are only a pilot signal. Only failures that remain after step 9 count toward a protocol-scoped model-hard evidence label.
