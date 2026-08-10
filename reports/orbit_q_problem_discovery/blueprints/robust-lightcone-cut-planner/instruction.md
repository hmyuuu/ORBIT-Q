# Problem 119: Robust observable-lightcone cut planning

Synthesize a capacity-feasible partition and one shared finite-shot allocation
for a hidden shallow many-qubit circuit.  The plan must be robust across every
supplied coherent calibration scenario and every supplied local Pauli
observable.  The full production circuit has 76 qubits, so constructing its
dense state is not a valid solution.

This is a pre-execution planning problem.  It scores the declared
quasiprobability/stratified variance model below; it does not ask you to invent
sample outcomes or claim that this proxy is an exact hardware MSE.

## Circuit and scenarios

`run_solution(config)` receives one `config["case"]`.  Qubits start in
`|0>`.  For scenario `u`, execute in this order:

1. on every qubit `q`, `RY((1+drive_scale)*initial_ry[q] + initial_shift)`
   followed by `RZ(initial_rz[q])`;
2. all supplied interaction records in list order;
3. on every qubit `q`, `RX((1+drive_scale)*final_rx[q] + final_shift)`
   followed by `RZ(final_rz[q])`.

An interaction record has a layer, two qubits, an axis in `XX`, `YY`, `ZZ`,
and base angle `theta`.  In scenario `u`, its effective angle is

```text
theta_u = (1 + interaction_scale[u]) * theta
          + layer_shifts[u][layer].
```

For `PP` in `XX`, `YY`, `ZZ`, the gate is

```text
RPP(theta_u) = exp(-i * theta_u * (P tensor P) / 2).
```

For every supplied single-site Pauli observable `o`, compute its exact pilot
expectation `E[u,o]` with TensorCircuit-NG.  The interaction layers are
matchings.  A gate belongs to the observable's backward lightcone when, on a
reverse layer scan, at least one endpoint is already in the support; both
endpoints then enter the support.  Gates outside that lightcone cannot affect
the pilot expectation and must not be charged to that observable.

## Partition artifact

Return exactly `fragment_count` fragments in the order of the supplied opaque
`fragment_ids`.  Their concatenated qubit lists must equal `layout_order`
exactly.  Thus every fragment is a contiguous interval of that hardware
layout.  Every interval must contain between `minimum_fragment_qubits` and
`maximum_fragment_qubits`, inclusive.

An interaction is cut precisely when its two qubits lie in different
fragments.  Return all cut gate IDs in original gate order.  Also return, for
each observable in input order, the intersection of the cut-gate list with
that observable's backward lightcone, again in original gate order.

## Robust variance model and shot allocation

For an effective cut `RPP(theta_u)`, use the independent single-gate QPD
factor

```text
gamma[u,e] = 1 + 2*abs(sin(theta_u)).
```

For partition `Pi`, the observable-specific squared overhead is

```text
G[u,o,Pi] = product over cut gates e in lightcone(o) of gamma[u,e]**2.
```

The case supplies execution strata `g`, a nonnegative
`stratum_mixture[o,g]` whose row sums to one, and scenario-dependent calibrated
`stratum_excess_variance[u,g]`.  With integer shots `n[g]`, define

```text
A[u,o,g] = G[u,o,Pi] * stratum_mixture[o,g]**2
           * (1 - E[u,o]**2 + stratum_excess_variance[u,g]),

V[u,o] = sum_g A[u,o,g] / n[g].
```

Every stratum receives at least `minimum_shots_per_stratum`, and the integer
allocations sum exactly to `total_shots`.  The same allocation is shared by all
scenarios and observables.  The joint robust objective is

```text
robust_variance = max over u,o of V[u,o].
```

Synthesize any feasible partition/allocation at or below the supplied
`maximum_robust_variance`.  The verifier deterministically derives that cutoff
for each hidden case and accepts better plans, but does not certify global
integer minimax optimality.  The generator's balanced-layout reference
allocation and a uniform shot split on any feasible partition are gated away
from the cutoff.

## Output

Return exactly

```python
{
    "case_id": str,
    "fragments": [
        {"fragment_id": str, "qubits": [int, ...]},
        ...,
    ],
    "cut_gate_ids": [str, ...],
    "lightcone_cut_gate_ids": [[str, ...], ...],
    "pilot_expectations": [[finite_real_for_observable, ...], ...],
    "shot_allocation": [
        {"stratum_id": str, "shots": int},
        ...,
    ],
    "predicted_variances": [[finite_real_for_observable, ...], ...],
    "robust_variance": finite_real,
    "maximum_log_overhead": finite_real,
}
```

Rows of `pilot_expectations` and `predicted_variances` follow scenario order;
columns follow observable order.  `maximum_log_overhead` is
`max_(u,o) log(G[u,o,Pi])`.  All identities, lists, pilot values, variances,
budgets, and certificates are independently recomputed.

## Submission

Create `/root/solution_119.py` with `run_solution(config)`.  TensorCircuit-NG
must perform every quantum pilot computation.  NumPy may be used for graph
bookkeeping and deterministic constrained optimization, but not to replace
the quantum calculation with a standalone simulator.  Do not form the full
76-qubit state.

The timed call must finish within 300 seconds and use at most 160 effective
non-empty, non-comment Python lines.  Do not inspect verifier files, hidden
seeds, `/tests`, `/logs`, rewards, secrets, or an oracle solution.
