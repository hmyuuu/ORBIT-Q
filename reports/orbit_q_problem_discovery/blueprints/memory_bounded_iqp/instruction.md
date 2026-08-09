# Problem 107: Memory-bounded IQP amplitude contraction

Compute selected amplitudes of hidden, shuffled 72-qubit IQP circuits and give
a checkable contraction-width certificate.  A dense statevector is impossible
at the production size.  The amplitude computation must build the declared
circuit with TensorCircuit-NG and contract its tensor network with a custom,
reusable contraction plan.

## Circuit and amplitudes

`run_solution(config)` receives `config["cases"]`.  Each case contains
`n_qubits`, real `rz_angles`, records `[u, v, theta]` in `rzz_edges`, selected
computational-basis `bitstrings`, and `logical_width_limit`.

For every case, start in `|0>**n`, then apply, in this exact order:

1. `H` on every qubit in ascending index order;
2. `RZ(rz_angles[q])` on every qubit in ascending index order;
3. `RZZ(theta)` for every edge record in the supplied order;
4. `H` on every qubit in ascending index order.

Use `RZ(a) = exp(-i a Z/2)` and
`RZZ(a) = exp(-i a Z tensor Z/2)`.  For each requested string `x`, compute the
scaled amplitude

```text
scaled_amplitude(x) = 2**(n_qubits/2) * <x|U|0...0>.
```

The scaling keeps chaotic-circuit amplitudes numerically well conditioned; it
does not change the contraction problem.

## Logical contraction certificate

Ignoring the one-body factors, form the undirected interaction graph whose
vertices are `0..n_qubits-1` and whose edges are the pairs in `rzz_edges`.
Return one full vertex-elimination order per case.  To score an order, maintain
the graph induced by the remaining vertices.  When eliminating `v`, connect
every pair of its current neighbors (fill edges), record the current degree of
`v`, then remove `v`.  The induced width is the maximum recorded degree.

The order must be a permutation of every vertex and its induced width must not
exceed `logical_width_limit`.  A complex binary factor eliminated at width `w`
has a largest joint table of `2**(w+1)` entries.  Return this exact peak-entry
certificate as well.  Vertex labels are independently shuffled in every
hidden case, so a hard-coded row-major order does not work.

## Output

Return NumPy-compatible values:

```python
{
    "scaled_amplitudes": complex array,      # (n_cases, n_queries)
    "elimination_orders": integer array,     # (n_cases, n_qubits)
    "induced_widths": integer array,         # (n_cases,)
    "peak_complex_entries": integer array,   # (n_cases,)
}
```

All hidden cases currently have the same number of qubits and queries.  The
evaluator independently reconstructs the IQP Ising partition function with
complex128 variable elimination and recomputes every certificate from the
submitted order.

## Submission

Create `/root/solution_107.py` with `run_solution(config)`.  The timed call must
finish within 300 seconds and the implementation must contain at most 160
effective non-empty, non-comment Python lines.  NumPy and installed generic
contraction optimizers may support planning, but the amplitude-producing
quantum tensor network must be built and executed with TensorCircuit-NG.  Do
not materialize a full statevector or replace the circuit with a standalone
NumPy/JAX Ising simulator.
