# Problem 103: Branch-sensitive equivalence of dynamic circuits

Implement a branch-sensitive equivalence checker for pairs of dynamic quantum
circuits.  Each circuit contains unitary segments, one or two computational-
basis mid-circuit measurements, branch-dependent feed-forward gates, and
optional reset-to-zero operations.

This is an *instrument* comparison.  Comparing only the final state after
averaging over measurement outcomes is incorrect: every classical history must
have the same probability and the same unnormalized conditional output state.

## Input

The evaluator calls `run_solution(config)`.  `config["cases"]` is a hidden,
seeded list.  Each case contains:

- `n_qubits` and a list of pure-state preparation circuits in `probes`;
- programs `program_a` and `program_b`;
- each program has `prefix`, `rounds`, and `final` gate lists;
- each round has `measure`, two `branches`, and a Boolean `reset` flag.

Gate records use `name`, `qubits`, and, for rotations, `theta`.  Supported names
are `h`, `x`, `s`, `sd`, `rx`, `ry`, `rz`, `cx`, and `cz`.

## Required computation

For every probe and every bit history, execute both programs with
TensorCircuit-NG.  Use TensorCircuit's explicit `post_select` (also exposed as
`mid_measurement`) operation for the selected Kraus/projector branch.  Its
state is deliberately unnormalized.  After a selected outcome `1`, a reset
means applying `X` before later operations.

For history \(h\), form the unnormalized branch density matrix

\[
R_h = |\psi_h\rangle\langle\psi_h|.
\]

The case distance is the maximum Frobenius norm
`||R_a - R_b||` over all supplied probes and classical histories.  A case is
equivalent exactly when this distance is at most
`config["equivalence_tolerance"]`.

Do not renormalize branch states, discard zero-probability histories, average
over histories, or replace the core computation with a raw NumPy simulator.
NumPy may be used for small reductions after TensorCircuit has executed the
branches.

## Output

Return a dictionary with NumPy-compatible values:

```python
{
    "instrument_distances": float array, shape (number_of_cases,),
    "equivalent": bool array, shape (number_of_cases,),
    "completeness_errors": float array, shape (number_of_cases,),
}
```

For one program and probe, the completeness error is
`abs(sum_h trace(R_h) - 1)`.  Report the maximum across the two programs and all
probes in each case.

## Submission

Create `/root/solution_103.py` with `run_solution(config)`.  The timed call must
finish within 180 seconds.  Keep the implementation at or below 160 effective
non-empty, non-comment Python lines.  Do not inspect evaluator secrets, hidden
oracle files, logs, or reward files.
