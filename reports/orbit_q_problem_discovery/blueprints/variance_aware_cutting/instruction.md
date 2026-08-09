# Problem 104: Variance-aware signed circuit cutting

Reconstruct hidden Pauli expectations by cutting one quantum wire between two
fragments.  Every upstream and downstream fragment evaluation must be executed
with TensorCircuit-NG.  The reconstruction is signed and uses a fixed finite
shot budget; simulating the uncut circuit is not a valid solution.

## Wire-cut identity

The left fragment produces a one-qubit reduced state at its boundary with Bloch
components \(r_P=\langle P\rangle\), for \(P\in\{X,Y,Z\}\).  Let
\(e_{P,+}\) and \(e_{P,-}\) be the requested downstream Pauli expectation when
the boundary input is prepared in the plus or minus eigenstate of \(P\).
Reconstruct

\[
E = \frac{e_{Z,+}+e_{Z,-}}{2}
  + \frac{1}{2}\sum_{P=X,Y,Z} r_P(e_{P,+}-e_{P,-}).
\]

The minus signs are essential.  The evaluator checks the fragment quantities
against an independent uncut statevector oracle.

## Hidden input

`run_solution(config)` receives `config["cases"]`.  Each hidden seeded case has
`n_left`, `n_right`, unitary gate lists `left_gates` and `right_gates`, a
downstream Pauli string `observable_ps`, `total_shots`, `min_shots`, and a
`sample_seed`.  Gate records support `h`, `x`, `s`, `sd`, `rx`, `ry`, `rz`,
`cx`, and `cz`.

Use component order

```text
rX, rY, rZ, eX+, eX-, eY+, eY-, eZ+, eZ-
```

For their exact means \(m_i\), use the first-order coefficients

```text
0.5(eX+ - eX-), 0.5(eY+ - eY-), 0.5(eZ+ - eZ-),
0.5 rX, -0.5 rX, 0.5 rY, -0.5 rY,
0.5(1 + rZ), 0.5(1 - rZ).
```

Allocate `min_shots` to every component.  Distribute the remaining shots in
proportion to `sqrt(coefficient**2 * (1 - mean**2) + 1e-14)`: take floors,
then give residual shots to descending fractional parts, breaking ties by the
component order above.

For deterministic finite-shot values, instantiate
`numpy.random.default_rng(sample_seed)`, visit components in the stated order,
and sample `k = rng.binomial(n, (1 + mean) / 2)`.  The sampled mean is
`(2*k - n)/n`.  Insert sampled values into the signed identity.

## Output

Return NumPy-compatible arrays:

```python
{
  "reconstructions":             shape (cases,),
  "allocations":                 shape (cases, 9), integer,
  "predicted_variances":         shape (cases,),
  "exact_reconstructions":       shape (cases,),
  "upstream_bloch":              shape (cases, 3),
  "downstream_responses":        shape (cases, 6),
}
```

The predicted variance is `sum(coefficient**2 * (1-mean**2) / shots)` for the
reported allocation.  Downstream response order is
`eX+, eX-, eY+, eY-, eZ+, eZ-`.

## Submission

Create `/root/solution_104.py` with `run_solution(config)`.  The timed call must
finish within 180 seconds and the implementation must contain at most 160
effective non-empty, non-comment Python lines.  NumPy is allowed for the
declared allocation, sampling, and recombination only; do not replace fragment
execution with a raw simulator or classical uncut calculation.
