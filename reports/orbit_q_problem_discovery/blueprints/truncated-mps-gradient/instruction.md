# Problem 111: Differentiable truncated-MPS sweep

Evaluate an explicitly truncated matrix-product-state (MPS) simulation and
differentiate its normalized spin energy.  The production instances are too
large for a dense statevector.  The state evolution and automatic gradient
must use TensorCircuit-NG's `MPSCircuit` with the declared hard bond cap.

## Input schema

`run_solution(config)` receives `config["cases"]`.  Each case is a dictionary
with the following fields:

- `n_qubits`: number of qubits;
- `max_bond_dimension`: the hard SVD rank cap `chi`;
- `parameters`: a real vector of length `3 * n_layers`;
- `y_offsets`, `y_scales`, `z_offsets`, `z_scales`: real arrays of shape
  `(n_layers, n_qubits)`;
- `entangler_offsets`, `entangler_scales`: real arrays of shape
  `(n_layers, n_qubits - 1)`;
- `energy_terms`: records `[weight, operators]`, where `operators` is a list of
  `[qubit, "X"]` or `[qubit, "Z"]` pairs; and
- `case_nonce`: an opaque anti-hardcoding integer.

All angles are in radians.  Qubit zero is the leftmost MPS site.

## Prescribed truncated sweep

Start from `|0>**n` with orthogonality center at site zero.  For layer `l`, set
`a = parameters[3*l:3*l+3]` and do the following in order:

1. On every site `q` in ascending order, apply
   `RY(y_offsets[l,q] + y_scales[l,q] * a[0])`, then
   `RZ(z_offsets[l,q] + z_scales[l,q] * a[1])`.
2. If `l` is even, move the center to site zero and sweep bonds
   `(0,1), (1,2), ..., (n-2,n-1)`.  On bond `(q,q+1)`, apply
   `RXX(entangler_offsets[l,q] + entangler_scales[l,q] * a[2])`, perform an
   SVD capped to the largest `chi` singular values, and leave the center at
   `q+1`.
3. If `l` is odd, move the center to site `n-1` and sweep the same bonds in
   reverse.  Apply `RZZ` with the same angle formula, truncate to `chi`, and
   leave the center at `q`.

Use `RP(alpha) = exp(-i alpha P/2)` for `P = Y, Z, XX, ZZ`.  Hard rank
truncation happens after every two-site gate, even when the current rank is
already at most `chi`.  After the final layer, normalize the MPS.

For each energy record, form the Pauli product in `operators`.  The energy is

```text
E(parameters) = sum(weight * normalized Pauli expectation).
```

Differentiate the *truncated algorithm itself*: retain exactly `chi` singular
values at an active truncation and all available singular values when the
two-site matrix rank is smaller than `chi`.  Derivatives pass through the
retained SVD factors.  Admitted evaluation cases have nondegenerate singular
values at every active truncation.

## Output

Return NumPy-compatible values:

```python
{
    "energies": real array,                # (n_cases,)
    "gradients": real array,               # (n_cases, n_parameters)
    "final_bond_dimensions": integer array # (n_cases, n_qubits - 1)
}
```

The evaluator independently implements complex128 TEBD, canonical-center
motion, hard SVD truncation, Pauli transfer contractions, and central finite
differences.  It also checks that truncation is active and that every submitted
bond dimension is exact.

## Submission

Create `/root/solution_111.py` with `run_solution(config)`.  The timed call
must finish within 300 seconds and the implementation must contain at most 160
effective non-empty, non-comment Python lines.  NumPy may format outputs, but
do not replace TensorCircuit's MPS evolution with a standalone simulator or a
dense statevector.  Use TensorCircuit/JAX automatic differentiation for the
gradient.
