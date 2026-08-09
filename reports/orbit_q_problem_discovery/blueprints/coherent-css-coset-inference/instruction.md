# Problem 113: Coherent degenerate CSS logical-coset inference

Infer syndrome-conditioned logical noise for two hidden `[[14,2,3]]` CSS
codes under a correlated coherent channel.  Each code is obtained from two
Steane blocks by a hidden CSS-preserving CNOT transform, qubit permutation,
and independent stabilizer-basis changes.  Do not assume the supplied rows
retain a block structure.

The task is not minimum-weight decoding.  Physical errors differing by a
stabilizer contribute coherently to the same syndrome and logical coset.  The
answer is a complex logical process matrix, and replacing coherent rotations
by Pauli probabilities gives a different result.

## Input code and conventions

`run_solution(config)` receives `config["cases"]`.  Every case contains:

- `n_qubits = 14`, `n_logicals = 2`;
- binary `x_stabilizers`, `z_stabilizers`, `logical_x`, and `logical_z` rows;
- `noise_branches`, a classical mixture of ordered coherent circuits;
- `syndrome_cases`, each with a measured syndrome and a declared physical
  Pauli recovery; and
- a shuffled `frame_candidates` list with opaque IDs and two-bit logical X/Z
  masks.

Binary products and sums are over GF(2).  Each stabilizer matrix has rank six,
and stabilizer rows all have positive sign.  With `H_X`, `H_Z`, `L_X`, and
`L_Z` denoting the four supplied matrices, the evaluator guarantees

```text
H_X H_Z^T = 0,  H_X L_Z^T = 0,  H_Z L_X^T = 0,
L_X L_Z^T = identity
```

modulo two.  The logical computational basis is fixed by

```text
|a0 a1>_L = 1/sqrt(|S_X|) sum_{g in rowspace(x_stabilizers)}
             |g + a0*logical_x[0] + a1*logical_x[1]>.
```

Qubit zero is the most-significant computational-basis bit.  For binary masks
`x,z`, use the Hermitian convention

```text
P(x,z) = i^(x dot z) X^x Z^z.
```

Each recovery contains length-`n_qubits` binary masks `x` and `z`.  Syndrome
bits list all X-stabilizer outcomes followed by all Z-stabilizer outcomes; bit
one means eigenvalue -1.  The supplied recovery has exactly that syndrome.
Its overall Pauli phase is fixed by the convention above.

## Correlated coherent channel

Each noise branch has probability `w` and an ordered `rotations` list.  A
record gives distinct `sites`, an equally long `paulis` string list, and an
angle theta.  Apply, in the supplied site order,

```text
U(theta,P) = exp(-i theta P / 2)
            = cos(theta/2) I - i sin(theta/2) P,
P = tensor_j paulis[j].
```

The physical channel is `E(rho)=sum_b w_b U_b rho U_b^dagger`.  Its rotations
include noncommuting one-qubit terms and correlated multiqubit terms.  State
evolution and the quantum overlap/contraction used to obtain
the syndrome likelihoods must use TensorCircuit-NG.

For syndrome record `s` with declared recovery `R_s`, define the unnormalized
logical Kraus matrix of branch `b` by

```text
K[s,b]_(v,u) = <v_L| R_s U_b |u_L>,  u,v in {00,01,10,11}.
```

For each shuffled frame candidate `f`, form its 4 by 4 Hermitian logical Pauli
matrix `P_f` using the same convention and define

```text
c[s,b,f] = Tr(P_f^dagger K[s,b]) / 4,
Chi_s[f,g] = sum_b w_b c[s,b,f] conj(c[s,b,g]),
p_s = Tr(Chi_s),
chi_s = Chi_s / p_s.
```

Thus `chi_s` is the normalized syndrome-conditioned logical process matrix in
the exact shuffled frame order.  Its diagonal is the vector of degenerate
logical-coset weights.  The best Pauli frame is the opaque ID at the largest
diagonal entry; use the first input occurrence only for an exact tie.

## Output

Preserve all input case, syndrome, and frame orders:

```python
{
    "cases": [
        {
            "case_id": str,
            "syndromes": [
                {
                    "syndrome_id": str,
                    "syndrome_probability": float,
                    "logical_chi": complex array,  # (16, 16)
                    "coset_weights": real array,   # (16,)
                    "best_frame_id": str,
                },
                ...,
            ],
        },
        ...,
    ]
}
```

The evaluator is TensorCircuit-free.  It independently builds the CSS logical
basis, evolves every coherent branch with complex128 NumPy statevectors,
projects each declared recovery, and checks the full complex process matrices,
probabilities, positivity, normalization, and opaque identities.

## Submission

Create `/root/solution_113.py` with `run_solution(config)`.  The timed call must
finish within 300 seconds and use at most 160 effective non-empty,
non-comment Python lines.  NumPy binary algebra and output formatting are
allowed, but do not replace TensorCircuit state evolution and quantum
likelihood contractions with a standalone simulator.  Do not inspect hidden
evaluator files, logs, rewards, or secrets.
