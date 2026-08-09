# Problem 116: Robust coherent toric recovery-portfolio selection

Select two recovery portfolios for periodic CSS memories under uncertain
coherent noise.  A portfolio selects one physical recovery for each measured
syndrome, declares its logical Pauli-frame class, and obeys a shared recovery
cost budget.  The objective is the worst-case syndrome-weighted conditional
entanglement fidelity across three noise scenarios.

This is not minimum-weight decoding and it is not a request to reconstruct a
fully specified channel.  Stabilizer-degenerate physical errors interfere
before probabilities are formed.  The cheapest recoveries and the
unconstrained highest-fidelity recoveries are deliberately incompatible with
the admitted budget.

## Periodic CSS memory

Each case is a toric CSS memory on an `L x L` periodic square lattice.  Data
qubits occupy its `2*L*L` edges:

```text
horizontal[y,x] joins vertex (y,x) to (y,x+1)
vertical[y,x]   joins vertex (y,x) to (y+1,x)
```

All indices are modulo `L`.  A binary Z recovery `R=(R_h,R_v)` has X-star
syndrome

```text
s[y,x] = R_h[y,x] xor R_h[y,x-1]
         xor R_v[y,x] xor R_v[y-1,x].
```

Every supplied recovery candidate has the declared syndrome.  Its hardware
cost is the binary inner product with `horizontal_recovery_cost` and
`vertical_recovery_cost`; the evaluator recomputes rather than trusts the
declared `hardware_cost` field.

For each syndrome, `anchor_candidate_id` fixes an affine origin.  The XOR of
another candidate with the anchor is a cycle.  Its relative homology bits are

```text
h0 = parity of horizontal cycle edges [y,L-1] over all y
h1 = parity of vertical cycle edges [L-1,x] over all x.
```

The shuffled `frame_candidates` map these two bits to opaque frame IDs.  The
reported frame is a declaration of the selected recovery's relative homology,
not a second correction that can change its fidelity.

## Coherent conditional fidelity

A noise scenario is a classical mixture of branches.  In branch `b`, every
edge `e` undergoes

```text
exp(-i theta[b,e] Z_e / 2)
  = cos(theta[b,e]/2) I - i sin(theta[b,e]/2) Z_e.
```

For selected recovery `R`, let `A[b,R,h]` be the coherent sum of physical-error
amplitudes over all Z errors that have the measured syndrome and that, after
`R`, differ from a stabilizer by toric homology `h in {00,01,10,11}`.  On this
periodic code each homology sector contains `2^(L*L-1)` plaquette-stabilizer
representatives.  Terms in the same sector must be added as complex
amplitudes, not as Pauli-twirled probabilities.

For scenario `u` with branch probabilities `p[b]`, define

```text
Chi[u,R] = sum_b p[b] A[b,R] A[b,R]^dagger
F[u,R]   = Chi[u,R][00,00] / Tr(Chi[u,R]).
```

For a complete portfolio `R_j`, the scored objective is

```text
robust_fidelity = min_u sum_j syndrome_weight[j] * F[u,R_j].
```

Select exactly one listed candidate per syndrome, keep input syndrome order,
and maximize this objective subject to

```text
sum_j hardware_cost(R_j) <= recovery_budget.
```

The evaluator independently obtains the exact optimum and accepts an objective
gap below `3e-9`.  It also checks the full recovery masks, syndromes, opaque
identities, frame declarations, total cost, and declared fidelity.

## Output

Return

```python
{
    "cases": [
        {
            "case_id": str,
            "selections": [
                {
                    "syndrome_id": str,
                    "candidate_id": str,
                    "recovery": {
                        "horizontal": list[list[int]],
                        "vertical": list[list[int]],
                    },
                    "frame_id": str,
                },
                ...,
            ],
            "robust_fidelity": float,
            "total_recovery_cost": int,
        },
        ...,
    ]
}
```

## Submission

Create `/root/solution_116.py` with `run_solution(config)`.  The phase-sensitive
degenerate-coset contraction used to evaluate recovery candidates must use
TensorCircuit-NG.  NumPy and other general libraries are allowed for binary
bookkeeping, cost accounting, and the discrete portfolio search, but a dense
98-qubit state or a standalone NumPy replacement for the quantum contraction
is not allowed.

The timed call must finish within 300 seconds and use at most 160 effective
non-empty, non-comment Python lines.  Preserve the exact case and syndrome
orders.  Do not inspect hidden evaluator files, logs, rewards, or secrets.
