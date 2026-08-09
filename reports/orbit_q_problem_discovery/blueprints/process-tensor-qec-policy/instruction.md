# Problem 118: Retained-memory QEC intervention-policy synthesis

Synthesize a compact portfolio of five-round recovery interventions for two
periodic CSS memories.  Each supplied syndrome history is evaluated under
three alternative non-Markovian process scenarios.  A retained two-level
memory coherently links the rounds, so selecting the best correction in each
round independently is generally wrong.  The interventions across all six
histories in a case also share one active implementation-cost budget.

This is a policy-synthesis problem, not channel reconstruction.  Return one
action sequence for every history.  The verifier generates new syndrome
histories, coherent angles, retained-memory rotations, opaque action IDs,
weights, and budgets from its private case seed.

## Periodic CSS memory

Each case has a toric CSS memory on an `L x L` periodic square lattice.  Data
qubits occupy its `2*L*L` edges:

```text
horizontal[y,x] joins vertex (y,x) to (y,x+1)
vertical[y,x]   joins vertex (y,x) to (y+1,x)
```

All indices are modulo `L`.  The X-star syndrome of a binary Z recovery
`R=(R_h,R_v)` is

```text
s[y,x] = R_h[y,x] xor R_h[y,x-1]
         xor R_v[y,x] xor R_v[y-1,x].
```

Every round record contains an `anchor_recovery` with exactly the declared
`observed_star_syndrome`.  A binary plaquette field `p[y,x]` has boundary

```text
B_h[y,x] = p[y,x] xor p[y-1,x]
B_v[y,x] = p[y,x] xor p[y,x-1].
```

The two noncontractible Z loops are a full horizontal row and a full vertical
column.  For sector `q = q0 + 2*q1`, define `L_q` by toggling
`horizontal[0,:]` when `q0=1` and `vertical[:,0]` when `q1=1`.

## Syndrome-conditioned spatial amplitudes

For process scenario `u`, round `t`, and retained-memory basis value
`m in {0,1}`, the config supplies one angle on every data edge.  Its coherent
Z rotation has weights

```text
w_e(0) = cos(theta_e/2)
w_e(1) = -i sin(theta_e/2).
```

For history `j`, let `R[j,t]` be that round's anchor.  The four coherent
logical-sector amplitudes are

```text
A[j,t,u,m,q] = (1/2) * sum over all p in {0,1}^(L*L)
               product over edges e of
               w_e((R[j,t] xor B(p) xor L_q)[e]).
```

The factor `1/2` removes the twofold redundancy between `p` and its global
complement.  At production size `L=5`, each amplitude therefore sums
`2^24` distinct stabilizer representatives.  Do not enumerate them and do not
construct a dense 50-qubit state.

## Retained-memory process instrument

The logical code space is two qubits.  For `q=q0+2*q1`, define

```text
P_q |b0,b1> = (-1)^(q0*b0 xor q1*b1) |b0,b1>.
```

For a real vector `v=(vx,vy,vz)`, use

```text
R(v) = Rz(vz) @ Ry(vy) @ Rx(vx),
Ra(x) = exp(-i*x*sigma_a/2).
```

The syndrome-conditioned process instrument in round `t` is the `8 x 8`
operator on logical space tensor retained memory

```text
M[j,t,u] = sum_q P_q tensor
           (R(memory_rotation[u,t])
            @ diag(A[j,t,u,0,q], A[j,t,u,1,q])).
```

An action record has an opaque `action_id`, `logical_frame in {0,1,2,3}`,
three-real `memory_kick`, and nonnegative integer `intervention_cost`.  Its
effective intervention is

```text
G[a] = P_logical_frame[a] tensor R(memory_kick[a]).
```

The retained-memory kick is a calibrated effective back-action of that
physical intervention; it is not permission to access an unknown environment
inside the solver.  For an action sequence `a[0],...,a[T-1]`, compose

```text
O[j,u,a] = G[a[T-1]] @ M[j,T-1,u] @ ... @ G[a[0]] @ M[j,0,u].
```

Scenario `u` supplies the initial memory state

```text
|mu_u> = cos(polar/2)|0> + exp(i*azimuth) sin(polar/2)|1>.
```

For final memory value `r`, set the `4 x 4` logical Kraus block

```text
K_r = <r| O[j,u,a] |mu_u>.
```

The conditional probability and entanglement fidelity are

```text
p[j,u,a] = (1/4) * sum_r Tr(K_r^dagger K_r),
F[j,u,a] = sum_r |Tr(K_r)|^2 / (16*p[j,u,a]).
```

For a history, score the worst process scenario:

```text
history_robust_fidelity[j] = min_u F[j,u,a_j].
```

For a complete case, maximize

```text
robust_objective = sum_j history_weight[j]
                   * history_robust_fidelity[j]
```

subject to

```text
sum over histories j and rounds t of intervention_cost[a_j[t]]
    <= shared_intervention_budget.
```

The five primitive actions and five rounds give `5^5` schedules per history.
The histories must be optimized jointly because of the shared budget.  Ties
within `1e-14` are resolved by smaller total cost and then lexicographic opaque
action-ID sequences.  The evaluator accepts any valid portfolio within
`3e-9` of its independently derived optimum.

## Output

Return exactly

```python
{
    "cases": [
        {
            "case_id": str,
            "policies": [
                {
                    "history_id": str,
                    "action_ids": [str, str, str, str, str],
                    "history_robust_fidelity": float,
                },
                ...,
            ],
            "robust_objective": float,
            "total_intervention_cost": int,
        },
        ...,
    ]
}
```

Preserve case and history order.  Every number must be finite, every action ID
must be listed in that case, the reported cost must equal the selected action
costs, and each reported fidelity is recomputed by the verifier.

## Submission

Create `/root/solution_118.py` with `run_solution(config)`.  The
stabilizer-degenerate spatial contractions that produce `A` must use
TensorCircuit-NG.  NumPy may be used for binary bookkeeping, the small
logical-memory operators, and deterministic discrete optimization, but not
to replace the quantum contraction with a standalone simulator or to
enumerate the `2^24` representatives.

The timed call must finish within 300 seconds and use at most 160 effective
non-empty, non-comment Python lines.  Do not inspect evaluator files, hidden
seeds, `/tests`, `/logs`, rewards, secrets, or an oracle solution.
