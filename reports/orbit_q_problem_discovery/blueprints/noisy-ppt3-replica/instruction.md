# Problem 115: Noisy p3-PPT certification with purified replicas

Certify bipartite entanglement after seeded local amplitude damping.  Each
production case purifies a 32-qubit noisy system with 32 environment qubits;
the corresponding dense purification has `2^64` amplitudes.  Prepare the
purification as an exact TensorCircuit-NG MPS and evaluate its two- and
three-replica permutation observables without constructing a statevector or a
density matrix.

## Purified circuit

`run_solution(config)` receives `config["cases"]`.  A case contains:

- `system_qubits` (`32` in admitted cases);
- `initial_bits`, the system computational-basis state;
- `layers`, lists of system-nearest-neighbor records
  `[left, theta, phi, eta]` in execution order;
- `amplitude_damping`, one probability `gamma_j` per system qubit;
- `partition_a` and `partition_b`, disjoint lists whose union is every system
  qubit;
- `certificate_tolerance`; and
- opaque `case_id` and `case_nonce` values.

Use the interleaved physical MPS order

```text
system_0, environment_0, system_1, environment_1, ... .
```

Every environment qubit starts in `|0>`.  System qubit `j` starts in
`|initial_bits[j]>`.  First apply all supplied system gates.  In the basis
`|00>, |01>, |10>, |11>`, a record specifies

```text
G(theta, phi, eta) =
[[1, 0, 0, 0],
 [0, cos(theta), -i exp(-i phi) sin(theta), 0],
 [0, -i exp(+i phi) sin(theta), cos(theta), 0],
 [0, 0, 0, exp(-i eta)]].
```

The gate acts on logical system sites `(left, left+1)`.  Because their
environment partners are interleaved, an exact adjacent-SWAP shuttle is one
valid MPS implementation.

After the system circuit, apply `G(arcsin(sqrt(gamma_j)), 0, 0)` to
`(system_j, environment_j)`.  Tracing the environment implements amplitude
damping: `|1,0>` maps to
`sqrt(1-gamma_j)|1,0> - i sqrt(gamma_j)|0,1>`.  Let `rho_AB` be the system
state after tracing every environment qubit.

No MPS truncation is permitted.

## Replica moments and PPT witness

Compute

```text
p2 = Tr[rho_AB^2],
p3 = Tr[(rho_AB^{T_B})^3].
```

Use implicit replica transfers formed from the TensorCircuit MPS tensors.  The
conventions are fixed by

```text
p2 = <Psi|<Psi| SWAP_(A union B) |Psi>|Psi>,

p3 = <Psi|^3 (V_A tensor V_B^{-1}) |Psi>^3.
```

The environment carries the identity permutation.  For `V_A`, the bra
physical indices of replicas `(1,2,3)` equal ket indices `(2,3,1)`.  For
`V_B^{-1}`, they equal `(3,1,2)`.  These opposite cycles are essential; using
the same cycle on both partitions computes `Tr[rho_AB^3]`, not the
partial-transpose moment.

If `rho_AB^{T_B}` were positive, its eigenvalues would be nonnegative and
Cauchy-Schwarz would imply `p3 >= p2^2`.  Therefore define

```text
ppt_gap = p2^2 - p3,
certified_npt = (ppt_gap > certificate_tolerance).
```

A positive gap is a sufficient NPT-entanglement certificate.  A nonpositive
gap is only “not certified”; it is not a proof of separability.

## Output

Return NumPy-compatible arrays, one entry per case:

```python
{
    "purity2": real array,
    "ppt_moment3": real array,
    "ppt_gap": real array,
    "certified_npt": boolean array,
}
```

The evaluator independently performs complex128 NumPy TEBD and explicit
two-/three-replica transfers.  A separately evolved dense 4-system +
4-environment canary verifies the Stinespring channel, the environment trace,
the partial transpose, and both permutation directions.

## Submission

Create `/root/solution_115.py` with `run_solution(config)`.  The timed call must
finish within 300 seconds and the implementation must contain at most 160
effective non-empty, non-comment Python lines.  NumPy may construct supplied
gate values and format outputs, but TensorCircuit must prepare the purified
MPS and perform both quantum replica contractions.  Do not construct the full
purified statevector, `rho_AB`, or explicit replicated states.
