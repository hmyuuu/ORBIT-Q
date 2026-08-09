# Problem 112: Charged-replica symmetry-resolved Renyi-2

Compute U(1)-resolved second Renyi entropies of fixed-charge 72-qubit circuit
states.  The state is supplied as a nearest-neighbor, number-conserving
brickwork circuit.  A dense statevector is impossible at the production size.
The state preparation and the one- and two-replica quantum tensor contractions
must use TensorCircuit-NG.

## Input and circuit

`run_solution(config)` receives `config["cases"]`.  Each case contains:

- `n_qubits`;
- `initial_bits`, a length-`n_qubits` computational-basis bit string;
- `layers`, each a list of records `[left, theta, phi, eta]` in execution order;
- `subsystem`, an unsorted list of distinct qubit indices defining `A`;
- `fourier_offset` beta and `fourier_size` K;
- `sector_queries`, the charge sectors whose entropies must be returned; and
- `case_nonce`, an opaque anti-hardcoding integer.

Qubit zero is the leftmost MPS site.  Start in the exact computational-basis
state `|initial_bits[0] ... initial_bits[n-1]>`.  For each layer and then each
record in its supplied order, apply a two-qubit gate to `(left, left+1)`.  In
the ordered basis `|00>, |01>, |10>, |11>`, the gate is

```text
G(theta, phi, eta) =
[[1, 0, 0, 0],
 [0, cos(theta), -i exp(-i phi) sin(theta), 0],
 [0, -i exp(+i phi) sin(theta), cos(theta), 0],
 [0, 0, 0, exp(-i eta)]].
```

Every gate preserves the occupation number.  The evaluator guarantees that
the initial state has a fixed total charge.  No MPS truncation is permitted.

## Charged moments

Let `rho_A = Tr_not-A |psi><psi|` and
`Q_A = sum(j in A) |1><1|_j`.  The fixed global charge implies
`[rho_A, Q_A] = 0`.  For

```text
alpha_k = beta + 2*pi*k/K,  k = 0,...,K-1,
```

compute the one-copy characteristic function and the charged second moment

```text
C1(alpha_k) = Tr[rho_A exp(+i alpha_k Q_A)],
C2(alpha_k) = Tr[rho_A^2 exp(+i alpha_k Q_A)].
```

The required two-copy identity, which also fixes every swap and phase
convention, is

```text
C2(alpha) = <psi|<psi| SWAP_A
            (exp(+i alpha Q_A) tensor I) |psi>|psi>.
```

`SWAP_A` swaps the two replicas only at sites in `A`.  The phase operator acts
on the first ket replica before that swap.  Sites outside `A` carry the
two-replica identity.  An implicit replica transfer made from TensorCircuit
MPS tensors is allowed; materializing a 144-qubit state is not.

All cases use `K = |A| + 1`, so charges `q=0,...,|A|` are resolved without
aliasing.  Reconstruct

```text
p(q)  = exp(-i beta q)/K * sum_k exp(-2*pi*i*k*q/K) C1(alpha_k),
Z2(q) = exp(-i beta q)/K * sum_k exp(-2*pi*i*k*q/K) C2(alpha_k).
```

Here `p(q) = Tr[P_q rho_A]` and `Z2(q) = Tr[(P_q rho_A)^2]` are real and
nonnegative.  For each `q` in `sector_queries`, return the normalized
symmetry-resolved entropy

```text
S2(q) = -log(Z2(q) / p(q)^2),
```

using the natural logarithm.  Admitted cases keep every queried `p(q)` and
`Z2(q)` safely above numerical-noise floors.

## Output

Return NumPy-compatible values:

```python
{
    "charged_moments": complex array,       # (n_cases, K), values C2(alpha_k)
    "sector_probabilities": real array,     # (n_cases, K), p(q)
    "sector_second_moments": real array,    # (n_cases, K), Z2(q)
    "resolved_renyi2": real array,          # (n_cases, n_sector_queries)
}
```

The evaluator independently evolves the same circuit with complex128 NumPy
TEBD, performs explicit replica transfers, checks the shifted Fourier
identities, and runs a separate dense reduced-density-matrix canary.

## Submission

Create `/root/solution_112.py` with `run_solution(config)`.  The timed call must
finish within 300 seconds and the implementation must contain at most 160
effective non-empty, non-comment Python lines.  NumPy may format and Fourier
transform outputs, but do not replace TensorCircuit's MPS state preparation or
replica quantum contraction with a standalone simulator.  Do not construct a
full statevector.
