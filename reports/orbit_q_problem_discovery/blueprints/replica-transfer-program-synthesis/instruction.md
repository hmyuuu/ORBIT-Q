# Problem 117: Replica-transfer program synthesis

Compile and execute a reusable tensor-network program for a bank of mixed-state
replica moments.  Each case purifies a 32-qubit noisy system with 32 environment
qubits, then asks forty-eight overlapping second-, third-, and fourth-order moment
queries.  Evaluating every query as a separate left-to-right contraction is not
an accepted solution: the returned program must share every common transfer
prefix and satisfy exact operation and live-memory certificates.

## Purified circuit

`run_solution(config)` receives `config["cases"]`.  Use the interleaved MPS
order

```text
system_0, environment_0, system_1, environment_1, ... .
```

All environment qubits start in `|0>`.  System qubit `j` starts in
`|initial_bits[j]>`.  Execute every supplied nearest-neighbor system-gate
record `[left, theta, phi, eta]` in layer and record order.  In the basis
`|00>, |01>, |10>, |11>`, its matrix is

```text
G(theta, phi, eta) =
[[1, 0, 0, 0],
 [0, cos(theta), -i exp(-i phi) sin(theta), 0],
 [0, -i exp(+i phi) sin(theta), cos(theta), 0],
 [0, 0, 0, exp(-i eta)]].
```

The gate acts on logical system sites `(left, left+1)`.  Exact adjacent SWAP
shuttling is one valid way to cross the interleaved environment site.  Then
apply `G(arcsin(sqrt(gamma_j)), 0, 0)` to adjacent
`(system_j, environment_j)` for each supplied damping probability.  Tracing
the environments gives the mixed system state.  Each case supplies
`replica_bond_capacity == 4`; the generated two-layer circuit has exact Schmidt
rank at most this value at every physical cut.  Use it as an exact MPS bond cap.
Discarding a nonzero Schmidt value is forbidden.

## Query semantics

A query supplies an `order` `k` in `{2,3,4}` and disjoint system-site lists
`partition_a` and `partition_b`.  Let `R=A union B`, let `rho_R` be the reduced
state on `R`, and define

```text
m_q = Tr[(rho_R^(T_B))^k].
```

Equivalently, take `k` purified replicas.  At an A site the bra replica index
`r` equals ket index `(r+1) mod k`; at a B site it equals ket index
`(r-1) mod k`.  Every system site outside `R` and every environment site uses
the identity replica permutation.  This mathematical convention is the only
replica-index map supplied; derive a generic contraction compiler rather than
hard-coding separate formulas for `k=2,3,4`.

## Required reusable program

For each case, form the physical-site selector string for every query:

```text
0 = identity, 1 = forward cycle, -1 = reverse cycle.
```

The program is the unique prefix trie of these strings, with separate implicit
roots for orders 2, 3, and 4.  Visit the implicit roots in increasing order;
within each root, number nodes by depth-first preorder with children ordered
`-1, 0, 1`.  A node record is

```text
[parent_node_id, physical_site, order, selector]
```

where `parent_node_id == -1` denotes the implicit root.  Node IDs are their
zero-based positions in the returned `nodes` list.  `terminal_nodes[q]` is the
last node of input query `q`.  Duplicate `(parent, selector)` children,
per-query copies of a shared prefix, reordered nodes, and incomplete paths are
invalid.

Execute the trie depth first.  Keep the parent transfer environment while a
child is evaluated, and release a child environment when leaving its subtree.
The certified peak is a conservative allocation certificate.  Charge every
nonterminal physical bond as `replica_bond_capacity**(2*k)` complex elements
for an order-`k` node, even if its realized bond is smaller; charge the final
scalar as one.  Sum the charged environments simultaneously live in the
depth-first traversal, including one element for an implicit root and both the
parent path and newly produced child.  Backend-internal contraction workspace
is not part of this graph certificate and is measured separately at benchmark
admission.

Return the exact trie certificate even if a different internal traversal would
produce the same moments.  Source audit must confirm that the submitted
moments were actually obtained through this shared TensorCircuit traversal;
fabricating the certificate after independent query contractions is a policy
failure.

## Output

Preserve case and query order and return exactly

```python
{
    "moments": [
        [finite_real_for_query_0, ...],
        ...,
    ],
    "programs": [
        {
            "nodes": [[parent, site, order, selector], ...],
            "terminal_nodes": [node_id_for_query_0, ...],
            "transfer_operation_count": int,
            "peak_live_complex_elements": int,
            "program_digest": "64 lowercase hexadecimal characters",
        },
        ...,
    ],
}
```

`program_digest` is SHA-256 over canonical JSON
`{"nodes": nodes, "terminal_nodes": terminal_nodes}` using sorted keys,
compact separators, UTF-8, and no NaN.  The evaluator reconstructs the unique
trie, both certificates, and all moments independently.  Each case also gives
upper bounds `max_transfer_operations` and
`max_peak_live_complex_elements`; a disjoint path per query exceeds the former.

## Submission

Create `/root/solution_117.py` with `run_solution(config)`.  TensorCircuit-NG
must prepare the purified MPS and perform every replica-transfer contraction.
NumPy may construct supplied gates, assemble the trie, hash its public program,
and format outputs.  Do not construct a dense purification, density matrix,
explicit replicated state, or a standalone NumPy quantum simulator.

The timed call must finish within 300 seconds and use at most 160 effective
non-empty, non-comment Python lines.  Do not inspect verifier files, hidden
holdouts, logs, rewards, or secrets.
