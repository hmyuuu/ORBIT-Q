# Problem 106: CSS encoder and propagated-fault verification synthesis

For each hidden instance, synthesize a compact Clifford encoder for a CSS
logical-zero state and an optimal set of ideal stabilizer checks that detects
every dangerous single fault in the declared encoder-fault model.  The two
instances are seeded, qubit-permuted members of the 7- and 15-qubit quantum
Hamming families.  Their generator bases and candidate-check identifiers are
scrambled independently.

This problem deliberately separates two claims.  The selected checks must
detect faults arising in state preparation.  Check measurements themselves
are ideal nondestructive Pauli measurements; faults inside those measurement
gadgets are outside the declared model.

## Input

The evaluator calls `run_solution(config)`.  `config["instances"]` contains:

- `instance_id`, `n_qubits`, and `x_generators` / `z_generators`;
- `gate_budget`, the maximum number of encoder gates;
- `candidate_checks`, each with an opaque `id`, `kind` (`x` or `z`), a binary
  `mask`, and its `coupling_cost`.

The X and Z rows are independent, mutually orthogonal, and together generate
the full positive-sign stabilizer of the requested logical-zero state.  A
valid H/CNOT encoder must prepare that state, including stabilizer signs; the
evaluator checks both its binary row space and its exact amplitudes.

## Declared single-fault model

Use qubit coordinates in ascending integer order.  Enumerate:

1. an `X` preparation fault on every qubit, inserted before the first gate;
2. `X`, `Z`, and `Y` after every H gate; and
3. all 15 nonidentity two-qubit Paulis after every CNOT gate.

Propagate each fault through the remaining encoder.  A final Pauli is
*correctable for this prepared state* when multiplying it by some target-state
stabilizer gives Pauli weight at most one.  A distinct final Pauli whose
minimum such weight is greater than one is dangerous.  Sort dangerous Pauli
labels lexicographically, using one character per qubit from `I`, `X`, `Y`,
and `Z`.

A candidate check detects a dangerous error exactly when its ideal expectation
on the faulty encoded state is negative.  Build the encoder and execute the
faulty states and candidate Pauli expectations with TensorCircuit-NG.  Binary
linear algebra may support circuit synthesis and error bookkeeping, but it
must not replace the TensorCircuit state/fault/check computation.

Among all candidate subsets covering every dangerous error, select the subset
with the fewest checks.  Break ties by the sum of `coupling_cost`, then by the
lexicographically sorted tuple of opaque check IDs.  Return selected IDs in
sorted order.

## Output

Return this dictionary, with one record per input instance:

```python
{
    "instances": [
        {
            "instance_id": str,
            "encoder": [
                {"name": "h", "qubits": [q]},
                {"name": "cx", "qubits": [control, target]},
                ...,
            ],
            "dangerous_errors": [str, ...],
            "fault_syndromes": bool array,
            "target_expectations": float array,
            "selected_check_ids": [str, ...],
        },
        ...,
    ]
}
```

Rows of `fault_syndromes` follow `dangerous_errors`; columns follow the input
`candidate_checks` order.  `target_expectations` uses that same candidate
order.  Its ideal values are all +1.

## Submission

Create `/root/solution_106.py` with `run_solution(config)`.  The timed call must
finish within 180 seconds.  Keep the implementation at or below 160 effective
non-empty, non-comment Python lines.  Do not inspect evaluator secrets, hidden
oracle files, logs, or reward files.
