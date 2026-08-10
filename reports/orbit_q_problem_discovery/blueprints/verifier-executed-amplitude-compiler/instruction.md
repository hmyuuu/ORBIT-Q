# Problem 121: Verifier-executed batched-amplitude compiler

Implement `run_solution(config)` in `/root/solution_121.py`.

Compile the supplied 64-qubit shallow circuit and 48 requested output
amplitudes into constant-free symbolic contraction bytecode.  Return the
program, not the amplitudes.  The verifier validates its provenance and index
semantics, terminates each ordinary worker process group, constructs fresh
numerical tensors only afterward, executes the program, and compares the
resulting amplitude lanes with an independent contraction oracle.  Submitted
numerical amplitude values are not accepted or scored.

This tracked bundle is design-only.  It is not authorized for Harbor, private
qualification, or a solver-model run.

## Circuit and TensorCircuit requirement

All 64 qubits start in `|0>`.  For physical qubit `q`, let its local record be
`local_gates[q]`.  Apply, in this order:

1. `RY(initial_ry)` and then `RZ(initial_rz)` on every qubit in increasing
   physical-qubit order;
2. every supplied `RZZ(theta)` entangler in list order (the `layer` fields
   describe four disjoint grid matchings);
3. `RX(final_rx)` and then `RY(final_ry)` on every qubit in increasing order.

The rotation convention is `RP(t) = exp(-i*t*P/2)`, and

```text
RZZ(t) = exp(-i*t*(Z tensor Z)/2).
```

Use TensorCircuit-NG centrally to construct this circuit and extract or
validate its native circuit/tensor representation.  `Circuit.to_qir()` is one
supported extraction API.  NumPy may be used for graph search, bookkeeping,
and resource arithmetic.  A handwritten quantum simulator, a different
quantum framework, or treating the supplied edge list as a replacement for
TensorCircuit extraction is not allowed.

The verifier uses the following factor-network semantics for execution.  For
the binary internal index `x_q`, define

```text
p_q[x]   = <x| RZ(initial_rz) RY(initial_ry) |0>
F_q[y,x] = <y| RY(final_ry) RX(final_rx) |x>
D_e[xu,xv] = exp(-i*theta_e*z(xu)*z(xv)/2),  z(0)=1, z(1)=-1.
```

For an output lane `L`, a site tensor is `p_q[x_q]*F_q[y_L,x_q]`.
An entangler source tensor is `D_e[x_u,x_v]`.  Summing every `x_q` in the
product of all 64 site tensors and all 112 entangler tensors gives that
lane's amplitude.

## Batched lanes and source provenance

The 48 query records form exactly six groups of eight.  Within a group the
first `shared_prefix_qubits` output bits agree.  Return the canonical groups
in lexicographic prefix order.  Each group record contains:

```python
{
    "prefix_bits": list[int],   # exact common prefix
    "query_ids": list[str],     # the group's eight IDs, sorted
    "lane_width": 8,
}
```

The sorted `query_ids` define lane order.  For a prefix qubit, its site source
has only index `[variable_ids[q]]`; its common output bit is in
`prefix_bits[q]`.  Every non-prefix site source has indices
`[variable_ids[q], "@lane"]`.  Thus `"@lane"` is one eight-valued hyperedge
shared by all non-prefix output tensors; it is never summed.  Eight separate
single-amplitude programs are not a batched program.

`source_tensor_ids` must list, without alteration, all `site_tensor_ids` in
physical-qubit order followed by all entangler `gate_id` values in supplied
list order.  A site ID refers to the site tensor above.  An entangler ID has
the two indices obtained by mapping its QIR qubits through `variable_ids` in
QIR order.  No literal tensor, scalar, gate matrix, amplitude, or other
numeric constant may appear in the artifact.

## Bytecode semantics

Choose exactly `required_slice_variables` distinct entries from
`slice_candidate_ids` and return them sorted as `slice_indices`.  For each
lane group, the executor evaluates all binary slice assignments in
lexicographic order and sums the four results.  Slicing fixes that named axis
in every source that contains it before bytecode execution.

Each bytecode step is exactly

```python
["ELIM", index_id, input_ids, output_id]
```

with the following meaning:

- `index_id` is one as-yet-uneliminated, unsliced binary variable;
- `input_ids` is the lexicographically sorted list of **every** currently live
  tensor containing that index;
- align equal named axes, multiply those inputs, sum `index_id`, and store the
  remaining axes under `output_id`;
- `"@lane"`, if present, is aligned and retained rather than summed;
- step `j` must use output ID `tmp-{j:03d}`.

Delete the inputs after each step.  Eliminate all 62 unsliced binary variables
exactly once.  `terminal_inputs` is the sorted list of every tensor left live;
after elimination their only possible named axis is `"@lane"`.  Broadcast
scalar terminal factors, multiply the terminal tensors lane-wise, and add the
four slice results.

Return one deterministic, canonical serialization of your chosen valid plan.
The verifier invokes `run_solution` once in each of two separately spawned,
scrubbed workers on the same exact config and requires byte-identical canonical
JSON in the pristine parent.  Slice IDs, operand IDs, groups, query IDs,
temporaries, and terminal IDs must follow the ordering rules above.  Any valid
path meeting the supplied resource caps is accepted; no particular path or
compiler search strategy is prescribed.

## Resource certificates

Every binary named index has dimension two.  `"@lane"` has dimension eight.
For an index set `S`, let

```text
size(S) = 2**(# binary indices in S) * (8 if "@lane" is in S else 1).
```

For one group and one slice assignment:

- live entries are the sum of `size(indices)` over all live tensors, initially
  after slicing and again after each step;
- maximum intermediate entries is the maximum size of any initial or produced
  tensor;
- if one elimination has `k` inputs with union scope `S` and eliminates `x`,
  its declared complex-FMA count is
  `(k-1)*size(S) + size(S without x)`;
- terminal multiplication costs `(k-1)*8` for `k` terminal inputs.

`peak_live_elements` and `max_intermediate_elements` are the one-group,
one-slice maxima.  Total `complex_fmas` multiplies the per-run sum by all six
groups and four slices.  Likewise, `eliminate_operations` is the step count
times 24, `terminal_contractions` is 24, and `slice_count` is 4.  All counts
are exact built-in integers, not estimates supplied on trust, and must not
exceed the corresponding caps in the case.

`program_digest` is lower-case SHA-256 of compact, sorted-key, `allow_nan=False`
JSON for the complete root artifact with the `certificates` field omitted.

## Exact return schema

Return one built-in dictionary with exactly these fields:

```python
{
    "schema": "orbit-q/batched-amplitude-elimination/v1",
    "case_id": str,
    "case_digest": str,
    "source_tensor_ids": list[str],
    "lane_groups": list[dict],
    "slice_indices": list[str],
    "steps": list[list],
    "terminal_inputs": list[str],
    "certificates": {
        "slice_count": int,
        "eliminate_operations": int,
        "terminal_contractions": int,
        "peak_live_elements": int,
        "max_intermediate_elements": int,
        "complex_fmas": int,
        "program_digest": str,
    },
}
```

Built-in lists, dictionaries, strings, and integers are required exactly;
Booleans are not integers.  Extra fields, alternate opcodes, stale or missing
IDs, reordered operands, duplicate slices, embedded constants, non-finite
values, submitted amplitudes, or false certificates fail.

Keep `/root/solution_121.py` at or below 160 effective non-empty,
non-comment Python lines and finish within 300 seconds.  Do not inspect
evaluator files, hidden seeds, tests, logs, rewards, or secrets.
