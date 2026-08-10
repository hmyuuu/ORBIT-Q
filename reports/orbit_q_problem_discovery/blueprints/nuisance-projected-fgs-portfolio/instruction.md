# Problem 122: Robust nuisance-projected FGS experiment portfolio

Implement `run_solution(config)` in `/root/solution_122.py`.

Design a sparse, integer shot allocation for identifying three parameters of a
fermionic-Gaussian quench in the presence of two nuisance parameters.  The
submission is an experiment portfolio, not an estimate of the parameters.  It
is scored by the classical Fisher information of the declared binary
measurements after projecting out nuisance directions.

## Quantum experiment

There are `config["n_sites"]` fermionic modes.  Initialize TensorCircuit-NG's
`FGSSimulator` with the modes in `config["filled"]` occupied.  The normalized
parameter vector `z` has order

```text
(hopping, pairing, staggered_potential, phase_offset, readout_contrast)
```

and maps to physical values by

```text
x = parameter_centers + parameter_half_widths * z.
```

For an action with phase `phi`, duration `dt`, measurement site `s`, and
three-entry control word `w`, apply three layers.  In layer `l`, set

```text
psi = phi + phase_offset + 0.37 * l * w[l].
```

Then apply, in order:

1. on every site `i`,
   `evol_cp(i, dt * mu * (-1)**i * (1 + 0.13*cos(psi + 0.31*i)))`;
2. on even bonds followed by odd bonds,
   `evol_hp(i, i+1, dt * J * w[l] * (1 + 0.11*sin(psi + 0.47*i)))`;
3. on the same even/odd bond order,
   `evol_sp(i, i+1, dt * Delta * exp(1j*(psi + 0.19*i)))`.

Let `q = real(C[s,s])` in TensorCircuit's FGS correlation convention.  The
observed outcome-zero probability is

```text
p = 0.5 + readout_contrast * (q - 0.5).
```

Every quantum probability used to design the portfolio must be obtained from
`tensorcircuit.FGSSimulator`.  NumPy and SciPy may perform the classical
Fisher-information and allocation work, but a standalone covariance or dense
Fock-state simulator is not an allowed replacement.

## Nuisance-projected information

For action `a`, use the gradient with respect to normalized `z`, not physical
coordinates:

```text
g_a = d p_a / d z,
I_a = outer(g_a, g_a) / (p_a * (1 - p_a)).
```

The reference evaluator uses a central step `2e-4` for the first four
coordinates and the exact linear derivative for contrast.  For integer shot
allocation `n`,

```text
I(n,z) = sum_a n[a] * I_a(z) + diag(0, 0, 0, 4, 9).
```

Partition the first three target coordinates from the final two nuisance
coordinates and form

```text
S = I_tt - I_tn @ inv(I_nn) @ I_nt,
D = logdet(S + 0.25 * eye(3)),
E = minimum_eigenvalue(S).
```

The hidden E-optimality gate compares
`(E + e_ratio_floor) / (E_baseline + e_ratio_floor)`; the reduced prototype
uses `e_ratio_floor = 1e-8`.  This small numerical floor is distinct from the
`0.25` log-determinant regularizer.

The eleven public design points are supplied in `training_points`.  Hidden
evaluation uses coupled corners and verifier-keyed interior points inside the
declared normalized radius.  A nominal-only or axis-only allocation is not
expected to pass.

## Return artifact

Return exactly three NumPy arrays:

```python
{
    "shot_allocation": np.ndarray,          # integer, shape (n_actions,)
    "public_logdet": np.ndarray,            # float64, shape (11,)
    "public_min_eigenvalue": np.ndarray,    # float64, shape (11,)
}
```

The evaluator rejects lists, Boolean/object/coercible arrays, non-finite
certificates, and extra keys.  The allocation must satisfy all of:

- exactly `shot_budget` shots;
- every entry is nonnegative and divisible by `allocation_granularity`;
- the number of active actions is in the declared inclusive range;
- every active action is between `minimum_active_shots` and
  `maximum_active_shots`;
- the action-cost dot product is at most `interrogation_budget_units`.

The two public certificates must agree with independent recomputation within
`certificate_tolerance`.  Hidden scores and hidden points must not be returned.

## Passing criteria and scope

Relative to `baseline_allocation`, every hidden point must jointly satisfy the
configured minimum log-determinant gain, minimum regularized E-optimality
ratio, maximum regularized condition number, and minimum active-action
probability margin. The intended benchmark runtime target is below 300
seconds, but runtime is recorded as a separate comparison dimension and does
not multiply or veto functional correctness. The solution must contain at
most 160 effective non-empty, non-comment Python lines. Strict runtime and
expert-p95 gates belong to the separately authorized admission protocol.

This tracked bundle is a reduced, deterministic, design-only prototype.  Its
local public development key is not secret trial material.  It is not
authorized for Harbor, private qualification, or a solver-model run.
