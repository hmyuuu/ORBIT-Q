# Hidden-ensemble leakage-aware robust GRAPE

Implement `run_solution(config)` in `/root/solution_109.py`.

Design a two-quadrature piecewise-constant pulse for a weakly anharmonic
three-level system.  The ordered basis is `|0>, |1>, |2>`.  For control slice
`j` and ensemble member `(delta, eta, zeta)`, the rotating-frame Hamiltonian is

```text
H_j = delta * N + (alpha + zeta) * P2
      + (1 + eta) * (ux[j] * X + uy[j] * Y) / 2,

N  = diag(0, 1, 2),        P2 = diag(0, 0, 1),
a  = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]],
X  = a + a^dagger,         Y = -i (a - a^dagger).
```

Here `delta` is detuning, `eta` is fractional drive-amplitude error, `zeta`
is an anharmonicity shift, and `alpha = config["anharmonicity"]`.  Apply
slices in increasing index order with

```text
U <- exp(-i * slice_duration * H_j) @ U,    U initially I_3.
```

The target on the computational subspace is

```text
T = exp(-i * target_angle * sigma_x / 2).
```

For each ensemble member define

```text
F = |trace(T^dagger * U[:2, :2])|^2 / 4,
infidelity = 1 - F,
leakage = 1 - sum(|U[r,c]|^2 for r,c in {0,1}) / 2.
```

The fidelity is invariant to a common phase.  The leakage is the average
population outside the computational subspace for computational-basis inputs.

## Public configuration schema

The evaluator passes a dictionary with these fields:

- `n_levels`: integer `3` and `n_slices`: the number of time slices;
- `slice_duration`, `anharmonicity`, and `target_angle`: real scalars with the
  meanings above;
- `uncertainty_bounds`: a dictionary whose `detuning`,
  `amplitude_fraction`, and `anharmonic_shift` entries are two-element
  `[lower, upper]` lists.  The pulse must work throughout this closed box;
- `training_ensemble`: a list of dictionaries.  Each has real `detuning`,
  `amplitude_fraction`, and `anharmonic_shift` fields.  These are sparse
  public anchor members, not an exhaustive validation set;
- `max_drive_amplitude`: the per-slice Euclidean bound
  `sqrt(ux[j]**2 + uy[j]**2)`;
- `max_slew_per_slice`: the Euclidean bound between adjacent control vectors;
- `max_edge_amplitude`: the Euclidean bound on the first and last controls;
- `maximum_worst_infidelity`, `maximum_p95_infidelity`, and
  `maximum_worst_leakage`: held-out acceptance thresholds;
- `optimization_seed`: a public integer for deterministic optimization.

The evaluator independently samples secret held-out members inside the
declared uncertainty box and also checks all eight coupled corners.  A pulse
optimized only for the nominal member or only for the sparse axis anchors is
not expected to pass.  No observations from the hidden ensemble are exposed.

## Conditioning notes

The public anchors lie on coordinate axes, whereas the worst members can be
coupled corners.  Mean infidelity alone is consequently poorly conditioned as
a robustness certificate; leakage must also be optimized rather than inferred
from the projected trace overlap.  Hard clipping after optimization can break
the slew bound and degrade a narrow worst-case margin.

## Return value

Return a dictionary with exactly these required entries:

```python
{
    "controls": np.ndarray,                 # shape (n_slices, 2), columns ux, uy
    "training_worst_infidelity": float,
    "training_worst_leakage": float,
}
```

The two reported training certificates must agree with direct evaluation on
`config["training_ensemble"]` to within `2e-6`.  The evaluator recomputes all
physics from the returned controls; an optimization history is not required.

The quantum propagation and any exact or automatic gradients used to design
the pulse must be implemented with the quantum software framework selected for
the run.  Framework-specific constraints are supplied separately.  NumPy,
JAX, and SciPy may be used for deterministic classical optimization, but must
not replace the framework's quantum evolution with a standalone simulator.

Results must be deterministic and finish within 300 seconds.  The solution
must not read evaluator files, hidden seeds, `/tests`, `/logs`, or any oracle
solution.
