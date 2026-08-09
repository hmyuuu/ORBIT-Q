# Problem 114: Liouvillian exceptional-point inference and response

Infer two driven open-qubit Liouvillians from exact multitime response data.
The cases lie on opposite sides of a genuine second-order Liouvillian
exceptional point, but are separated far enough that all scored modes are
diagonalizable.  You must preserve the non-Hermitian left/right-mode response;
replacing the generator by a normal or Pauli-diagonal model gives a different
answer.

## Physical model

For each case, the unknown parameters appear in this fixed order:

```text
[gamma_down, gamma_up, gamma_phi, drive_omega, frame_alpha, frame_beta].
```

All rates and `drive_omega` are positive and every parameter lies within the
supplied `parameter_bounds`.  Define

```text
V = Rz(frame_alpha) Ry(frame_beta),
sigma_minus = |0><1|,
D[A](rho) = A rho A^dagger - (A^dagger A rho + rho A^dagger A)/2.
```

The continuous-time laboratory-frame generator is

```text
L(rho) = -i [V (drive_omega X/2) V^dagger, rho]
         + gamma_down D[V sigma_minus V^dagger](rho)
         + gamma_up D[V sigma_minus^dagger V^dagger](rho)
         + (gamma_phi/2) D[V Z V^dagger](rho).
```

One time step is the exact CPTP map `exp(time_step * L)`, not a first-order
Euler step.  Constructing its small superoperator or Choi/Kraus support with
NumPy/SciPy is allowed.  Applying the channel to probe density matrices and
performing the quantum observable contractions and channel tomography must use
TensorCircuit-NG, for example `DMCircuit` plus `apply_general_kraus`.  Do not
replace the timed quantum evolution by a standalone NumPy affine-state
simulator.

The initial state of a probe with `preparation = [polar, azimuth]` is

```text
Rz(azimuth) Ry(polar) |0>.
```

At every integer in `readout_steps`, form the laboratory Bloch vector
`r = [Tr(X rho), Tr(Y rho), Tr(Z rho)]`.  If `M` is the supplied 3 by 3
`measurement_axes` matrix, that readout contributes `M @ r`.  Values are
ordered first by the supplied readout-step order and then by matrix-row order.
Training probes include `observed_features`; held-out probes do not.

## Exceptional point and modal response

In the canonical frame, the linear Bloch generator is

```text
G0 = [[-Gamma2,       0,       0],
      [      0, -Gamma2,  -Omega],
      [      0,    Omega, -Gamma1]],

Gamma1 = gamma_down + gamma_up,
Gamma2 = Gamma1/2 + gamma_phi,
Omega  = drive_omega.
```

The laboratory matrix is `G = R(V) G0 R(V)^T`.  Its exceptional-point
discriminant and signed distance in drive strength are

```text
D = (Gamma1 - Gamma2)^2 - 4 Omega^2,
signed_ep_offset = Omega - abs(Gamma1 - Gamma2)/2.
```

`D = 0` is defective for these admitted cases.  The verifier chooses one case
with `D > 0` and one with `D < 0`, while keeping `|D|` and every eigenvalue gap
above declared numerical floors.

Return Liouvillian eigenvalues in exactly this order:

```text
mu_plus  = -(Gamma1 + Gamma2)/2 + sqrt(D + 0j)/2,
mu_minus = -(Gamma1 + Gamma2)/2 - sqrt(D + 0j)/2,
mu_transverse = -Gamma2,
```

where the complex principal square root is used.  For these three distinct
values, define the gauge-invariant spectral projectors

```text
P_j = product(k != j) (G - mu_k I)/(mu_j - mu_k).
```

Each response record supplies real laboratory vectors `source = b` and
`readout = m`.  Preserve response-record order and return

```text
modal_residues[pair,j] = m^T P_j b,
susceptibilities[pair,k] = m^T (i*omega_k*I - G)^(-1) b
```

for the supplied frequency order.  The two forms must obey
`chi(omega) = sum_j residue_j/(i*omega-mu_j)`.  Also return the physical steady
Bloch vector of the inferred affine Liouvillian.

## Output

Preserve case order and opaque case IDs.  Return NumPy-compatible values with
exactly this layout:

```python
{
    "cases": [
        {
            "case_id": str,
            "estimated_parameters": real array,       # (6,)
            "training_rmse": float,
            "heldout_features": real array,           # (2, 4, 3)
            "liouvillian_matrix": real array,          # (3, 3), lab frame
            "steady_bloch": real array,                # (3,)
            "ep_discriminant": float,
            "signed_ep_offset": float,
            "liouvillian_eigenvalues": complex array,  # (3,), declared order
            "modal_residues": complex array,           # (3, 3)
            "susceptibilities": complex array,         # (3, 5)
        },
        ...,
    ]
}
```

`training_rmse` is the root mean squared residual over every supplied training
feature.  The evaluator recomputes it and all held-out responses independently
from a complex128 NumPy superoperator.  A separate analytic Bloch/Kraus canary
checks vectorization, complete positivity, trace preservation, frame rotation,
and exceptional-point conventions.

## Submission

Create `/root/solution_114.py` containing `run_solution(config)`.  The timed
call must finish within 300 seconds and the implementation must contain at
most 160 effective non-empty, non-comment Python lines.  Do not inspect hidden
evaluator files, logs, rewards, or secrets.
