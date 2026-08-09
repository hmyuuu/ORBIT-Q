# Problem 101: Robust mixed-state SLD-QFIM probe design

## Goal

Design a two-qubit probe for simultaneous estimation of three parameters under
anisotropic Pauli noise.  A candidate is judged by the symmetric logarithmic
derivative quantum Fisher information matrix (SLD-QFIM) over a deterministic
ensemble of sensing points and noise perturbations.  The evaluator also uses
held-out ensemble members that are not supplied to the solution.

The task is constructive: return the six probe-circuit parameters.  Do not
return a claimed QFIM or score; the evaluator independently reconstructs the
mixed states, differentiates them, and calculates every score.

## Probe circuit

Qubit 0 is the leftmost tensor factor.  Starting in `|00>`, apply in order

1. `RY(a0)` on qubit 0;
2. `RZ(a1)` on qubit 0;
3. `RY(a2)` on qubit 1;
4. `RZ(a3)` on qubit 1;
5. `RXX(a4)` on qubits `(0, 1)`;
6. `RYY(a5)` on qubits `(0, 1)`.

All rotations use `R_P(t) = exp(-i t P / 2)`.

## Parameter encoding and noise

For each sensing parameter `theta[p]`, `p = 0, 1, 2`, let row `w` be
`config["generator_weights"][p]`.  Apply these gates in order:

1. `RX(2 * theta[p] * w[0])` on qubit 0;
2. `RY(2 * theta[p] * w[1])` on qubit 1;
3. `RZZ(2 * theta[p] * w[2])` on `(0, 1)`;
4. `RZ(2 * theta[p] * w[3])` on qubit 1;
5. `RXX(2 * theta[p] * w[4])` on `(0, 1)`.

After all three parameter blocks, apply independent anisotropic Pauli channels
to qubits 0 and 1.  For row `(px, py, pz)`,

$$
\mathcal E(\rho)=(1-p_x-p_y-p_z)\rho+p_xX\rho X+p_yY\rho Y+p_zZ\rho Z.
$$

Every supplied or hidden channel has positive probabilities and produces a
full-rank state with smallest eigenvalue above `config["support_floor"]`.
Thus the state support is fixed over the entire scored domain.

## SLD-QFIM and robust score

For `a, b` in `{0, 1, 2}`, the SLD operators satisfy

$$
\partial_a\rho=(\rho L_a+L_a\rho)/2,
\qquad F_{ab}=\operatorname{Re}\operatorname{Tr}(\partial_a\rho L_b).
$$

At each ensemble member define

$$s(F)=\log\det(F+0.04 I_3).$$

The robust score is the minimum `s(F)` over the evaluated ensemble.  A valid
probe must improve this score over the fixed zero-parameter probe while also
maintaining a minimum QFIM eigenvalue on held-out members.

## Configuration

The evaluator calls `run_solution(config)` for several deterministic instances:

```python
{
    "generator_weights": <real array, shape (3, 5)>,
    "training_theta": <real array, shape (4, 3)>,
    "training_noise": <real array, shape (3, 2, 3)>,
    "support_floor": 2e-5,
    "score_regularizer": 0.04,
    "required_score_gain": 0.32,
    "required_min_qfim_eigenvalue": 0.025,
    "max_steps": 650,
    "learning_rate": 0.035,
    "instance_fingerprint": <hexadecimal string>,
}
```

The Cartesian product of the four training sensing points and three training
noise arrays is available for optimization.  Hidden sensing points and noise
arrays are generated independently from the same declared ranges.

## Solution interface

Create `solution_101.py` exposing:

```python
def run_solution(config):
    return {"probe_parameters": parameters}
```

`parameters` must be a finite real NumPy-compatible array of shape `(6,)`,
with principal representatives in `[-pi, pi]`.  The result dictionary must
contain no precomputed scores, density matrices, QFIMs, or hidden data.

## Passing criteria

Every default instance must satisfy all of the following under the independent
evaluator:

- exact result shape, finite values, and principal phase representatives;
- density matrices are Hermitian, unit trace, and have eigenvalues at least
  `support_floor` on every held-out member;
- every QFIM is real symmetric and positive semidefinite to numerical tolerance;
- held-out robust-score gain over the zero probe is at least `0.32`;
- the smallest held-out QFIM eigenvalue is at least `0.025`;
- analytic SLD and spectral SLD formulas agree on evaluator canary members;
- all timed solution calls together finish within 300 seconds.

The intended expert runtime target is below 180 seconds.  The solution may use
any quantum software framework; framework constraints are supplied separately
by the benchmark runner.
