# Problem 102: Gauge-fixed quantum signal-processing phase synthesis

## Goal

Recover a reflection-symmetric list of phase factors for each supplied complex
parity polynomial.  The returned phases must reproduce the polynomial as the
upper-left entry of the quantum signal-processing (QSP) sequence below.  The
evaluator generates several deterministic instances and checks the response on
held-out points that are not included in the configuration.

This is an inverse synthesis task.  Returning polynomial samples or a fitted
classical polynomial is not sufficient: the submitted artifact is the QSP
phase sequence itself.

## Exact convention

For `x` in `[-1, 1]`, define

$$
W(x)=\begin{pmatrix}x&i\sqrt{1-x^2}\\i\sqrt{1-x^2}&x\end{pmatrix},\qquad
Z(\phi)=\operatorname{diag}(e^{i\phi},e^{-i\phi}).
$$

For degree `d` and `d + 1` real phases, the ordered QSP unitary is

$$
U_{\boldsymbol\phi}(x)=Z(\phi_0)
\prod_{k=1}^{d}\left[W(x)Z(\phi_k)\right],
$$

where products are applied from left to right in increasing `k`.  The target
response is

$$P(x)=[U_{\boldsymbol\phi}(x)]_{00}.$$

The phase vector must obey reflection symmetry
`phi[k] == phi[d-k]`.  Phases are interpreted modulo `2*pi`; return the
principal representatives in `[-pi, pi]`.

## Configuration

The evaluator calls `run_solution(config)` separately for multiple generated
instances.  Each configuration has this schema:

```python
{
    "degree": 8,
    "target_chebyshev_real": <float array, shape (degree + 1,)>,
    "target_chebyshev_imag": <float array, shape (degree + 1,)>,
    "training_x": <float array with at least 2 * degree + 3 entries>,
    "response_tolerance": 1.5e-3,
    "instance_fingerprint": <hexadecimal string>,
    "max_iterations": 900,
}
```

The target polynomial is completely specified by its Chebyshev coefficients,
with NumPy's convention
`P(x) = numpy.polynomial.chebyshev.chebval(x, coefficients)`.  Coefficients of
the wrong parity are numerically zero.  Every generated target is guaranteed
to have at least one reflection-symmetric QSP realization in this convention.
The fingerprint is an identifier only and carries no numerical information.

## Solution interface

Create `solution_102.py` exposing:

```python
def run_solution(config):
    return {"phases": phases}
```

`phases` must be a finite real NumPy-compatible array of shape
`(config["degree"] + 1,)`.  Do not return target responses, hidden-grid values,
or evaluator-specific data.

## Passing criteria

For every generated instance, the independent evaluator requires:

- the exact phase-vector shape and finite real values;
- principal representatives in `[-pi, pi]`;
- reflection-symmetry error at most `2e-5`;
- held-out response root-mean-square error at most `7.5e-4`;
- held-out maximum response error at most `1.5e-3`;
- QSP unitarity defect at most `2e-12` under the evaluator's independent
  matrix implementation;
- parity-identity error at most `2e-10`.

The default evaluation uses three independently generated degree-8 targets.
The timed solution calls together must finish within 300 seconds.  The intended
expert runtime target is below 180 seconds.

The solution may use any quantum software framework.  Framework constraints
are supplied separately by the benchmark runner.
