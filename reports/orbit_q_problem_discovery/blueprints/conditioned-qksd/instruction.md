# Problem 108: Readout-calibrated, conditioned fermionic QKSD

Reconstruct a finite-shot quantum Krylov subspace diagonalization (QKSD) from
asymmetric-readout Hadamard-test records, stabilize its indefinite overlap
matrix by canonical orthogonalization, and reproduce phase-sensitive control
moments of the same large fermionic experiment with TensorCircuit-NG.

The production cases contain 56 and 60 fermionic modes.  A dense qubit
statevector is therefore not an admissible representation.  The intended
quantum representation is TensorCircuit's fermionic-Gaussian simulator.

## Fermionic experiment

For one case, let

```text
H = sum_i onsite[i] c_i^dagger c_i
    + sum_[i,j,re,im] ((re + i im) c_i^dagger c_j
                       + (re - i im) c_j^dagger c_i).
```

The initial Slater determinant occupies exactly the modes in `filled`.  Define

```text
U = exp(-i * time_step * H)
|phi_l> = U**l |phi_0>
g_l = <phi_0|phi_l>
h_l = <phi_0|H|phi_l>.
```

Complex Peierls phases are present.  A magnitude-only Gaussian-state overlap
is insufficient: both real and imaginary parts of each control moment are
scored.  Global phase conventions must be consistent with the displayed
second-quantized evolution.

The QKSD basis is `|phi_0>, ..., |phi_(K-1)>`, where
`K = krylov_dimension`.  Because `H` commutes with `U`, its exact projected
matrices are Hermitian Toeplitz:

```text
S[i,j] = g_(j-i)             if j >= i, else conj(g_(i-j))
Hk[i,j] = h_(j-i)            if j >= i, else conj(h_(i-j)).
```

## Raw Hadamard-test records and readout calibration

The QKSD moments are supplied only as finite-shot counts.  Counts always have
the form `[observed_plus, observed_minus]`; their signed mean is

```text
m(counts) = (observed_plus - observed_minus) / sum(counts).
```

There are four measurement channels in this exact order:

```text
0: Re(g_l), 1: Im(g_l), 2: Re(h_l / B), 3: Im(h_l / B),
```

where `B = hamiltonian_normalization` is the declared sector block-encoding
normalization.  `calibration_counts[channel, 0]` was obtained from a prepared
`+1` eigenvalue and `calibration_counts[channel, 1]` from a prepared `-1`
eigenvalue.  For each channel compute

```text
r_plus  = m(calibration_counts[channel, 0])
r_minus = m(calibration_counts[channel, 1])
a = (r_plus - r_minus) / 2
b = (r_plus + r_minus) / 2
calibrated(raw) = clip((m(raw) - b) / a, -1, 1).
```

Apply this independently to the two quadratures in `overlap_counts[lag]` and
`hamiltonian_counts[lag]`; multiply the latter by `B`.  Enforce the structural
identities `g_0 = 1 + 0j` and `Im(h_0) = 0`, then form the two Toeplitz matrices
above and explicitly Hermitize each as `(A + A^dagger)/2`.

## Canonical overlap filtering

Independent shot noise makes the sampled overlap matrix indefinite and nearly
singular.  Do not pass it directly to a generalized eigensolver.

Let `s, V = eigh(S)` in ascending eigenvalue order.  Let
`minimum_overlap_shots` be the smallest total count among every quadrature
record in `overlap_counts`.  Use the prescribed noise cutoff

```text
cutoff = noise_filter_z * sqrt(K / minimum_overlap_shots).
```

Retain exactly the modes with `s > cutoff`.  With

```text
X = V[:, retained] / sqrt(s[retained])
H_eff = (X^dagger Hk X + (X^dagger Hk X)^dagger) / 2,
```

return the three smallest eigenvalues of `H_eff`, the retained rank, the
cutoff, and the retained overlap condition number
`max(s[retained]) / min(s[retained])`.  The seeded cases have at least three
retained modes and a nonzero spectral margin around the cutoff.

## TensorCircuit control moments

The finite-shot data do not contain the noiseless control targets.  For every
integer in `control_lags`, independently evolve the supplied Slater determinant
under the displayed Hamiltonian with TensorCircuit-NG and return the complex
`g_l` and `h_l`.  The control lags include both in-window and extrapolation
points.  The core state evolution must use TensorCircuit's
`FGSSimulator`/fermionic-Gaussian APIs; do not replace it with a standalone
NumPy/SciPy one-particle simulator or a dense qubit statevector.

For a number-conserving Slater representation with occupied-orbital matrices
`Q_0` and `Q_l`, the phase-sensitive identities

```text
g_l = det(Q_0^dagger Q_l)
h_l = g_l * trace((Q_0^dagger Q_l)^(-1) Q_0^dagger A Q_l)
```

may be used after TensorCircuit evolution; `A` is the one-particle matrix of
`H`.  Solve the linear system rather than explicitly constructing the inverse.

## Configuration and output

`run_solution(config)` receives two hidden seeded cases.  Each case contains
the fields described above plus a nonce.  Return NumPy-compatible arrays:

```python
{
    "ritz_energies": float array,                 # (2, 3)
    "retained_ranks": integer array,              # (2,)
    "overlap_cutoffs": float array,               # (2,)
    "retained_condition_numbers": float array,    # (2,)
    "overlap_spectra": float array,               # (2, K)
    "sampled_overlap_matrices": complex array,    # (2, K, K)
    "sampled_hamiltonian_matrices": complex array,# (2, K, K)
    "control_overlaps": complex array,            # (2, n_controls)
    "control_hamiltonian_moments": complex array, # (2, n_controls)
}
```

The evaluator independently rebuilds the one-particle evolution, generates all
hidden shot records, repeats calibration and canonical filtering, and checks a
separate dense Fock-space identity canary.  Seeds change the lattice
relabeling, disorder, complex hopping phases, occupied modes, measurement
counts, and readout response.

Create `/root/solution_108.py` with `run_solution(config)`.  The implementation
must have at most 160 effective non-empty, non-comment Python lines and the
timed call must finish within 300 seconds.

## Numerical and statistical cautions

- Calibration divides by a measured readout slope, so using nominal response
  probabilities or pooling channels biases every Toeplitz diagonal.
- Shot noise destroys positive semidefiniteness; negative overlap eigenvalues
  are expected and are evidence that the conditioning path is active.
- Filtering must occur before whitening.  Taking square roots of the raw
  overlap spectrum or using a generic generalized eigensolver is unstable.
- Near-zero Slater transition overlaps can make transition densities fragile;
  use complex128, determinants, and linear solves consistently.
- Fermionic-Gaussian overlap helpers that discard the U(1) phase cannot satisfy
  the complex control-moment checks.
