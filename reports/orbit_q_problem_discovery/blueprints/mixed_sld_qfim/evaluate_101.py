"""Independent NumPy/SciPy evaluator for Problem 101."""

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import expm


DEFAULT_SEEDS = (101021, 101033, 101051)
BASE_SEED_OFFSETS = tuple(seed - DEFAULT_SEEDS[0] for seed in DEFAULT_SEEDS)
SUPPORT_FLOOR = 2e-5
REQUIRED_SCORE_GAIN = 0.32
REQUIRED_MIN_QFIM_EIGENVALUE = 0.025
SLD_IDENTITY_TOLERANCE = 3e-7
EXPERT_RUNTIME_TARGET_SECONDS = 180.0
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
PAULIS = (X, Y, Z)


def seeds_from_base(base_seed):
    """Expand one verifier-only base seed to the three-instance schedule."""

    if base_seed < 0:
        raise ValueError("base seed must be non-negative")
    return tuple(base_seed + offset for offset in BASE_SEED_OFFSETS)


def default_seeds():
    """Preserve the local canaries unless the verifier supplies one base seed."""

    base_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    return DEFAULT_SEEDS if base_seed is None else seeds_from_base(int(base_seed))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def case_digest(value):
    """Hash the exact ordered public configuration under the pinned runtime."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def rotation(pauli, theta, _expm=expm, _float=float):
    return _expm(-0.5j * _float(theta) * pauli)


def apply_unitary(rho, unitary):
    return unitary @ rho @ unitary.conj().T


def local(pauli, qubit, _kron=np.kron, _i2=I2):
    return _kron(pauli, _i2) if qubit == 0 else _kron(_i2, pauli)


def oracle_density(
    probe,
    theta,
    weights,
    noise,
    _zeros=np.zeros,
    _complex128=np.complex128,
    _local=local,
    _x=X,
    _y=Y,
    _z=Z,
    _kron=np.kron,
    _apply_unitary=apply_unitary,
    _rotation=rotation,
    _range=range,
    _float=float,
    _sum=np.sum,
    _paulis=PAULIS,
    _zip=zip,
):
    rho = _zeros((4, 4), dtype=_complex128)
    rho[0, 0] = 1.0
    probe_gates = (
        (_local(_y, 0), probe[0]),
        (_local(_z, 0), probe[1]),
        (_local(_y, 1), probe[2]),
        (_local(_z, 1), probe[3]),
        (_kron(_x, _x), probe[4]),
        (_kron(_y, _y), probe[5]),
    )
    for generator, angle in probe_gates:
        rho = _apply_unitary(rho, _rotation(generator, angle))
    for p in _range(3):
        gates = (
            (_local(_x, 0), 2 * theta[p] * weights[p, 0]),
            (_local(_y, 1), 2 * theta[p] * weights[p, 1]),
            (_kron(_z, _z), 2 * theta[p] * weights[p, 2]),
            (_local(_z, 1), 2 * theta[p] * weights[p, 3]),
            (_kron(_x, _x), 2 * theta[p] * weights[p, 4]),
        )
        for generator, angle in gates:
            rho = _apply_unitary(rho, _rotation(generator, angle))
    for qubit in _range(2):
        original = rho
        rho = (1.0 - _float(_sum(noise[qubit]))) * original
        for probability, pauli in _zip(noise[qubit], _paulis):
            embedded = _local(pauli, qubit)
            rho = rho + probability * (embedded @ original @ embedded)
    return rho


def finite_derivatives(
    probe,
    theta,
    weights,
    noise,
    step=2e-5,
    _zeros=np.zeros,
    _oracle_density=oracle_density,
    _range=range,
):
    derivatives = []
    for index in _range(3):
        delta = _zeros(3)
        delta[index] = step
        plus = _oracle_density(probe, theta + delta, weights, noise)
        minus = _oracle_density(probe, theta - delta, weights, noise)
        derivatives.append((plus - minus) / (2.0 * step))
    return derivatives


def qfim_spectral(
    rho,
    derivatives,
    _eigh=np.linalg.eigh,
    _empty=np.empty,
    _float=float,
    _sum=np.sum,
    _real=np.real,
    _range=range,
):
    values, vectors = _eigh(rho)
    transformed = [
        vectors.conj().T @ derivative @ vectors for derivative in derivatives
    ]
    qfim = _empty((3, 3), dtype=_float)
    denominator = values[:, None] + values[None, :]
    for a in _range(3):
        for b in _range(3):
            numerator = transformed[a] * transformed[b].T
            qfim[a, b] = _float(_sum(2.0 * _real(numerator) / denominator))
    return 0.5 * (qfim + qfim.T)


def qfim_sld(
    rho,
    derivatives,
    _eye=np.eye,
    _complex128=np.complex128,
    _kron=np.kron,
    _solve=np.linalg.solve,
    _array=np.array,
    _real=np.real,
    _trace=np.trace,
    _range=range,
):
    identity = _eye(4, dtype=_complex128)
    linear_map = 0.5 * (_kron(identity, rho) + _kron(rho.T, identity))
    slds = [
        _solve(linear_map, derivative.reshape(-1, order="F")).reshape(4, 4, order="F")
        for derivative in derivatives
    ]
    qfim = _array(
        [
            [_real(_trace(derivatives[a] @ slds[b])) for b in _range(3)]
            for a in _range(3)
        ]
    )
    return 0.5 * (qfim + qfim.T)


def generate_case(seed):
    rng = np.random.default_rng(seed)
    base = np.array(
        [
            [1.0, 0.20, 0.36, -0.12, 0.18],
            [0.14, 1.0, -0.24, 0.31, 0.22],
            [0.23, -0.16, 0.92, 0.18, 0.47],
        ]
    )
    weights = base + rng.normal(0.0, 0.055, base.shape)
    training_theta = rng.uniform(-0.48, 0.48, size=(4, 3))
    base_noise = np.array([[0.030, 0.022, 0.016], [0.024, 0.034, 0.019]])
    training_noise = np.stack(
        [
            np.clip(base_noise + rng.normal(0.0, 0.0035, base_noise.shape), 0.009, 0.05)
            for _ in range(3)
        ]
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "weights": weights.round(12).tolist(),
                "theta": training_theta.round(12).tolist(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    return {
        "generator_weights": weights,
        "training_theta": training_theta,
        "training_noise": training_noise,
        "support_floor": SUPPORT_FLOOR,
        "score_regularizer": 0.04,
        "required_score_gain": REQUIRED_SCORE_GAIN,
        "required_min_qfim_eigenvalue": REQUIRED_MIN_QFIM_EIGENVALUE,
        "max_steps": 650,
        "learning_rate": 0.035,
        "instance_fingerprint": fingerprint,
    }


def hidden_ensemble(seed, config):
    rng = np.random.default_rng(seed ^ 0x713B)
    theta = rng.uniform(-0.53, 0.53, size=(7, 3))
    center = np.mean(np.asarray(config["training_noise"]), axis=0)
    noise = np.stack(
        [
            np.clip(center + rng.normal(0.0, 0.0045, center.shape), 0.008, 0.052)
            for _ in range(5)
        ]
    )
    return [(point, channel) for point in theta for channel in noise]


def probe_metrics(
    probe,
    ensemble,
    config,
    _asarray=np.asarray,
    _inf=np.inf,
    _oracle_density=oracle_density,
    _finite_derivatives=finite_derivatives,
    _qfim_spectral=qfim_spectral,
    _eigvalsh=np.linalg.eigvalsh,
    _float=float,
    _min=min,
    _slogdet=np.linalg.slogdet,
    _eye=np.eye,
    _norm=np.linalg.norm,
    _abs=abs,
    _trace=np.trace,
    _qfim_sld=qfim_sld,
    _max=max,
    _np_max=np.max,
    _enumerate=enumerate,
):
    weights = _asarray(config["generator_weights"])
    scores = []
    min_eigenvalues = []
    min_support = _inf
    identity_errors = []
    physical = True
    for item, (theta, noise) in _enumerate(ensemble):
        rho = _oracle_density(probe, theta, weights, noise)
        derivatives = _finite_derivatives(probe, theta, weights, noise)
        qfim = _qfim_spectral(rho, derivatives)
        rho_values = _eigvalsh(rho)
        qfim_values = _eigvalsh(qfim)
        min_support = _min(min_support, _float(rho_values[0]))
        min_eigenvalues.append(_float(qfim_values[0]))
        scores.append(_float(_slogdet(qfim + config["score_regularizer"] * _eye(3))[1]))
        physical &= _norm(rho - rho.conj().T) <= 2e-10
        physical &= _abs(_trace(rho) - 1.0) <= 2e-10
        physical &= _norm(qfim - qfim.T) <= 3e-8 and qfim_values[0] >= -2e-7
        if item in (0, 17, 34):
            identity_errors.append(
                _float(_np_max(_abs(qfim - _qfim_sld(rho, derivatives))))
            )
    return {
        "robust_score": _min(scores),
        "minimum_qfim_eigenvalue": _min(min_eigenvalues),
        "minimum_support_eigenvalue": min_support,
        "sld_identity_error": _max(identity_errors),
        "physical": physical,
    }


def evaluate_case(
    module,
    config,
    ensemble,
    *,
    _clock=time.perf_counter,
    _deepcopy=copy.deepcopy,
    _asarray=np.asarray,
    _float_type=float,
    _all=np.all,
    _isfinite=np.isfinite,
    _bool=bool,
    _probe_metrics=probe_metrics,
    _negative_inf=-np.inf,
    _positive_inf=np.inf,
    _zeros=np.zeros,
    _isinstance=isinstance,
    _dict=dict,
    _set=set,
    _abs=np.abs,
    _pi=float(np.pi),
    _sld_tolerance=SLD_IDENTITY_TOLERANCE,
    _real_numeric_kinds=frozenset(("f", "i", "u")),
):
    solution_config = _deepcopy(config)
    started = _clock()
    result = module.run_solution(solution_config)
    elapsed = _clock() - started
    keys_valid = _isinstance(result, _dict) and _set(result) == {"probe_parameters"}
    raw_probe = _asarray(result["probe_parameters"] if keys_valid else ())
    real_numeric = raw_probe.dtype.kind in _real_numeric_kinds
    probe = _asarray(raw_probe, dtype=_float_type) if real_numeric else _zeros(0)
    valid_shape = real_numeric and probe.shape == (6,)
    finite = _bool(valid_shape and _all(_isfinite(probe)))
    candidate = (
        _probe_metrics(probe, ensemble, config)
        if finite
        else {
            "robust_score": _negative_inf,
            "minimum_qfim_eigenvalue": _negative_inf,
            "minimum_support_eigenvalue": _negative_inf,
            "sld_identity_error": _positive_inf,
            "physical": False,
        }
    )
    baseline = _probe_metrics(_zeros(6), ensemble, config)
    score_gain = candidate["robust_score"] - baseline["robust_score"]
    criteria = {
        "result keys exactly probe_parameters": keys_valid,
        "probe shape": valid_shape,
        "finite principal parameters": finite
        and _bool(_all(_abs(probe) <= _pi + 1e-12)),
        "physical fixed-support states": candidate["physical"]
        and candidate["minimum_support_eigenvalue"] >= config["support_floor"],
        "SLD and spectral identities agree": candidate["sld_identity_error"]
        <= _sld_tolerance,
        "held-out robust score gain": score_gain >= config["required_score_gain"],
        "held-out minimum QFIM eigenvalue": candidate["minimum_qfim_eigenvalue"]
        >= config["required_min_qfim_eigenvalue"],
    }
    return criteria, {
        "fingerprint": config["instance_fingerprint"],
        "elapsed": elapsed,
        "score_gain": score_gain,
        **candidate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_101")
    parser.add_argument("--seeds", default=",".join(map(str, default_seeds())))
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds or seeds != seeds_from_base(seeds[0]):
        raise ValueError("--seeds must be an ordered (b, b + 12, b + 30) schedule")
    verifier_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    if verifier_seed is not None and int(verifier_seed) != seeds[0]:
        raise ValueError("--seeds cannot override the verifier-authorized base seed")
    configs = [generate_case(seed) for seed in seeds]
    hidden_ensembles = [
        hidden_ensemble(seed, config) for seed, config in zip(seeds, configs)
    ]
    digest = case_digest(configs)
    case_inputs = tuple(zip(configs, hidden_ensembles))
    trusted_evaluate_case = evaluate_case
    trusted_clock = time.perf_counter
    trusted_print = print
    trusted_json_dumps = json.dumps
    trusted_all = all
    trusted_min = min
    trusted_max = max
    trusted_float = float
    trusted_system_exit = SystemExit
    admission_thresholds = (
        REQUIRED_SCORE_GAIN,
        REQUIRED_MIN_QFIM_EIGENVALUE,
        SUPPORT_FLOOR,
        SLD_IDENTITY_TOLERANCE,
        EXPERT_RUNTIME_TARGET_SECONDS,
    )
    trusted_print(
        trusted_json_dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": seeds[0],
                    "case_digest": digest,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    sys.argv[:] = [sys.argv[0]]
    module = importlib.import_module(args.solution)

    total_time = 0.0
    all_pass = True
    case_metrics = []
    trusted_print("Problem 101 evaluation")
    trusted_print(f"Solution module: {args.solution}")
    for config, ensemble in case_inputs:
        criteria, metrics = trusted_evaluate_case(
            module, config, ensemble, _clock=trusted_clock
        )
        case_metrics.append(metrics)
        total_time += metrics["elapsed"]
        all_pass &= trusted_all(criteria.values())
        trusted_print(
            f"Instance {metrics['fingerprint']}: gain={metrics['score_gain']:.6f}, "
            f"min-qfim={metrics['minimum_qfim_eigenvalue']:.6f}, "
            f"min-support={metrics['minimum_support_eigenvalue']:.3e}"
        )
        for name, passed in criteria.items():
            trusted_print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    trusted_print(f"End-to-end solution time: {total_time:.2f}s")
    trusted_print(
        "Runtime is reported separately and does not change functional correctness"
    )
    if all_pass:
        required_gain, required_qfim, support_floor, sld_tolerance, runtime_target = (
            admission_thresholds
        )
        admission = {
            "orbit_q_expert_admission_metrics": {
                "schema_version": 1,
                "protocol_seed": seeds[0],
                "case_digest": digest,
                "metrics": [
                    {
                        "metric": "minimum_heldout_robust_score_gain",
                        "direction": "at_least",
                        "observed": trusted_float(
                            trusted_min(row["score_gain"] for row in case_metrics)
                        ),
                        "threshold": required_gain,
                    },
                    {
                        "metric": "minimum_heldout_qfim_eigenvalue",
                        "direction": "at_least",
                        "observed": trusted_float(
                            trusted_min(
                                row["minimum_qfim_eigenvalue"] for row in case_metrics
                            )
                        ),
                        "threshold": required_qfim,
                    },
                    {
                        "metric": "minimum_support_eigenvalue",
                        "direction": "at_least",
                        "observed": trusted_float(
                            trusted_min(
                                row["minimum_support_eigenvalue"]
                                for row in case_metrics
                            )
                        ),
                        "threshold": support_floor,
                    },
                    {
                        "metric": "maximum_sld_spectral_identity_error",
                        "direction": "at_most",
                        "observed": trusted_float(
                            trusted_max(
                                row["sld_identity_error"] for row in case_metrics
                            )
                        ),
                        "threshold": sld_tolerance,
                    },
                    {
                        "metric": "total_solution_runtime_seconds",
                        "direction": "at_most",
                        "observed": trusted_float(total_time),
                        "threshold": runtime_target,
                    },
                ],
            }
        }
        trusted_print(
            trusted_json_dumps(admission, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    trusted_print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    raise trusted_system_exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
