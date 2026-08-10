from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import os
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT_ROOT = ROOT / "reports" / "orbit_q_problem_discovery" / "blueprints"
PRIMARY_SEED_ENV = "ORBIT_Q_CANDIDATE_SEED"
SINGLE_SEED_EVALUATORS = (
    ("dynamic_branch_equivalence", 103, 1032026, "ORBIT_DYNAMIC_SEED"),
    ("variance_aware_cutting", 104, 1042026, "ORBIT_CUTTING_SEED"),
    ("conditioned_fgs_inverse", 105, 1052026, None),
    ("css-state-verification", 106, 1062026, "ORBIT_CSS_SEED"),
    ("memory_bounded_iqp", 107, 1072026, "ORBIT_CONTRACTION_SEED"),
    ("conditioned-qksd", 108, 1082026, "ORBIT_QKSD_SEED"),
    ("robust-leakage-grape", 109, 1092026, None),
    ("nonmarkovian-channel-inverse", 110, 1102026, None),
    ("truncated-mps-gradient", 111, 1112026, "ORBIT_MPS_SEED"),
    ("charged-renyi-replica", 112, 1122026, "ORBIT_RENYI_SEED"),
    ("coherent-css-coset-inference", 113, 1132026, "ORBIT_COHERENT_CSS_SEED"),
    ("liouvillian-ep-response", 114, 1142026, "ORBIT_LIOUVILLIAN_EP_SEED"),
)
CASE_IDENTITY_EVALUATORS = (
    ("mixed_sld_qfim", 101),
    ("qsp_phase_synthesis", 102),
    *(tuple((slug, problem_id) for slug, problem_id, _, _ in SINGLE_SEED_EVALUATORS)),
)


def load_seed_api(slug: str, problem_id: int) -> SimpleNamespace:
    """Execute only the dependency-free seed declarations from an evaluator."""

    path = BLUEPRINT_ROOT / slug / f"evaluate_{problem_id}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    declarations = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"DEFAULT_SEED", "DEFAULT_SEEDS", "BASE_SEED_OFFSETS"}
            for target in node.targets
        ):
            declarations.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "default_seed",
            "default_seeds",
            "seeds_from_base",
        }:
            declarations.append(node)
    namespace = {"os": os}
    module = ast.fix_missing_locations(ast.Module(body=declarations, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def load_evaluator(slug: str, problem_id: int):
    path = BLUEPRINT_ROOT / slug / f"evaluate_{problem_id}.py"
    spec = importlib.util.spec_from_file_location(
        f"orbit_q_case_identity_{problem_id}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_qfim_protocol_api() -> ModuleType:
    """Load candidate 101's protocol control flow without numeric dependencies."""

    path = BLUEPRINT_ROOT / "mixed_sld_qfim" / "evaluate_101.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    declarations = []
    constant_names = {
        "DEFAULT_SEEDS",
        "BASE_SEED_OFFSETS",
        "SUPPORT_FLOOR",
        "REQUIRED_SCORE_GAIN",
        "REQUIRED_MIN_QFIM_EIGENVALUE",
        "SLD_IDENTITY_TOLERANCE",
        "EXPERT_RUNTIME_TARGET_SECONDS",
    }
    function_names = {
        "seeds_from_base",
        "default_seeds",
        "_json_default",
        "case_digest",
        "main",
    }
    for node in tree.body:
        if isinstance(node, ast.Import) and all(
            alias.name
            in {"argparse", "hashlib", "importlib", "json", "os", "sys", "time"}
            for alias in node.names
        ):
            declarations.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constant_names
            for target in node.targets
        ):
            declarations.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            declarations.append(node)
    module = ModuleType("orbit_q_qfim_protocol_101")
    protocol_tree = ast.fix_missing_locations(
        ast.Module(body=declarations, type_ignores=[])
    )
    exec(compile(protocol_tree, str(path), "exec"), module.__dict__)
    return module


def load_grape_protocol_api() -> ModuleType:
    """Load candidate 109's import boundary without numerical dependencies."""

    path = BLUEPRINT_ROOT / "robust-leakage-grape" / "evaluate_109.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    declarations = []
    import_names = {
        "argparse",
        "copy",
        "hashlib",
        "importlib",
        "json",
        "os",
        "sys",
        "time",
    }
    for node in tree.body:
        if isinstance(node, ast.Import) and all(
            alias.name in import_names for alias in node.names
        ):
            declarations.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            declarations.append(node)
    module = ModuleType("orbit_q_grape_protocol_109")
    protocol_tree = ast.fix_missing_locations(
        ast.Module(body=declarations, type_ignores=[])
    )
    exec(compile(protocol_tree, str(path), "exec"), module.__dict__)
    return module


def digest_for_seed(module, problem_id: int, seed: int) -> str:
    if problem_id == 101:
        seeds = module.seeds_from_base(seed)
        return module.case_digest(
            [module.generate_case(case_seed) for case_seed in seeds]
        )
    if problem_id == 102:
        seeds = module.seeds_from_base(seed)
        configs = [module.generate_case(case_seed)[0] for case_seed in seeds]
        return module.case_digest(configs)
    if problem_id in {103, 104, 106, 107, 108, 111, 112, 113}:
        return module.build_config(seed)["case_digest"]
    if problem_id == 114:
        config, _ = module._configuration(seed)
        return module._case_digest(config)
    generated = module._configuration(seed)
    config = generated[0] if isinstance(generated, tuple) else generated
    return module.case_digest(config)


@pytest.mark.parametrize(
    ("slug", "problem_id", "local_default", "legacy_env"),
    SINGLE_SEED_EVALUATORS,
)
def test_single_seed_evaluators_prefer_verifier_seed_and_preserve_fallbacks(
    slug: str,
    problem_id: int,
    local_default: int,
    legacy_env: str | None,
    monkeypatch,
) -> None:
    evaluator = load_seed_api(slug, problem_id)
    monkeypatch.delenv(PRIMARY_SEED_ENV, raising=False)
    if legacy_env is not None:
        monkeypatch.delenv(legacy_env, raising=False)
    assert evaluator.default_seed() == local_default

    if legacy_env is not None:
        monkeypatch.setenv(legacy_env, "771103")
        assert evaluator.default_seed() == 771103

    monkeypatch.setenv(PRIMARY_SEED_ENV, "880019")
    assert evaluator.default_seed() == 880019


def test_qsp_base_seed_expands_to_documented_three_instance_schedule(
    monkeypatch,
) -> None:
    evaluator = load_seed_api("qsp_phase_synthesis", 102)
    monkeypatch.delenv(PRIMARY_SEED_ENV, raising=False)
    assert evaluator.default_seeds() == evaluator.DEFAULT_SEEDS

    monkeypatch.setenv(PRIMARY_SEED_ENV, "880019")
    assert evaluator.default_seeds() == (880019, 880031, 880047)
    assert evaluator.seeds_from_base(880019) == evaluator.default_seeds()


@pytest.mark.parametrize(
    ("slug", "problem_id"),
    (("mixed_sld_qfim", 101), ("qsp_phase_synthesis", 102))
    + tuple((slug, problem_id) for slug, problem_id, _, _ in SINGLE_SEED_EVALUATORS),
)
def test_eligible_blueprints_document_one_verifier_base_seed(
    slug: str, problem_id: int
) -> None:
    metadata = json.loads((BLUEPRINT_ROOT / slug / "blueprint.json").read_text())
    protocol = metadata["seed_protocol"]

    assert metadata["problem_id"] == problem_id
    assert protocol["deterministic"] is True
    assert protocol["verifier_env"] == PRIMARY_SEED_ENV
    assert protocol["cli_override"] in {"--seed", "--seeds"}
    assert protocol["base_seed_mapping"]
    assert protocol["determinism_scope"]


@pytest.mark.parametrize(("slug", "problem_id"), CASE_IDENTITY_EVALUATORS)
def test_public_case_digest_is_stable_and_changes_with_seed(
    slug: str, problem_id: int
) -> None:
    pytest.importorskip("numpy", reason="case generators require the evaluator runtime")
    if problem_id in {101, 109}:
        pytest.importorskip(
            "scipy", reason="the candidate evaluator imports its SciPy oracle"
        )
    evaluator = load_evaluator(slug, problem_id)
    seed = 700_000 + problem_id
    first = digest_for_seed(evaluator, problem_id, seed)
    repeated = digest_for_seed(evaluator, problem_id, seed)
    changed = digest_for_seed(evaluator, problem_id, seed + 1)

    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert repeated == first
    assert changed != first


@pytest.mark.parametrize(("slug", "problem_id"), CASE_IDENTITY_EVALUATORS)
def test_case_identity_is_emitted_and_documented(slug: str, problem_id: int) -> None:
    source = (BLUEPRINT_ROOT / slug / f"evaluate_{problem_id}.py").read_text()
    metadata = json.loads((BLUEPRINT_ROOT / slug / "blueprint.json").read_text())
    protocol = metadata["seed_protocol"]

    assert source.count('"orbit_q_case_identity"') == 1
    assert '"protocol_seed"' in source
    assert '"case_digest"' in source
    assert "sha-256" in protocol["case_identity"].lower()
    assert "case_digest" in protocol["case_identity"]
    assert "full" in protocol["case_identity"].lower()
    assert "structured" in protocol["case_identity"].lower()
    note = protocol["pinned_numpy_note"].lower()
    assert "numpy 2.2.6" in note
    assert "seed alone" in note


@pytest.mark.parametrize(("slug", "problem_id"), CASE_IDENTITY_EVALUATORS)
def test_case_identity_is_flushed_before_untrusted_solution_execution(
    slug: str, problem_id: int
) -> None:
    source = (BLUEPRINT_ROOT / slug / f"evaluate_{problem_id}.py").read_text()
    tree = ast.parse(source)
    function_name = "main" if problem_id in {101, 102, 105, 109, 110} else "evaluate"
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None

    marker_offset = function_source.index('"orbit_q_case_identity"')
    untrusted_offset = function_source.index(
        "importlib.import_module" if problem_id in {101, 102} else "run_solution"
    )
    assert marker_offset < untrusted_offset
    assert "flush=True" in function_source[marker_offset:untrusted_offset]
    assert (
        'os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)'
        in function_source[marker_offset:untrusted_offset]
    )


def test_qsp_rejects_env_and_explicit_schedule_mismatch(monkeypatch) -> None:
    pytest.importorskip("numpy", reason="QSP evaluator imports NumPy")
    evaluator = load_evaluator("qsp_phase_synthesis", 102)
    monkeypatch.setenv(PRIMARY_SEED_ENV, "102031")
    monkeypatch.setattr(
        "sys.argv", ["evaluate_102.py", "--seeds", "902031,902043,902059"]
    )

    with pytest.raises(ValueError, match="verifier-authorized base seed"):
        evaluator.main()


def test_mixed_qfim_base_seed_expands_to_documented_three_instance_schedule(
    monkeypatch,
) -> None:
    evaluator = load_seed_api("mixed_sld_qfim", 101)
    monkeypatch.delenv(PRIMARY_SEED_ENV, raising=False)
    assert evaluator.default_seeds() == evaluator.DEFAULT_SEEDS

    monkeypatch.setenv(PRIMARY_SEED_ENV, "880019")
    assert evaluator.default_seeds() == (880019, 880031, 880049)
    assert evaluator.seeds_from_base(880019) == evaluator.default_seeds()


def test_mixed_qfim_rejects_env_and_explicit_schedule_mismatch(monkeypatch) -> None:
    evaluator = load_qfim_protocol_api()
    monkeypatch.setenv(PRIMARY_SEED_ENV, "101021")
    monkeypatch.setattr(
        "sys.argv", ["evaluate_101.py", "--seeds", "901021,901033,901051"]
    )

    with pytest.raises(ValueError, match="verifier-authorized base seed"):
        evaluator.main()


def test_mixed_qfim_emits_exact_admission_policy_and_scrubs_before_import(
    monkeypatch, capsys
) -> None:
    evaluator = load_qfim_protocol_api()
    base_seed = 750101
    schedule = evaluator.seeds_from_base(base_seed)
    observed_import_state = {}
    intercepted_after_import = []
    intercepted_evaluators = []
    intercepted_helpers = []
    intercepted_clocks = []
    completed_cases = []
    hidden_cases = []
    trusted_clock_calls = []
    trusted_clock_values = iter((100.0, 141.0, 200.0, 243.0, 300.0, 342.0))
    real_print = builtins.print

    def trusted_clock():
        trusted_clock_calls.append(True)
        return next(trusted_clock_values)

    def intercepted_evaluate(*args, **kwargs):
        intercepted_evaluators.append((args, kwargs))
        raise AssertionError("untrusted import replaced __main__.evaluate_case")

    def intercepted_helper(*args, **kwargs):
        intercepted_helpers.append((args, kwargs))
        raise AssertionError("untrusted import reached a replaced oracle helper")

    def intercepted_clock():
        intercepted_clocks.append(True)
        raise AssertionError("untrusted import replaced time.perf_counter")

    def fake_import(name):
        assert hidden_cases == list(schedule)
        observed_import_state["name"] = name
        observed_import_state["seed"] = os.environ.get(PRIMARY_SEED_ENV)
        observed_import_state["argv"] = list(__import__("sys").argv)

        def intercept(*args, **kwargs):
            intercepted_after_import.append(
                (len(completed_cases), " ".join(map(str, args)))
            )
            return real_print(*args, **kwargs)

        monkeypatch.setattr(builtins, "print", intercept)
        main_module = __import__("__main__")
        assert main_module is evaluator
        monkeypatch.setattr(main_module, "evaluate_case", intercepted_evaluate)
        for helper_name in (
            "probe_metrics",
            "oracle_density",
            "finite_derivatives",
            "qfim_spectral",
            "qfim_sld",
            "hidden_ensemble",
        ):
            monkeypatch.setattr(
                main_module, helper_name, intercepted_helper, raising=False
            )
        monkeypatch.setattr(evaluator.time, "perf_counter", intercepted_clock)
        return SimpleNamespace(run_solution=lambda config: None)

    rows = {
        schedule[0]: (0.91, 0.83, 0.0030, 4.0e-12),
        schedule[1]: (0.72, 0.64, 0.0040, 7.0e-12),
        schedule[2]: (0.81, 0.77, 0.0050, 5.0e-12),
    }

    def fake_hidden(seed, config):
        hidden_cases.append(seed)
        return {"protocol_seed": seed}

    def fake_evaluate(module, config, ensemble, *, _clock):
        seed = ensemble["protocol_seed"]
        gain, qfim, support, identity = rows[seed]
        started = _clock()
        module.run_solution(config)
        elapsed = _clock() - started
        completed_cases.append(seed)
        return {"synthetic protocol pass": True}, {
            "fingerprint": config["instance_fingerprint"],
            "elapsed": elapsed,
            "score_gain": gain,
            "minimum_qfim_eigenvalue": qfim,
            "minimum_support_eigenvalue": support,
            "sld_identity_error": identity,
            "physical": True,
            "robust_score": 1.0,
        }

    monkeypatch.setattr(evaluator.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        evaluator,
        "generate_case",
        lambda seed: {"instance_fingerprint": "opaque-protocol-fingerprint"},
        raising=False,
    )
    monkeypatch.setattr(evaluator, "evaluate_case", fake_evaluate, raising=False)
    monkeypatch.setattr(evaluator, "hidden_ensemble", fake_hidden, raising=False)
    monkeypatch.setattr(evaluator.time, "perf_counter", trusted_clock)
    monkeypatch.setitem(__import__("sys").modules, "__main__", evaluator)
    monkeypatch.setenv(PRIMARY_SEED_ENV, str(base_seed))
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            "evaluate_101.py",
            "--solution",
            "candidate_solution",
            "--seeds",
            ",".join(map(str, schedule)),
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        evaluator.main()
    assert stopped.value.code == 0
    assert observed_import_state == {
        "name": "candidate_solution",
        "seed": None,
        "argv": ["evaluate_101.py"],
    }
    assert intercepted_evaluators == []
    assert intercepted_helpers == []
    assert intercepted_clocks == []
    assert len(trusted_clock_calls) == 2 * len(schedule)
    private_schedule_tokens = {str(value) for value in schedule}
    assert all(
        completed == len(schedule)
        for completed, message in intercepted_after_import
        if any(token in message for token in private_schedule_tokens)
    )

    structured = []
    for line in capsys.readouterr().out.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            structured.append(value)
    identity_rows = [
        row["orbit_q_case_identity"]
        for row in structured
        if "orbit_q_case_identity" in row
    ]
    admission_rows = [
        row["orbit_q_expert_admission_metrics"]
        for row in structured
        if "orbit_q_expert_admission_metrics" in row
    ]
    assert len(identity_rows) == len(admission_rows) == 1
    expected_digest = evaluator.case_digest(
        [evaluator.generate_case(seed) for seed in schedule]
    )
    assert identity_rows[0] == {
        "protocol_seed": base_seed,
        "case_digest": expected_digest,
    }
    admission = admission_rows[0]
    assert admission["protocol_seed"] == base_seed
    assert admission["case_digest"] == expected_digest
    assert admission["metrics"] == [
        {
            "metric": "minimum_heldout_robust_score_gain",
            "direction": "at_least",
            "observed": 0.72,
            "threshold": 0.32,
        },
        {
            "metric": "minimum_heldout_qfim_eigenvalue",
            "direction": "at_least",
            "observed": 0.64,
            "threshold": 0.025,
        },
        {
            "metric": "minimum_support_eigenvalue",
            "direction": "at_least",
            "observed": 0.003,
            "threshold": 2e-5,
        },
        {
            "metric": "maximum_sld_spectral_identity_error",
            "direction": "at_most",
            "observed": 7e-12,
            "threshold": 3e-7,
        },
        {
            "metric": "total_solution_runtime_seconds",
            "direction": "at_most",
            "observed": 126.0,
            "threshold": 180.0,
        },
    ]


def test_mixed_qfim_oracle_prebinds_the_post_import_trust_chain() -> None:
    source = (BLUEPRINT_ROOT / "mixed_sld_qfim" / "evaluate_101.py").read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    evaluate_case = functions["evaluate_case"]
    evaluate_source = ast.get_source_segment(source, evaluate_case)
    assert evaluate_source is not None
    assert "solution_config = _deepcopy(config)" in evaluate_source
    assert "module.run_solution(solution_config)" in evaluate_source
    assert "_probe_metrics(probe, ensemble, config)" in evaluate_source
    assert "hidden_ensemble" not in evaluate_source
    assert "_clock=time.perf_counter" in evaluate_source
    assert "_probe_metrics=probe_metrics" in evaluate_source

    required_helper_defaults = {
        "rotation": {"expm"},
        "local": {"np.kron"},
        "oracle_density": {"local", "apply_unitary", "rotation"},
        "finite_derivatives": {"oracle_density"},
        "qfim_spectral": {"np.linalg.eigh"},
        "qfim_sld": {"np.linalg.solve"},
        "probe_metrics": {
            "oracle_density",
            "finite_derivatives",
            "qfim_spectral",
            "qfim_sld",
        },
    }
    for function_name, expected_defaults in required_helper_defaults.items():
        defaults = {
            ast.unparse(default)
            for default in functions[function_name].args.defaults
            + functions[function_name].args.kw_defaults
            if default is not None
        }
        assert expected_defaults <= defaults

    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main)
    assert main_source is not None
    import_offset = main_source.index("importlib.import_module")
    for binding in (
        "trusted_evaluate_case = evaluate_case",
        "trusted_clock = time.perf_counter",
        "trusted_print = print",
        "trusted_json_dumps = json.dumps",
    ):
        assert main_source.index(binding) < import_offset


@pytest.mark.parametrize("kind", ("numeric_string", "complex", "boolean"))
def test_mixed_qfim_rejects_non_real_numeric_probe_dtypes(kind: str) -> None:
    np = pytest.importorskip("numpy", reason="QFIM evaluator imports NumPy")
    evaluator = load_evaluator("mixed_sld_qfim", 101)
    invalid = {
        "numeric_string": ["0.1"] * 6,
        "complex": np.ones(6, dtype=np.complex128),
        "boolean": [True] * 6,
    }[kind]
    clock = iter((10.0, 11.0))

    def fake_metrics(probe, ensemble, config):
        return {
            "robust_score": 0.0,
            "minimum_qfim_eigenvalue": 0.0,
            "minimum_support_eigenvalue": 1.0,
            "sld_identity_error": 0.0,
            "physical": True,
        }

    criteria, _ = evaluator.evaluate_case(
        SimpleNamespace(run_solution=lambda config: {"probe_parameters": invalid}),
        {
            "instance_fingerprint": "opaque",
            "support_floor": 2e-5,
            "required_score_gain": 0.32,
            "required_min_qfim_eigenvalue": 0.025,
        },
        [],
        _clock=lambda: next(clock),
        _probe_metrics=fake_metrics,
    )
    assert criteria["result keys exactly probe_parameters"] is True
    assert criteria["probe shape"] is False
    assert criteria["finite principal parameters"] is False


def test_robust_grape_scrubs_seed_channels_before_untrusted_import(
    monkeypatch, capsys
) -> None:
    evaluator = load_grape_protocol_api()
    seed = (1 << 62) + 750_109
    observed_import_state = {}

    def run_solution(config):
        raise RuntimeError("stop after observing scrubbed import state")

    def fake_import(name):
        observed_import_state["name"] = name
        observed_import_state["seed"] = os.environ.get(PRIMARY_SEED_ENV)
        observed_import_state["argv"] = list(__import__("sys").argv)
        return SimpleNamespace(run_solution=run_solution)

    monkeypatch.setattr(evaluator.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        evaluator,
        "default_seed",
        lambda: 1092026,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "_configuration",
        lambda selected: {"protocol_seed": selected},
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "case_digest",
        lambda config: "a" * 64,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator,
        "_heldout",
        lambda config, selected: [{"precomputed_for": selected}],
        raising=False,
    )
    monkeypatch.setattr(evaluator, "_metrics", lambda *args: None, raising=False)
    monkeypatch.setenv(PRIMARY_SEED_ENV, str(seed))
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            "evaluate_109.py",
            "--solution",
            "candidate_solution",
            "--seed",
            str(seed),
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        evaluator.main()
    assert stopped.value.code == 1
    assert observed_import_state == {
        "name": "candidate_solution",
        "seed": None,
        "argv": ["evaluate_109.py"],
    }

    structured = []
    for line in capsys.readouterr().out.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            structured.append(value)
    identity_rows = [
        row["orbit_q_case_identity"]
        for row in structured
        if "orbit_q_case_identity" in row
    ]
    assert len(identity_rows) == 1
    assert identity_rows[0] == {
        "protocol_seed": seed,
        "case_digest": "a" * 64,
    }


def test_robust_grape_rejects_env_and_explicit_seed_mismatch(monkeypatch) -> None:
    evaluator = load_grape_protocol_api()
    monkeypatch.setattr(evaluator, "default_seed", lambda: 1092026, raising=False)
    monkeypatch.setenv(PRIMARY_SEED_ENV, "1092026")
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["evaluate_109.py", "--seed", "9092026"],
    )

    with pytest.raises(ValueError, match="verifier-authorized seed"):
        evaluator.main()


def test_robust_grape_scrubs_argv_after_identity_before_import() -> None:
    source = (BLUEPRINT_ROOT / "robust-leakage-grape" / "evaluate_109.py").read_text()
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    function_source = ast.get_source_segment(source, main)
    assert function_source is not None

    identity_offset = function_source.index('"orbit_q_case_identity"')
    heldout_offset = function_source.index(
        "hidden_members = _heldout(config, args.seed)"
    )
    copy_offset = function_source.index("solution_config = copy.deepcopy(config)")
    scrub_offset = function_source.index("sys.argv[:] = [sys.argv[0]]")
    import_offset = function_source.index("importlib.import_module")
    assert heldout_offset < copy_offset < identity_offset < scrub_offset < import_offset


def test_robust_grape_seals_rng_config_and_output_before_untrusted_import(
    monkeypatch, capsys
) -> None:
    evaluator = load_grape_protocol_api()

    class FakeControls:
        shape = (44, 2)

    difference_marker = object()
    fake_np = SimpleNamespace(
        asarray=lambda value, dtype=None: value,
        diff=lambda value, axis=0: difference_marker,
        isfinite=lambda value: [True],
        all=all,
        max=max,
        quantile=lambda values, quantile: max(values),
        linalg=SimpleNamespace(
            norm=lambda value, axis=1: (
                [0.0] * (43 if value is difference_marker else 44)
            )
        ),
    )
    evaluator.np = fake_np
    seed = (1 << 62) + 811_109
    digest = "b" * 64
    trusted_config = {
        "n_slices": 44,
        "training_ensemble": [{"detuning": 0.0}],
        "uncertainty_bounds": {
            "detuning": [-0.1, 0.1],
            "amplitude_fraction": [-0.07, 0.07],
            "anharmonic_shift": [-0.2, 0.2],
        },
        "max_drive_amplitude": 3.15,
        "max_slew_per_slice": 1.15,
        "max_edge_amplitude": 0.035,
        "maximum_worst_infidelity": 0.00125,
        "maximum_p95_infidelity": 0.0007,
        "maximum_worst_leakage": 0.00015,
        "optimization_seed": int(seed % (2**31 - 1)),
    }
    hidden_members = [{"private_member": True}]
    events = []
    intercepted_prints = []
    intercepted_serializations = []
    intercepted_metrics = []
    intercepted_clocks = []
    trusted_clock_values = iter((100.0, 101.0))
    received_solution_config = {}

    def fake_heldout(config, selected):
        assert config is trusted_config
        assert selected == seed
        events.append("heldout_precomputed")
        return hidden_members

    def run_solution(config):
        received_solution_config["value"] = config
        assert config is not trusted_config
        assert config["training_ensemble"] is not trusted_config["training_ensemble"]
        config["n_slices"] = 1
        config["training_ensemble"].append({"detuning": 99.0})
        config["maximum_p95_infidelity"] = -1.0
        return {
            "controls": FakeControls(),
            "training_worst_infidelity": 0.0002,
            "training_worst_leakage": 0.00001,
        }

    def intercepted_heldout(*args, **kwargs):
        events.append("heldout_intercepted_after_import")
        raise AssertionError("held-out RNG was invoked after untrusted import")

    def intercepted_print(*args, **kwargs):
        intercepted_prints.append((args, kwargs))

    def intercepted_dumps(*args, **kwargs):
        intercepted_serializations.append((args, kwargs))
        raise AssertionError("post-import serialization used a mutable global")

    def replaced_metrics(*args, **kwargs):
        intercepted_metrics.append((args, kwargs))
        raise AssertionError("untrusted import replaced __main__._metrics")

    def replaced_clock():
        intercepted_clocks.append(True)
        raise AssertionError("untrusted import replaced time.perf_counter")

    def fake_import(name):
        assert name == "candidate_solution"
        assert events == ["heldout_precomputed"]
        assert os.environ.get(PRIMARY_SEED_ENV) is None
        assert __import__("sys").argv == ["evaluate_109.py"]
        monkeypatch.setattr(evaluator, "_heldout", intercepted_heldout)
        monkeypatch.setattr(builtins, "print", intercepted_print)
        monkeypatch.setattr(evaluator.json, "dumps", intercepted_dumps)
        main_module = __import__("__main__")
        assert main_module is evaluator
        monkeypatch.setattr(main_module, "_metrics", replaced_metrics)
        monkeypatch.setattr(evaluator.time, "perf_counter", replaced_clock)
        return SimpleNamespace(run_solution=run_solution)

    def fake_metrics(controls, config, members):
        assert config is trusted_config
        assert config["n_slices"] == 44
        assert config["maximum_p95_infidelity"] == 0.0007
        if members is trusted_config["training_ensemble"]:
            return [0.0001, 0.0002], [0.00001, 0.000009]
        assert members is hidden_members
        return [0.0003, 0.0004], [0.00002, 0.000018]

    monkeypatch.setattr(
        evaluator, "_configuration", lambda selected: trusted_config, raising=False
    )
    monkeypatch.setattr(evaluator, "default_seed", lambda: 1092026, raising=False)
    monkeypatch.setattr(evaluator, "_heldout", fake_heldout, raising=False)
    monkeypatch.setattr(evaluator, "case_digest", lambda config: digest, raising=False)
    monkeypatch.setattr(evaluator, "_metrics", fake_metrics, raising=False)
    monkeypatch.setattr(
        evaluator.time, "perf_counter", lambda: next(trusted_clock_values)
    )
    monkeypatch.setattr(evaluator.importlib, "import_module", fake_import)
    monkeypatch.setitem(__import__("sys").modules, "__main__", evaluator)
    monkeypatch.setenv(PRIMARY_SEED_ENV, str(seed))
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["evaluate_109.py", "--solution", "candidate_solution", "--seed", str(seed)],
    )

    with pytest.raises(SystemExit) as stopped:
        evaluator.main()
    assert stopped.value.code == 0
    assert events == ["heldout_precomputed"]
    assert intercepted_prints == []
    assert intercepted_serializations == []
    assert intercepted_metrics == []
    assert intercepted_clocks == []
    assert received_solution_config["value"] is not trusted_config
    assert trusted_config["n_slices"] == 44
    assert trusted_config["training_ensemble"] == [{"detuning": 0.0}]
    assert trusted_config["maximum_p95_infidelity"] == 0.0007

    output = capsys.readouterr().out
    assert f"Case seed: {seed}" not in output
    structured = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            structured.append(value)
    identity = next(
        row["orbit_q_case_identity"]
        for row in structured
        if "orbit_q_case_identity" in row
    )
    admission = next(
        row["orbit_q_expert_admission_metrics"]
        for row in structured
        if "orbit_q_expert_admission_metrics" in row
    )
    assert identity == {"protocol_seed": seed, "case_digest": digest}
    assert admission["protocol_seed"] == seed
    assert admission["case_digest"] == digest


def test_robust_grape_documents_public_modulo_optimizer_seed() -> None:
    source = (BLUEPRINT_ROOT / "robust-leakage-grape" / "evaluate_109.py").read_text()
    assert "Public solver-local randomness only: base seed modulo 2**31 - 1" in source
    assert '"optimization_seed": int(seed % (2**31 - 1))' in source


@pytest.mark.parametrize(("slug", "problem_id"), CASE_IDENTITY_EVALUATORS)
def test_runtime_is_reported_but_not_part_of_functional_correctness(
    slug: str, problem_id: int
) -> None:
    source = (BLUEPRINT_ROOT / slug / f"evaluate_{problem_id}.py").read_text().lower()

    assert '"runtime below' not in source
    assert "timed execution within" not in source


@pytest.mark.parametrize(
    ("slug", "problem_id"),
    (
        ("mixed_sld_qfim", 101),
        ("qsp_phase_synthesis", 102),
        ("dynamic_branch_equivalence", 103),
        ("variance_aware_cutting", 104),
    ),
)
def test_candidate_main_exits_nonzero_when_functional_checks_fail(
    slug: str, problem_id: int
) -> None:
    source = (BLUEPRINT_ROOT / slug / f"evaluate_{problem_id}.py").read_text()
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main)

    assert main_source is not None
    exit_callable = "trusted_system_exit" if problem_id == 101 else "SystemExit"
    assert f"raise {exit_callable}(0 if" in main_source
    assert "else 1" in main_source
