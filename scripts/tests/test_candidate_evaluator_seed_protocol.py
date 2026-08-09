from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

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


def digest_for_seed(module, problem_id: int, seed: int) -> str:
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
    (("qsp_phase_synthesis", 102),)
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
    if problem_id == 109:
        pytest.importorskip(
            "scipy", reason="the robust-control evaluator imports its SciPy oracle"
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
    function_name = "main" if problem_id in {102, 105, 109, 110} else "evaluate"
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None

    marker_offset = function_source.index('"orbit_q_case_identity"')
    untrusted_offset = function_source.index(
        "importlib.import_module" if problem_id == 102 else "run_solution"
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
    assert "raise SystemExit(0 if" in main_source
    assert "else 1" in main_source
