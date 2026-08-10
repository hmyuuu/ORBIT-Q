from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_candidate_task as materializer  # noqa: E402
import run_harbor_candidate as runner  # noqa: E402
from reports.orbit_q_problem_discovery import reserve_registry  # noqa: E402


DISCOVERY = ROOT / "reports" / "orbit_q_problem_discovery"
RESERVES = (
    (
        "coherent-toric-recovery-portfolio",
        "r116_coherent_toric_recovery_portfolio--post_shortlist_reserve--"
        "robust_selection",
        116,
    ),
    (
        "replica-transfer-program-synthesis",
        "r117_replica_transfer_program_synthesis--post_shortlist_reserve--"
        "shared_contraction_dag",
        117,
    ),
    (
        "process-tensor-qec-policy",
        "r118_process_tensor_qec_policy--post_shortlist_reserve--robust_synthesis",
        118,
    ),
    (
        "robust-lightcone-cut-planner",
        "r119_robust_lightcone_cut_planner--post_shortlist_reserve--"
        "partition_allocation_synthesis",
        119,
    ),
    (
        "robust-flagged-css-gadget",
        "r120_flagged_css_gadget_synthesis--post_shortlist_reserve--"
        "robust_fault_tolerant_implementation",
        120,
    ),
    (
        "verifier-executed-amplitude-compiler",
        "r121_verifier_executed_batched_amplitude_compiler--post_shortlist_reserve--"
        "symbolic_contraction_program",
        121,
    ),
    (
        "nuisance-projected-fgs-portfolio",
        "r122_nuisance_projected_fgs_experiment_portfolio--post_shortlist_reserve--"
        "robust_integer_design",
        122,
    ),
)
PRIMARY_RESERVE = RESERVES[0]
ARTIFACT_CASES = tuple(
    (slug, relative)
    for slug, _, problem_id in RESERVES
    for relative in (
        "blueprint.json",
        "instruction.md",
        f"evaluate_{problem_id}.py",
        f"expert/solution_{problem_id}.py",
    )
)


def copy_registry_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "discovery"
    workspace.mkdir()
    for name in (
        "candidates.jsonl",
        "shortlist.json",
        "summary.json",
        reserve_registry.REGISTRY_FILENAME,
        "post_snapshot_reserve_registry.schema.json",
    ):
        shutil.copy2(DISCOVERY / name, workspace / name)
    for slug, _, _ in RESERVES:
        shutil.copytree(
            DISCOVERY / "blueprints" / slug,
            workspace / "blueprints" / slug,
        )
    return workspace


def read_registry(workspace: Path) -> dict:
    return json.loads((workspace / reserve_registry.REGISTRY_FILENAME).read_text())


def write_rehashed_registry(workspace: Path, document: dict) -> None:
    for entry in document["reserves"]:
        entry["entry_hash"] = reserve_registry.entry_hash(entry)
    document["registry_hash"] = reserve_registry.registry_hash(document)
    (workspace / reserve_registry.REGISTRY_FILENAME).write_text(
        json.dumps(document, indent=2) + "\n"
    )


def materialize_reserve(workspace: Path, tmp_path: Path, slug: str) -> Path:
    return materializer.materialize_candidate_task(
        workspace / "blueprints" / slug,
        tmp_path / "candidate-tasks",
        workspace,
    )


@pytest.mark.parametrize(("slug", "candidate_id", "problem_id"), RESERVES)
def test_registry_and_materialization_bind_exact_design_only_reserve(
    tmp_path: Path, slug: str, candidate_id: str, problem_id: int
) -> None:
    workspace = copy_registry_workspace(tmp_path)
    document = reserve_registry.load_verified_registry(workspace)
    entries = {entry["candidate_id"]: entry for entry in document["reserves"]}
    assert set(entries) == {reserve[1] for reserve in RESERVES}
    entry = entries[candidate_id]
    assert entry["slug"] == slug
    assert entry["problem_id"] == problem_id
    assert entry["lifecycle_status"] == "design_only"
    assert entry["audit_status"]["execution_authorized"] is False

    frozen_before = {
        name: (workspace / name).read_bytes() for name in reserve_registry.FROZEN_FILES
    }
    task_dir = materialize_reserve(workspace, tmp_path, slug)
    metadata = json.loads((task_dir / "candidate_metadata.json").read_text())
    binding = metadata["discovery_binding"]
    assert binding["contract"] == "post_snapshot_reserve_design_contract"
    assert binding["execution_status"] == "design_only"
    assert binding["registry_entry_hash"] == entry["entry_hash"]
    assert binding["blueprint_bundle_sha256"] == entry["blueprint_bundle_sha256"]
    assert binding["registry_payload_sha256"] == document["registry_hash"]
    evaluator_source = (
        workspace / "blueprints" / slug / f"evaluate_{problem_id}.py"
    ).read_text()
    requires_admission_marker = (
        materializer.EXPERT_ADMISSION_METRICS_KEY in evaluator_source
    )
    assert (
        task_dir / "tests" / materializer.EXPERT_ADMISSION_PROTOCOL_FILE
    ).is_file() is requires_admission_marker
    assert materializer.validate_materialized_task_contract(
        task_dir,
        workspace,
        require_expert_admission=requires_admission_marker,
    )["blueprint_path"] == str((workspace / "blueprints" / slug).resolve())
    assert frozen_before == {
        name: (workspace / name).read_bytes() for name in reserve_registry.FROZEN_FILES
    }


def test_registry_rejects_payload_hash_drift(tmp_path: Path) -> None:
    workspace = copy_registry_workspace(tmp_path)
    document = read_registry(workspace)
    document["reserves"][0]["audit_status"]["hardness_status"] = "claimed"
    (workspace / reserve_registry.REGISTRY_FILENAME).write_text(json.dumps(document))

    with pytest.raises(
        reserve_registry.ReserveRegistryError, match="registry_hash does not match"
    ):
        reserve_registry.load_verified_registry(workspace)


def test_materialized_binding_rejects_registry_file_byte_drift(
    tmp_path: Path,
) -> None:
    workspace = copy_registry_workspace(tmp_path)
    task_dir = materialize_reserve(workspace, tmp_path, PRIMARY_RESERVE[0])
    document = read_registry(workspace)
    # Canonical payload hashes still validate, but the tracked registry file
    # bytes no longer match the materialized provenance binding.
    (workspace / reserve_registry.REGISTRY_FILENAME).write_text(
        json.dumps(document, separators=(",", ":")) + "\n"
    )
    reserve_registry.load_verified_registry(workspace)

    with pytest.raises(materializer.CandidateTaskError, match="binding is stale"):
        materializer.validate_materialized_task_contract(
            task_dir, workspace, require_expert_admission=False
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("candidate_id", "second-post-snapshot-reserve", "duplicate reserve slug"),
        (
            "slug",
            "second-reserve-slug",
            "duplicate reserve candidate_id",
        ),
    ),
)
def test_registry_rejects_duplicate_ids_and_slugs_after_valid_rehash(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    workspace = copy_registry_workspace(tmp_path)
    document = read_registry(workspace)
    duplicate = copy.deepcopy(document["reserves"][0])
    duplicate[field] = replacement
    document["reserves"].append(duplicate)
    write_rehashed_registry(workspace, document)

    with pytest.raises(reserve_registry.ReserveRegistryError, match=message):
        reserve_registry.load_verified_registry(workspace)


def test_registry_rejects_duplicate_problem_id_after_valid_rehash(
    tmp_path: Path,
) -> None:
    workspace = copy_registry_workspace(tmp_path)
    document = read_registry(workspace)
    duplicate = copy.deepcopy(document["reserves"][0])
    duplicate["candidate_id"] = "distinct-post-snapshot-reserve"
    duplicate["slug"] = "distinct-reserve-slug"
    document["reserves"].append(duplicate)
    write_rehashed_registry(workspace, document)

    with pytest.raises(
        reserve_registry.ReserveRegistryError, match="duplicate reserve problem_id"
    ):
        reserve_registry.load_verified_registry(workspace)


def test_registry_rejects_blueprint_numeric_id_mismatch(tmp_path: Path) -> None:
    workspace = copy_registry_workspace(tmp_path)
    metadata_path = workspace / "blueprints" / PRIMARY_RESERVE[0] / "blueprint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["numeric_id"] += 1
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    document = read_registry(workspace)
    entry = document["reserves"][0]
    metadata_hash = reserve_registry.file_sha256(metadata_path)
    entry["artifacts"]["blueprint_metadata"]["sha256"] = metadata_hash
    entry["blueprint_bundle_sha256"] = reserve_registry.blueprint_bundle_hash(
        entry["artifacts"]
    )
    write_rehashed_registry(workspace, document)

    with pytest.raises(
        reserve_registry.ReserveRegistryError,
        match="identity differs from blueprint metadata",
    ):
        reserve_registry.load_verified_registry(workspace)


def test_reserve_119_keeps_runtime_separate_and_rejects_seed_override() -> None:
    source = (
        DISCOVERY / "blueprints" / "robust-lightcone-cut-planner" / "evaluate_119.py"
    ).read_text()
    assert '"timed call within 300 seconds"' not in source
    assert (
        "Runtime is reported separately and does not change functional correctness"
        in source
    )
    assert 'verifier_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")' in source
    assert "int(verifier_seed) != args.seed" in source
    assert (
        'raise ValueError("--seed cannot override the verifier-authorized seed")'
        in source
    )


@pytest.mark.parametrize(("slug", "relative"), ARTIFACT_CASES)
def test_registry_rejects_stale_blueprint_artifact_bytes(
    tmp_path: Path, slug: str, relative: str
) -> None:
    workspace = copy_registry_workspace(tmp_path)
    artifact = workspace / "blueprints" / slug / relative
    artifact.write_bytes(artifact.read_bytes() + b"\n# unregistered drift\n")

    with pytest.raises(reserve_registry.ReserveRegistryError, match="artifact drifted"):
        reserve_registry.load_verified_registry(workspace)


@pytest.mark.parametrize("name", reserve_registry.FROZEN_FILES)
def test_registry_rejects_frozen_snapshot_drift(tmp_path: Path, name: str) -> None:
    workspace = copy_registry_workspace(tmp_path)
    path = workspace / name
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(
        reserve_registry.ReserveRegistryError,
        match=f"frozen discovery snapshot drifted: {name}",
    ):
        reserve_registry.load_verified_registry(workspace)


def test_registry_and_materializer_enforce_blueprint_tree_containment(
    tmp_path: Path,
) -> None:
    workspace = copy_registry_workspace(tmp_path)
    document = read_registry(workspace)
    document["reserves"][0]["blueprint_path"] = "../outside-blueprint"
    write_rehashed_registry(workspace, document)
    with pytest.raises(
        reserve_registry.ReserveRegistryError, match="contained relative"
    ):
        reserve_registry.load_verified_registry(workspace)

    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    linked_workspace = copy_registry_workspace(linked_root)
    linked_blueprint = linked_workspace / "blueprints" / "linked-reserve"
    linked_blueprint.symlink_to(linked_workspace / "blueprints" / PRIMARY_RESERVE[0])
    linked_document = read_registry(linked_workspace)
    linked_document["reserves"][0]["blueprint_path"] = "blueprints/linked-reserve"
    write_rehashed_registry(linked_workspace, linked_document)
    with pytest.raises(reserve_registry.ReserveRegistryError, match="symbolic link"):
        reserve_registry.load_verified_registry(linked_workspace)

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_workspace = copy_registry_workspace(clean_root)
    outside = tmp_path / "outside-copy"
    shutil.copytree(clean_workspace / "blueprints" / PRIMARY_RESERVE[0], outside)
    with pytest.raises(
        materializer.CandidateTaskError, match="exact tracked blueprint path"
    ):
        materializer.materialize_candidate_task(
            outside, tmp_path / "outside-stage", clean_workspace
        )


@pytest.mark.parametrize(("slug", "candidate_id", "problem_id"), RESERVES)
@pytest.mark.parametrize("expert_only", (False, True))
def test_safe_runner_refuses_design_only_reserve_before_any_execution_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expert_only: bool,
    slug: str,
    candidate_id: str,
    problem_id: int,
) -> None:
    workspace = copy_registry_workspace(tmp_path)
    task_dir = materialize_reserve(workspace, tmp_path, slug)
    assert (
        json.loads((task_dir / "candidate_metadata.json").read_text())["candidate_id"]
        == candidate_id
    )

    def forbidden_authorization(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("design-only reserve reached model authorization")

    monkeypatch.setattr(runner, "authorized_model_execution", forbidden_authorization)
    argv = [
        "--task-dir",
        str(task_dir),
        "--workspace",
        str(workspace),
        "--candidate-seed",
        f"{problem_id}2026",
        "--jobs-dir",
        str(tmp_path / "candidate-jobs"),
    ]
    if expert_only:
        argv.extend(
            [
                "--expert-only",
                "--container-image-digest",
                "sha256:" + "e" * 64,
            ]
        )
    else:
        argv.extend(["--manifest", str(tmp_path / "never-read-manifest.json")])

    with pytest.raises(materializer.CandidateTaskError, match="design_only"):
        runner.build_harbor_command(runner.parse_args(argv))
