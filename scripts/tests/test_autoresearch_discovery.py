"""Tests for the deterministic offline autoresearch discovery stages."""

from __future__ import annotations

import hashlib
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reports.orbit_q_problem_discovery.autoresearch_discovery import (
    ArtifactRef,
    ArtifactStore,
    DiscoveryError,
    canonical_json_bytes,
    cli,
    freeze_search_plan,
    ingest_raw_manifests,
    run_discovery,
)


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_ROOT = ROOT / "reports" / "orbit_q_problem_discovery"
SCHEMAS = (
    "autoresearch_query_plan.schema.json",
    "autoresearch_raw_hits.schema.json",
    "autoresearch_problem_source_record.schema.json",
    "autoresearch_artifact.schema.json",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plan() -> dict:
    return {
        "schema_version": 1,
        "plan_id": "quantum-source-search-v1",
        "purpose": "Find source-grounded quantum benchmark problem leads.",
        "search_execution": {
            "pipeline_mode": "planning_only_no_network",
            "intended_result_custody": "operator_provided_raw_manifest",
        },
        "queries": [
            {
                "query_id": "q-qec",
                "query_text": "tensor network decoding coherent quantum noise",
                "provider_hint": "openalex",
                "requested_limit": 100,
                "source_types": ["paper"],
                "facets": {
                    "quantum_domains": ["qec"],
                    "situations": ["noise"],
                    "evidence_roles": ["primary-method"],
                },
            },
            {
                "query_id": "q-dynamics",
                "query_text": "differentiable variational open quantum dynamics",
                "provider_hint": "crossref",
                "requested_limit": 100,
                "source_types": ["paper"],
                "facets": {
                    "quantum_domains": ["variational"],
                    "situations": ["dynamic"],
                    "evidence_roles": ["primary-method"],
                },
            },
        ],
        "deduplication": {"near_duplicate_threshold_percent": 60},
        "coverage": {
            "sample_size": 2,
            "targets": {
                "quantum_domain": ["qec", "variational"],
                "situation": ["dynamic", "noise"],
                "source_type": ["paper"],
                "license_status": ["known"],
                "search_execution_claim": [
                    "not_claimed",
                    "operator_attested_external_execution",
                ],
            },
        },
    }


def _source(
    *,
    title: str,
    doi: str,
    url_suffix: str,
    authors: list[str] | None = None,
) -> dict:
    return {
        "title": title,
        "url": f"https://example.org/{url_suffix}?utm_source=test",
        "source_type": "paper",
        "authors": authors or ["Ada Researcher", "Bo Scientist"],
        "publisher": "Example Quantum Society",
        "identifiers": {"doi": doi},
    }


def _version(label: str, *, content_seed: str | None) -> dict:
    return {
        "version_type": "version-of-record",
        "version_label": label,
        "version_locator": f"https://example.org/version/{label}",
        "observed_at": "2026-08-10T00:00:00Z",
        "content_sha256": (
            hashlib.sha256(content_seed.encode()).hexdigest() if content_seed else None
        ),
    }


def _license(*, known: bool) -> dict:
    if known:
        return {
            "status": "known",
            "spdx_id": "CC-BY-4.0",
            "evidence_url": "https://example.org/license/cc-by-4.0",
            "redistribution_allowed": True,
        }
    return {
        "status": "unknown",
        "spdx_id": None,
        "evidence_url": None,
        "redistribution_allowed": None,
    }


def _problem(
    *,
    domain: str,
    situation: str,
    archetype: str,
    objective: str,
    keywords: list[str],
) -> dict:
    return {
        "quantum_domain": domain,
        "situation": situation,
        "task_archetype": archetype,
        "objective": objective,
        "tensorcircuit_path": "Use TensorCircuit gates and contractions for the core computation.",
        "evidence_role": "primary-method",
        "keywords": keywords,
    }


def _manifest_a(plan_sha256: str) -> dict:
    common_problem = _problem(
        domain="qec",
        situation="noise",
        archetype="decoder",
        objective="Decode coherent quantum noise with a tensor network decoder.",
        keywords=["coherent-noise", "decoder", "tensor-network"],
    )
    return {
        "schema_version": 1,
        "manifest_id": "manual-openalex-page-1",
        "search_plan_artifact_sha256": plan_sha256,
        "collection": {
            "input_custody": "operator_provided",
            "collection_method": "manual_curation",
            "provider_name": "openalex",
            "provider_version": "metadata snapshot 2026-08",
            "adapter_version": "manual-export/1",
            "observed_at": "2026-08-10T00:00:00Z",
            "search_execution_claim": "not_claimed",
        },
        "hits": [
            {
                "hit_id": "hit-alpha-a",
                "query_id": "q-qec",
                "rank": 1,
                "source": _source(
                    title="Tensor network decoding under coherent quantum noise",
                    doi="10.1000/alpha",
                    url_suffix="alpha",
                ),
                "version": _version("alpha-v1", content_seed="alpha-content"),
                "license": _license(known=True),
                "problem": common_problem,
            },
            {
                "hit_id": "hit-alpha-b",
                "query_id": "q-qec",
                "rank": 2,
                "source": _source(
                    title="Tensor network decoding under coherent quantum noise",
                    doi="https://doi.org/10.1000/ALPHA",
                    url_suffix="alpha/",
                ),
                "version": _version("alpha-v1", content_seed="alpha-content"),
                "license": _license(known=True),
                "problem": common_problem,
            },
        ],
    }


def _manifest_b(plan_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "manifest_id": "external-crossref-page-1",
        "search_plan_artifact_sha256": plan_sha256,
        "collection": {
            "input_custody": "operator_provided",
            "collection_method": "external_adapter_export",
            "provider_name": "crossref",
            "provider_version": "rest-api-2026-08",
            "adapter_version": "operator-adapter/3",
            "observed_at": "2026-08-10T00:05:00Z",
            "search_execution_claim": "operator_attested_external_execution",
            "execution_attestation": {
                "executed_at": "2026-08-10T00:04:00Z",
                "executor_id": "operator-adapter",
                "raw_response_sha256": hashlib.sha256(b"raw-response").hexdigest(),
            },
        },
        "hits": [
            {
                "hit_id": "hit-beta",
                "query_id": "q-qec",
                "rank": 3,
                "source": _source(
                    title="Tensor network decoder for coherent quantum noise",
                    doi="10.1000/beta",
                    url_suffix="beta",
                ),
                "version": _version("beta-v2", content_seed=None),
                "license": _license(known=False),
                "problem": _problem(
                    domain="qec",
                    situation="noise",
                    archetype="decoder",
                    objective="Decode coherent quantum noise with tensor network decoders.",
                    keywords=["coherent-noise", "decoder", "tensor-network"],
                ),
            },
            {
                "hit_id": "hit-gamma",
                "query_id": "q-dynamics",
                "rank": 1,
                "source": _source(
                    title="Differentiable variational simulation of open quantum dynamics",
                    doi="10.1000/gamma",
                    url_suffix="gamma",
                ),
                "version": _version("gamma-v1", content_seed="gamma-content"),
                "license": _license(known=True),
                "problem": _problem(
                    domain="variational",
                    situation="dynamic",
                    archetype="simulation",
                    objective="Simulate differentiable open-system dynamics with variational circuits.",
                    keywords=["differentiable", "open-system", "variational"],
                ),
            },
        ],
    }


def _workspace(
    tmp_path: Path, name: str = "workspace"
) -> tuple[Path, ArtifactStore, Path]:
    workspace = tmp_path / name
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "tasks").mkdir()
    store = ArtifactStore(workspace, ".artifacts/discovery")
    plan_path = workspace / "inputs" / "plan.json"
    _write_json(plan_path, _plan())
    return workspace, store, plan_path


def _fixture_inputs(
    tmp_path: Path, name: str = "workspace"
) -> tuple[Path, ArtifactStore, Path, Path, Path, ArtifactRef]:
    workspace, store, plan_path = _workspace(tmp_path, name)
    plan_ref = freeze_search_plan(store, plan_path)
    manifest_a = workspace / "inputs" / "raw-a.json"
    manifest_b = workspace / "inputs" / "raw-b.json"
    _write_json(manifest_a, _manifest_a(plan_ref.sha256))
    _write_json(manifest_b, _manifest_b(plan_ref.sha256))
    return workspace, store, plan_path, manifest_a, manifest_b, plan_ref


def _artifact_files(store: ArtifactStore) -> dict[str, bytes]:
    return {
        str(path.relative_to(store.root)): path.read_bytes()
        for path in sorted(store.root.rglob("*.json"))
    }


def test_end_to_end_is_offline_content_addressed_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, store, plan_path, manifest_a, manifest_b, _ = _fixture_inputs(tmp_path)

    def network_forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("discovery pipeline attempted network access")

    monkeypatch.setattr(socket, "socket", network_forbidden)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    first = run_discovery(store, plan_path, [manifest_a, manifest_b])
    first_files = _artifact_files(store)
    mtimes = {
        path: (store.root / path).stat().st_mtime_ns for path in sorted(first_files)
    }

    second = run_discovery(store, plan_path, [manifest_b, manifest_a])
    assert first == second
    assert _artifact_files(store) == first_files
    assert {
        path: (store.root / path).stat().st_mtime_ns for path in sorted(first_files)
    } == mtimes

    for relative_path, content in first_files.items():
        digest = Path(relative_path).stem
        assert hashlib.sha256(content).hexdigest() == digest
        assert content.endswith(b"\n")
        assert content == canonical_json_bytes(json.loads(content))

    other_workspace, other_store, other_plan_path = _workspace(tmp_path, "other")
    other_plan_ref = freeze_search_plan(other_store, other_plan_path)
    other_a = other_workspace / "inputs" / "raw-a.json"
    other_b = other_workspace / "inputs" / "raw-b.json"
    _write_json(other_a, _manifest_a(other_plan_ref.sha256))
    _write_json(other_b, _manifest_b(other_plan_ref.sha256))
    other = run_discovery(other_store, other_plan_path, [other_b, other_a])
    assert first == other
    assert first_files == _artifact_files(other_store)


def test_dedupe_and_coverage_preserve_records_and_provenance(tmp_path: Path) -> None:
    _, store, plan_path, manifest_a, manifest_b, _ = _fixture_inputs(tmp_path)
    refs = run_discovery(store, plan_path, [manifest_a, manifest_b])
    normalize = store.read(refs["normalize"])
    dedupe = store.read(refs["dedupe"])
    sample = store.read(refs["coverage_sample"])

    assert normalize["record_count"] == 4
    assert sorted(
        len(group["member_record_ids"]) for group in dedupe["exact_groups"]
    ) == [
        1,
        1,
        2,
    ]
    assert sorted(
        len(group["member_record_ids"]) for group in dedupe["near_groups"]
    ) == [
        1,
        3,
    ]
    assert dedupe["near_duplicate_edges"]
    assert sample["selected_count"] == 2
    assert not sample["unmet_targets"]
    selected = [item["record"] for item in sample["selected"]]
    assert {record["problem"]["quantum_domain"] for record in selected} == {
        "qec",
        "variational",
    }
    assert {record["provenance"]["search_execution_claim"] for record in selected} == {
        "not_claimed",
        "operator_attested_external_execution",
    }
    for record in selected:
        assert set(record) == {
            "record_id",
            "source",
            "version",
            "license",
            "provenance",
            "problem",
        }
        assert record["source"]["identity_keys"]
        assert record["version"]["version_locator"].startswith("https://")
        assert record["provenance"]["pipeline_executed_web_search"] is False
        assert (
            record["provenance"]["execution_evidence_verification"]
            == "not_performed_operator_attestation_only"
        )


def test_plan_and_raw_artifacts_distinguish_planning_from_execution_claims(
    tmp_path: Path,
) -> None:
    _, store, plan_path, manifest_a, manifest_b, plan_ref = _fixture_inputs(tmp_path)
    ingest_ref = ingest_raw_manifests(store, plan_ref, [manifest_a, manifest_b])
    plan_artifact = store.read(plan_ref)
    ingest = store.read(ingest_ref)
    raw_artifacts = [
        store.read(ArtifactRef(value["stage"], value["sha256"]))
        for value in ingest["raw_manifests"]
    ]

    assert plan_artifact["execution_status"] == "query_plan_only_not_executed"
    assert plan_artifact["pipeline_executed_web_search"] is False
    assert ingest["input_custody"] == "operator_provided_raw_manifests"
    assert ingest["pipeline_executed_web_search"] is False
    assert {
        item["manifest"]["collection"]["search_execution_claim"]
        for item in raw_artifacts
    } == {
        "not_claimed",
        "operator_attested_external_execution",
    }
    assert all(item["pipeline_executed_web_search"] is False for item in raw_artifacts)
    assert all(
        item["execution_evidence_verification"]
        == "not_performed_operator_attestation_only"
        for item in raw_artifacts
    )


def test_manifest_must_bind_plan_and_attested_execution_needs_evidence(
    tmp_path: Path,
) -> None:
    workspace, store, _, manifest_a, _, plan_ref = _fixture_inputs(tmp_path)
    wrong = _manifest_a("f" * 64)
    wrong_path = workspace / "inputs" / "wrong-plan.json"
    _write_json(wrong_path, wrong)
    with pytest.raises(DiscoveryError, match="not bound"):
        ingest_raw_manifests(store, plan_ref, [wrong_path])

    missing = _manifest_b(plan_ref.sha256)
    missing["collection"].pop("execution_attestation")
    missing_path = workspace / "inputs" / "missing-attestation.json"
    _write_json(missing_path, missing)
    with pytest.raises(DiscoveryError, match="must be an object"):
        ingest_raw_manifests(store, plan_ref, [missing_path])

    duplicate_path = workspace / "inputs" / "duplicate.json"
    _write_json(duplicate_path, _manifest_a(plan_ref.sha256))
    with pytest.raises(DiscoveryError, match="duplicate manifest_id"):
        ingest_raw_manifests(store, plan_ref, [manifest_a, duplicate_path])


def test_paths_are_contained_and_canonical_tasks_are_forbidden(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "tasks").mkdir()
    with pytest.raises(DiscoveryError, match="canonical tasks tree"):
        ArtifactStore(workspace, "tasks/discovery-output")

    store = ArtifactStore(workspace, ".artifacts/discovery")
    outside = tmp_path / "outside-plan.json"
    _write_json(outside, _plan())
    with pytest.raises(DiscoveryError, match="inside workspace"):
        freeze_search_plan(store, outside)

    task_plan = workspace / "tasks" / "plan.json"
    _write_json(task_plan, _plan())
    with pytest.raises(DiscoveryError, match="canonical tasks tree"):
        freeze_search_plan(store, task_plan)

    symlink = workspace / "escaped-plan.json"
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(DiscoveryError, match="inside workspace"):
        freeze_search_plan(store, symlink)


def test_artifact_tampering_and_secret_like_inputs_fail_closed(tmp_path: Path) -> None:
    workspace, store, plan_path = _workspace(tmp_path)
    plan_ref = freeze_search_plan(store, plan_path)
    plan_artifact_path = store.artifact_path(plan_ref)
    plan_artifact_path.write_bytes(plan_artifact_path.read_bytes() + b" ")
    with pytest.raises(DiscoveryError, match="digest mismatch"):
        store.read(plan_ref)

    secret_plan = _plan()
    secret_plan["queries"][0]["query_text"] = "quantum sk-abcdefghijklmnopqrstuv"
    secret_path = workspace / "inputs" / "secret-plan.json"
    _write_json(secret_path, secret_plan)
    with pytest.raises(DiscoveryError, match="secret-like"):
        freeze_search_plan(store, secret_path)


def test_atomic_content_address_write_is_safe_under_concurrency(tmp_path: Path) -> None:
    _, store, plan_path = _workspace(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = list(
            executor.map(lambda _: freeze_search_plan(store, plan_path), range(32))
        )
    assert len(set(refs)) == 1
    content = store.artifact_path(refs[0]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == refs[0].sha256


def test_cli_plan_and_run_emit_machine_readable_refs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, store, plan_path, manifest_a, manifest_b, plan_ref = _fixture_inputs(
        tmp_path
    )
    result = cli(
        [
            "plan",
            "--workspace-root",
            str(workspace),
            "--store-root",
            str(store.root),
            "--plan",
            str(plan_path),
        ]
    )
    assert result == 0
    plan_output = json.loads(capsys.readouterr().out)
    assert plan_output["search_plan"]["sha256"] == plan_ref.sha256

    result = cli(
        [
            "run",
            "--workspace-root",
            str(workspace),
            "--store-root",
            str(store.root),
            "--plan",
            str(plan_path),
            "--raw-manifest",
            str(manifest_b),
            "--raw-manifest",
            str(manifest_a),
        ]
    )
    assert result == 0
    run_output = json.loads(capsys.readouterr().out)
    assert run_output["pipeline_executed_web_search"] is False
    assert set(run_output["artifacts"]) == {
        "search_plan",
        "raw_ingest",
        "normalize",
        "dedupe",
        "coverage_sample",
    }
    assert all(
        not value["relative_path"].startswith("tasks/")
        for value in run_output["artifacts"].values()
    )


def test_published_schemas_are_strict_json_and_describe_pipeline_contract() -> None:
    for filename in SCHEMAS:
        content = (DISCOVERY_ROOT / filename).read_bytes()
        schema = json.loads(content)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"].startswith("https://orbit-q.local/schemas/")
        assert schema["type"] == "object"
        assert (
            content == json.dumps(schema, ensure_ascii=False, indent=2).encode() + b"\n"
        )

    query_schema = json.loads(
        (DISCOVERY_ROOT / "autoresearch_query_plan.schema.json").read_text()
    )
    raw_schema = json.loads(
        (DISCOVERY_ROOT / "autoresearch_raw_hits.schema.json").read_text()
    )
    record_schema = json.loads(
        (DISCOVERY_ROOT / "autoresearch_problem_source_record.schema.json").read_text()
    )
    assert (
        query_schema["properties"]["search_execution"]["properties"]["pipeline_mode"][
            "const"
        ]
        == "planning_only_no_network"
    )
    assert (
        raw_schema["$defs"]["collection"]["properties"]["input_custody"]["const"]
        == "operator_provided"
    )
    assert {
        "source",
        "version",
        "license",
        "provenance",
        "problem",
    }.issubset(record_schema["required"])
