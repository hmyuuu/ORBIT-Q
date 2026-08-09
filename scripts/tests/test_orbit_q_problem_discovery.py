"""Regression tests for the future-problem discovery and review gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reports.orbit_q_problem_discovery.pipeline import (
    CONCEPT_REVIEW_ROLES,
    FAILURE_AUDIT_ROLES,
    PILOT_REVIEW_ROLES,
    audit_model_trial,
    authorized_model_execution,
    authorize_model_test,
    build_workspace,
    generate_candidates,
    model_hardness_status,
    record_model_trial,
    record_prototype,
    record_review,
    reserve_model_trial_seed,
    select_shortlist,
    validate_catalog,
    _harbor_legacy_task_checksum,
    _harbor_task_digest,
    _path_sha256,
)
from scripts.materialize_candidate_task import (
    materialize_candidate_task,
    trusted_verifier_harness_sha256,
)
from scripts.run_problem_discovery import _nonblank_text


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY = ROOT / "reports" / "orbit_q_problem_discovery"


def _catalog() -> dict:
    return json.loads((DISCOVERY / "catalog.json").read_text(encoding="utf-8"))


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "discovery"
    workspace.mkdir()
    shutil.copy2(DISCOVERY / "catalog.json", workspace / "catalog.json")
    build_workspace(workspace)
    return workspace


def _artifact_root(workspace: Path) -> Path:
    root = workspace.parent / ".artifacts" / "problem-discovery"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_artifact(path: Path, content: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _payload_sha256(value: dict, hash_field: str) -> str:
    payload = {key: item for key, item in value.items() if key != hash_field}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _prototype_bundle(workspace: Path, candidate_id: str) -> Path:
    state = json.loads((workspace / "review_state.json").read_text())
    candidate_hash = state["candidates"][candidate_id]["candidate_hash"]
    evidence_root = _artifact_root(workspace) / "prototype"
    expert_source = "".join(f"value_{index} = {index}\n" for index in range(140))
    blueprint = workspace / "blueprints" / "fixture-candidate"
    (blueprint / "expert").mkdir(parents=True, exist_ok=True)
    (blueprint / "instruction.md").write_text("Solve the fixture challenge.\n")
    (blueprint / "blueprint.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "title": "Fixture candidate",
                "problem_id": 901,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (blueprint / "expert" / "solution_901.py").write_text(expert_source)
    (blueprint / "evaluate_901.py").write_text(
        'EVALUATOR = "orbit_q_expert_admission_metrics"\n'
    )
    candidate_tasks = evidence_root / "candidate-tasks"
    if candidate_tasks.exists():
        shutil.rmtree(candidate_tasks)
    task_bundle = materialize_candidate_task(
        blueprint,
        candidate_tasks,
        workspace,
    )
    prompt_path = evidence_root / "prompt.md"
    prompt_path.write_text("TensorCircuit prompt\n")
    artifacts = {
        "expert_baseline": {
            "path": str(task_bundle / "solution" / "solution_901.py"),
            "sha256": hashlib.sha256(expert_source.encode()).hexdigest(),
            "author": "expert-author",
        },
        "independent_oracle": {
            **_write_artifact(evidence_root / "oracle.py", "ORACLE = True\n"),
            "author": "oracle-author",
        },
        "evaluator": {
            "path": str(task_bundle / "tests" / "evaluate_901.py"),
            "sha256": hashlib.sha256(
                'EVALUATOR = "orbit_q_expert_admission_metrics"\n'.encode()
            ).hexdigest(),
            "author": "evaluator-author",
        },
        "instruction": {
            "path": str(task_bundle / "instruction.md"),
            "sha256": hashlib.sha256(
                (task_bundle / "instruction.md").read_bytes()
            ).hexdigest(),
            "author": "specification-author",
        },
        "candidate_metadata": {
            "path": str(task_bundle / "candidate_metadata.json"),
            "sha256": hashlib.sha256(
                (task_bundle / "candidate_metadata.json").read_bytes()
            ).hexdigest(),
            "author": "metadata-author",
        },
        "task_bundle": {
            "path": str(task_bundle),
            "sha256": _path_sha256(task_bundle),
            "author": "task-author",
        },
        "framework_prompt": {
            "path": str(prompt_path),
            "sha256": hashlib.sha256(b"TensorCircuit prompt\n").hexdigest(),
            "author": "prompt-author",
        },
    }
    image_digest = "sha256:" + "e" * 64
    verifier_harness_sha256 = trusted_verifier_harness_sha256()
    source_commit = "0123456789abcdef"
    canary = _write_artifact(
        evidence_root / "canary.json",
        json.dumps(
            {
                "passed": True,
                "container_image_digest": image_digest,
                "source_commit": source_commit,
            },
            indent=2,
        )
        + "\n",
    )
    verifier_runs = []
    for index in range(20):
        runtime = 40.0 + index / 10
        peak_memory = 2048.0 + index
        log = _write_artifact(
            evidence_root / f"verifier-{index}.json",
            json.dumps(
                {
                    "id": f"verifier-{index}",
                    "finished_at": "2026-08-09T00:00:00Z",
                    "verifier_result": {
                        "rewards": {
                            "reward": 1.0,
                            "functional_score": 1.0,
                            "static_policy_score": 1.0,
                            "llm_audit_score": 1.0,
                            "runtime_sec": runtime,
                        }
                    },
                    "orbit_q_attestation": {
                        "attestation_type": "operator_attested",
                        "attestor_id": "prototype-runner",
                        "container_image_digest": image_digest,
                        "peak_memory_mb": peak_memory,
                    },
                },
                indent=2,
            )
            + "\n",
        )
        verifier_runs.append(
            {
                "job_id": f"verifier-{index}",
                "passed": True,
                "runtime_sec": runtime,
                "peak_memory_mb": peak_memory,
                "container_image_digest": image_digest,
                "log": log,
            }
        )
    docker_image = "challenge-benchmark-quantum-tensorcircuit:py311"
    harbor_version = "0.20.0"
    harbor_task_digest = _harbor_task_digest(task_bundle)
    task_checksum = _harbor_legacy_task_checksum(task_bundle)
    expert_prequalified_cases = []
    for seed in range(105):
        case_digest = hashlib.sha256(
            f"{candidate_id}:{seed}:public-case".encode()
        ).hexdigest()
        admission = {
            "status": "admitted",
            "minimum_margin": 0.25,
            "margins": [
                {
                    "metric": "normalized_accuracy_margin",
                    "direction": "at_least",
                    "observed": 1.25,
                    "threshold": 1.0,
                    "absolute_margin": 0.25,
                    "passed": True,
                }
            ],
        }
        threshold_policy = [
            {
                "metric": margin["metric"],
                "direction": margin["direction"],
                "threshold": float(margin["threshold"]),
            }
            for margin in admission["margins"]
        ]
        threshold_policy_sha256 = hashlib.sha256(
            json.dumps(
                threshold_policy,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        admission_record_payload = {
            "schema_version": 1,
            "producer": "orbit_q_score_submission_from_candidate_evaluator",
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash,
            "protocol_seed": seed,
            "case_digest": case_digest,
            "expert_baseline_sha256": artifacts["expert_baseline"]["sha256"],
            "evaluator_sha256": artifacts["evaluator"]["sha256"],
            "task_bundle_sha256": artifacts["task_bundle"]["sha256"],
            "framework_prompt_sha256": artifacts["framework_prompt"]["sha256"],
            "verifier_harness_sha256": verifier_harness_sha256,
            "container_image_digest": image_digest,
            "expert_passed": True,
            "threshold_policy_sha256": threshold_policy_sha256,
            "admission": admission,
        }
        admission_record = _write_artifact(
            evidence_root / f"expert-admission-{seed}.json",
            json.dumps(admission_record_payload, indent=2) + "\n",
        )
        verifier_env = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "REQUIRED_QUANTUM_FRAMEWORK": "tensorcircuit",
            "ORBIT_Q_NON_HARDNESS_RUN": "expert_prequalification",
            "ORBIT_Q_EXPERT_PREQUALIFICATION_MODE": "expert_only_oracle",
            "ORBIT_Q_CANDIDATE_ID": candidate_id,
            "ORBIT_Q_CANDIDATE_HASH": candidate_hash,
            "ORBIT_Q_CANDIDATE_SEED": str(seed),
            "ORBIT_Q_CONTAINER_IMAGE_DIGEST": image_digest,
            "ORBIT_Q_EXPERT_BASELINE_SHA256": artifacts["expert_baseline"]["sha256"],
            "ORBIT_Q_EVALUATOR_SHA256": artifacts["evaluator"]["sha256"],
            "ORBIT_Q_TASK_BUNDLE_SHA256": artifacts["task_bundle"]["sha256"],
            "ORBIT_Q_FRAMEWORK_PROMPT_SHA256": artifacts["framework_prompt"]["sha256"],
            "ORBIT_Q_VERIFIER_HARNESS_SHA256": verifier_harness_sha256,
        }
        expert_config = {
            "task": {"path": str(task_bundle)},
            "timeout_multiplier": 1.0,
            "install_only": False,
            "skills": [],
            "agent": {
                "name": "oracle",
                "import_path": None,
                "model_name": None,
                "kwargs": {},
                "skills": [],
                "mcp_servers": [],
                "extra_allowed_hosts": [],
            },
            "environment": {
                "import_path": "adapters.framework_docker:FrameworkDockerEnvironment",
                "cpu_enforcement_policy": "limit",
                "memory_enforcement_policy": "limit",
                "override_cpus": 8,
                "override_memory_mb": 8192,
                "kwargs": {
                    "framework": "tensorcircuit",
                    "docker_image": f"{docker_image}@{image_digest}",
                },
            },
            "verifier": {
                "import_path": "adapters.codex_para_verifier:CodexParaVerifier",
                "kwargs": {"audit_model": "gpt-5", "expert_only": True},
                "env": verifier_env,
                "disable": False,
            },
        }
        sidecar_config = {
            "task": expert_config["task"],
            "environment": expert_config["environment"],
            "verifier": expert_config["verifier"],
        }
        trial_lock = {
            "schema_version": 1,
            "task": {
                "name": task_bundle.name,
                "type": "local",
                "digest": f"sha256:{harbor_task_digest}",
                "path": str(task_bundle),
            },
            "install_only": False,
            "skills": [],
            "timeout_multiplier": 1.0,
            "extra_instructions": [
                {
                    "path": str(prompt_path),
                    "digest": f"sha256:{artifacts['framework_prompt']['sha256']}",
                }
            ],
            "agent": expert_config["agent"],
            "environment": expert_config["environment"],
            "verifier": expert_config["verifier"],
        }
        job_lock = {
            "schema_version": 2,
            "harbor": {"version": harbor_version, "is_editable": False},
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "trials": [trial_lock],
        }
        raw_evidence = {
            "harbor_config": _write_artifact(
                evidence_root / f"expert-config-{seed}.json",
                json.dumps(sidecar_config, indent=2) + "\n",
            ),
            "harbor_lock": _write_artifact(
                evidence_root / f"expert-trial-lock-{seed}.json",
                json.dumps(trial_lock, indent=2) + "\n",
            ),
            "harbor_job_lock": _write_artifact(
                evidence_root / f"expert-job-lock-{seed}.json",
                json.dumps(job_lock, indent=2) + "\n",
            ),
            "functional_output": _write_artifact(
                evidence_root / f"expert-functional-{seed}.txt",
                json.dumps(
                    {
                        "orbit_q_case_identity": {
                            "protocol_seed": seed,
                            "case_digest": case_digest,
                        }
                    }
                )
                + "\nOverall: PASS\n",
            ),
            "admission_record": admission_record,
        }
        job_id = f"expert-job-{seed}"
        raw_result = {
            "id": job_id,
            "finished_at": "2026-08-09T00:00:00Z",
            "task_checksum": task_checksum,
            "config": expert_config,
            "agent_info": {
                "name": "oracle",
                "version": "1.0.0",
                "model_info": None,
            },
            "agent_result": {
                "n_input_tokens": None,
                "n_cache_tokens": None,
                "n_output_tokens": None,
                "cost_usd": None,
                "rollout_details": None,
                "metadata": None,
            },
            "verifier_result": {
                "rewards": {
                    "reward": 1.0,
                    "functional_score": 1.0,
                    "static_policy_score": 1.0,
                    "llm_audit_score": 1.0,
                    "runtime_sec": 30.0 + seed / 100,
                    **{
                        f"expert_admission_sha256_word_{index}": int(
                            admission_record["sha256"][offset : offset + 8], 16
                        )
                        for index, offset in enumerate(range(0, 64, 8))
                    },
                },
            },
            "exception_info": None,
        }
        raw_evidence["raw_job"] = _write_artifact(
            evidence_root / f"expert-raw-job-{seed}.json",
            json.dumps(raw_result, indent=2) + "\n",
        )
        ordered_hashes = {
            name: raw_evidence[name]["sha256"]
            for name in (
                "raw_job",
                "harbor_config",
                "harbor_lock",
                "harbor_job_lock",
                "functional_output",
                "admission_record",
            )
        }
        run_evidence_sha256 = hashlib.sha256(
            json.dumps(
                {"job_id": job_id, "artifact_sha256s": ordered_hashes},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        expert_prequalified_cases.append(
            {
                "job_id": job_id,
                "expert_passed": True,
                "runtime_sec": 30.0 + seed / 100,
                "protocol_seed": seed,
                "case_digest": case_digest,
                "admission": admission,
                "threshold_policy_sha256": threshold_policy_sha256,
                "run_evidence_sha256": run_evidence_sha256,
                "raw_evidence": raw_evidence,
            }
        )
    oracle_runs = [
        {
            "runtime_sec": 20.0 + index,
            "log": _write_artifact(
                evidence_root / f"oracle-{index}.json",
                json.dumps(
                    {
                        "passed": True,
                        "runtime_sec": 20.0 + index,
                        "oracle_method": "independent-density-or-covariance-oracle",
                    },
                    indent=2,
                )
                + "\n",
            ),
        }
        for index in range(3)
    ]
    alternatives = [
        {
            "accepted": True,
            "log": _write_artifact(
                evidence_root / f"alternative-{index}.json",
                json.dumps(
                    {
                        "accepted": True,
                        "implementation_style": f"valid-style-{index}",
                    },
                    indent=2,
                )
                + "\n",
            ),
        }
        for index in range(2)
    ]
    mutations = [
        {
            "name": f"mutation-{index}",
            "rejected": True,
            "log": _write_artifact(
                evidence_root / f"mutation-{index}.json",
                json.dumps(
                    {
                        "rejected": True,
                        "mutation_type": f"mutation-class-{index}",
                    },
                    indent=2,
                )
                + "\n",
            ),
        }
        for index in range(6)
    ]
    bundle = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "artifacts": artifacts,
        "environment": {
            "container_image_digest": image_digest,
            "docker_image": docker_image,
            "harbor_version": harbor_version,
            "harbor_task_digest": harbor_task_digest,
            "task_checksum": task_checksum,
            "source_commit": source_commit,
            "observed_cpu_count": 8,
            "observed_memory_mb": 8192,
        },
        "public_api_canary": {"passed": True, "log": canary},
        "verifier_runs": verifier_runs,
        "expert_prequalified_cases": expert_prequalified_cases,
        "independent_oracle_runs": oracle_runs,
        "gold_effective_lines": 140,
        "valid_alternatives": alternatives,
        "mutation_tests": mutations,
    }
    bundle_path = evidence_root / "evidence.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle_path


def _protocol_config(workspace: Path) -> Path:
    path = _artifact_root(workspace) / "protocol.json"
    state = json.loads((workspace / "review_state.json").read_text())
    prototype = next(
        (
            row.get("prototype", {})
            for row in state["candidates"].values()
            if row.get("prototype")
        ),
        {},
    )
    config = {
        "solver_agent": "codex-para",
        "model": "gpt-5.6-sol",
        "provider_snapshot": "provider-build-2026-08-09",
        "reasoning_effort": "max",
        "token_budget": 5000,
        "token_budget_metric": "input_plus_output_tokens",
        "token_budget_enforcement": "post_run_admission_only",
        "wall_time_sec": 1800,
        "solver_agent_version": "0.146.0",
        "harbor_version": "0.20.0",
        "audit_model": "gpt-5",
        "build_identity_policy": "require_runtime_observation_for_hardness",
        "tools_policy": "harbor_default_no_mcp",
        "network_policy": "public",
        "proxy_policy": "none",
        "hardware_class": "8cpu-8gib-test",
        "cpus": int(prototype.get("observed_cpu_count", 8)),
        "memory_mb": int(prototype.get("observed_memory_mb", 8192)),
        "docker_image": "challenge-benchmark-quantum-tensorcircuit:py311",
        "container_image_digest": prototype.get(
            "container_image_digest", "sha256:" + "e" * 64
        ),
        "task_bundle_sha256": prototype.get("task_bundle_sha256", "a" * 64),
        "harbor_task_digest": prototype.get("harbor_task_digest", "b" * 64),
        "task_checksum": prototype.get("task_checksum", "c" * 64),
        "framework_prompt_sha256": prototype.get("framework_prompt_sha256", "d" * 64),
        "force_auth_json": True,
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _approved_workspace(tmp_path: Path) -> tuple[Path, str, Path]:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed evidence",
        )
    record_prototype(
        workspace, candidate_id, _prototype_bundle(workspace, candidate_id)
    )
    for role in PILOT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "pilot",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed frozen pilot",
        )
    return workspace, candidate_id, _protocol_config(workspace)


def _record_failure_trials(
    workspace: Path,
    candidate_id: str,
    manifest_path: Path,
    seeds: list[int],
    *,
    trial_prefix: str = "trial",
) -> None:
    for index, seed in enumerate(seeds):
        trial_id = f"{trial_prefix}-{index}"
        record_model_trial(
            workspace,
            candidate_id,
            manifest_path,
            trial_id,
            _trial_result(workspace, manifest_path, index, seed, passed=False),
        )
        for audit_index, role in enumerate(FAILURE_AUDIT_ROLES):
            audit_model_trial(
                workspace,
                candidate_id,
                trial_id,
                role,
                "algorithm_design",
                f"failure-auditor-{audit_index}",
                "trace inspected",
            )


def _trial_result(
    workspace: Path,
    manifest_path: Path,
    index: int,
    seed: int,
    *,
    passed: bool,
    input_tokens: int = 1000,
    output_tokens: int = 100,
) -> Path:
    manifest = json.loads(manifest_path.read_text())
    result_root = _artifact_root(workspace) / "trials"
    run_key = manifest["manifest_hash"][:12]
    reserve_model_trial_seed(
        workspace,
        manifest_path,
        seed,
        f"fixture-{run_key}-{index}",
    )
    score = 1.0 if passed else 0.0
    protocol = manifest["protocol_config"]
    case_digest = next(
        case["case_digest"]
        for case in manifest["expert_prequalified_cases"]
        if case["protocol_seed"] == seed
    )
    agent_kwargs = {
        "reasoning_effort": protocol["reasoning_effort"],
        "version": protocol["solver_agent_version"],
        "force_auth_json": True,
    }
    verifier_kwargs = {
        "audit_model": protocol["audit_model"],
        "force_auth_json": True,
    }
    verifier_env = {
        "REQUIRED_QUANTUM_FRAMEWORK": "tensorcircuit",
        "ORBIT_Q_CANDIDATE_SEED": str(seed),
        "ORBIT_Q_AUTHORIZATION_MANIFEST_HASH": manifest["manifest_hash"],
        "ORBIT_Q_CONTAINER_IMAGE_DIGEST": protocol["container_image_digest"],
        "ORBIT_Q_TASK_BUNDLE_SHA256": protocol["task_bundle_sha256"],
        "ORBIT_Q_PROVIDER_SNAPSHOT": protocol["provider_snapshot"],
        "ORBIT_Q_TOKEN_BUDGET": str(protocol["token_budget"]),
        "ORBIT_Q_TOKEN_BUDGET_ENFORCEMENT": protocol["token_budget_enforcement"],
        "ORBIT_Q_BUILD_IDENTITY_POLICY": protocol["build_identity_policy"],
        "ORBIT_Q_NETWORK_POLICY": protocol["network_policy"],
        "ORBIT_Q_PROXY_POLICY": protocol["proxy_policy"],
    }
    agent_config = {
        "import_path": "adapters.codex_para:CodexPara",
        "model_name": manifest["model"],
        "skills": [],
        "kwargs": agent_kwargs,
        "env": {},
        "mcp_servers": [],
    }
    environment_config = {
        "import_path": "adapters.framework_docker:FrameworkDockerEnvironment",
        "cpu_enforcement_policy": "limit",
        "memory_enforcement_policy": "limit",
        "override_cpus": protocol["cpus"],
        "override_memory_mb": protocol["memory_mb"],
        "kwargs": {
            "framework": "tensorcircuit",
            "docker_image": (
                f"{protocol['docker_image']}@{protocol['container_image_digest']}"
            ),
        },
    }
    verifier_config = {
        "import_path": "adapters.codex_para_verifier:CodexParaVerifier",
        "kwargs": verifier_kwargs,
        "env": verifier_env,
    }
    harbor_config_payload = {
        "timeout_multiplier": 1.0,
        "agent_timeout_multiplier": 1.0,
        "agent": agent_config,
        "environment": environment_config,
        "verifier": verifier_config,
    }
    trial_lock_payload = {
        "schema_version": 1,
        "task": {
            "name": f"candidate-{manifest['candidate_id']}",
            "type": "local",
            "digest": f"sha256:{protocol['harbor_task_digest']}",
            "path": "/frozen/candidate-task",
        },
        "install_only": False,
        "skills": [],
        "timeout_multiplier": 1.0,
        "extra_instructions": [
            {
                "path": "/frozen/tensorcircuit.md",
                "digest": f"sha256:{protocol['framework_prompt_sha256']}",
            }
        ],
        "agent": agent_config,
        "environment": environment_config,
        "verifier": verifier_config,
    }
    job_lock_payload = {
        "schema_version": 2,
        "harbor": {"version": protocol["harbor_version"], "is_editable": False},
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "trials": [trial_lock_payload],
    }
    raw_result = {
        "id": f"job-{run_key}-{index}",
        "finished_at": "2026-08-09T00:00:00Z",
        "task_checksum": protocol["task_checksum"],
        "config": harbor_config_payload,
        "agent_info": {
            "name": "codex",
            "version": protocol["solver_agent_version"],
            "model_info": {
                "name": manifest["model"],
                "provider": protocol["provider_snapshot"],
            },
        },
        "verifier_info": {"model_info": {"name": protocol["audit_model"]}},
        "agent_result": {
            "n_input_tokens": input_tokens,
            "n_output_tokens": output_tokens,
        },
        "verifier_result": {
            "rewards": {
                "reward": score,
                "functional_score": score,
                "static_policy_score": 1.0,
                "llm_audit_score": 1.0,
                "runtime_sec": 30.0 if passed else 60.0,
            }
        },
        "exception_info": None,
        "orbit_q_attestation": {
            "attestation_type": "operator_attested",
            "attestor_id": "test-runner",
            "attested_at": "2026-08-09T00:00:00Z",
            "manifest_hash": manifest["manifest_hash"],
            "candidate_hash": manifest["candidate_hash"],
            "protocol_config_sha256": manifest["protocol_config_sha256"],
        },
    }
    raw_job = _write_artifact(
        result_root / f"raw-job-{run_key}-{index}.json",
        json.dumps(raw_result, indent=2) + "\n",
    )
    transcript = _write_artifact(
        result_root / f"transcript-{run_key}-{index}.log",
        f"solver transcript {index}\n",
    )
    harbor_config = _write_artifact(
        result_root / f"config-{run_key}-{index}.json",
        json.dumps(harbor_config_payload, indent=2) + "\n",
    )
    harbor_lock = _write_artifact(
        result_root / f"trial-lock-{run_key}-{index}.json",
        json.dumps(trial_lock_payload, indent=2) + "\n",
    )
    harbor_job_lock = _write_artifact(
        result_root / f"job-lock-{run_key}-{index}.json",
        json.dumps(job_lock_payload, indent=2) + "\n",
    )
    functional_output = _write_artifact(
        result_root / f"functional-{run_key}-{index}.txt",
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": seed,
                    "case_digest": case_digest,
                }
            }
        )
        + "\nOverall: PASS\n",
    )
    result = {
        "schema_version": 2,
        "manifest_hash": manifest["manifest_hash"],
        "candidate_hash": manifest["candidate_hash"],
        "model": manifest["model"],
        "protocol_config_sha256": manifest["protocol_config_sha256"],
        "container_image_digest": manifest["prototype"]["container_image_digest"],
        "framework_prompt_sha256": manifest["prototype"]["framework_prompt_sha256"],
        "task_bundle_sha256": manifest["prototype"]["task_bundle_sha256"],
        "harbor_task_digest": protocol["harbor_task_digest"],
        "task_checksum": protocol["task_checksum"],
        "job_id": f"job-{run_key}-{index}",
        "protocol_seed": seed,
        "case_digest": case_digest,
        "solver_input_tokens": input_tokens,
        "solver_output_tokens": output_tokens,
        "solver_total_tokens": input_tokens + output_tokens,
        "token_budget_compliant": (
            input_tokens + output_tokens <= protocol["token_budget"]
        ),
        "hard_token_limit_enforced": False,
        "solver_provider_identity_verified": True,
        "audit_model_identity_verified": True,
        "execution_status": "completed",
        "reward": score,
        "functional_score": score,
        "static_policy_score": 1.0,
        "llm_audit_score": 1.0,
        "runtime_sec": 30.0 if passed else 60.0,
        "raw_job": raw_job,
        "solver_transcript": transcript,
        "harbor_config": harbor_config,
        "harbor_lock": harbor_lock,
        "harbor_job_lock": harbor_job_lock,
        "functional_output": functional_output,
    }
    path = result_root / f"trial-result-{run_key}-{index}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def _rehash_trial_artifact(result_path: Path, artifact_name: str) -> dict:
    result = json.loads(result_path.read_text())
    artifact_path = Path(result[artifact_name]["path"])
    result[artifact_name]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _rewrite_expert_run_artifact(
    run: dict, artifact_name: str, value: dict | str
) -> None:
    item = run["raw_evidence"][artifact_name]
    path = Path(item["path"])
    content = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
    path.write_text(content, encoding="utf-8")
    item["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    _refresh_expert_run_hash(run)


def _refresh_expert_run_hash(run: dict) -> None:
    artifact_hashes = {
        name: evidence["sha256"] for name, evidence in run["raw_evidence"].items()
    }
    run["run_evidence_sha256"] = hashlib.sha256(
        json.dumps(
            {"job_id": run["job_id"], "artifact_sha256s": artifact_hashes},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def test_catalog_expands_to_auditable_thousand_regimes() -> None:
    catalog = _catalog()
    assert validate_catalog(catalog) == []
    candidates = generate_candidates(catalog)
    assert len(candidates) == 1000
    assert len({row["id"] for row in candidates}) == 1000
    assert len({row["family_id"] for row in candidates}) == 50
    assert {row["empirical_status"] for row in candidates} == {"untested"}
    assert all(row["empirical_pass_rate"] is None for row in candidates)
    assert all(len(row["candidate_hash"]) == 64 for row in candidates)
    assert all(
        row["candidate_hash"] == _payload_sha256(row, "candidate_hash")
        for row in candidates
    )


def test_shortlist_is_ten_diverse_human_review_candidates() -> None:
    catalog = _catalog()
    candidates = generate_candidates(catalog)
    shortlist = select_shortlist(catalog, candidates)
    assert [row["rank"] for row in shortlist] == list(range(1, 11))
    assert len({row["family_id"] for row in shortlist}) == 10
    assert len({row["domain"] for row in shortlist}) >= 8
    assert {row["situation"]["id"] for row in shortlist} == {
        "forward",
        "inverse",
        "finite_shot_noise",
        "adaptive",
        "robust_ensemble",
    }
    assert {row["screen_status"] for row in shortlist} == {"shortlist_eligible"}
    candidates[-1]["risk_flags"]["stochastic_flake"] += 1
    with pytest.raises(ValueError, match="complete payload"):
        select_shortlist(catalog, candidates)


def test_machine_readable_contracts_are_valid_json() -> None:
    for name in (
        "research_protocol.json",
        "prototype_evidence.schema.json",
        "trial_result.schema.json",
        "protocol_config.example.json",
    ):
        assert isinstance(json.loads((DISCOVERY / name).read_text()), dict)
    protocol = json.loads((DISCOVERY / "research_protocol.json").read_text())
    assert protocol["stage_order"] == [
        "search",
        "ingest",
        "normalize",
        "dedupe",
        "coverage-sample",
        "score",
        "human-review",
    ]
    assert len({row["id"] for row in protocol["query_strategies"]}) == 5
    assert protocol["semantic_deduplication"]["automatic_action"].startswith(
        "Create review clusters only"
    )
    assert protocol["promotion_contract"]["automatic_catalog_edit_allowed"] is False
    assert (
        protocol["quota_policy"]["coverage_sample"][
            "expected_candidate_contract_count"
        ]["value"]
        == 1000
    )


def test_cli_normalizes_human_identity_and_note_text() -> None:
    assert _nonblank_text("  One   Person  ") == "One Person"
    assert _nonblank_text("  reviewed   evidence  ") == "reviewed evidence"
    with pytest.raises(argparse.ArgumentTypeError, match="non-whitespace"):
        _nonblank_text(" \t ")


def test_fresh_build_is_deterministic_and_has_clean_ledger(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = {
        name: (workspace / name).read_bytes()
        for name in (
            "candidates.jsonl",
            "shortlist.json",
            "shortlist.md",
            "summary.json",
            "review_state.json",
        )
    }
    build_workspace(workspace)
    assert first == {name: (workspace / name).read_bytes() for name in first}
    assert b"stale_candidate_hash" not in first["review_state.json"]
    ledger = json.loads(first["review_state.json"])
    assert set(ledger["candidates"]) == set(ledger["active_shortlist_ids"])


def test_concurrent_reviews_do_not_lose_updates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]

    def review(role: str) -> None:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"concurrent-{role}",
            "independently reviewed",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(review, CONCEPT_REVIEW_ROLES))
    state = json.loads((workspace / "review_state.json").read_text())
    record = state["candidates"][candidate_id]
    assert set(record["concept_reviews"]) == set(CONCEPT_REVIEW_ROLES)
    review_events = [
        event
        for event in state["event_log"]
        if event.get("event") == "review" and event.get("candidate_id") == candidate_id
    ]
    assert len(review_events) == 3


def test_concurrent_rebuild_and_review_preserve_ledger_update(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]

    def rebuild() -> None:
        build_workspace(workspace)

    def review() -> None:
        record_review(
            workspace,
            candidate_id,
            "concept",
            CONCEPT_REVIEW_ROLES[0],
            "approve",
            "concurrent-rebuild-reviewer",
            "reviewed during rebuild",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(rebuild), pool.submit(review))
        for future in futures:
            future.result()
    state = json.loads((workspace / "review_state.json").read_text())
    assert (
        CONCEPT_REVIEW_ROLES[0] in state["candidates"][candidate_id]["concept_reviews"]
    )


def test_real_test_stays_blocked_until_verified_human_gates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    config = _protocol_config(workspace)
    output = _artifact_root(workspace) / "manifests" / "blocked.json"
    with pytest.raises(PermissionError):
        authorize_model_test(
            workspace, candidate_id, config, "pilot", list(range(5)), output
        )

    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed",
        )
    with pytest.raises(FileNotFoundError):
        record_prototype(
            workspace,
            candidate_id,
            _artifact_root(workspace) / "missing-evidence.json",
        )
    bundle_path = _prototype_bundle(workspace, candidate_id)
    bundle = json.loads(bundle_path.read_text())
    bundle["valid_alternatives"][1]["log"] = bundle["valid_alternatives"][0]["log"]
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    with pytest.raises(ValueError, match="valid-alternative logs and styles"):
        record_prototype(workspace, candidate_id, bundle_path)


@pytest.mark.parametrize(
    "relative",
    (
        "tests/score_submission.py",
        "tests/test.sh",
        "tests/static_policy.py",
        "tests/audit_codex.py",
        "solution/solve.sh",
        "task.toml",
    ),
)
def test_record_prototype_rejects_self_consistent_staged_substitution(
    tmp_path: Path, relative: str
) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed evidence",
        )
    bundle_path = _prototype_bundle(workspace, candidate_id)
    bundle = json.loads(bundle_path.read_text())
    task_bundle = Path(bundle["artifacts"]["task_bundle"]["path"])
    target = task_bundle / relative
    target.write_bytes(target.read_bytes() + b"\n# staged substitution\n")
    bundle["artifacts"]["task_bundle"]["sha256"] = _path_sha256(task_bundle)
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")

    with pytest.raises(ValueError, match="trusted source contract"):
        record_prototype(workspace, candidate_id, bundle_path)


def test_record_prototype_rejects_extra_file_and_shell_mode_drift(
    tmp_path: Path,
) -> None:
    for variant in ("extra", "mode"):
        variant_root = tmp_path / variant
        variant_root.mkdir()
        workspace = _workspace(variant_root)
        candidate_id = json.loads((workspace / "shortlist.json").read_text())[
            "candidates"
        ][0]["id"]
        for role in CONCEPT_REVIEW_ROLES:
            record_review(
                workspace,
                candidate_id,
                "concept",
                role,
                "approve",
                f"reviewer-{role}",
                "reviewed evidence",
            )
        bundle_path = _prototype_bundle(workspace, candidate_id)
        bundle = json.loads(bundle_path.read_text())
        task_bundle = Path(bundle["artifacts"]["task_bundle"]["path"])
        if variant == "extra":
            (task_bundle / "tests" / "operator-summary.py").write_text(
                "ADMITTED = True\n"
            )
        else:
            (task_bundle / "tests" / "test.sh").chmod(0o644)
        bundle["artifacts"]["task_bundle"]["sha256"] = _path_sha256(task_bundle)
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")

        with pytest.raises(ValueError, match="trusted source contract"):
            record_prototype(workspace, candidate_id, bundle_path)


def test_prototype_prequalified_cases_fail_closed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed evidence",
        )
    bundle_path = _prototype_bundle(workspace, candidate_id)
    original = json.loads(bundle_path.read_text())

    too_few = {
        **original,
        "expert_prequalified_cases": original["expert_prequalified_cases"][:24],
    }
    bundle_path.write_text(json.dumps(too_few, indent=2) + "\n")
    with pytest.raises(ValueError, match="at least 25"):
        record_prototype(workspace, candidate_id, bundle_path)

    invalid_margin = json.loads(json.dumps(original))
    invalid_margin["expert_prequalified_cases"][0]["admission"]["minimum_margin"] = 0.20
    bundle_path.write_text(json.dumps(invalid_margin, indent=2) + "\n")
    with pytest.raises(ValueError, match="declared expert run summary"):
        record_prototype(workspace, candidate_id, bundle_path)

    invalid_digest = json.loads(json.dumps(original))
    invalid_digest["expert_prequalified_cases"][0]["case_digest"] = "abc"
    bundle_path.write_text(json.dumps(invalid_digest, indent=2) + "\n")
    with pytest.raises(ValueError, match="declared expert run summary"):
        record_prototype(workspace, candidate_id, bundle_path)

    bundle_path.write_text(json.dumps(original, indent=2) + "\n")
    status = record_prototype(workspace, candidate_id, bundle_path)
    assert status["prototype_gate"]["expert_prequalified_cases_valid"] is True


def test_expert_prequalification_is_derived_from_raw_run_evidence(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"raw-run-reviewer-{role}",
            "reviewed expert-run evidence contract",
        )

    mutation_names = (
        "fabricated-summary",
        "functional-output",
        "admission-record",
        "raw-job",
        "job-lock",
        "task-binding",
        "expert-binding",
        "evaluator-binding",
        "image-binding",
        "declared-margin",
    )
    for mutation_name in mutation_names:
        bundle_path = _prototype_bundle(workspace, candidate_id)
        bundle = json.loads(bundle_path.read_text())
        run = bundle["expert_prequalified_cases"][0]
        if mutation_name == "fabricated-summary":
            run["job_id"] = "fabricated-job"
            _refresh_expert_run_hash(run)
        elif mutation_name == "functional-output":
            _rewrite_expert_run_artifact(
                run,
                "functional_output",
                json.dumps(
                    {
                        "orbit_q_case_identity": {
                            "protocol_seed": run["protocol_seed"],
                            "case_digest": run["case_digest"],
                        }
                    }
                )
                + "\nOverall: FAIL\n",
            )
        elif mutation_name == "admission-record":
            path = Path(run["raw_evidence"]["admission_record"]["path"])
            payload = json.loads(path.read_text())
            payload["admission"]["margins"][0]["observed"] = 0.8
            _rewrite_expert_run_artifact(run, "admission_record", payload)
        elif mutation_name == "raw-job":
            path = Path(run["raw_evidence"]["raw_job"]["path"])
            payload = json.loads(path.read_text())
            payload["finished_at"] = None
            _rewrite_expert_run_artifact(run, "raw_job", payload)
        elif mutation_name == "job-lock":
            path = Path(run["raw_evidence"]["harbor_job_lock"]["path"])
            payload = json.loads(path.read_text())
            payload["retry"]["max_retries"] = 1
            _rewrite_expert_run_artifact(run, "harbor_job_lock", payload)
        elif mutation_name == "task-binding":
            path = Path(run["raw_evidence"]["harbor_lock"]["path"])
            payload = json.loads(path.read_text())
            payload["task"]["digest"] = "sha256:" + "9" * 64
            _rewrite_expert_run_artifact(run, "harbor_lock", payload)
        elif mutation_name in {
            "expert-binding",
            "evaluator-binding",
            "image-binding",
        }:
            field = {
                "expert-binding": "expert_baseline_sha256",
                "evaluator-binding": "evaluator_sha256",
                "image-binding": "container_image_digest",
            }[mutation_name]
            path = Path(run["raw_evidence"]["admission_record"]["path"])
            payload = json.loads(path.read_text())
            payload[field] = (
                "sha256:" + "9" * 64 if field == "container_image_digest" else "9" * 64
            )
            _rewrite_expert_run_artifact(run, "admission_record", payload)
        else:
            run["admission"]["minimum_margin"] = 0.1
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
        with pytest.raises(ValueError):
            record_prototype(workspace, candidate_id, bundle_path)


def test_authorization_is_subset_of_exact_prequalified_cases(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    with pytest.raises(PermissionError, match="without exact expert prequalification"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            [0, 1, 2, 3, 999],
            _artifact_root(workspace) / "manifests" / "unqualified.json",
        )

    manifest_path = _artifact_root(workspace) / "manifests" / "qualified.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", [0, 1, 2, 3, 4], manifest_path
    )
    assert [
        case["protocol_seed"] for case in manifest["expert_prequalified_cases"]
    ] == [0, 1, 2, 3, 4]
    assert all(
        len(case["case_digest"]) == 64
        and case["admission"]["status"] == "admitted"
        and case["admission"]["minimum_margin"] > 0
        for case in manifest["expert_prequalified_cases"]
    )
    canonical_cases = json.dumps(
        manifest["expert_prequalified_cases"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert (
        manifest["expert_prequalified_cases_sha256"]
        == hashlib.sha256(canonical_cases).hexdigest()
    )


@pytest.mark.parametrize("entrypoint", ("authorize", "reserve", "import", "hardness"))
def test_every_model_gate_reparses_expert_run_files(
    tmp_path: Path, entrypoint: str
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / f"{entrypoint}.json"
    manifest = None
    result_path = None
    if entrypoint != "authorize":
        manifest = authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            [0, 1, 2, 3, 4],
            manifest_path,
        )
    if entrypoint == "import":
        result_path = _trial_result(workspace, manifest_path, 880, 0, passed=False)

    state = json.loads((workspace / "review_state.json").read_text())
    prototype = state["candidates"][candidate_id]["prototype"]
    raw_job = Path(
        prototype["expert_prequalified_cases"][0]["raw_evidence"]["raw_job"]["path"]
    )
    raw_job.write_text(raw_job.read_text() + "\n")

    if entrypoint == "authorize":
        with pytest.raises(PermissionError):
            authorize_model_test(
                workspace,
                candidate_id,
                config,
                "pilot",
                [0, 1, 2, 3, 4],
                manifest_path,
            )
    elif entrypoint == "reserve":
        with pytest.raises(PermissionError):
            reserve_model_trial_seed(workspace, manifest_path, 0, "drifted-run")
    elif entrypoint == "import":
        assert result_path is not None
        with pytest.raises(PermissionError):
            record_model_trial(
                workspace, candidate_id, manifest_path, "drifted-run", result_path
            )
    else:
        assert manifest is not None
        status = model_hardness_status(
            workspace, candidate_id, manifest["manifest_hash"]
        )
        assert status["conclusion"] == "evidence_drift_or_gate_revoked"


def test_five_failures_are_only_a_pilot_signal(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "pilot.json"
    seeds = [11, 12, 13, 14, 15]
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", seeds, manifest_path
    )
    _record_failure_trials(workspace, candidate_id, manifest_path, seeds)
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "pilot_hardness_signal_ready"
    assert status["schedule_complete"] is True
    assert status["pass_at_k"] == 0
    with pytest.raises(PermissionError, match="precommitted"):
        record_model_trial(
            workspace,
            candidate_id,
            manifest_path,
            "optional-extra",
            _trial_result(workspace, manifest_path, 99, 99, passed=False),
        )


def test_any_raw_pass_blocks_hardness_before_audit(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "pass.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    record_model_trial(
        workspace,
        candidate_id,
        manifest_path,
        "trial-pass",
        _trial_result(workspace, manifest_path, 100, 0, passed=True),
    )
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "solved_in_at_least_one_valid_trial"


def test_typed_summary_cannot_override_raw_harbor_result(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "tamper.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    result_path = _trial_result(workspace, manifest_path, 101, 0, passed=True)
    result = json.loads(result_path.read_text())
    result["reward"] = 0.0
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    with pytest.raises(ValueError, match="disagrees with raw Harbor evidence"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "tampered", result_path
        )


def test_confirmation_requires_twenty_audited_substantive_failures(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    pilot_path = _artifact_root(workspace) / "manifests" / "qualifying-pilot.json"
    pilot_seeds = [100, 101, 102, 103, 104]
    pilot = authorize_model_test(
        workspace, candidate_id, config, "pilot", pilot_seeds, pilot_path
    )
    _record_failure_trials(
        workspace,
        candidate_id,
        pilot_path,
        pilot_seeds,
        trial_prefix="pilot",
    )
    manifest_path = _artifact_root(workspace) / "manifests" / "confirmation.json"
    seeds = list(range(20))
    manifest = authorize_model_test(
        workspace,
        candidate_id,
        config,
        "confirmation",
        seeds,
        manifest_path,
        pilot["manifest_hash"],
    )
    _record_failure_trials(workspace, candidate_id, manifest_path, seeds)
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert (
        status["conclusion"] == "protocol_scoped_hardness_blocked_by_unenforced_limits"
    )
    assert status["admission_limitations"] == [
        "submitted_solution_process_isolation_unavailable",
        "hard_token_execution_cap_unavailable",
    ]
    assert status["valid_trials"] == 20
    state = json.loads((workspace / "review_state.json").read_text())
    pilot_raw_path = Path(
        state["candidates"][candidate_id]["model_trials"]["pilot-0"]["raw_job"]["path"]
    )
    original_pilot_raw = pilot_raw_path.read_bytes()
    pilot_raw_path.write_bytes(original_pilot_raw + b"\n")
    linked_drift = model_hardness_status(
        workspace, candidate_id, manifest["manifest_hash"]
    )
    assert linked_drift["conclusion"] == "evidence_drift_or_gate_revoked"
    assert (
        "qualifying_pilot_evidence_invalid_or_changed"
        in linked_drift["evidence_violations"]
    )
    pilot_raw_path.write_bytes(original_pilot_raw)
    assert (
        model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])[
            "conclusion"
        ]
        == "protocol_scoped_hardness_blocked_by_unenforced_limits"
    )
    expert_path = Path(
        state["candidates"][candidate_id]["prototype"]["expert_baseline_path"]
    )
    expert_path.write_text(expert_path.read_text() + "changed = True\n")
    drifted = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert drifted["conclusion"] == "evidence_drift_or_gate_revoked"
    assert drifted["authorization_active"] is False
    assert "current_human_or_prototype_gate_failed" in drifted["evidence_violations"]


def test_authorization_recomputes_shortlist_content_hash(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    shortlist_path = workspace / "shortlist.json"
    shortlist = json.loads(shortlist_path.read_text())
    candidate = next(
        row for row in shortlist["candidates"] if row["id"] == candidate_id
    )
    candidate["objective"] += " tampered"
    shortlist_path.write_text(json.dumps(shortlist, indent=2) + "\n")
    with pytest.raises(ValueError, match="hash does not match"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            list(range(5)),
            _artifact_root(workspace) / "tampered-shortlist.json",
        )


def test_authorization_validates_source_candidate_payload_binding(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    candidate_path = workspace / "candidates.jsonl"
    rows = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    candidate = next(row for row in rows if row["id"] == candidate_id)
    candidate["risk_flags"]["runtime_overflow"] += 1
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="complete payload"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            list(range(5)),
            _artifact_root(workspace) / "candidate-payload-tampered.json",
        )

    candidate["candidate_hash"] = _payload_sha256(candidate, "candidate_hash")
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="not bound to its source row"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            list(range(5)),
            _artifact_root(workspace) / "candidate-rehashed-tamper.json",
        )


def test_confirmation_and_safe_output_are_enforced(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    with pytest.raises(ValueError, match="at least 20"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "confirmation",
            list(range(19)),
            _artifact_root(workspace) / "too-few.json",
        )
    with pytest.raises(ValueError, match="must stay under"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            list(range(5)),
            tmp_path / "unsafe-output.json",
        )


def test_execution_requires_active_manifest_and_one_unused_seed(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "execute.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", [10, 11, 12, 13, 14], manifest_path
    )
    state = json.loads((workspace / "review_state.json").read_text())
    prototype = state["candidates"][candidate_id]["prototype"]
    spec = authorized_model_execution(
        workspace,
        manifest_path,
        Path(prototype["task_bundle_path"]),
        Path(prototype["framework_prompt_path"]),
        10,
    )
    assert spec["manifest_hash"] == manifest["manifest_hash"]
    assert spec["protocol_seed"] == 10
    selected = next(
        case
        for case in manifest["expert_prequalified_cases"]
        if case["protocol_seed"] == 10
    )
    assert spec["case_digest"] == selected["case_digest"]
    assert spec["expert_admission"] == selected["admission"]
    wrong_task = _artifact_root(workspace) / "wrong-task.bundle"
    wrong_task.write_text("different task\n")
    with pytest.raises(PermissionError, match="task hash"):
        authorized_model_execution(
            workspace,
            manifest_path,
            wrong_task,
            Path(prototype["framework_prompt_path"]),
            11,
        )
    wrong_prompt = _artifact_root(workspace) / "wrong-prompt.md"
    wrong_prompt.write_text("different prompt\n")
    with pytest.raises(PermissionError, match="prompt hash"):
        authorized_model_execution(
            workspace,
            manifest_path,
            Path(prototype["task_bundle_path"]),
            wrong_prompt,
            11,
        )
    with pytest.raises(PermissionError, match="precommitted"):
        authorized_model_execution(
            workspace,
            manifest_path,
            Path(prototype["task_bundle_path"]),
            Path(prototype["framework_prompt_path"]),
            99,
        )
    reserve_model_trial_seed(workspace, manifest_path, 10, "execution-10")
    with pytest.raises(PermissionError, match="consumed or reserved"):
        authorized_model_execution(
            workspace,
            manifest_path,
            Path(prototype["task_bundle_path"]),
            Path(prototype["framework_prompt_path"]),
            10,
        )

    replacement_path = _artifact_root(workspace) / "manifests" / "replacement.json"
    authorize_model_test(
        workspace,
        candidate_id,
        config,
        "pilot",
        [20, 21, 22, 23, 24],
        replacement_path,
    )
    with pytest.raises(PermissionError, match="inactive or superseded"):
        authorized_model_execution(
            workspace,
            manifest_path,
            Path(prototype["task_bundle_path"]),
            Path(prototype["framework_prompt_path"]),
            11,
        )


def test_import_and_hardness_revalidate_exact_prequalified_case(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "case-binding.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", [0, 1, 2, 3, 4], manifest_path
    )
    result_path = _trial_result(workspace, manifest_path, 700, 0, passed=False)
    result = json.loads(result_path.read_text())
    wrong_digest = "f" * 64
    functional_path = Path(result["functional_output"]["path"])
    functional_path.write_text(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": 0,
                    "case_digest": wrong_digest,
                }
            }
        )
        + "\nOverall: PASS\n"
    )
    result["functional_output"]["sha256"] = hashlib.sha256(
        functional_path.read_bytes()
    ).hexdigest()
    result["case_digest"] = wrong_digest
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    with pytest.raises(PermissionError, match="exact expert-prequalified case"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "wrong-case", result_path
        )

    correct_result = _trial_result(workspace, manifest_path, 701, 1, passed=False)
    record_model_trial(
        workspace, candidate_id, manifest_path, "bound-case", correct_result
    )
    raw_job = Path(
        manifest["expert_prequalified_cases"][1]["raw_evidence"]["raw_job"]["path"]
    )
    raw_job.write_text(raw_job.read_text() + "\n")
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "evidence_drift_or_gate_revoked"
    assert "manifest_or_protocol_unverifiable" in status["evidence_violations"]


def test_seed_reservation_is_atomic_across_concurrent_runners(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "race.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", [30, 31, 32, 33, 34], manifest_path
    )

    def reserve(execution_id: str) -> str:
        try:
            reserve_model_trial_seed(workspace, manifest_path, 30, execution_id)
        except PermissionError:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, ("race-a", "race-b")))
    assert sorted(outcomes) == ["rejected", "reserved"]


def test_concurrent_authorizations_preserve_revocation_ledger(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    artifact_root = _artifact_root(workspace)

    def authorize(index: int) -> str:
        seeds = list(range(index * 5, index * 5 + 5))
        manifest = authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            seeds,
            artifact_root / "manifests" / f"concurrent-{index}.json",
        )
        return manifest["manifest_hash"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest_hashes = list(pool.map(authorize, (0, 1)))
    state = json.loads((workspace / "review_state.json").read_text())
    record = state["candidates"][candidate_id]
    rows = [
        row
        for row in record["authorizations"]
        if row.get("manifest_hash") in manifest_hashes
    ]
    assert len(rows) == 2
    assert record["active_authorization_hash"] in manifest_hashes
    assert sum("revoked_at" in row for row in rows) == 1
    revoked = next(row for row in rows if "revoked_at" in row)
    assert revoked["revocation_reason"] == "superseded_by_new_authorization"
    assert (
        len(
            [
                event
                for event in state["event_log"]
                if event.get("event") == "model_test_authorized"
                and event.get("manifest_hash") in manifest_hashes
            ]
        )
        == 2
    )


def test_concurrent_audits_do_not_lose_updates(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "audit-race.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", [0, 1, 2, 3, 4], manifest_path
    )
    record_model_trial(
        workspace,
        candidate_id,
        manifest_path,
        "audit-race-trial",
        _trial_result(workspace, manifest_path, 800, 0, passed=False),
    )

    def audit(item: tuple[int, str]) -> None:
        index, role = item
        audit_model_trial(
            workspace,
            candidate_id,
            "audit-race-trial",
            role,
            "algorithm_design",
            f"concurrent-auditor-{index}",
            "trace inspected independently",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(audit, enumerate(FAILURE_AUDIT_ROLES)))
    state = json.loads((workspace / "review_state.json").read_text())
    audits = state["candidates"][candidate_id]["model_trials"]["audit-race-trial"][
        "human_audits"
    ]
    assert set(audits) == set(FAILURE_AUDIT_ROLES)


def test_concurrent_pass_import_and_authorization_cannot_bypass_veto(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "pass-race.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", [0, 1, 2, 3, 4], manifest_path
    )
    pass_result = _trial_result(workspace, manifest_path, 900, 0, passed=True)

    def import_pass() -> str:
        try:
            record_model_trial(
                workspace,
                candidate_id,
                manifest_path,
                "concurrent-pass",
                pass_result,
            )
        except PermissionError:
            return "pass-import-rejected"
        return "pass-imported"

    def issue_replacement() -> str:
        try:
            authorize_model_test(
                workspace,
                candidate_id,
                config,
                "pilot",
                [10, 11, 12, 13, 14],
                _artifact_root(workspace) / "manifests" / "pass-race-replacement.json",
            )
        except PermissionError:
            return "authorization-rejected"
        return "authorization-issued"

    with ThreadPoolExecutor(max_workers=2) as pool:
        import_future = pool.submit(import_pass)
        authorize_future = pool.submit(issue_replacement)
        outcomes = {import_future.result(), authorize_future.result()}
    assert outcomes in (
        {"pass-imported", "authorization-rejected"},
        {"pass-import-rejected", "authorization-issued"},
    )
    assert outcomes != {"pass-imported", "authorization-issued"}


def test_confirmation_rejects_missing_incomplete_and_overlapping_pilot(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    with pytest.raises(ValueError, match="pilot-manifest-hash"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "confirmation",
            list(range(20, 40)),
            _artifact_root(workspace) / "manifests" / "missing-link.json",
        )

    pilot_path = _artifact_root(workspace) / "manifests" / "pilot-link.json"
    pilot_seeds = [1, 2, 3, 4, 5]
    pilot = authorize_model_test(
        workspace, candidate_id, config, "pilot", pilot_seeds, pilot_path
    )
    with pytest.raises(PermissionError, match="zero-pass, two-auditor"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "confirmation",
            list(range(20, 40)),
            _artifact_root(workspace) / "manifests" / "incomplete-pilot.json",
            pilot["manifest_hash"],
        )
    _record_failure_trials(workspace, candidate_id, pilot_path, pilot_seeds)
    with pytest.raises(ValueError, match="disjoint"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "confirmation",
            [1, *range(20, 39)],
            _artifact_root(workspace) / "manifests" / "overlap.json",
            pilot["manifest_hash"],
        )


def test_any_candidate_model_pass_vetoes_later_hardness_authorization(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "solved-pilot.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    record_model_trial(
        workspace,
        candidate_id,
        manifest_path,
        "solving-trial",
        _trial_result(workspace, manifest_path, 501, 0, passed=True),
    )
    state_path = workspace / "review_state.json"
    state = json.loads(state_path.read_text())
    del state["candidates"][candidate_id]["model_trials"]["solving-trial"]
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    with pytest.raises(PermissionError, match="prior valid pass"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            list(range(10, 15)),
            _artifact_root(workspace) / "manifests" / "post-pass.json",
        )


def test_machine_config_controls_seed_model_and_full_case_digest(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "machine.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )

    attestation_only = _trial_result(workspace, manifest_path, 600, 0, passed=False)
    normalized = json.loads(attestation_only.read_text())
    raw_path = Path(normalized["raw_job"]["path"])
    raw = json.loads(raw_path.read_text())
    raw["orbit_q_attestation"]["protocol_seed"] = 999999
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    _rehash_trial_artifact(attestation_only, "raw_job")
    trial = record_model_trial(
        workspace,
        candidate_id,
        manifest_path,
        "attestation-supplemental",
        attestation_only,
    )
    assert trial["protocol_seed"] == 0

    wrong_model = _trial_result(workspace, manifest_path, 601, 1, passed=False)
    normalized = json.loads(wrong_model.read_text())
    raw_path = Path(normalized["raw_job"]["path"])
    raw = json.loads(raw_path.read_text())
    raw["config"]["agent"]["model_name"] = "different-model"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    _rehash_trial_artifact(wrong_model, "raw_job")
    with pytest.raises(ValueError, match="Harbor config disagrees"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "wrong-machine-model", wrong_model
        )

    short_digest = _trial_result(workspace, manifest_path, 602, 2, passed=False)
    normalized = json.loads(short_digest.read_text())
    functional_path = Path(normalized["functional_output"]["path"])
    identity = json.loads(functional_path.read_text().splitlines()[0])
    identity["orbit_q_case_identity"]["case_digest"] = "a" * 16
    functional_path.write_text(json.dumps(identity) + "\n")
    _rehash_trial_artifact(short_digest, "functional_output")
    with pytest.raises(ValueError, match="full 64-hex digest"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "short-case-digest", short_digest
        )

    plain_harbor = _trial_result(workspace, manifest_path, 603, 3, passed=False)
    normalized = json.loads(plain_harbor.read_text())
    raw_path = Path(normalized["raw_job"]["path"])
    raw = json.loads(raw_path.read_text())
    del raw["orbit_q_attestation"]
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    _rehash_trial_artifact(plain_harbor, "raw_job")
    trial = record_model_trial(
        workspace, candidate_id, manifest_path, "unaltered-harbor-shape", plain_harbor
    )
    assert trial["protocol_seed"] == 3

    unobserved_identity = _trial_result(workspace, manifest_path, 604, 4, passed=False)
    normalized = json.loads(unobserved_identity.read_text())
    raw_path = Path(normalized["raw_job"]["path"])
    raw = json.loads(raw_path.read_text())
    raw["agent_info"]["model_info"]["provider"] = None
    del raw["verifier_info"]
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    normalized["solver_provider_identity_verified"] = False
    normalized["audit_model_identity_verified"] = False
    unobserved_identity.write_text(json.dumps(normalized, indent=2) + "\n")
    _rehash_trial_artifact(unobserved_identity, "raw_job")
    trial = record_model_trial(
        workspace,
        candidate_id,
        manifest_path,
        "runtime-build-unobserved",
        unobserved_identity,
    )
    assert trial["solver_provider_identity_verified"] is False
    assert trial["audit_model_identity_verified"] is False


def test_over_budget_failures_do_not_count_but_a_pass_still_vetoes(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "budget.json"
    seeds = list(range(5))
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", seeds, manifest_path
    )
    for index, seed in enumerate(seeds):
        trial_id = f"budget-{index}"
        record_model_trial(
            workspace,
            candidate_id,
            manifest_path,
            trial_id,
            _trial_result(
                workspace,
                manifest_path,
                700 + index,
                seed,
                passed=False,
                input_tokens=6000,
            ),
        )
        for audit_index, role in enumerate(FAILURE_AUDIT_ROLES):
            audit_model_trial(
                workspace,
                candidate_id,
                trial_id,
                role,
                "algorithm_design",
                f"budget-auditor-{audit_index}",
                "trace inspected",
            )
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "incomplete_or_invalid_trial_schedule"
    assert status["substantive_failures"] == 0
    assert status["excluded_failures"] == 5

    solved_root = tmp_path / "solved"
    solved_root.mkdir()
    solved_workspace, solved_candidate, solved_config = _approved_workspace(solved_root)
    solved_manifest_path = (
        _artifact_root(solved_workspace) / "manifests" / "over-budget-pass.json"
    )
    solved_manifest = authorize_model_test(
        solved_workspace,
        solved_candidate,
        solved_config,
        "pilot",
        list(range(5)),
        solved_manifest_path,
    )
    record_model_trial(
        solved_workspace,
        solved_candidate,
        solved_manifest_path,
        "over-budget-pass",
        _trial_result(
            solved_workspace,
            solved_manifest_path,
            800,
            0,
            passed=True,
            input_tokens=6000,
        ),
    )
    solved = model_hardness_status(
        solved_workspace, solved_candidate, solved_manifest["manifest_hash"]
    )
    assert solved["conclusion"] == "solved_in_at_least_one_valid_trial"


def test_timeout_override_and_retry_bypasses_are_rejected(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "controls.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )

    timeout_result = _trial_result(workspace, manifest_path, 900, 0, passed=False)
    normalized = json.loads(timeout_result.read_text())
    raw_path = Path(normalized["raw_job"]["path"])
    raw = json.loads(raw_path.read_text())
    raw["config"]["agent"]["override_timeout_sec"] = 1
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    _rehash_trial_artifact(timeout_result, "raw_job")
    with pytest.raises(ValueError, match="timeout/trajectory override"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "short-timeout", timeout_result
        )

    retry_result = _trial_result(workspace, manifest_path, 901, 1, passed=False)
    normalized = json.loads(retry_result.read_text())
    job_lock_path = Path(normalized["harbor_job_lock"]["path"])
    job_lock = json.loads(job_lock_path.read_text())
    job_lock["retry"]["max_retries"] = 1
    job_lock_path.write_text(json.dumps(job_lock, indent=2) + "\n")
    _rehash_trial_artifact(retry_result, "harbor_job_lock")
    with pytest.raises(ValueError, match="disable retries"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "retry-enabled", retry_result
        )

    mount_result = _trial_result(workspace, manifest_path, 902, 2, passed=False)
    normalized = json.loads(mount_result.read_text())
    raw_path = Path(normalized["raw_job"]["path"])
    raw = json.loads(raw_path.read_text())
    raw["config"]["environment"]["mounts"] = ["/host:/container"]
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    _rehash_trial_artifact(mount_result, "raw_job")
    with pytest.raises(ValueError, match="unauthorized expansion"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "mount-injected", mount_result
        )


def test_reviewers_and_failure_audits_are_append_only(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    first_role, second_role = CONCEPT_REVIEW_ROLES[:2]
    with pytest.raises(ValueError, match="nonempty identity"):
        record_review(
            workspace, candidate_id, "concept", first_role, "approve", " ", "ok"
        )
    with pytest.raises(ValueError, match="nonempty text"):
        record_review(
            workspace,
            candidate_id,
            "concept",
            first_role,
            "approve",
            "first person",
            " \t ",
        )
    record_review(
        workspace,
        candidate_id,
        "concept",
        first_role,
        "approve",
        "  One   Person  ",
        "  reviewed   carefully  ",
    )
    state = json.loads((workspace / "review_state.json").read_text())
    stored = state["candidates"][candidate_id]["concept_reviews"][first_role]
    assert stored["reviewer"] == "One Person"
    assert stored["note"] == "reviewed carefully"
    with pytest.raises(ValueError, match="distinct reviewer"):
        record_review(
            workspace,
            candidate_id,
            "concept",
            second_role,
            "approve",
            "  one   person  ",
            "ok",
        )
    with pytest.raises(ValueError, match="append-only"):
        record_review(
            workspace,
            candidate_id,
            "concept",
            first_role,
            "reject",
            "different-person",
            "changed mind",
        )


def test_gate_mutation_explicitly_revokes_active_authorization(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "revoke.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    state_path = workspace / "review_state.json"
    state = json.loads(state_path.read_text())
    del state["candidates"][candidate_id]["pilot_reviews"]["independent_reviewer"]
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    record_review(
        workspace,
        candidate_id,
        "pilot",
        "independent_reviewer",
        "approve",
        "replacement independent reviewer",
        "replacement review after package mutation",
    )
    state = json.loads(state_path.read_text())
    record = state["candidates"][candidate_id]
    assert "active_authorization_hash" not in record
    authorization = next(
        row
        for row in record["authorizations"]
        if row["manifest_hash"] == manifest["manifest_hash"]
    )
    assert authorization["revocation_reason"] == "pilot_review_mutated"
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["authorization_active"] is False
    assert status["conclusion"] == "authorization_revoked_or_superseded"


def test_prototype_mutation_explicitly_revokes_active_authorization(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    state_path = workspace / "review_state.json"
    state = json.loads(state_path.read_text())
    prototype_path = Path(
        state["candidates"][candidate_id]["prototype"]["evidence_bundle_path"]
    )
    manifest_path = _artifact_root(workspace) / "manifests" / "prototype-revoke.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    state = json.loads(state_path.read_text())
    state["candidates"][candidate_id]["prototype"] = {}
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    record_prototype(workspace, candidate_id, prototype_path)
    state = json.loads(state_path.read_text())
    record = state["candidates"][candidate_id]
    assert "active_authorization_hash" not in record
    authorization = next(
        row
        for row in record["authorizations"]
        if row["manifest_hash"] == manifest["manifest_hash"]
    )
    assert authorization["revocation_reason"] == "prototype_mutated"


def test_hardness_status_fails_closed_on_candidate_and_manifest_drift(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "file-drift.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    candidate_path = workspace / "candidates.jsonl"
    original_candidates = candidate_path.read_text()
    rows = [json.loads(line) for line in original_candidates.splitlines()]
    candidate = next(row for row in rows if row["id"] == candidate_id)
    candidate["evidence_class"] = "tampered"
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    candidate_drift = model_hardness_status(
        workspace, candidate_id, manifest["manifest_hash"]
    )
    assert candidate_drift["authorization_active"] is False
    assert "candidate_content_unverifiable" in candidate_drift["evidence_violations"]

    candidate_path.write_text(original_candidates)
    original_manifest = manifest_path.read_text()
    changed_manifest = json.loads(original_manifest)
    changed_manifest["execution_policy"]["frozen_prompt_required"] = False
    manifest_path.write_text(json.dumps(changed_manifest, indent=2) + "\n")
    manifest_drift = model_hardness_status(
        workspace, candidate_id, manifest["manifest_hash"]
    )
    assert manifest_drift["authorization_active"] is False
    assert "manifest_or_protocol_unverifiable" in manifest_drift["evidence_violations"]


def test_failure_auditor_identity_and_note_are_normalized(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "audit-text.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    record_model_trial(
        workspace,
        candidate_id,
        manifest_path,
        "trial-audit-text",
        _trial_result(workspace, manifest_path, 200, 0, passed=False),
    )
    with pytest.raises(ValueError, match="nonempty identity"):
        audit_model_trial(
            workspace,
            candidate_id,
            "trial-audit-text",
            FAILURE_AUDIT_ROLES[0],
            "algorithm_design",
            " \t ",
            "trace inspected",
        )
    with pytest.raises(ValueError, match="nonempty text"):
        audit_model_trial(
            workspace,
            candidate_id,
            "trial-audit-text",
            FAILURE_AUDIT_ROLES[0],
            "algorithm_design",
            "audit person",
            "  ",
        )
    audit = audit_model_trial(
        workspace,
        candidate_id,
        "trial-audit-text",
        FAILURE_AUDIT_ROLES[0],
        "algorithm_design",
        "  Audit   Person  ",
        "  inspected   complete trace  ",
    )
    assert audit["reviewer"] == "Audit Person"
    assert audit["note"] == "inspected complete trace"


def test_hardness_status_fails_closed_on_trial_ledger_drift(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "trial-drift.json"
    seeds = list(range(5))
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", seeds, manifest_path
    )
    _record_failure_trials(workspace, candidate_id, manifest_path, seeds)
    ready = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert ready["conclusion"] == "pilot_hardness_signal_ready"
    assert ready["authorization_active"] is True

    state_path = workspace / "review_state.json"
    state = json.loads(state_path.read_text())
    state["candidates"][candidate_id]["model_trials"]["trial-0"]["passed"] = True
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    drifted = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert drifted["authorization_active"] is False
    assert drifted["conclusion"] == "evidence_drift_or_gate_revoked"
    assert "trial_ledger_changed:trial-0" in drifted["evidence_violations"]
    assert "trial_pass_derivation_changed:trial-0" in drifted["evidence_violations"]
