"""Build and gate a source-grounded pool of future ORBIT-Q task concepts.

The screening score produced here is a design prior.  It is deliberately kept
separate from empirical model results: no candidate can be called "unsolved"
until an expert baseline, verifier-only smoke test, human pilot approval, and
repeated frozen-model trials have all happened.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCORE_NAMES = (
    "agent_difficulty_prior",
    "tensorcircuit_path_confidence",
    "scientific_value",
    "verifier_feasibility",
    "runtime_fit",
    "determinism",
    "novelty",
    "anti_shortcut",
)

SCORE_WEIGHTS = {
    "agent_difficulty_prior": 0.22,
    "tensorcircuit_path_confidence": 0.16,
    "scientific_value": 0.12,
    "verifier_feasibility": 0.16,
    "runtime_fit": 0.10,
    "determinism": 0.08,
    "novelty": 0.10,
    "anti_shortcut": 0.06,
}

REQUIRED_PROTOCOL_CONFIG_FIELDS = (
    "model",
    "provider_snapshot",
    "reasoning_effort",
    "token_budget",
    "wall_time_sec",
    "solver_agent_version",
    "harbor_version",
    "audit_model",
    "tools_policy",
    "network_policy",
    "hardware_class",
)

CONCEPT_REVIEW_ROLES = (
    "quantum_scientist",
    "tensorcircuit_expert",
    "verifier_engineer",
)
PILOT_REVIEW_ROLES = ("benchmark_owner", "independent_reviewer")
FAILURE_AUDIT_ROLES = ("failure_auditor_primary", "failure_auditor_secondary")
TRIAL_EXECUTION_STATUSES = (
    "completed",
    "agent_timeout",
    "verifier_timeout",
    "infrastructure_error",
    "authentication_error",
)
SUBSTANTIVE_FAILURE_CLASSES = (
    "quantum_reasoning",
    "algorithm_design",
    "tensorcircuit_implementation",
    "numerical_method",
    "resource_strategy",
)
EXCLUDED_FAILURE_CLASSES = (
    "infrastructure",
    "authentication",
    "policy_refusal",
    "ambiguous_specification",
    "framework_impossibility",
    "evaluator_defect",
    "indeterminate",
)
REQUIRED_PROTOTYPE_FIELDS = (
    "evidence_bundle_path",
    "evidence_bundle_sha256",
    "expert_baseline_path",
    "expert_baseline_sha256",
    "independent_oracle_path",
    "independent_oracle_sha256",
    "evaluator_path",
    "evaluator_sha256",
    "task_bundle_path",
    "task_bundle_sha256",
    "framework_prompt_path",
    "container_image_digest",
    "framework_prompt_sha256",
    "source_commit",
    "artifact_authors",
    "independent_artifact_authors",
    "public_api_canary_passed",
    "public_api_canary_log_sha256",
    "expert_runtime_p95_sec",
    "independent_oracle_runtime_sec",
    "gold_effective_lines",
    "reproducibility_runs",
    "observed_cpu_count",
    "observed_memory_mb",
    "expert_peak_memory_mb",
    "verifier_only_passed",
    "valid_alternatives_accepted",
    "mutation_tests_rejected",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _normalized_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty identity")
    return " ".join(value.split())


def _identity_key(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _append_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    event_log = state.setdefault("event_log", [])
    previous = event_log[-1].get("event_hash", "0" * 64) if event_log else "0" * 64
    chained = {**event, "previous_event_hash": previous}
    chained["event_hash"] = _canonical_hash(chained, length=64)
    event_log.append(chained)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(row for row in path.rglob("*") if row.is_file())
    if not files:
        raise ValueError(f"evidence directory is empty: {path}")
    for file_path in files:
        if file_path.is_symlink():
            raise ValueError(
                f"symlinks are not allowed in evidence bundles: {file_path}"
            )
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(file_path)))
    return digest.hexdigest()


def _repository_root(root: Path) -> Path:
    if root.parent.name == "reports":
        return root.parents[1]
    return root.parent


def _safe_manifest_output(root: Path, output: Path) -> Path:
    allowed_root = (
        _repository_root(root) / ".artifacts" / "problem-discovery"
    ).resolve()
    resolved = output.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(f"run manifests must stay under {allowed_root}")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite an existing manifest: {resolved}")
    return resolved


def _safe_evidence_path(root: Path, base: Path, value: str) -> Path:
    allowed_root = (
        _repository_root(root) / ".artifacts" / "problem-discovery"
    ).resolve()
    path = Path(value)
    resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(f"evidence must stay under {allowed_root}: {resolved}")
    return resolved


def _verified_evidence_item(
    root: Path, base: Path, item: dict[str, Any], label: str
) -> tuple[Path, str]:
    if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
        raise ValueError(f"{label} needs path and sha256")
    path = _safe_evidence_path(root, base, item["path"])
    actual = _path_sha256(path)
    if actual != item["sha256"]:
        raise ValueError(f"{label} SHA-256 does not match {path}")
    return path, actual


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _effective_python_lines(path: Path) -> int:
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    if not files or any(file_path.suffix != ".py" for file_path in files):
        raise ValueError(
            "expert baseline must be a Python file or Python-only directory"
        )
    return sum(
        1
        for file_path in files
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _parse_prototype_verifier_result(path: Path, image_digest: str) -> dict[str, Any]:
    result = _read_json_object(path, "prototype verifier result")
    if result.get("finished_at") in (None, ""):
        raise ValueError("prototype verifier result is not finished")
    job_id = _normalized_identity(result.get("id"), "prototype verifier job id")
    rewards = result.get("verifier_result", {}).get("rewards")
    if not isinstance(rewards, dict):
        raise ValueError("prototype verifier result is missing rewards")
    scores: dict[str, float] = {}
    for name in (
        "reward",
        "functional_score",
        "static_policy_score",
        "llm_audit_score",
    ):
        value = rewards.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"prototype verifier {name} must be numeric in [0, 1]")
        scores[name] = float(value)
    expected = (
        scores["functional_score"]
        * scores["static_policy_score"]
        * scores["llm_audit_score"]
    )
    if abs(scores["reward"] - expected) > 1e-9 or any(
        scores[name] <= 0
        for name in (
            "reward",
            "functional_score",
            "static_policy_score",
            "llm_audit_score",
        )
    ):
        raise ValueError("prototype verifier result is not a compound pass")
    runtime = rewards.get("runtime_sec")
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or runtime <= 0
    ):
        raise ValueError("prototype verifier runtime must be positive")
    attestation = result.get("orbit_q_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("prototype verifier result needs orbit_q_attestation")
    if attestation.get("attestation_type") != "operator_attested":
        raise ValueError("prototype verifier evidence must be operator-attested")
    _normalized_identity(attestation.get("attestor_id"), "prototype attestor")
    if attestation.get("container_image_digest") != image_digest:
        raise ValueError("prototype verifier image digest does not match")
    peak_memory = attestation.get("peak_memory_mb")
    if (
        not isinstance(peak_memory, (int, float))
        or isinstance(peak_memory, bool)
        or peak_memory <= 0
    ):
        raise ValueError("prototype verifier peak memory must be positive")
    return {
        "job_id": job_id,
        "runtime_sec": float(runtime),
        "peak_memory_mb": float(peak_memory),
    }


def _clamp_score(value: int) -> int:
    return max(1, min(5, value))


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    """Return catalog invariant violations without mutating the input."""
    errors: list[str] = []
    sources = catalog.get("sources", [])
    source_ids = [row.get("id") for row in sources]
    if len(source_ids) != len(set(source_ids)):
        errors.append("source ids must be unique")
    for source in sources:
        if not source.get("url") or not source.get("title"):
            errors.append(f"source {source.get('id')} needs title and url")
        if source.get("evidence_class") not in {
            "official",
            "literature",
            "survey_inference",
        }:
            errors.append(f"source {source.get('id')} has invalid evidence class")

    families = catalog.get("families", [])
    family_ids = [row.get("id") for row in families]
    if len(families) != 50:
        errors.append("catalog must contain exactly 50 source-grounded families")
    if len(family_ids) != len(set(family_ids)):
        errors.append("family ids must be unique")
    source_id_set = set(source_ids)
    priorities: list[int] = []
    for family in families:
        missing = [name for name in SCORE_NAMES if name not in family.get("scores", {})]
        if missing:
            errors.append(f"family {family.get('id')} missing scores: {missing}")
        for name, value in family.get("scores", {}).items():
            if name in SCORE_NAMES and (
                not isinstance(value, int) or not 1 <= value <= 5
            ):
                errors.append(f"family {family.get('id')} invalid {name}: {value}")
        unknown_sources = set(family.get("source_ids", [])) - source_id_set
        if unknown_sources:
            errors.append(
                f"family {family.get('id')} references unknown sources: "
                f"{sorted(unknown_sources)}"
            )
        role_sources = {
            source_id
            for values in family.get("source_roles", {}).values()
            for source_id in values
        }
        if role_sources - set(family.get("source_ids", [])):
            errors.append(
                f"family {family.get('id')} source roles must reference source_ids"
            )
        priority = family.get("shortlist_priority")
        if priority is not None:
            priorities.append(priority)
            if not family.get("recommended_scale") or not family.get(
                "recommended_situation"
            ):
                errors.append(
                    f"shortlisted family {family.get('id')} needs a recommended regime"
                )

    if sorted(priorities) != list(range(1, 11)):
        errors.append("shortlist priorities must be exactly 1 through 10")

    scales = catalog.get("scale_regimes", [])
    situations = catalog.get("situations", [])
    if len(scales) != 4 or len(situations) != 5:
        errors.append("catalog needs four scales and five situations")
    if len({row.get("id") for row in scales}) != len(scales):
        errors.append("scale ids must be unique")
    if len({row.get("id") for row in situations}) != len(situations):
        errors.append("situation ids must be unique")
    for axis in (*scales, *situations):
        unknown = set(axis.get("score_deltas", {})) - set(SCORE_NAMES)
        if unknown:
            errors.append(f"axis {axis.get('id')} has unknown score deltas: {unknown}")

    if len(families) * len(scales) * len(situations) != 1000:
        errors.append("family x scale x situation matrix must contain 1,000 candidates")
    return list(dict.fromkeys(errors))


def _candidate_status(scores: dict[str, int], design_score: float) -> str:
    if scores["tensorcircuit_path_confidence"] < 3:
        return "hold_framework_obstruction"
    if (
        min(
            scores["verifier_feasibility"],
            scores["runtime_fit"],
            scores["determinism"],
        )
        < 3
    ):
        return "hold_feasibility"
    if scores["agent_difficulty_prior"] >= 4 and design_score >= 76:
        return "shortlist_eligible"
    return "sampled"


def generate_candidates(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand 50 families across four scales and five situations."""
    errors = validate_catalog(catalog)
    if errors:
        raise ValueError("invalid discovery catalog: " + "; ".join(errors))

    candidates: list[dict[str, Any]] = []
    source_index = {row["id"]: row for row in catalog["sources"]}
    for family in catalog["families"]:
        source_snapshot_hash = _canonical_hash(
            [source_index[source_id] for source_id in family["source_ids"]],
            length=64,
        )
        for scale in catalog["scale_regimes"]:
            for situation in catalog["situations"]:
                scores = dict(family["scores"])
                for axis in (scale, situation):
                    for name, delta in axis.get("score_deltas", {}).items():
                        scores[name] = _clamp_score(scores[name] + delta)
                design_score = round(
                    sum(scores[name] * SCORE_WEIGHTS[name] for name in SCORE_NAMES)
                    * 20,
                    1,
                )
                candidate_id = f"{family['id']}--{scale['id']}--{situation['id']}"
                core = {
                    "id": candidate_id,
                    "family_id": family["id"],
                    "title": family["title"],
                    "domain": family["domain"],
                    "mechanism": family["mechanism"],
                    "objective": family["objective"],
                    "scale": scale,
                    "situation": situation,
                    "source_ids": family["source_ids"],
                    "source_roles": family.get(
                        "source_roles", {"scientific_basis": family["source_ids"]}
                    ),
                    "source_snapshot_hash": source_snapshot_hash,
                    "tensorcircuit_path": family["tensorcircuit_path"],
                    "oracle_design": family["oracle_design"],
                    "novelty_note": family["novelty_note"],
                    "current_task_overlap": family.get("current_task_overlap", []),
                    "scores": scores,
                    "design_score": design_score,
                }
                candidate = {
                    **core,
                    "candidate_hash": _canonical_hash(core, length=64),
                    "screen_status": _candidate_status(scores, design_score),
                    "empirical_status": "untested",
                    "empirical_pass_rate": None,
                    "risk_flags": {
                        "framework_obstruction": 6
                        - scores["tensorcircuit_path_confidence"],
                        "oracle_weakness": 6 - scores["verifier_feasibility"],
                        "runtime_overflow": 6 - scores["runtime_fit"],
                        "stochastic_flake": 6 - scores["determinism"],
                    },
                    "evidence_class": "survey_inference",
                }
                candidates.append(candidate)
    if len(candidates) != 1000 or len({row["id"] for row in candidates}) != 1000:
        raise AssertionError("candidate generation must yield 1,000 unique rows")
    return candidates


def select_shortlist(
    catalog: dict[str, Any], candidates: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Choose one human-designed regime for each of ten diverse top families."""
    index = {row["id"]: row for row in candidates}
    selected: list[dict[str, Any]] = []
    priority_families = sorted(
        (row for row in catalog["families"] if row.get("shortlist_priority")),
        key=lambda row: row["shortlist_priority"],
    )
    for family in priority_families:
        candidate_id = (
            f"{family['id']}--{family['recommended_scale']}--"
            f"{family['recommended_situation']}"
        )
        candidate = dict(index[candidate_id])
        if candidate["screen_status"] != "shortlist_eligible":
            raise ValueError(f"preferred candidate is not eligible: {candidate_id}")
        candidate.update(
            {
                "rank": family["shortlist_priority"],
                "why_plausibly_hard": family["why_plausibly_hard"],
                "falsification_check": family["falsification_check"],
                "review_questions": family["review_questions"],
            }
        )
        candidate["screen_candidate_hash"] = candidate.pop("candidate_hash")
        candidate["candidate_hash"] = _canonical_hash(candidate, length=64)
        selected.append(candidate)
    if len(selected) != 10:
        raise AssertionError("shortlist must contain exactly ten candidates")
    return selected


def _verified_shortlist_candidate(root: Path, candidate_id: str) -> dict[str, Any]:
    candidates = _read_json(root / "shortlist.json").get("candidates", [])
    candidate = next((row for row in candidates if row.get("id") == candidate_id), None)
    if candidate is None:
        raise KeyError(f"candidate is not in the current shortlist: {candidate_id}")
    recorded_hash = candidate.get("candidate_hash")
    unsigned = {
        key: value for key, value in candidate.items() if key != "candidate_hash"
    }
    if recorded_hash != _canonical_hash(unsigned, length=64):
        raise ValueError("shortlist candidate hash does not match its reviewed content")
    return candidate


def _summary(
    catalog: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    shortlist = select_shortlist(catalog, candidates)
    return {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "family_count": len(catalog["families"]),
        "source_count": len(catalog["sources"]),
        "shortlist_count": len(shortlist),
        "empirically_tested_count": 0,
        "evidence_classes": ["official", "literature", "survey_inference"],
        "domains": dict(sorted(Counter(row["domain"] for row in candidates).items())),
        "screen_statuses": dict(
            sorted(Counter(row["screen_status"] for row in candidates).items())
        ),
        "score_weights": SCORE_WEIGHTS,
        "catalog_source_snapshot_hash": _canonical_hash(catalog["sources"], length=64),
        "shortlist_snapshot": _canonical_hash(
            [(row["id"], row["candidate_hash"]) for row in shortlist], length=64
        ),
        "claim_boundary": (
            "Scores are design priors, not measured GPT-5.6-sol outcomes. "
            "No real solver trials were run during discovery."
        ),
    }


def _shortlist_markdown(
    catalog: dict[str, Any], shortlist: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    source_index = {row["id"]: row for row in catalog["sources"]}
    lines = [
        "# ORBIT-Q future-problem shortlist",
        "",
        "> **Claim boundary:** This is a source-grounded design screen. The scores are",
        "> priors, not evidence that GPT-5.6-sol failed. Real model tests remain blocked",
        "> until the human and expert-baseline gates pass.",
        "",
        "## Screening result",
        "",
        f"The deterministic matrix contains **{summary['candidate_count']:,} candidates**",
        f"from **{summary['family_count']} families**, four scale regimes, and five",
        "situations. Ten candidates were selected for design review; none has been",
        "empirically tested.",
        "",
        "| Rank | Candidate | Domain | Design prior | TC path | Verifier |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in shortlist:
        scores = row["scores"]
        lines.append(
            f"| {row['rank']} | `{row['id']}` — {row['title']} | {row['domain']} | "
            f"{row['design_score']:.1f}/100 | "
            f"{scores['tensorcircuit_path_confidence']}/5 | "
            f"{scores['verifier_feasibility']}/5 |"
        )

    for row in shortlist:
        lines.extend(
            [
                "",
                f"## {row['rank']}. {row['title']}",
                "",
                f"- Candidate: `{row['id']}` (`{row['candidate_hash']}`)",
                f"- Contract: {row['objective']} {row['scale']['description']} "
                f"{row['situation']['description']}",
                f"- TensorCircuit path: {row['tensorcircuit_path']}",
                f"- Oracle: {row['oracle_design']}",
                f"- Why plausibly hard: {row['why_plausibly_hard']}",
                f"- Falsification check: {row['falsification_check']}",
                f"- Current-suite relation: {row['novelty_note']}",
                "- Human review questions: " + "; ".join(row["review_questions"]),
                "- Evidence roles: "
                + "; ".join(
                    f"{role}="
                    + ", ".join(source_index[source_id]["title"] for source_id in ids)
                    for role, ids in row["source_roles"].items()
                ),
                "- Sources: "
                + ", ".join(
                    f"[{source_index[source_id]['title']}]"
                    f"({source_index[source_id]['url']})"
                    for source_id in row["source_ids"]
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Human-gated path to a real test",
            "",
            "1. Three concept roles approve: quantum scientist, TensorCircuit expert, and verifier engineer.",
            "2. An expert implementation and independent evaluator are built outside `tasks/`.",
            "3. Public-API canary and 20 verifier-only reproductions pass in the pinned image.",
            "4. Gold code is at most 160 effective lines; expert p95 is at most 180 seconds; the independent oracle is at most 240 seconds.",
            "5. Benchmark owner and independent reviewer approve the frozen pilot package.",
            "6. The CLI emits a hash-bound run manifest. It never launches Harbor itself.",
            "7. Run a precommitted five-trial GPT-5.6-sol pilot under one frozen protocol.",
            "8. Two independent auditors classify every failure and exclude infrastructure, auth, policy refusal, ambiguous spec, and framework impossibility.",
            "9. A model-hard evidence label requires a fresh precommitted confirmation with at least 20 valid trials and zero passes.",
            "",
            "Five failures are only a pilot signal. Only failures that remain after step 9 count toward a protocol-scoped model-hard evidence label.",
            "",
        ]
    )
    return "\n".join(lines)


def _initial_review_state(
    shortlist: list[dict[str, Any]], existing: dict[str, Any] | None
) -> dict[str, Any]:
    state = existing or {
        "schema_version": 1,
        "candidates": {},
        "event_log": [],
    }
    state.pop("source_snapshot_hash", None)
    active_ids = {row["id"] for row in shortlist}
    for row in shortlist:
        fresh_record = {
            "candidate_hash": row["candidate_hash"],
            "concept_reviews": {},
            "prototype": {},
            "pilot_reviews": {},
            "authorizations": [],
            "model_trials": {},
        }
        record = state["candidates"].setdefault(row["id"], fresh_record)
        record.setdefault("authorizations", [])
        record.setdefault("model_trials", {})
        if record.get("candidate_hash") != row["candidate_hash"]:
            has_evidence = any(
                record.get(name)
                for name in (
                    "concept_reviews",
                    "prototype",
                    "pilot_reviews",
                    "authorizations",
                    "model_trials",
                )
            )
            if has_evidence:
                archived = {
                    **record,
                    "invalidated_at": _utc_now(),
                    "superseded_by_candidate_hash": row["candidate_hash"],
                }
                state.setdefault("archived_candidate_versions", {}).setdefault(
                    row["id"], []
                ).append(archived)
                _append_event(
                    state,
                    {
                        "event": "candidate_version_invalidated",
                        "candidate_id": row["id"],
                        "prior_candidate_hash": record.get("candidate_hash"),
                        "candidate_hash": row["candidate_hash"],
                        "timestamp": archived["invalidated_at"],
                    },
                )
            state["candidates"][row["id"]] = fresh_record
    for candidate_id in list(state["candidates"]):
        if candidate_id in active_ids:
            continue
        record = state["candidates"][candidate_id]
        has_evidence = any(
            record.get(name)
            for name in (
                "concept_reviews",
                "prototype",
                "pilot_reviews",
                "authorizations",
                "model_trials",
            )
        )
        if has_evidence:
            archived_at = _utc_now()
            state.setdefault("archived_candidate_versions", {}).setdefault(
                candidate_id, []
            ).append({**record, "invalidated_at": archived_at})
            _append_event(
                state,
                {
                    "event": "candidate_removed_from_shortlist",
                    "candidate_id": candidate_id,
                    "candidate_hash": record.get("candidate_hash"),
                    "timestamp": archived_at,
                },
            )
        del state["candidates"][candidate_id]
    state["active_shortlist_ids"] = sorted(active_ids)
    state["shortlist_source_snapshot_hash"] = _canonical_hash(
        sorted({row["source_snapshot_hash"] for row in shortlist}), length=64
    )
    state["shortlist_snapshot"] = _canonical_hash(
        [(row["id"], row["candidate_hash"]) for row in shortlist], length=64
    )
    return state


def build_workspace(root: Path) -> dict[str, Any]:
    """Build deterministic artifacts and preserve any existing human reviews."""
    catalog = _read_json(root / "catalog.json")
    candidates = generate_candidates(catalog)
    shortlist = select_shortlist(catalog, candidates)
    summary = _summary(catalog, candidates)

    (root / "candidates.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in candidates
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "shortlist.json",
        {
            "schema_version": 1,
            "claim_boundary": summary["claim_boundary"],
            "candidates": shortlist,
        },
    )
    _write_json(root / "summary.json", summary)
    (root / "shortlist.md").write_text(
        _shortlist_markdown(catalog, shortlist, summary), encoding="utf-8"
    )

    review_path = root / "review_state.json"
    existing = _read_json(review_path) if review_path.exists() else None
    _write_json(review_path, _initial_review_state(shortlist, existing))
    return summary


def _decision_status(reviews: dict[str, Any], roles: Iterable[str]) -> dict[str, Any]:
    roles = tuple(roles)
    missing = [role for role in roles if role not in reviews]
    rejected = [
        role for role in roles if reviews.get(role, {}).get("decision") == "reject"
    ]
    approved = [
        role for role in roles if reviews.get(role, {}).get("decision") == "approve"
    ]
    return {
        "passed": not missing and not rejected and len(approved) == len(roles),
        "missing_roles": missing,
        "rejected_roles": rejected,
        "approved_roles": approved,
    }


def _gate_fingerprint(record: dict[str, Any]) -> str:
    return _canonical_hash(
        {
            "candidate_hash": record["candidate_hash"],
            "concept_reviews": record.get("concept_reviews", {}),
            "prototype": record.get("prototype", {}),
            "pilot_reviews": record.get("pilot_reviews", {}),
        },
        length=64,
    )


def _reviewer_identities(record: dict[str, Any]) -> set[str]:
    identities = {
        review.get("reviewer")
        for group in ("concept_reviews", "pilot_reviews")
        for review in record.get(group, {}).values()
    }
    identities.update(record.get("prototype", {}).get("artifact_authors", []))
    return {_identity_key(identity) for identity in identities if identity}


def candidate_gate_status(state: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    record = state.get("candidates", {}).get(candidate_id)
    if record is None:
        raise KeyError(f"candidate is not in review state: {candidate_id}")
    concept = _decision_status(record.get("concept_reviews", {}), CONCEPT_REVIEW_ROLES)
    prototype = record.get("prototype", {})
    prototype_missing = [
        name
        for name in REQUIRED_PROTOTYPE_FIELDS
        if name not in prototype or prototype[name] in (None, "")
    ]

    def in_range(name: str, lower: float, upper: float) -> bool:
        value = prototype.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and lower <= value <= upper
        )

    def is_hex(value: Any, lengths: tuple[int, ...]) -> bool:
        return (
            isinstance(value, str)
            and len(value) in lengths
            and all(character in "0123456789abcdef" for character in value.lower())
        )

    expert_runtime_ok = in_range("expert_runtime_p95_sec", 0.000001, 180)
    oracle_runtime_ok = in_range("independent_oracle_runtime_sec", 0.000001, 240)
    line_count_ok = in_range("gold_effective_lines", 1, 160)
    reproducibility_ok = in_range("reproducibility_runs", 20, 10_000)
    resource_envelope_ok = (
        in_range("observed_cpu_count", 1, 10_000)
        and in_range("observed_memory_mb", 1, 10_000_000)
        and in_range("expert_peak_memory_mb", 0.000001, 10_000_000)
        and prototype.get("expert_peak_memory_mb", float("inf"))
        <= 0.70 * prototype.get("observed_memory_mb", 0)
    )
    public_api_ok = prototype.get("public_api_canary_passed") is True
    verifier_ok = prototype.get("verifier_only_passed") is True
    artifact_hashes_ok = all(
        is_hex(prototype.get(name), (64,))
        for name in (
            "expert_baseline_sha256",
            "independent_oracle_sha256",
            "evaluator_sha256",
            "task_bundle_sha256",
            "framework_prompt_sha256",
            "evidence_bundle_sha256",
        )
    )
    artifacts_still_match = True
    for path_name, hash_name in (
        ("expert_baseline_path", "expert_baseline_sha256"),
        ("independent_oracle_path", "independent_oracle_sha256"),
        ("evaluator_path", "evaluator_sha256"),
        ("task_bundle_path", "task_bundle_sha256"),
        ("framework_prompt_path", "framework_prompt_sha256"),
        ("evidence_bundle_path", "evidence_bundle_sha256"),
    ):
        try:
            artifacts_still_match = artifacts_still_match and (
                _path_sha256(Path(prototype[path_name])) == prototype[hash_name]
            )
        except (KeyError, FileNotFoundError, ValueError, OSError):
            artifacts_still_match = False
    supporting_evidence_still_matches = False
    try:
        evidence_bundle_path = Path(prototype["evidence_bundle_path"])
        evidence_bundle = _read_json(evidence_bundle_path)
        evidence_base = evidence_bundle_path.parent

        def bundle_item_matches(item: dict[str, Any]) -> bool:
            item_path = Path(item["path"])
            if not item_path.is_absolute():
                item_path = evidence_base / item_path
            return _path_sha256(item_path) == item["sha256"]

        supporting_items = [evidence_bundle["public_api_canary"]["log"]]
        supporting_items.extend(run["log"] for run in evidence_bundle["verifier_runs"])
        supporting_items.extend(
            run["log"] for run in evidence_bundle["independent_oracle_runs"]
        )
        supporting_items.extend(
            run["log"] for run in evidence_bundle["valid_alternatives"]
        )
        supporting_items.extend(run["log"] for run in evidence_bundle["mutation_tests"])
        supporting_evidence_still_matches = all(
            bundle_item_matches(item) for item in supporting_items
        )
    except (
        KeyError,
        TypeError,
        FileNotFoundError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ):
        supporting_evidence_still_matches = False
    independent_artifacts_ok = (
        len(
            {
                _identity_key(author)
                for author in prototype.get("independent_artifact_authors", [])
            }
        )
        == 3
        and len(
            {
                prototype.get("expert_baseline_sha256"),
                prototype.get("independent_oracle_sha256"),
                prototype.get("evaluator_sha256"),
            }
        )
        == 3
    )
    verifier_challenge_ok = in_range(
        "valid_alternatives_accepted", 2, 10_000
    ) and in_range("mutation_tests_rejected", 6, 10_000)
    image_digest = prototype.get("container_image_digest", "")
    image_digest_ok = (
        isinstance(image_digest, str)
        and image_digest.startswith("sha256:")
        and is_hex(image_digest.removeprefix("sha256:"), (64,))
    )
    source_commit = prototype.get("source_commit")
    source_commit_ok = (
        isinstance(source_commit, str)
        and 7 <= len(source_commit) <= 64
        and all(character in "0123456789abcdef" for character in source_commit.lower())
    )
    prototype_passed = not prototype_missing and all(
        (
            public_api_ok,
            verifier_ok,
            artifact_hashes_ok,
            artifacts_still_match,
            supporting_evidence_still_matches,
            independent_artifacts_ok,
            verifier_challenge_ok,
            image_digest_ok,
            source_commit_ok,
            expert_runtime_ok,
            oracle_runtime_ok,
            line_count_ok,
            reproducibility_ok,
            resource_envelope_ok,
        )
    )
    pilot = _decision_status(record.get("pilot_reviews", {}), PILOT_REVIEW_ROLES)
    return {
        "candidate_id": candidate_id,
        "concept_gate": concept,
        "prototype_gate": {
            "passed": prototype_passed,
            "missing_fields": prototype_missing,
            "public_api_canary_passed": public_api_ok,
            "artifact_hashes_valid": artifact_hashes_ok,
            "artifacts_still_match": artifacts_still_match,
            "supporting_evidence_still_matches": supporting_evidence_still_matches,
            "independent_artifacts_valid": independent_artifacts_ok,
            "verifier_challenge_passed": verifier_challenge_ok,
            "container_image_digest_valid": image_digest_ok,
            "source_commit_valid": source_commit_ok,
            "expert_p95_within_180s": expert_runtime_ok,
            "independent_oracle_within_240s": oracle_runtime_ok,
            "gold_within_160_effective_lines": line_count_ok,
            "at_least_20_reproducibility_runs": reproducibility_ok,
            "resource_envelope_valid": resource_envelope_ok,
            "verifier_only_passed": verifier_ok,
        },
        "pilot_gate": pilot,
        "ready_for_model_test": concept["passed"]
        and prototype_passed
        and pilot["passed"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def record_review(
    root: Path,
    candidate_id: str,
    gate: str,
    role: str,
    decision: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    roles = CONCEPT_REVIEW_ROLES if gate == "concept" else PILOT_REVIEW_ROLES
    if gate not in {"concept", "pilot"} or role not in roles:
        raise ValueError(f"invalid {gate} review role: {role}")
    reviewer = _normalized_identity(reviewer, "reviewer")
    note = _normalized_identity(note, "review note")
    path = root / "review_state.json"
    state = _read_json(path)
    record = state["candidates"][candidate_id]
    key = "concept_reviews" if gate == "concept" else "pilot_reviews"
    if role in record[key]:
        raise ValueError(
            "reviews are append-only; revise the candidate to review again"
        )
    if _identity_key(reviewer) in _reviewer_identities(record):
        raise ValueError("each review role requires a distinct reviewer identity")
    if role == "independent_reviewer" and _identity_key(reviewer) in {
        _identity_key(author)
        for author in record.get("prototype", {}).get("artifact_authors", [])
    }:
        raise ValueError("the independent reviewer cannot author a prototype artifact")
    review = {
        "decision": decision,
        "reviewer": reviewer,
        "note": note,
        "timestamp": _utc_now(),
    }
    record[key][role] = review
    _append_event(
        state,
        {
            "event": "review",
            "candidate_id": candidate_id,
            "gate": gate,
            "role": role,
            **review,
        },
    )
    _write_json(path, state)
    return candidate_gate_status(state, candidate_id)


def record_prototype(
    root: Path, candidate_id: str, evidence_bundle_path: Path
) -> dict[str, Any]:
    """Verify a raw prototype bundle and record only derived gate evidence."""
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    record = state["candidates"][candidate_id]
    if record.get("prototype"):
        raise ValueError(
            "prototype evidence is append-only; revise the candidate to replace it"
        )
    concept_status = _decision_status(
        record.get("concept_reviews", {}), CONCEPT_REVIEW_ROLES
    )
    if not concept_status["passed"]:
        raise PermissionError("concept reviews must pass before prototype evidence")

    bundle_path = _safe_evidence_path(
        root, evidence_bundle_path.parent, str(evidence_bundle_path)
    )
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("prototype evidence bundle must use schema_version=1")
    if bundle.get("candidate_id") != candidate_id:
        raise ValueError("prototype bundle candidate_id does not match")
    if bundle.get("candidate_hash") != record["candidate_hash"]:
        raise ValueError("prototype bundle candidate hash is stale")
    base = bundle_path.parent

    artifact_records = bundle.get("artifacts", {})
    artifact_names = (
        "expert_baseline",
        "independent_oracle",
        "evaluator",
        "task_bundle",
        "framework_prompt",
    )
    artifacts: dict[str, dict[str, str]] = {}
    for name in artifact_names:
        item = artifact_records.get(name, {})
        artifact_path, artifact_hash = _verified_evidence_item(root, base, item, name)
        author = _normalized_identity(item.get("author"), f"{name} author")
        artifacts[name] = {
            "path": str(artifact_path),
            "sha256": artifact_hash,
            "author": author,
        }
    independent_names = ("expert_baseline", "independent_oracle", "evaluator")
    independent_authors = [artifacts[name]["author"] for name in independent_names]
    independent_hashes = [artifacts[name]["sha256"] for name in independent_names]
    if len({_identity_key(author) for author in independent_authors}) != len(
        independent_authors
    ):
        raise ValueError("baseline, oracle, and evaluator require distinct authors")
    if len(set(independent_hashes)) != len(independent_hashes):
        raise ValueError("baseline, oracle, and evaluator must be distinct artifacts")

    environment = bundle.get("environment", {})
    image_digest = environment.get("container_image_digest", "")
    if not (
        isinstance(image_digest, str)
        and image_digest.startswith("sha256:")
        and len(image_digest.removeprefix("sha256:")) == 64
        and all(
            character in "0123456789abcdef"
            for character in image_digest.removeprefix("sha256:").lower()
        )
    ):
        raise ValueError("environment needs a full container image SHA-256 digest")
    source_commit = environment.get("source_commit")
    if not (
        isinstance(source_commit, str)
        and 7 <= len(source_commit) <= 64
        and all(character in "0123456789abcdef" for character in source_commit.lower())
    ):
        raise ValueError("environment source_commit must be hexadecimal")
    observed_cpu_count = environment.get("observed_cpu_count")
    observed_memory_mb = environment.get("observed_memory_mb")
    if (
        not isinstance(observed_cpu_count, int)
        or isinstance(observed_cpu_count, bool)
        or observed_cpu_count < 1
    ):
        raise ValueError("observed_cpu_count must be a positive integer")
    if (
        not isinstance(observed_memory_mb, (int, float))
        or isinstance(observed_memory_mb, bool)
        or observed_memory_mb <= 0
    ):
        raise ValueError("observed_memory_mb must be positive")

    canary = bundle.get("public_api_canary", {})
    if canary.get("passed") is not True:
        raise ValueError("public API canary did not pass")
    canary_path, canary_log_hash = _verified_evidence_item(
        root, base, canary.get("log", {}), "public API canary log"
    )
    canary_result = _read_json_object(canary_path, "public API canary result")
    if (
        canary_result.get("passed") is not True
        or canary_result.get("container_image_digest") != image_digest
        or canary_result.get("source_commit") != source_commit
    ):
        raise ValueError(
            "public API canary result does not match the frozen environment"
        )

    verifier_runs = bundle.get("verifier_runs", [])
    if not isinstance(verifier_runs, list) or len(verifier_runs) < 20:
        raise ValueError("prototype bundle needs at least 20 verifier-only runs")
    verifier_job_ids: list[str] = []
    verifier_log_hashes: list[str] = []
    runtimes: list[float] = []
    peak_memories: list[float] = []
    verifier_records: list[dict[str, Any]] = []
    for index, run in enumerate(verifier_runs):
        log_path, log_hash = _verified_evidence_item(
            root, base, run.get("log", {}), f"verifier run {index} log"
        )
        derived = _parse_prototype_verifier_result(log_path, image_digest)
        declared = {
            "job_id": run.get("job_id"),
            "runtime_sec": float(run["runtime_sec"])
            if isinstance(run.get("runtime_sec"), (int, float))
            and not isinstance(run.get("runtime_sec"), bool)
            else None,
            "peak_memory_mb": float(run["peak_memory_mb"])
            if isinstance(run.get("peak_memory_mb"), (int, float))
            and not isinstance(run.get("peak_memory_mb"), bool)
            else None,
        }
        if (
            run.get("passed") is not True
            or run.get("container_image_digest") != image_digest
            or declared != derived
        ):
            raise ValueError(
                f"verifier run {index} disagrees with its structured result"
            )
        job_id = derived["job_id"]
        runtime = derived["runtime_sec"]
        peak_memory = derived["peak_memory_mb"]
        verifier_job_ids.append(job_id)
        verifier_log_hashes.append(log_hash)
        runtimes.append(float(runtime))
        peak_memories.append(float(peak_memory))
        verifier_records.append(
            {
                "job_id": job_id,
                "runtime_sec": float(runtime),
                "peak_memory_mb": float(peak_memory),
                "log_sha256": log_hash,
            }
        )
    if len(set(verifier_job_ids)) != len(verifier_job_ids):
        raise ValueError("verifier run job IDs must be unique")
    if len(set(verifier_log_hashes)) != len(verifier_log_hashes):
        raise ValueError("verifier run logs must be unique")

    oracle_runs = bundle.get("independent_oracle_runs", [])
    if not isinstance(oracle_runs, list) or len(oracle_runs) < 3:
        raise ValueError(
            "prototype bundle needs at least three independent-oracle runs"
        )
    oracle_runtimes: list[float] = []
    oracle_log_hashes: list[str] = []
    for index, run in enumerate(oracle_runs):
        runtime = run.get("runtime_sec")
        if (
            not isinstance(runtime, (int, float))
            or isinstance(runtime, bool)
            or runtime <= 0
        ):
            raise ValueError(f"oracle run {index} has invalid runtime")
        log_path, log_hash = _verified_evidence_item(
            root, base, run.get("log", {}), f"oracle run {index} log"
        )
        oracle_result = _read_json_object(log_path, f"oracle run {index} result")
        if (
            oracle_result.get("passed") is not True
            or oracle_result.get("runtime_sec") != runtime
            or not isinstance(oracle_result.get("oracle_method"), str)
            or not oracle_result["oracle_method"].strip()
        ):
            raise ValueError(f"oracle run {index} disagrees with its structured result")
        oracle_runtimes.append(float(runtime))
        oracle_log_hashes.append(log_hash)
    if len(set(oracle_log_hashes)) != len(oracle_log_hashes):
        raise ValueError("independent-oracle run logs must be unique")

    gold_effective_lines = bundle.get("gold_effective_lines")
    if not isinstance(gold_effective_lines, int) or isinstance(
        gold_effective_lines, bool
    ):
        raise ValueError("gold_effective_lines must be an integer")
    measured_effective_lines = _effective_python_lines(
        Path(artifacts["expert_baseline"]["path"])
    )
    if measured_effective_lines != gold_effective_lines:
        raise ValueError("gold_effective_lines does not match the expert baseline")
    alternatives = bundle.get("valid_alternatives", [])
    if not isinstance(alternatives, list) or len(alternatives) < 2:
        raise ValueError(
            "at least two independent valid implementation styles are required"
        )
    alternative_hashes: list[str] = []
    alternative_styles: list[str] = []
    for index, alternative in enumerate(alternatives):
        if alternative.get("accepted") is not True:
            raise ValueError(f"valid alternative {index} was not accepted")
        log_path, log_hash = _verified_evidence_item(
            root, base, alternative.get("log", {}), f"valid alternative {index} log"
        )
        result = _read_json_object(log_path, f"valid alternative {index} result")
        style = _normalized_identity(
            result.get("implementation_style"), f"valid alternative {index} style"
        )
        if result.get("accepted") is not True:
            raise ValueError(f"valid alternative {index} result was not accepted")
        alternative_hashes.append(log_hash)
        alternative_styles.append(style.casefold())
    if len(set(alternative_hashes)) != len(alternative_hashes) or len(
        set(alternative_styles)
    ) != len(alternative_styles):
        raise ValueError("valid-alternative logs and styles must be unique")

    mutation_tests = bundle.get("mutation_tests", [])
    if not isinstance(mutation_tests, list) or len(mutation_tests) < 6:
        raise ValueError("at least six verifier mutation tests are required")
    mutation_hashes: list[str] = []
    mutation_types: list[str] = []
    for index, mutation in enumerate(mutation_tests):
        if mutation.get("rejected") is not True:
            raise ValueError(f"verifier mutation {index} escaped rejection")
        log_path, log_hash = _verified_evidence_item(
            root, base, mutation.get("log", {}), f"mutation test {index} log"
        )
        result = _read_json_object(log_path, f"mutation test {index} result")
        mutation_type = _normalized_identity(
            result.get("mutation_type"), f"mutation test {index} type"
        )
        if result.get("rejected") is not True:
            raise ValueError(f"mutation test {index} result escaped rejection")
        mutation_hashes.append(log_hash)
        mutation_types.append(mutation_type.casefold())
    if len(set(mutation_hashes)) != len(mutation_hashes) or len(
        set(mutation_types)
    ) != len(mutation_types):
        raise ValueError("verifier-mutation logs and types must be unique")

    runtime_p95 = sorted(runtimes)[math.ceil(0.95 * len(runtimes)) - 1]
    oracle_runtime_p95 = sorted(oracle_runtimes)[
        math.ceil(0.95 * len(oracle_runtimes)) - 1
    ]
    peak_memory = max(peak_memories)
    if peak_memory > 0.70 * float(observed_memory_mb):
        raise ValueError("expert peak memory exceeds the 70% admission headroom")

    evidence = {
        "evidence_bundle_path": str(bundle_path),
        "evidence_bundle_sha256": _file_sha256(bundle_path),
        "expert_baseline_path": artifacts["expert_baseline"]["path"],
        "expert_baseline_sha256": artifacts["expert_baseline"]["sha256"],
        "independent_oracle_path": artifacts["independent_oracle"]["path"],
        "independent_oracle_sha256": artifacts["independent_oracle"]["sha256"],
        "evaluator_path": artifacts["evaluator"]["path"],
        "evaluator_sha256": artifacts["evaluator"]["sha256"],
        "task_bundle_path": artifacts["task_bundle"]["path"],
        "task_bundle_sha256": artifacts["task_bundle"]["sha256"],
        "framework_prompt_path": artifacts["framework_prompt"]["path"],
        "framework_prompt_sha256": artifacts["framework_prompt"]["sha256"],
        "artifact_authors": sorted(
            {artifacts[name]["author"] for name in artifact_names}, key=str.casefold
        ),
        "independent_artifact_authors": independent_authors,
        "container_image_digest": image_digest,
        "source_commit": source_commit,
        "public_api_canary_passed": True,
        "public_api_canary_log_sha256": canary_log_hash,
        "expert_runtime_p95_sec": runtime_p95,
        "independent_oracle_runtime_sec": oracle_runtime_p95,
        "gold_effective_lines": gold_effective_lines,
        "reproducibility_runs": len(verifier_runs),
        "observed_cpu_count": observed_cpu_count,
        "observed_memory_mb": float(observed_memory_mb),
        "expert_peak_memory_mb": peak_memory,
        "verifier_only_passed": True,
        "valid_alternatives_accepted": len(alternatives),
        "mutation_tests_rejected": len(mutation_tests),
        "verifier_runs": verifier_records,
        "independent_oracle_log_sha256s": oracle_log_hashes,
        "valid_alternative_log_sha256s": alternative_hashes,
        "mutation_test_log_sha256s": mutation_hashes,
        "notes": bundle.get("notes", ""),
    }
    recorded_at = _utc_now()
    record["prototype"] = {**evidence, "recorded_at": recorded_at}
    _append_event(
        state,
        {
            "event": "prototype_evidence",
            "candidate_id": candidate_id,
            "timestamp": recorded_at,
            "evidence_hash": _canonical_hash(evidence, length=64),
        },
    )
    _write_json(state_path, state)
    return candidate_gate_status(state, candidate_id)


def authorize_model_test(
    root: Path,
    candidate_id: str,
    protocol_config_path: Path,
    stage: str,
    seed_schedule: list[int],
    output: Path,
) -> dict[str, Any]:
    if stage not in {"pilot", "confirmation"}:
        raise ValueError("stage must be pilot or confirmation")
    minimum_trials = 5 if stage == "pilot" else 20
    if len(seed_schedule) < minimum_trials:
        raise ValueError(f"{stage} evidence requires at least {minimum_trials} trials")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_schedule
    ) or len(seed_schedule) != len(set(seed_schedule)):
        raise ValueError("the precommitted seed schedule must contain unique integers")
    protocol_config_path = _safe_evidence_path(
        root, protocol_config_path.parent, str(protocol_config_path)
    )
    if not protocol_config_path.is_file():
        raise FileNotFoundError(
            f"protocol config does not exist: {protocol_config_path}"
        )
    protocol_config = _read_json(protocol_config_path)
    missing_protocol = [
        name
        for name in REQUIRED_PROTOCOL_CONFIG_FIELDS
        if protocol_config.get(name) in (None, "")
        or (
            isinstance(protocol_config.get(name), str)
            and protocol_config[name].startswith("<")
        )
    ]
    if missing_protocol:
        raise ValueError(f"protocol config missing required fields: {missing_protocol}")
    for name in ("token_budget", "wall_time_sec"):
        value = protocol_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"protocol {name} must be a positive integer")
    output = _safe_manifest_output(root, output)
    state = _read_json(root / "review_state.json")
    status = candidate_gate_status(state, candidate_id)
    if not status["ready_for_model_test"]:
        raise PermissionError(
            "real model test remains human-gated: " + json.dumps(status, sort_keys=True)
        )
    candidate = _verified_shortlist_candidate(root, candidate_id)
    record = state["candidates"][candidate_id]
    if record["candidate_hash"] != candidate["candidate_hash"]:
        raise PermissionError(
            "candidate content changed after review; re-review is required"
        )
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "candidate_hash": candidate["candidate_hash"],
        "source_snapshot_hash": candidate["source_snapshot_hash"],
        "model": protocol_config["model"],
        "stage": stage,
        "independent_trials": len(seed_schedule),
        "seed_schedule": seed_schedule,
        "protocol_config_path": str(protocol_config_path),
        "protocol_config_sha256": _file_sha256(protocol_config_path),
        "protocol_config": protocol_config,
        "authorized_at": _utc_now(),
        "shortlist_snapshot": state["shortlist_snapshot"],
        "gate_fingerprint": _gate_fingerprint(record),
        "prototype": record["prototype"],
        "concept_reviews": record["concept_reviews"],
        "pilot_reviews": record["pilot_reviews"],
        "execution_policy": {
            "minimum_pilot_trials": 5,
            "minimum_confirmation_trials": 20,
            "frozen_prompt_required": True,
            "frozen_image_digest_required": True,
            "separate_verifier_required": True,
            "human_failure_classification_required": True,
            "excluded_failure_classes": [
                "infrastructure",
                "authentication",
                "policy_refusal",
                "ambiguous_specification",
                "framework_impossibility",
            ],
        },
    }
    manifest["manifest_hash"] = _canonical_hash(manifest, length=64)
    _write_json(output, manifest)
    record.setdefault("authorizations", []).append(
        {
            "manifest_hash": manifest["manifest_hash"],
            "manifest_path": str(output),
            "model": protocol_config["model"],
            "stage": stage,
            "independent_trials": len(seed_schedule),
            "seed_schedule": seed_schedule,
            "protocol_config_sha256": manifest["protocol_config_sha256"],
            "authorized_at": manifest["authorized_at"],
            "gate_fingerprint": manifest["gate_fingerprint"],
        }
    )
    record["active_authorization_hash"] = manifest["manifest_hash"]
    _append_event(
        state,
        {
            "event": "model_test_authorized",
            "candidate_id": candidate_id,
            "manifest_hash": manifest["manifest_hash"],
            "timestamp": manifest["authorized_at"],
        },
    )
    _write_json(root / "review_state.json", state)
    return manifest


def _verified_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    recorded_hash = manifest.get("manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if recorded_hash != _canonical_hash(unsigned, length=64):
        raise ValueError("run manifest hash does not match its contents")
    return manifest


def _harbor_execution_status(result: dict[str, Any]) -> str:
    """Derive a coarse execution status from an immutable Harbor trial result."""
    if result.get("finished_at") in (None, ""):
        raise ValueError("raw Harbor trial is not finished")
    exception = result.get("exception_info")
    if not exception:
        return "completed"
    exception_type = str(exception.get("exception_type", "")).lower()
    message = str(exception.get("exception_message", "")).lower()
    if "timeout" in exception_type and "agent" in exception_type:
        return "agent_timeout"
    if "timeout" in exception_type and "verifier" in exception_type:
        return "verifier_timeout"
    authentication_markers = (
        "authentication",
        "unauthorized",
        "api key",
        "invalid token",
        "usage limit",
        "quota",
        "http 401",
        "http 403",
        " 401 ",
        " 403 ",
    )
    if any(marker in message for marker in authentication_markers):
        return "authentication_error"
    return "infrastructure_error"


def _parse_attested_harbor_result(
    raw_job_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Extract trial facts from Harbor output instead of trusting a typed summary."""
    if not raw_job_path.is_file():
        raise ValueError("raw_job must point to one Harbor trial result.json file")
    result = _read_json(raw_job_path)
    if not isinstance(result, dict):
        raise ValueError("raw Harbor trial result must be a JSON object")
    attestation = result.get("orbit_q_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("raw Harbor result needs an orbit_q_attestation block")
    if attestation.get("attestation_type") != "operator_attested":
        raise ValueError("raw Harbor result must declare operator_attested evidence")
    _normalized_identity(attestation.get("attestor_id"), "Harbor attestor")
    _normalized_identity(attestation.get("attested_at"), "Harbor attestation time")
    if attestation.get("protocol_config") != manifest["protocol_config"]:
        raise ValueError(
            "Harbor attestation does not bind the complete protocol config"
        )
    attested_bindings = {
        "manifest_hash": manifest["manifest_hash"],
        "candidate_hash": manifest["candidate_hash"],
        "protocol_config_sha256": manifest["protocol_config_sha256"],
        "container_image_digest": manifest["prototype"]["container_image_digest"],
        "framework_prompt_sha256": manifest["prototype"]["framework_prompt_sha256"],
        "task_bundle_sha256": manifest["prototype"]["task_bundle_sha256"],
    }
    mismatched = [
        name
        for name, expected in attested_bindings.items()
        if attestation.get(name) != expected
    ]
    if mismatched:
        raise ValueError(
            "raw Harbor attestation does not match frozen manifest fields: "
            f"{mismatched}"
        )
    seed = attestation.get("protocol_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("raw Harbor attestation protocol_seed must be an integer")

    config = result.get("config", {})
    agent_config = config.get("agent", {})
    verifier_config = config.get("verifier", {})
    model = agent_config.get("model_name")
    if model != manifest["model"]:
        raise ValueError("raw Harbor model does not match the frozen manifest")
    reasoning_effort = agent_config.get("kwargs", {}).get("reasoning_effort")
    if reasoning_effort != manifest["protocol_config"]["reasoning_effort"]:
        raise ValueError("raw Harbor reasoning effort does not match the protocol")
    audit_model = verifier_config.get("kwargs", {}).get("audit_model")
    if audit_model != manifest["protocol_config"]["audit_model"]:
        raise ValueError("raw Harbor audit model does not match the protocol")

    rewards = result.get("verifier_result", {}).get("rewards")
    if not isinstance(rewards, dict):
        raise ValueError("raw Harbor result is missing verifier rewards")
    job_id = result.get("id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("raw Harbor result needs a trial UUID in id")
    return {
        **attested_bindings,
        "model": model,
        "job_id": job_id,
        "protocol_seed": seed,
        "execution_status": _harbor_execution_status(result),
        "reward": rewards.get("reward"),
        "functional_score": rewards.get("functional_score"),
        "static_policy_score": rewards.get("static_policy_score"),
        "llm_audit_score": rewards.get("llm_audit_score"),
        "runtime_sec": rewards.get("runtime_sec"),
    }


def record_model_trial(
    root: Path,
    candidate_id: str,
    manifest_path: Path,
    trial_id: str,
    result_path: Path,
) -> dict[str, Any]:
    """Record raw Harbor evidence without interpreting why a failure occurred."""
    manifest = _verified_manifest(manifest_path)
    if manifest.get("candidate_id") != candidate_id:
        raise ValueError("manifest candidate does not match the requested candidate")
    result_path = _safe_evidence_path(root, result_path.parent, str(result_path))
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    evidence = _read_json(result_path)
    if evidence.get("schema_version") != 1:
        raise ValueError("trial result must use schema_version=1")
    raw_job_path, raw_job_hash = _verified_evidence_item(
        root, result_path.parent, evidence.get("raw_job", {}), "raw Harbor job"
    )
    transcript_path, transcript_hash = _verified_evidence_item(
        root,
        result_path.parent,
        evidence.get("solver_transcript", {}),
        "solver transcript",
    )
    parsed = _parse_attested_harbor_result(raw_job_path, manifest)
    mismatched = [name for name, value in parsed.items() if evidence.get(name) != value]
    if mismatched:
        raise ValueError(
            f"normalized trial result disagrees with raw Harbor evidence: {mismatched}"
        )
    if evidence.get("execution_status") not in TRIAL_EXECUTION_STATUSES:
        raise ValueError("invalid trial execution status")
    for name in (
        "reward",
        "functional_score",
        "static_policy_score",
        "llm_audit_score",
    ):
        value = evidence.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be numeric in [0, 1]")
    expected_reward = (
        evidence["functional_score"]
        * evidence["static_policy_score"]
        * evidence["llm_audit_score"]
    )
    if abs(evidence["reward"] - expected_reward) > 1e-9:
        raise ValueError("reward must equal functional * static policy * LLM audit")
    if evidence["execution_status"] != "completed" and evidence["reward"] != 0:
        raise ValueError("an incomplete execution cannot carry a positive reward")
    runtime = evidence.get("runtime_sec")
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or runtime < -1
    ):
        raise ValueError(
            "runtime_sec must be -1 for missing data or a nonnegative number"
        )
    result_hash = _file_sha256(result_path)
    if not isinstance(evidence.get("job_id"), str) or not evidence["job_id"].strip():
        raise ValueError("job_id must be a nonempty string")
    if not isinstance(evidence.get("protocol_seed"), int) or isinstance(
        evidence["protocol_seed"], bool
    ):
        raise ValueError("protocol_seed must be an integer")

    if evidence["protocol_seed"] not in manifest.get("seed_schedule", []):
        raise ValueError("protocol_seed was not precommitted in the run manifest")
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    record = state["candidates"][candidate_id]
    current_candidate = _verified_shortlist_candidate(root, candidate_id)
    if current_candidate["candidate_hash"] != manifest["candidate_hash"]:
        raise PermissionError("current shortlist content differs from the manifest")
    if not any(
        row.get("manifest_hash") == manifest["manifest_hash"]
        for row in record.get("authorizations", [])
    ):
        raise PermissionError("manifest was not authorized in this review ledger")
    if record.get("active_authorization_hash") != manifest["manifest_hash"]:
        raise PermissionError("authorization was superseded by a newer manifest")
    if _gate_fingerprint(record) != manifest.get("gate_fingerprint"):
        raise PermissionError(
            "an upstream review or prototype changed after authorization"
        )
    if not candidate_gate_status(state, candidate_id)["ready_for_model_test"]:
        raise PermissionError("current human gates no longer authorize model testing")
    if record["candidate_hash"] != manifest["candidate_hash"]:
        raise PermissionError("candidate changed after this run was authorized")
    if trial_id in record.setdefault("model_trials", {}):
        raise ValueError(f"trial id already exists: {trial_id}")
    if any(
        row.get("manifest_hash") == manifest["manifest_hash"]
        and (
            row.get("job_id") == evidence["job_id"]
            or row.get("protocol_seed") == evidence["protocol_seed"]
            or row.get("result_sha256") == result_hash
        )
        for row in record["model_trials"].values()
    ):
        raise ValueError(
            "job_id, protocol_seed, and result_sha256 must be unique within a manifest"
        )

    passed = (
        evidence["execution_status"] == "completed"
        and evidence["reward"] > 0
        and evidence["functional_score"] > 0
        and evidence["static_policy_score"] > 0
        and evidence["llm_audit_score"] > 0
    )
    trial = {
        **evidence,
        "result_path": str(result_path),
        "result_sha256": result_hash,
        "raw_job": {"path": str(raw_job_path), "sha256": raw_job_hash},
        "solver_transcript": {
            "path": str(transcript_path),
            "sha256": transcript_hash,
        },
        "trial_id": trial_id,
        "candidate_hash": record["candidate_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "model": manifest["model"],
        "passed": passed,
        "recorded_at": _utc_now(),
        "human_audits": {},
    }
    record["model_trials"][trial_id] = trial
    _append_event(
        state,
        {
            "event": "model_trial_recorded",
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            "manifest_hash": manifest["manifest_hash"],
            "timestamp": trial["recorded_at"],
        },
    )
    _write_json(state_path, state)
    return trial


def audit_model_trial(
    root: Path,
    candidate_id: str,
    trial_id: str,
    role: str,
    failure_class: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    """Attach an independent human explanation to one immutable raw trial."""
    if role not in FAILURE_AUDIT_ROLES:
        raise ValueError("invalid failure-audit role")
    allowed = ("success",) + SUBSTANTIVE_FAILURE_CLASSES + EXCLUDED_FAILURE_CLASSES
    if failure_class not in allowed:
        raise ValueError("invalid human failure class")
    reviewer = _normalized_identity(reviewer, "failure auditor")
    note = _normalized_identity(note, "failure-audit note")
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    trial = state["candidates"][candidate_id]["model_trials"][trial_id]
    if role in trial.get("human_audits", {}):
        raise ValueError("failure audits are append-only")
    if trial["passed"] != (failure_class == "success"):
        raise ValueError(
            "success classification must agree with the compound pass result"
        )
    status = trial["execution_status"]
    allowed_by_status = {
        "infrastructure_error": {"infrastructure"},
        "authentication_error": {"authentication"},
        "verifier_timeout": {"infrastructure", "evaluator_defect", "indeterminate"},
        "agent_timeout": {"resource_strategy", "infrastructure", "indeterminate"},
    }
    if status in allowed_by_status and failure_class not in allowed_by_status[status]:
        raise ValueError(f"{failure_class} is incompatible with {status}")
    other_reviewers = {
        row.get("reviewer")
        for audit_role, row in trial.get("human_audits", {}).items()
        if audit_role != role
    }
    if _identity_key(reviewer) in {
        _identity_key(identity) for identity in other_reviewers if identity
    }:
        raise ValueError("the two failure-audit roles require distinct reviewers")
    record = state["candidates"][candidate_id]
    if _identity_key(reviewer) in _reviewer_identities(record):
        raise ValueError(
            "failure auditors must be independent of prior reviewers and authors"
        )
    audit = {
        "failure_class": failure_class,
        "reviewer": reviewer,
        "note": note,
        "timestamp": _utc_now(),
    }
    trial.setdefault("human_audits", {})[role] = audit
    _append_event(
        state,
        {
            "event": "model_trial_audited",
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            "role": role,
            **audit,
        },
    )
    _write_json(state_path, state)
    return audit


def model_hardness_status(
    root: Path, candidate_id: str, manifest_hash: str
) -> dict[str, Any]:
    """Summarize protocol-scoped evidence without claiming universal inability."""
    state = _read_json(root / "review_state.json")
    record = state["candidates"][candidate_id]
    authorization = next(
        (
            row
            for row in record.get("authorizations", [])
            if row.get("manifest_hash") == manifest_hash
        ),
        None,
    )
    if authorization is None:
        raise KeyError("unknown authorization manifest")
    if record.get("active_authorization_hash") != manifest_hash:
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "conclusion": "authorization_revoked_or_superseded",
            "claim_boundary": "No model-hardness conclusion is valid for an inactive manifest.",
        }
    if _gate_fingerprint(record) != authorization.get("gate_fingerprint"):
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "conclusion": "authorization_revoked_by_gate_change",
            "claim_boundary": "Upstream evidence changed after this manifest was issued.",
        }
    evidence_violations: list[str] = []
    try:
        candidate = _verified_shortlist_candidate(root, candidate_id)
        if candidate["candidate_hash"] != record["candidate_hash"]:
            evidence_violations.append("candidate_hash_changed")
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        evidence_violations.append("candidate_content_unverifiable")
    if not candidate_gate_status(state, candidate_id)["ready_for_model_test"]:
        evidence_violations.append("current_human_or_prototype_gate_failed")
    manifest: dict[str, Any] | None = None
    try:
        manifest = _verified_manifest(Path(authorization["manifest_path"]))
        if manifest.get("manifest_hash") != manifest_hash:
            evidence_violations.append("manifest_hash_changed")
        protocol_path = Path(manifest["protocol_config_path"])
        if _file_sha256(protocol_path) != manifest["protocol_config_sha256"]:
            evidence_violations.append("protocol_config_changed")
    except (KeyError, FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        evidence_violations.append("manifest_or_protocol_unverifiable")
    trials = [
        row
        for row in record.get("model_trials", {}).values()
        if row.get("manifest_hash") == manifest_hash
    ]
    if manifest is not None:
        for trial in trials:
            try:
                result_path = Path(trial["result_path"])
                if _file_sha256(result_path) != trial["result_sha256"]:
                    evidence_violations.append(
                        f"trial_result_changed:{trial.get('trial_id')}"
                    )
                    continue
                raw_job_path = Path(trial["raw_job"]["path"])
                transcript_path = Path(trial["solver_transcript"]["path"])
                if _path_sha256(raw_job_path) != trial["raw_job"]["sha256"]:
                    evidence_violations.append(
                        f"raw_job_changed:{trial.get('trial_id')}"
                    )
                if (
                    _path_sha256(transcript_path)
                    != trial["solver_transcript"]["sha256"]
                ):
                    evidence_violations.append(
                        f"solver_transcript_changed:{trial.get('trial_id')}"
                    )
                parsed = _parse_attested_harbor_result(raw_job_path, manifest)
                if any(trial.get(name) != value for name, value in parsed.items()):
                    evidence_violations.append(
                        f"raw_job_disagrees_with_ledger:{trial.get('trial_id')}"
                    )
            except (
                KeyError,
                FileNotFoundError,
                ValueError,
                OSError,
                json.JSONDecodeError,
            ):
                evidence_violations.append(
                    f"trial_evidence_unverifiable:{trial.get('trial_id')}"
                )
    if evidence_violations:
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "conclusion": "evidence_drift_or_gate_revoked",
            "evidence_violations": sorted(set(evidence_violations)),
            "claim_boundary": (
                "No model-hardness conclusion is valid while current candidate, "
                "gate, manifest, protocol, or trial evidence cannot be revalidated."
            ),
        }
    unique_trials: list[dict[str, Any]] = []
    seen_runs: set[tuple[Any, Any]] = set()
    for trial in sorted(trials, key=lambda row: row["trial_id"]):
        identity = (trial.get("job_id"), trial.get("protocol_seed"))
        if identity not in seen_runs:
            unique_trials.append(trial)
            seen_runs.add(identity)

    def consensus_class(trial: dict[str, Any]) -> str | None:
        audits = trial.get("human_audits", {})
        if any(role not in audits for role in FAILURE_AUDIT_ROLES):
            return None
        classes = {audits[role].get("failure_class") for role in FAILURE_AUDIT_ROLES}
        return next(iter(classes)) if len(classes) == 1 else None

    passes = [row for row in unique_trials if row["passed"]]
    consensus_audited = [
        row for row in unique_trials if consensus_class(row) is not None
    ]
    conflicting_audits = [
        row
        for row in unique_trials
        if all(role in row.get("human_audits", {}) for role in FAILURE_AUDIT_ROLES)
        and consensus_class(row) is None
    ]
    substantive = [
        row
        for row in consensus_audited
        if consensus_class(row) in SUBSTANTIVE_FAILURE_CLASSES
    ]
    excluded = [
        row
        for row in consensus_audited
        if consensus_class(row) in EXCLUDED_FAILURE_CLASSES
    ]
    required = authorization["independent_trials"]
    scheduled_seeds = set(authorization["seed_schedule"])
    recorded_seeds = {row["protocol_seed"] for row in unique_trials}
    missing_seeds = sorted(scheduled_seeds - recorded_seeds)
    unexpected_seeds = sorted(recorded_seeds - scheduled_seeds)
    schedule_complete = (
        not missing_seeds and not unexpected_seeds and len(unique_trials) == required
    )
    valid_count = len(passes) + len(substantive)
    if passes:
        conclusion = "solved_in_at_least_one_valid_trial"
    elif schedule_complete and len(substantive) == required:
        conclusion = (
            "pilot_hardness_signal_ready"
            if authorization["stage"] == "pilot"
            else "protocol_scoped_model_hard_evidence_ready"
        )
    else:
        conclusion = "incomplete_or_invalid_trial_schedule"
    return {
        "candidate_id": candidate_id,
        "manifest_hash": manifest_hash,
        "required_valid_trials": required,
        "stage": authorization["stage"],
        "scheduled_seeds": sorted(scheduled_seeds),
        "missing_seeds": missing_seeds,
        "unexpected_seeds": unexpected_seeds,
        "schedule_complete": schedule_complete,
        "recorded_unique_trials": len(unique_trials),
        "consensus_audited_trials": len(consensus_audited),
        "conflicting_audits": len(conflicting_audits),
        "valid_trials": valid_count,
        "substantive_failures": len(substantive),
        "excluded_failures": len(excluded),
        "passes": len(passes),
        "pass_at_k": 1 if passes else 0,
        "conclusion": conclusion,
        "claim_boundary": (
            "This conclusion applies only to the frozen manifest and model. "
            "It is evidence under a protocol, not proof of universal inability."
        ),
    }
