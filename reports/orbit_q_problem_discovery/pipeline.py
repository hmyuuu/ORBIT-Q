"""Build and gate a source-grounded pool of future ORBIT-Q task concepts.

The screening score produced here is a design prior.  It is deliberately kept
separate from empirical model results: no candidate can be called "unsolved"
until an expert baseline, verifier-only smoke test, human pilot approval, and
repeated frozen-model trials have all happened.
"""

from __future__ import annotations

import hashlib
import json
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

CONCEPT_REVIEW_ROLES = (
    "quantum_scientist",
    "tensorcircuit_expert",
    "verifier_engineer",
)
PILOT_REVIEW_ROLES = ("benchmark_owner", "independent_reviewer")
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
    "expert_baseline_path",
    "expert_baseline_sha256",
    "independent_oracle_path",
    "independent_oracle_sha256",
    "evaluator_path",
    "evaluator_sha256",
    "verifier_job_id",
    "container_image_digest",
    "framework_prompt_sha256",
    "source_commit",
    "public_api_canary_passed",
    "expert_runtime_p95_sec",
    "independent_oracle_runtime_sec",
    "gold_effective_lines",
    "reproducibility_runs",
    "observed_cpu_count",
    "observed_memory_mb",
    "expert_peak_memory_mb",
    "verifier_only_passed",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


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
            if name in SCORE_NAMES and (not isinstance(value, int) or not 1 <= value <= 5):
                errors.append(f"family {family.get('id')} invalid {name}: {value}")
        unknown_sources = set(family.get("source_ids", [])) - source_id_set
        if unknown_sources:
            errors.append(
                f"family {family.get('id')} references unknown sources: "
                f"{sorted(unknown_sources)}"
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
    if min(
        scores["verifier_feasibility"],
        scores["runtime_fit"],
        scores["determinism"],
    ) < 3:
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
    for family in catalog["families"]:
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
                    "tensorcircuit_path": family["tensorcircuit_path"],
                    "oracle_design": family["oracle_design"],
                    "novelty_note": family["novelty_note"],
                    "current_task_overlap": family.get("current_task_overlap", []),
                    "scores": scores,
                    "design_score": design_score,
                }
                candidate = {
                    **core,
                    "candidate_hash": _canonical_hash(core),
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
        selected.append(candidate)
    if len(selected) != 10:
        raise AssertionError("shortlist must contain exactly ten candidates")
    return selected


def _summary(catalog: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
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
        "shortlist_snapshot": _canonical_hash(
            [(row["id"], row["candidate_hash"]) for row in shortlist], length=24
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
            "7. Run at least five independent GPT-5.6-sol trials under one frozen prompt/image/protocol.",
            "8. Human audit excludes infrastructure, auth, policy refusal, ambiguous spec, and framework impossibility failures.",
            "",
            "Only failures that remain after step 8 count toward evidence that a task is model-hard.",
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
    active_ids = {row["id"] for row in shortlist}
    for row in shortlist:
        record = state["candidates"].setdefault(
            row["id"],
            {
                "candidate_hash": row["candidate_hash"],
                "concept_reviews": {},
                "prototype": {},
                "pilot_reviews": {},
                "authorizations": [],
                "model_trials": {},
            },
        )
        record.setdefault("authorizations", [])
        record.setdefault("model_trials", {})
        if record.get("candidate_hash") != row["candidate_hash"]:
            record["stale_candidate_hash"] = record.get("candidate_hash")
            record["candidate_hash"] = row["candidate_hash"]
            record["concept_reviews"] = {}
            record["prototype"] = {}
            record["pilot_reviews"] = {}
            record["authorizations"] = []
            record["model_trials"] = {}
    state["active_shortlist_ids"] = sorted(active_ids)
    state["shortlist_snapshot"] = _canonical_hash(
        [(row["id"], row["candidate_hash"]) for row in shortlist], length=24
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
        <= prototype.get("observed_memory_mb", 0)
    )
    public_api_ok = prototype.get("public_api_canary_passed") is True
    verifier_ok = prototype.get("verifier_only_passed") is True
    artifact_hashes_ok = all(
        is_hex(prototype.get(name), (64,))
        for name in (
            "expert_baseline_sha256",
            "independent_oracle_sha256",
            "evaluator_sha256",
            "framework_prompt_sha256",
        )
    )
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
        "ready_for_model_test": concept["passed"] and prototype_passed and pilot["passed"],
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
    path = root / "review_state.json"
    state = _read_json(path)
    record = state["candidates"][candidate_id]
    key = "concept_reviews" if gate == "concept" else "pilot_reviews"
    review = {
        "decision": decision,
        "reviewer": reviewer,
        "note": note,
        "timestamp": _utc_now(),
    }
    record[key][role] = review
    state["event_log"].append(
        {"event": "review", "candidate_id": candidate_id, "gate": gate, "role": role, **review}
    )
    _write_json(path, state)
    return candidate_gate_status(state, candidate_id)


def record_prototype(root: Path, candidate_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    path = root / "review_state.json"
    state = _read_json(path)
    record = state["candidates"][candidate_id]
    record["prototype"] = {**evidence, "recorded_at": _utc_now()}
    state["event_log"].append(
        {
            "event": "prototype_evidence",
            "candidate_id": candidate_id,
            "timestamp": _utc_now(),
            "evidence_hash": _canonical_hash(evidence),
        }
    )
    _write_json(path, state)
    return candidate_gate_status(state, candidate_id)


def authorize_model_test(
    root: Path,
    candidate_id: str,
    model: str,
    trials: int,
    output: Path,
) -> dict[str, Any]:
    if trials < 5:
        raise ValueError("model-hardness evidence requires at least five independent trials")
    state = _read_json(root / "review_state.json")
    status = candidate_gate_status(state, candidate_id)
    if not status["ready_for_model_test"]:
        raise PermissionError(
            "real model test remains human-gated: "
            + json.dumps(status, sort_keys=True)
        )
    candidate = next(
        row
        for row in _read_json(root / "shortlist.json")["candidates"]
        if row["id"] == candidate_id
    )
    record = state["candidates"][candidate_id]
    if record["candidate_hash"] != candidate["candidate_hash"]:
        raise PermissionError("candidate content changed after review; re-review is required")
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "candidate_hash": candidate["candidate_hash"],
        "model": model,
        "independent_trials": trials,
        "authorized_at": _utc_now(),
        "shortlist_snapshot": state["shortlist_snapshot"],
        "prototype": record["prototype"],
        "concept_reviews": record["concept_reviews"],
        "pilot_reviews": record["pilot_reviews"],
        "execution_policy": {
            "minimum_pilot_trials": 5,
            "recommended_confirmation_trials": 10,
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
    manifest["manifest_hash"] = _canonical_hash(manifest, length=24)
    _write_json(output, manifest)
    record.setdefault("authorizations", []).append(
        {
            "manifest_hash": manifest["manifest_hash"],
            "manifest_path": str(output),
            "model": model,
            "independent_trials": trials,
            "authorized_at": manifest["authorized_at"],
        }
    )
    state["event_log"].append(
        {
            "event": "model_test_authorized",
            "candidate_id": candidate_id,
            "manifest_hash": manifest["manifest_hash"],
            "timestamp": manifest["authorized_at"],
        }
    )
    _write_json(root / "review_state.json", state)
    return manifest


def _verified_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    recorded_hash = manifest.get("manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if recorded_hash != _canonical_hash(unsigned, length=24):
        raise ValueError("run manifest hash does not match its contents")
    return manifest


def record_model_trial(
    root: Path,
    candidate_id: str,
    manifest_path: Path,
    trial_id: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Record raw Harbor evidence without interpreting why a failure occurred."""
    if evidence.get("execution_status") not in TRIAL_EXECUTION_STATUSES:
        raise ValueError("invalid trial execution status")
    for name in (
        "reward",
        "functional_score",
        "static_policy_score",
        "llm_audit_score",
    ):
        value = evidence.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be numeric in [0, 1]")
    runtime = evidence.get("runtime_sec")
    if not isinstance(runtime, (int, float)) or isinstance(runtime, bool) or runtime < -1:
        raise ValueError("runtime_sec must be -1 for missing data or a nonnegative number")
    result_hash = evidence.get("result_sha256")
    if not (
        isinstance(result_hash, str)
        and len(result_hash) == 64
        and all(character in "0123456789abcdef" for character in result_hash.lower())
    ):
        raise ValueError("result_sha256 must be a 64-character hexadecimal digest")

    manifest = _verified_manifest(manifest_path)
    if manifest.get("candidate_id") != candidate_id:
        raise ValueError("manifest candidate does not match the requested candidate")
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    record = state["candidates"][candidate_id]
    if not any(
        row.get("manifest_hash") == manifest["manifest_hash"]
        for row in record.get("authorizations", [])
    ):
        raise PermissionError("manifest was not authorized in this review ledger")
    if record["candidate_hash"] != manifest["candidate_hash"]:
        raise PermissionError("candidate changed after this run was authorized")
    if trial_id in record.setdefault("model_trials", {}):
        raise ValueError(f"trial id already exists: {trial_id}")

    passed = (
        evidence["execution_status"] == "completed"
        and evidence["reward"] > 0
        and evidence["functional_score"] > 0
        and evidence["static_policy_score"] > 0
        and evidence["llm_audit_score"] > 0
    )
    trial = {
        **evidence,
        "trial_id": trial_id,
        "candidate_hash": record["candidate_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "model": manifest["model"],
        "passed": passed,
        "recorded_at": _utc_now(),
        "human_audit": {},
    }
    record["model_trials"][trial_id] = trial
    state["event_log"].append(
        {
            "event": "model_trial_recorded",
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            "manifest_hash": manifest["manifest_hash"],
            "timestamp": trial["recorded_at"],
        }
    )
    _write_json(state_path, state)
    return trial


def audit_model_trial(
    root: Path,
    candidate_id: str,
    trial_id: str,
    failure_class: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    """Attach an independent human explanation to one immutable raw trial."""
    allowed = ("success",) + SUBSTANTIVE_FAILURE_CLASSES + EXCLUDED_FAILURE_CLASSES
    if failure_class not in allowed:
        raise ValueError("invalid human failure class")
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    trial = state["candidates"][candidate_id]["model_trials"][trial_id]
    if trial["passed"] != (failure_class == "success"):
        raise ValueError("success classification must agree with the compound pass result")
    audit = {
        "failure_class": failure_class,
        "reviewer": reviewer,
        "note": note,
        "timestamp": _utc_now(),
    }
    trial["human_audit"] = audit
    state["event_log"].append(
        {
            "event": "model_trial_audited",
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            **audit,
        }
    )
    _write_json(state_path, state)
    return audit


def model_hardness_status(
    state: dict[str, Any], candidate_id: str, manifest_hash: str
) -> dict[str, Any]:
    """Summarize protocol-scoped evidence without claiming universal inability."""
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
    trials = [
        row
        for row in record.get("model_trials", {}).values()
        if row.get("manifest_hash") == manifest_hash
    ]
    unique_trials: list[dict[str, Any]] = []
    seen_runs: set[tuple[Any, Any]] = set()
    for trial in sorted(trials, key=lambda row: row["trial_id"]):
        identity = (trial.get("job_id"), trial.get("protocol_seed"))
        if identity not in seen_runs:
            unique_trials.append(trial)
            seen_runs.add(identity)
    audited = [row for row in unique_trials if row.get("human_audit")]
    passes = [row for row in audited if row["passed"]]
    substantive = [
        row
        for row in audited
        if row["human_audit"].get("failure_class") in SUBSTANTIVE_FAILURE_CLASSES
    ]
    excluded = [
        row
        for row in audited
        if row["human_audit"].get("failure_class") in EXCLUDED_FAILURE_CLASSES
    ]
    required = authorization["independent_trials"]
    valid_count = len(passes) + len(substantive)
    if passes:
        conclusion = "solved_in_at_least_one_valid_trial"
    elif len(substantive) >= required:
        conclusion = "protocol_scoped_model_hard_evidence_ready"
    else:
        conclusion = "insufficient_valid_trials"
    return {
        "candidate_id": candidate_id,
        "manifest_hash": manifest_hash,
        "required_valid_trials": required,
        "recorded_unique_trials": len(unique_trials),
        "audited_trials": len(audited),
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
