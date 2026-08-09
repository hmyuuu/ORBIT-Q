"""Deterministic validation for post-snapshot reserve design records.

The reserve registry is deliberately not an authorization ledger.  Every
entry in schema version 1 is ``design_only`` and therefore ineligible for both
expert prequalification and model trials.  Its only purpose is to bind a
post-snapshot blueprint to exact tracked bytes without editing the frozen
1,000-candidate discovery snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any


REGISTRY_FILENAME = "post_snapshot_reserve_registry.json"
REGISTRY_KIND = "orbit_q_post_snapshot_reserve_design_registry"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ARTIFACT_ROLES = (
    "blueprint_metadata",
    "instruction",
    "evaluator",
    "expert_solution",
)
FROZEN_FILES = ("candidates.jsonl", "shortlist.json", "summary.json")


class ReserveRegistryError(ValueError):
    """Raised when a reserve registry or its tracked blueprint has drifted."""


def canonical_sha256(value: Any) -> str:
    """Hash one JSON value using the repository's canonical JSON convention."""

    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry_hash(entry: dict[str, Any]) -> str:
    """Return the content hash of an entry, excluding its stored hash field."""

    return canonical_sha256({k: v for k, v in entry.items() if k != "entry_hash"})


def registry_hash(document: dict[str, Any]) -> str:
    """Return the content hash of a registry, excluding its stored hash field."""

    return canonical_sha256({k: v for k, v in document.items() if k != "registry_hash"})


def blueprint_bundle_hash(artifacts: dict[str, dict[str, str]]) -> str:
    """Hash the path-sensitive, role-independent blueprint file manifest."""

    manifest = sorted(
        (
            {"path": item["path"], "sha256": item["sha256"]}
            for item in artifacts.values()
        ),
        key=lambda item: item["path"],
    )
    return canonical_sha256(manifest)


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReserveRegistryError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReserveRegistryError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ReserveRegistryError(f"{label} must contain a JSON object")
    return value


def _relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ReserveRegistryError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ReserveRegistryError(f"{label} must be a contained relative path")
    if path.as_posix() != value:
        raise ReserveRegistryError(f"{label} must use normalized POSIX syntax")
    return path


def _contained_path(base: Path, relative: PurePosixPath, label: str) -> Path:
    base = base.resolve()
    path = base.joinpath(*relative.parts)
    cursor = base
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ReserveRegistryError(f"{label} must not traverse a symbolic link")
    resolved = path.resolve()
    if resolved != base and not resolved.is_relative_to(base):
        raise ReserveRegistryError(f"{label} escapes its tracked root")
    return resolved


def _validate_audit_status(value: Any) -> None:
    expected = {
        "classification": "design_only",
        "scientific_design_review": "completed_with_blockers",
        "local_oracle_status": "validated_nonbinding",
        "tensorcircuit_expert_status": "not_executed",
        "pinned_image_status": "not_executed",
        "harbor_status": "not_executed",
        "solver_model_status": "not_executed",
        "hardness_status": "no_claim",
        "execution_authorized": False,
    }
    if value != expected:
        raise ReserveRegistryError(
            "design-only reserve audit_status must preserve every non-execution "
            "and no-hardness-claim assertion"
        )


def _frozen_candidate_ids(workspace: Path) -> set[str]:
    path = workspace / "candidates.jsonl"
    identifiers: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReserveRegistryError(
            f"cannot read frozen candidates.jsonl: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReserveRegistryError(
                f"invalid frozen candidates.jsonl line {line_number}"
            ) from exc
        candidate_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ReserveRegistryError(
                f"frozen candidates.jsonl line {line_number} lacks an id"
            )
        identifiers.add(candidate_id)
    return identifiers


def load_verified_registry(workspace: Path) -> dict[str, Any]:
    """Load and rehash the registry, frozen snapshot, and every blueprint byte."""

    workspace = workspace.resolve()
    registry_path = workspace / REGISTRY_FILENAME
    document = _json_object(registry_path, "post-snapshot reserve registry")
    if set(document) != {
        "schema_version",
        "registry_kind",
        "frozen_snapshot",
        "reserves",
        "registry_hash",
    }:
        raise ReserveRegistryError("reserve registry has unknown or missing fields")
    if document["schema_version"] != 1 or document["registry_kind"] != REGISTRY_KIND:
        raise ReserveRegistryError("unsupported reserve registry schema or kind")
    if not SHA256_RE.fullmatch(str(document.get("registry_hash", ""))):
        raise ReserveRegistryError("reserve registry_hash must be lowercase SHA-256")
    if registry_hash(document) != document["registry_hash"]:
        raise ReserveRegistryError("reserve registry_hash does not match its payload")

    frozen = document.get("frozen_snapshot")
    expected_frozen_keys = {f"{name}_sha256" for name in FROZEN_FILES}
    if not isinstance(frozen, dict) or set(frozen) != expected_frozen_keys:
        raise ReserveRegistryError("reserve registry frozen_snapshot shape is invalid")
    for name in FROZEN_FILES:
        expected = frozen[f"{name}_sha256"]
        path = workspace / name
        if not SHA256_RE.fullmatch(str(expected)):
            raise ReserveRegistryError(f"frozen {name} hash must be lowercase SHA-256")
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ReserveRegistryError(f"frozen discovery snapshot drifted: {name}")

    reserves = document.get("reserves")
    if not isinstance(reserves, list) or not reserves:
        raise ReserveRegistryError("reserve registry requires at least one entry")
    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    frozen_ids = _frozen_candidate_ids(workspace)
    blueprint_root = (workspace / "blueprints").resolve()
    if (workspace / "blueprints").is_symlink() or not blueprint_root.is_dir():
        raise ReserveRegistryError(
            "tracked blueprints root must be a regular directory"
        )

    for index, entry in enumerate(reserves):
        label = f"reserve entry {index}"
        if not isinstance(entry, dict) or set(entry) != {
            "candidate_id",
            "slug",
            "problem_id",
            "lifecycle_status",
            "blueprint_path",
            "artifacts",
            "blueprint_bundle_sha256",
            "audit_status",
            "entry_hash",
        }:
            raise ReserveRegistryError(f"{label} has unknown or missing fields")
        candidate_id = entry.get("candidate_id")
        slug = entry.get("slug")
        problem_id = entry.get("problem_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ReserveRegistryError(f"{label} candidate_id is invalid")
        if candidate_id in seen_ids:
            raise ReserveRegistryError(
                f"duplicate reserve candidate_id: {candidate_id}"
            )
        if candidate_id in frozen_ids:
            raise ReserveRegistryError(
                f"post-snapshot reserve collides with frozen candidate: {candidate_id}"
            )
        if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
            raise ReserveRegistryError(f"{label} slug is invalid")
        if slug in seen_slugs:
            raise ReserveRegistryError(f"duplicate reserve slug: {slug}")
        if (
            isinstance(problem_id, bool)
            or not isinstance(problem_id, int)
            or problem_id <= 0
        ):
            raise ReserveRegistryError(f"{label} problem_id must be a positive integer")
        if entry.get("lifecycle_status") != "design_only":
            raise ReserveRegistryError(
                f"{label} must remain design_only; this registry cannot authorize runs"
            )
        _validate_audit_status(entry.get("audit_status"))
        if not SHA256_RE.fullmatch(str(entry.get("entry_hash", ""))):
            raise ReserveRegistryError(f"{label} entry_hash must be lowercase SHA-256")
        if entry_hash(entry) != entry["entry_hash"]:
            raise ReserveRegistryError(f"{label} entry_hash does not match its payload")

        relative_blueprint = _relative_path(
            entry["blueprint_path"], f"{label} blueprint_path"
        )
        blueprint_dir = _contained_path(
            workspace, relative_blueprint, f"{label} blueprint_path"
        )
        if (
            blueprint_dir == blueprint_root
            or not blueprint_dir.is_relative_to(blueprint_root)
            or blueprint_dir.is_symlink()
            or not blueprint_dir.is_dir()
        ):
            raise ReserveRegistryError(
                f"{label} blueprint_path must name a contained regular blueprint directory"
            )

        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_ROLES):
            raise ReserveRegistryError(f"{label} artifact manifest shape is invalid")
        expected_paths = {
            "blueprint_metadata": "blueprint.json",
            "instruction": "instruction.md",
            "evaluator": f"evaluate_{problem_id}.py",
            "expert_solution": f"expert/solution_{problem_id}.py",
        }
        declared_paths: set[str] = set()
        for role in ARTIFACT_ROLES:
            artifact = artifacts[role]
            if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
                raise ReserveRegistryError(f"{label} {role} artifact shape is invalid")
            relative_artifact = _relative_path(
                artifact["path"], f"{label} {role} artifact path"
            )
            if relative_artifact.as_posix() != expected_paths[role]:
                raise ReserveRegistryError(
                    f"{label} {role} artifact path is unexpected"
                )
            expected_hash = artifact["sha256"]
            if not SHA256_RE.fullmatch(str(expected_hash)):
                raise ReserveRegistryError(f"{label} {role} hash is invalid")
            artifact_path = _contained_path(
                blueprint_dir, relative_artifact, f"{label} {role} artifact"
            )
            if (
                artifact_path.is_symlink()
                or not artifact_path.is_file()
                or file_sha256(artifact_path) != expected_hash
            ):
                raise ReserveRegistryError(f"{label} tracked artifact drifted: {role}")
            declared_paths.add(relative_artifact.as_posix())

        actual_paths: set[str] = set()
        for directory, directory_names, file_names in os.walk(
            blueprint_dir, followlinks=False
        ):
            directory_path = Path(directory)
            if any((directory_path / name).is_symlink() for name in directory_names):
                raise ReserveRegistryError(f"{label} blueprint contains a symlink")
            for name in file_names:
                path = directory_path / name
                if path.is_symlink() or not path.is_file():
                    raise ReserveRegistryError(
                        f"{label} blueprint contains an invalid file"
                    )
                actual_paths.add(path.relative_to(blueprint_dir).as_posix())
        if actual_paths != declared_paths:
            raise ReserveRegistryError(
                f"{label} blueprint file set differs from its artifact manifest"
            )
        if (
            not SHA256_RE.fullmatch(str(entry.get("blueprint_bundle_sha256", "")))
            or blueprint_bundle_hash(artifacts) != entry["blueprint_bundle_sha256"]
        ):
            raise ReserveRegistryError(f"{label} blueprint bundle hash is invalid")

        metadata = _json_object(blueprint_dir / "blueprint.json", f"{label} metadata")
        if (
            metadata.get("candidate_id") != candidate_id
            or metadata.get("slug") != slug
            or metadata.get("problem_id") != problem_id
        ):
            raise ReserveRegistryError(
                f"{label} identity differs from blueprint metadata"
            )
        seen_ids.add(candidate_id)
        seen_slugs.add(slug)
    return document


def verified_reserve_entry(
    workspace: Path, candidate_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return one exact registry entry and document, or ``None`` when unregistered."""

    registry_path = workspace.resolve() / REGISTRY_FILENAME
    if not registry_path.exists():
        return None
    # Keep the frozen shortlist/screen execution path independent of unrelated
    # reserve design work.  A candidate named by the raw registry must pass the
    # complete validation below; candidates not named there do not acquire a
    # registry dependency or execution semantics.
    raw = _json_object(registry_path, "post-snapshot reserve registry")
    raw_reserves = raw.get("reserves")
    if not isinstance(raw_reserves, list) or not any(
        isinstance(entry, dict) and entry.get("candidate_id") == candidate_id
        for entry in raw_reserves
    ):
        return None
    document = load_verified_registry(workspace)
    matches = [
        entry for entry in document["reserves"] if entry["candidate_id"] == candidate_id
    ]
    if not matches:
        return None
    if len(matches) != 1:  # Defensive; load_verified_registry already rejects this.
        raise ReserveRegistryError(f"duplicate reserve candidate_id: {candidate_id}")
    return matches[0], document
