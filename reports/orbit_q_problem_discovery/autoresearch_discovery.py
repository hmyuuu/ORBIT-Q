"""Deterministic, offline discovery stages for quantum problem-source leads.

This module deliberately does not contain a web client.  It freezes query plans,
ingests operator-provided metadata manifests, normalizes source records, groups
duplicates, and takes a deterministic coverage sample.  Every persisted artifact
is addressed by the SHA-256 digest of its exact canonical JSON bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


ARTIFACT_SCHEMA_VERSION = 1
STAGE_VERSION = "orbit-q-autoresearch-discovery/1"
MAX_INPUT_BYTES = 8 * 1024 * 1024
ARTIFACT_STAGES = frozenset(
    {
        "search-plan",
        "raw-manifest",
        "raw-ingest",
        "normalized-record",
        "normalize",
        "dedupe",
        "coverage-sample",
    }
)
COVERAGE_DIMENSIONS = frozenset(
    {
        "quantum_domain",
        "situation",
        "task_archetype",
        "evidence_role",
        "source_type",
        "license_status",
        "search_execution_claim",
    }
)
COLLECTION_METHODS = frozenset(
    {"manual_curation", "provider_export", "external_adapter_export"}
)
SEARCH_EXECUTION_CLAIMS = frozenset(
    {"not_claimed", "operator_attested_external_execution"}
)
LICENSE_STATUSES = frozenset({"known", "unknown", "restricted"})
TRACKING_QUERY_KEYS = frozenset(
    {"fbclid", "gclid", "mc_cid", "mc_eid", "ref_src", "ref_url"}
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\b"),
)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "via",
        "with",
    }
)


class DiscoveryError(ValueError):
    """Raised when discovery inputs, paths, or artifacts are invalid."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Stable reference to a canonical artifact."""

    stage: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"sha256": self.sha256, "stage": self.stage}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical byte representation used by the artifact store."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise DiscoveryError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc
    return (encoded + "\n").encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_contained_path(
    workspace_root: Path,
    path: Path,
    *,
    must_exist: bool,
    label: str,
) -> Path:
    candidate = path if path.is_absolute() else workspace_root / path
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise DiscoveryError(f"cannot resolve {label}: {exc}") from exc
    if not _is_within(resolved, workspace_root):
        raise DiscoveryError(f"{label} must stay inside workspace root")
    tasks_root = (workspace_root / "tasks").resolve(strict=False)
    if resolved == tasks_root or _is_within(resolved, tasks_root):
        raise DiscoveryError(f"{label} must not be inside the canonical tasks tree")
    return resolved


class ArtifactStore:
    """Atomic content-addressed JSON store contained in one workspace."""

    def __init__(self, workspace_root: Path | str, store_root: Path | str):
        workspace = Path(workspace_root).resolve(strict=True)
        if not workspace.is_dir():
            raise DiscoveryError("workspace root must be a directory")
        root = _resolve_contained_path(
            workspace,
            Path(store_root),
            must_exist=False,
            label="artifact store",
        )
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve(strict=True) != root:
            raise DiscoveryError("artifact store changed identity during creation")
        self.workspace_root = workspace
        self.root = root

    def input_path(self, path: Path | str, *, label: str) -> Path:
        resolved = _resolve_contained_path(
            self.workspace_root,
            Path(path),
            must_exist=True,
            label=label,
        )
        if not resolved.is_file():
            raise DiscoveryError(f"{label} must be a regular file")
        return resolved

    def artifact_path(self, ref: ArtifactRef) -> Path:
        _validate_ref(ref)
        return self.root / ref.stage / f"{ref.sha256}.json"

    def write(self, stage: str, body: Mapping[str, Any]) -> ArtifactRef:
        if stage not in ARTIFACT_STAGES:
            raise DiscoveryError(f"unknown artifact stage: {stage}")
        if "stage" in body or "artifact_schema_version" in body:
            raise DiscoveryError("artifact body must not override envelope fields")
        artifact = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "stage": stage,
            "stage_version": STAGE_VERSION,
            **body,
        }
        content = canonical_json_bytes(artifact)
        ref = ArtifactRef(stage=stage, sha256=sha256_bytes(content))
        stage_root = self.root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        destination = self.artifact_path(ref)
        if destination.exists():
            existing = destination.read_bytes()
            if existing != content or sha256_bytes(existing) != ref.sha256:
                raise DiscoveryError(
                    f"content-address collision or tampering at {destination}"
                )
            return ref

        descriptor, temporary_name = tempfile.mkstemp(
            dir=stage_root,
            prefix=f".{ref.sha256}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o644)
            if destination.exists():
                existing = destination.read_bytes()
                if existing != content:
                    raise DiscoveryError(
                        f"content-address collision or tampering at {destination}"
                    )
                temporary.unlink()
            else:
                os.replace(temporary, destination)
            if destination.read_bytes() != content:
                raise DiscoveryError("atomic artifact write verification failed")
        finally:
            if temporary.exists():
                temporary.unlink()
        return ref

    def read(self, ref: ArtifactRef) -> dict[str, Any]:
        path = self.artifact_path(ref)
        if not path.is_file():
            raise DiscoveryError(f"artifact does not exist: {ref.stage}/{ref.sha256}")
        content = path.read_bytes()
        if sha256_bytes(content) != ref.sha256:
            raise DiscoveryError(f"artifact digest mismatch: {ref.stage}/{ref.sha256}")
        value = _decode_json(content, label="artifact")
        if not isinstance(value, dict):
            raise DiscoveryError("artifact root must be an object")
        if value.get("stage") != ref.stage:
            raise DiscoveryError("artifact stage does not match its reference")
        if value.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise DiscoveryError("unsupported artifact schema version")
        if value.get("stage_version") != STAGE_VERSION:
            raise DiscoveryError("unsupported artifact stage version")
        return value


def _validate_ref(ref: ArtifactRef) -> None:
    if ref.stage not in ARTIFACT_STAGES:
        raise DiscoveryError(f"unknown artifact stage: {ref.stage}")
    if not SHA256_RE.fullmatch(ref.sha256):
        raise DiscoveryError(
            "artifact digest must be 64 lowercase hexadecimal characters"
        )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiscoveryError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DiscoveryError(f"non-finite JSON number is forbidden: {value}")


def _decode_json(content: bytes, *, label: str) -> Any:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiscoveryError(f"{label} must be UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise DiscoveryError(f"invalid {label} JSON: {exc}") from exc


def _load_input_json(
    store: ArtifactStore, path: Path | str, *, label: str
) -> tuple[Any, str]:
    resolved = store.input_path(path, label=label)
    size = resolved.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise DiscoveryError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
    content = resolved.read_bytes()
    _scan_secret_like_values(content)
    return _decode_json(content, label=label), sha256_bytes(content)


def _scan_secret_like_values(content: bytes) -> None:
    text = content.decode("utf-8", errors="ignore")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise DiscoveryError("input contains a secret-like credential pattern")
    if re.search(r"https?://[^/@\s:]+:[^/@\s]+@", text, flags=re.IGNORECASE):
        raise DiscoveryError("credential-bearing URLs are forbidden")


def _expect_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DiscoveryError(f"{label} must be an object")
    return value


def _expect_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - value.keys())
    extra = sorted(value.keys() - allowed)
    if missing:
        raise DiscoveryError(f"{label} is missing fields: {', '.join(missing)}")
    if extra:
        raise DiscoveryError(f"{label} has unsupported fields: {', '.join(extra)}")


def _text(value: Any, *, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str):
        raise DiscoveryError(f"{label} must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise DiscoveryError(f"{label} must not be blank")
    if len(normalized) > maximum:
        raise DiscoveryError(f"{label} exceeds {maximum} characters")
    return normalized


def _optional_text(value: Any, *, label: str, maximum: int = 4096) -> str | None:
    if value is None:
        return None
    return _text(value, label=label, maximum=maximum)


def _category(value: Any, *, label: str) -> str:
    source = _text(value, label=label, maximum=80)
    normalized = source.lower().replace(" ", "-")
    if source != normalized:
        raise DiscoveryError(
            f"{label} must already use canonical lowercase-hyphen form"
        )
    if not CATEGORY_RE.fullmatch(normalized):
        raise DiscoveryError(
            f"{label} must normalize to lowercase letters, digits, '.', '_' or '-'"
        )
    return normalized


def _integer(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiscoveryError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise DiscoveryError(f"{label} must be between {minimum} and {maximum}")
    return value


def _sha256(value: Any, *, label: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise DiscoveryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: Any, *, label: str) -> str:
    normalized = _text(value, label=label, maximum=20)
    if not TIMESTAMP_RE.fullmatch(normalized):
        raise DiscoveryError(f"{label} must use UTC YYYY-MM-DDTHH:MM:SSZ")
    return normalized


def _sorted_unique_texts(
    value: Any,
    *,
    label: str,
    category: bool = False,
    maximum_items: int = 64,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DiscoveryError(f"{label} must be a non-empty list")
    if len(value) > maximum_items:
        raise DiscoveryError(f"{label} exceeds {maximum_items} items")
    normalizer = _category if category else _text
    normalized = [normalizer(item, label=f"{label}[]") for item in value]
    if len(set(normalized)) != len(normalized):
        raise DiscoveryError(f"{label} must not contain duplicates")
    return sorted(normalized)


def _https_url(value: Any, *, label: str) -> str:
    raw = _text(value, label=label, maximum=2048)
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise DiscoveryError(f"{label} must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise DiscoveryError(f"{label} must not contain user information")
    return raw


def _canonical_url(value: Any, *, label: str) -> str:
    raw = _https_url(value, label=label)
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = host if port in (None, 443) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query_items = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, item))
    query = urlencode(sorted(query_items))
    return urlunsplit(("https", netloc, path, query, ""))


def _normalize_identifier(kind: str, value: Any) -> str:
    normalized = _text(value, label=f"source.identifiers.{kind}", maximum=256)
    lowered = normalized.lower()
    if kind == "doi":
        lowered = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", lowered)
        if not lowered.startswith("10.") or "/" not in lowered:
            raise DiscoveryError("DOI identifier is malformed")
        return lowered
    if kind == "arxiv":
        lowered = re.sub(
            r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)", "", lowered
        )
        lowered = lowered.removesuffix(".pdf")
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", lowered):
            raise DiscoveryError("arXiv identifier is malformed")
        return lowered
    if kind == "pmid":
        if not normalized.isdigit():
            raise DiscoveryError("PMID identifier must contain only digits")
        return normalized
    raise DiscoveryError(f"unsupported identifier kind: {kind}")


def _normalize_query_plan(value: Any) -> dict[str, Any]:
    plan = _expect_object(value, label="query plan")
    _expect_keys(
        plan,
        required={
            "schema_version",
            "plan_id",
            "purpose",
            "search_execution",
            "queries",
            "deduplication",
            "coverage",
        },
        label="query plan",
    )
    if plan["schema_version"] != 1:
        raise DiscoveryError("query plan schema_version must be 1")
    execution = _expect_object(plan["search_execution"], label="search_execution")
    _expect_keys(
        execution,
        required={"pipeline_mode", "intended_result_custody"},
        label="search_execution",
    )
    if execution["pipeline_mode"] != "planning_only_no_network":
        raise DiscoveryError(
            "search_execution.pipeline_mode must be planning_only_no_network"
        )
    if execution["intended_result_custody"] != "operator_provided_raw_manifest":
        raise DiscoveryError(
            "search_execution.intended_result_custody must be operator_provided_raw_manifest"
        )

    queries_value = plan["queries"]
    if not isinstance(queries_value, list) or not queries_value:
        raise DiscoveryError("queries must be a non-empty list")
    normalized_queries = []
    query_ids: set[str] = set()
    for index, item in enumerate(queries_value):
        query = _expect_object(item, label=f"queries[{index}]")
        _expect_keys(
            query,
            required={
                "query_id",
                "query_text",
                "provider_hint",
                "requested_limit",
                "source_types",
                "facets",
            },
            label=f"queries[{index}]",
        )
        query_id = _category(query["query_id"], label=f"queries[{index}].query_id")
        if query_id in query_ids:
            raise DiscoveryError(f"duplicate query_id: {query_id}")
        query_ids.add(query_id)
        facets = _expect_object(query["facets"], label=f"queries[{index}].facets")
        _expect_keys(
            facets,
            required={"quantum_domains", "situations", "evidence_roles"},
            label=f"queries[{index}].facets",
        )
        normalized_queries.append(
            {
                "facets": {
                    "evidence_roles": _sorted_unique_texts(
                        facets["evidence_roles"],
                        label=f"queries[{index}].facets.evidence_roles",
                        category=True,
                    ),
                    "quantum_domains": _sorted_unique_texts(
                        facets["quantum_domains"],
                        label=f"queries[{index}].facets.quantum_domains",
                        category=True,
                    ),
                    "situations": _sorted_unique_texts(
                        facets["situations"],
                        label=f"queries[{index}].facets.situations",
                        category=True,
                    ),
                },
                "provider_hint": _category(
                    query["provider_hint"], label=f"queries[{index}].provider_hint"
                ),
                "query_id": query_id,
                "query_text": _text(
                    query["query_text"],
                    label=f"queries[{index}].query_text",
                    maximum=2048,
                ),
                "requested_limit": _integer(
                    query["requested_limit"],
                    label=f"queries[{index}].requested_limit",
                    minimum=1,
                    maximum=10000,
                ),
                "source_types": _sorted_unique_texts(
                    query["source_types"],
                    label=f"queries[{index}].source_types",
                    category=True,
                ),
            }
        )

    deduplication = _expect_object(plan["deduplication"], label="deduplication")
    _expect_keys(
        deduplication,
        required={"near_duplicate_threshold_percent"},
        label="deduplication",
    )
    threshold = _integer(
        deduplication["near_duplicate_threshold_percent"],
        label="deduplication.near_duplicate_threshold_percent",
        minimum=1,
        maximum=100,
    )

    coverage = _expect_object(plan["coverage"], label="coverage")
    _expect_keys(coverage, required={"sample_size", "targets"}, label="coverage")
    targets = _expect_object(coverage["targets"], label="coverage.targets")
    if not targets:
        raise DiscoveryError("coverage.targets must not be empty")
    if not set(targets).issubset(COVERAGE_DIMENSIONS):
        unsupported = sorted(set(targets) - COVERAGE_DIMENSIONS)
        raise DiscoveryError(
            f"unsupported coverage dimensions: {', '.join(unsupported)}"
        )
    normalized_targets = {
        dimension: _sorted_unique_texts(
            targets[dimension],
            label=f"coverage.targets.{dimension}",
            category=True,
        )
        for dimension in sorted(targets)
    }
    return {
        "coverage": {
            "sample_size": _integer(
                coverage["sample_size"],
                label="coverage.sample_size",
                minimum=1,
                maximum=1000,
            ),
            "targets": normalized_targets,
        },
        "deduplication": {"near_duplicate_threshold_percent": threshold},
        "plan_id": _category(plan["plan_id"], label="plan_id"),
        "purpose": _text(plan["purpose"], label="purpose", maximum=2048),
        "queries": sorted(normalized_queries, key=lambda item: item["query_id"]),
        "schema_version": 1,
        "search_execution": {
            "intended_result_custody": "operator_provided_raw_manifest",
            "pipeline_mode": "planning_only_no_network",
        },
    }


def _normalize_collection(value: Any) -> dict[str, Any]:
    collection = _expect_object(value, label="collection")
    _expect_keys(
        collection,
        required={
            "input_custody",
            "collection_method",
            "provider_name",
            "provider_version",
            "adapter_version",
            "observed_at",
            "search_execution_claim",
        },
        optional={"execution_attestation"},
        label="collection",
    )
    if collection["input_custody"] != "operator_provided":
        raise DiscoveryError("collection.input_custody must be operator_provided")
    method = collection["collection_method"]
    if method not in COLLECTION_METHODS:
        raise DiscoveryError("unsupported collection.collection_method")
    claim = collection["search_execution_claim"]
    if claim not in SEARCH_EXECUTION_CLAIMS:
        raise DiscoveryError("unsupported collection.search_execution_claim")
    attestation_value = collection.get("execution_attestation")
    if claim == "not_claimed":
        if attestation_value is not None:
            raise DiscoveryError(
                "not_claimed collection must not include an attestation"
            )
        attestation = None
    else:
        attestation_object = _expect_object(
            attestation_value, label="collection.execution_attestation"
        )
        _expect_keys(
            attestation_object,
            required={"executed_at", "executor_id", "raw_response_sha256"},
            label="collection.execution_attestation",
        )
        attestation = {
            "executed_at": _timestamp(
                attestation_object["executed_at"],
                label="collection.execution_attestation.executed_at",
            ),
            "executor_id": _category(
                attestation_object["executor_id"],
                label="collection.execution_attestation.executor_id",
            ),
            "raw_response_sha256": _sha256(
                attestation_object["raw_response_sha256"],
                label="collection.execution_attestation.raw_response_sha256",
            ),
        }
    return {
        "adapter_version": _text(
            collection["adapter_version"],
            label="collection.adapter_version",
            maximum=128,
        ),
        "collection_method": method,
        "execution_attestation": attestation,
        "input_custody": "operator_provided",
        "observed_at": _timestamp(
            collection["observed_at"], label="collection.observed_at"
        ),
        "provider_name": _category(
            collection["provider_name"], label="collection.provider_name"
        ),
        "provider_version": _text(
            collection["provider_version"],
            label="collection.provider_version",
            maximum=128,
        ),
        "search_execution_claim": claim,
    }


def _normalize_source(value: Any, *, label: str) -> dict[str, Any]:
    source = _expect_object(value, label=label)
    _expect_keys(
        source,
        required={"title", "url", "source_type", "authors", "publisher", "identifiers"},
        label=label,
    )
    authors_value = source["authors"]
    if not isinstance(authors_value, list) or not authors_value:
        raise DiscoveryError(f"{label}.authors must be a non-empty list")
    authors = [
        _text(item, label=f"{label}.authors[]", maximum=256) for item in authors_value
    ]
    identifiers_value = _expect_object(
        source["identifiers"], label=f"{label}.identifiers"
    )
    if not set(identifiers_value).issubset({"arxiv", "doi", "pmid"}):
        raise DiscoveryError(
            f"{label}.identifiers contains an unsupported identifier kind"
        )
    identifiers = {
        kind: _normalize_identifier(kind, identifiers_value[kind])
        for kind in sorted(identifiers_value)
    }
    return {
        "authors": authors,
        "canonical_url": _canonical_url(source["url"], label=f"{label}.url"),
        "identifiers": identifiers,
        "publisher": _text(
            source["publisher"], label=f"{label}.publisher", maximum=256
        ),
        "source_type": _category(source["source_type"], label=f"{label}.source_type"),
        "title": _text(source["title"], label=f"{label}.title", maximum=1024),
    }


def _normalize_version(value: Any, *, label: str) -> dict[str, Any]:
    version = _expect_object(value, label=label)
    _expect_keys(
        version,
        required={
            "version_type",
            "version_label",
            "version_locator",
            "observed_at",
            "content_sha256",
        },
        label=label,
    )
    return {
        "content_sha256": _sha256(
            version["content_sha256"], label=f"{label}.content_sha256", nullable=True
        ),
        "observed_at": _timestamp(version["observed_at"], label=f"{label}.observed_at"),
        "version_label": _text(
            version["version_label"], label=f"{label}.version_label", maximum=128
        ),
        "version_locator": _canonical_url(
            version["version_locator"], label=f"{label}.version_locator"
        ),
        "version_type": _category(
            version["version_type"], label=f"{label}.version_type"
        ),
    }


def _normalize_license(value: Any, *, label: str) -> dict[str, Any]:
    license_value = _expect_object(value, label=label)
    _expect_keys(
        license_value,
        required={"status", "spdx_id", "evidence_url", "redistribution_allowed"},
        label=label,
    )
    status = license_value["status"]
    if status not in LICENSE_STATUSES:
        raise DiscoveryError(f"{label}.status is unsupported")
    spdx_id = _optional_text(
        license_value["spdx_id"], label=f"{label}.spdx_id", maximum=80
    )
    evidence_url = license_value["evidence_url"]
    if evidence_url is not None:
        evidence_url = _canonical_url(evidence_url, label=f"{label}.evidence_url")
    redistribution = license_value["redistribution_allowed"]
    if redistribution is not None and not isinstance(redistribution, bool):
        raise DiscoveryError(f"{label}.redistribution_allowed must be boolean or null")
    if status == "known" and spdx_id is None:
        raise DiscoveryError(f"{label}.spdx_id is required when status is known")
    if status == "unknown" and (spdx_id is not None or redistribution is not None):
        raise DiscoveryError(
            f"{label} unknown status must not assert license permissions"
        )
    if status == "restricted" and redistribution is not False:
        raise DiscoveryError(
            f"{label} restricted status must set redistribution_allowed=false"
        )
    return {
        "evidence_url": evidence_url,
        "redistribution_allowed": redistribution,
        "spdx_id": spdx_id,
        "status": status,
    }


def _normalize_problem(value: Any, *, label: str) -> dict[str, Any]:
    problem = _expect_object(value, label=label)
    _expect_keys(
        problem,
        required={
            "quantum_domain",
            "situation",
            "task_archetype",
            "objective",
            "tensorcircuit_path",
            "evidence_role",
            "keywords",
        },
        label=label,
    )
    return {
        "evidence_role": _category(
            problem["evidence_role"], label=f"{label}.evidence_role"
        ),
        "keywords": _sorted_unique_texts(
            problem["keywords"], label=f"{label}.keywords", category=True
        ),
        "objective": _text(
            problem["objective"], label=f"{label}.objective", maximum=2048
        ),
        "quantum_domain": _category(
            problem["quantum_domain"], label=f"{label}.quantum_domain"
        ),
        "situation": _category(problem["situation"], label=f"{label}.situation"),
        "task_archetype": _category(
            problem["task_archetype"], label=f"{label}.task_archetype"
        ),
        "tensorcircuit_path": _text(
            problem["tensorcircuit_path"],
            label=f"{label}.tensorcircuit_path",
            maximum=2048,
        ),
    }


def _normalize_raw_manifest(
    value: Any,
    *,
    expected_plan_sha256: str,
    query_ids: set[str],
) -> dict[str, Any]:
    manifest = _expect_object(value, label="raw hit manifest")
    _expect_keys(
        manifest,
        required={
            "schema_version",
            "manifest_id",
            "search_plan_artifact_sha256",
            "collection",
            "hits",
        },
        label="raw hit manifest",
    )
    if manifest["schema_version"] != 1:
        raise DiscoveryError("raw hit manifest schema_version must be 1")
    bound_plan = _sha256(
        manifest["search_plan_artifact_sha256"],
        label="search_plan_artifact_sha256",
    )
    if bound_plan != expected_plan_sha256:
        raise DiscoveryError(
            "raw hit manifest is not bound to this search-plan artifact"
        )
    hits_value = manifest["hits"]
    if not isinstance(hits_value, list) or not hits_value:
        raise DiscoveryError("raw hit manifest hits must be a non-empty list")
    collection = _normalize_collection(manifest["collection"])
    hits: list[dict[str, Any]] = []
    hit_ids: set[str] = set()
    ranks: set[tuple[str, int]] = set()
    for index, item in enumerate(hits_value):
        hit = _expect_object(item, label=f"hits[{index}]")
        _expect_keys(
            hit,
            required={
                "hit_id",
                "query_id",
                "rank",
                "source",
                "version",
                "license",
                "problem",
            },
            label=f"hits[{index}]",
        )
        hit_id = _category(hit["hit_id"], label=f"hits[{index}].hit_id")
        if hit_id in hit_ids:
            raise DiscoveryError(f"duplicate hit_id: {hit_id}")
        hit_ids.add(hit_id)
        query_id = _category(hit["query_id"], label=f"hits[{index}].query_id")
        if query_id not in query_ids:
            raise DiscoveryError(f"hit references unknown query_id: {query_id}")
        rank = _integer(
            hit["rank"], label=f"hits[{index}].rank", minimum=1, maximum=1000000
        )
        if (query_id, rank) in ranks:
            raise DiscoveryError(f"duplicate rank {rank} for query {query_id}")
        ranks.add((query_id, rank))
        hits.append(
            {
                "hit_id": hit_id,
                "license": _normalize_license(
                    hit["license"], label=f"hits[{index}].license"
                ),
                "problem": _normalize_problem(
                    hit["problem"], label=f"hits[{index}].problem"
                ),
                "query_id": query_id,
                "rank": rank,
                "source": _normalize_source(
                    hit["source"], label=f"hits[{index}].source"
                ),
                "version": _normalize_version(
                    hit["version"], label=f"hits[{index}].version"
                ),
            }
        )
    return {
        "collection": collection,
        "hits": sorted(
            hits, key=lambda hit: (hit["query_id"], hit["rank"], hit["hit_id"])
        ),
        "manifest_id": _category(manifest["manifest_id"], label="manifest_id"),
        "schema_version": 1,
        "search_plan_artifact_sha256": bound_plan,
    }


def freeze_search_plan(store: ArtifactStore, plan_path: Path | str) -> ArtifactRef:
    """Validate and freeze a query plan without executing any search."""

    raw, input_sha256 = _load_input_json(store, plan_path, label="query plan")
    plan = _normalize_query_plan(raw)
    return store.write(
        "search-plan",
        {
            "execution_status": "query_plan_only_not_executed",
            "input_sha256": input_sha256,
            "pipeline_executed_web_search": False,
            "plan": plan,
        },
    )


def ingest_raw_manifests(
    store: ArtifactStore,
    search_plan_ref: ArtifactRef,
    manifest_paths: Sequence[Path | str],
) -> ArtifactRef:
    """Freeze operator-provided raw manifests and build an immutable ingest index."""

    if not manifest_paths:
        raise DiscoveryError("at least one raw hit manifest is required")
    plan_artifact = store.read(search_plan_ref)
    if search_plan_ref.stage != "search-plan":
        raise DiscoveryError("raw ingest requires a search-plan artifact")
    query_ids = {query["query_id"] for query in plan_artifact["plan"]["queries"]}
    manifest_refs: list[ArtifactRef] = []
    seen_manifest_ids: set[str] = set()
    for path in manifest_paths:
        raw, input_sha256 = _load_input_json(store, path, label="raw hit manifest")
        manifest = _normalize_raw_manifest(
            raw,
            expected_plan_sha256=search_plan_ref.sha256,
            query_ids=query_ids,
        )
        if manifest["manifest_id"] in seen_manifest_ids:
            raise DiscoveryError(f"duplicate manifest_id: {manifest['manifest_id']}")
        seen_manifest_ids.add(manifest["manifest_id"])
        manifest_refs.append(
            store.write(
                "raw-manifest",
                {
                    "execution_evidence_verification": "not_performed_operator_attestation_only",
                    "input_sha256": input_sha256,
                    "manifest": manifest,
                    "pipeline_executed_web_search": False,
                    "search_plan": search_plan_ref.as_dict(),
                },
            )
        )
    sorted_refs = sorted(manifest_refs, key=lambda ref: ref.sha256)
    return store.write(
        "raw-ingest",
        {
            "input_custody": "operator_provided_raw_manifests",
            "pipeline_executed_web_search": False,
            "raw_manifests": [ref.as_dict() for ref in sorted_refs],
            "search_plan": search_plan_ref.as_dict(),
        },
    )


def _ref_from_dict(value: Any, *, expected_stage: str, label: str) -> ArtifactRef:
    ref_value = _expect_object(value, label=label)
    _expect_keys(ref_value, required={"stage", "sha256"}, label=label)
    ref = ArtifactRef(stage=ref_value["stage"], sha256=ref_value["sha256"])
    _validate_ref(ref)
    if ref.stage != expected_stage:
        raise DiscoveryError(f"{label} must reference stage {expected_stage}")
    return ref


def _source_identity_keys(
    source: Mapping[str, Any], version: Mapping[str, Any]
) -> list[str]:
    identifiers = source["identifiers"]
    keys = []
    if "doi" in identifiers:
        keys.append(f"doi:{identifiers['doi']}")
    if "arxiv" in identifiers:
        base = re.sub(r"v\d+$", "", identifiers["arxiv"])
        keys.append(f"arxiv:{base}")
    if "pmid" in identifiers:
        keys.append(f"pmid:{identifiers['pmid']}")
    if version["content_sha256"] is not None:
        keys.append(f"content-sha256:{version['content_sha256']}")
    if not keys:
        keys.append(f"url:{source['canonical_url']}")
    return sorted(keys)


def _record_id(record_without_id: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(record_without_id))


def normalize_ingest(store: ArtifactStore, ingest_ref: ArtifactRef) -> ArtifactRef:
    """Create normalized problem-source record artifacts from one raw ingest."""

    if ingest_ref.stage != "raw-ingest":
        raise DiscoveryError("normalize requires a raw-ingest artifact")
    ingest = store.read(ingest_ref)
    plan_ref = _ref_from_dict(
        ingest["search_plan"],
        expected_stage="search-plan",
        label="raw-ingest.search_plan",
    )
    record_refs: list[ArtifactRef] = []
    record_ids: set[str] = set()
    for manifest_ref_value in ingest["raw_manifests"]:
        manifest_ref = _ref_from_dict(
            manifest_ref_value,
            expected_stage="raw-manifest",
            label="raw-ingest.raw_manifests[]",
        )
        manifest_artifact = store.read(manifest_ref)
        manifest = manifest_artifact["manifest"]
        collection = manifest["collection"]
        for hit in manifest["hits"]:
            record_without_id = {
                "license": hit["license"],
                "problem": hit["problem"],
                "provenance": {
                    "adapter_version": collection["adapter_version"],
                    "collection_method": collection["collection_method"],
                    "execution_attestation": collection["execution_attestation"],
                    "execution_evidence_verification": (
                        "not_performed_operator_attestation_only"
                    ),
                    "hit_id": hit["hit_id"],
                    "input_custody": collection["input_custody"],
                    "manifest_id": manifest["manifest_id"],
                    "observed_at": collection["observed_at"],
                    "pipeline_executed_web_search": False,
                    "provider_name": collection["provider_name"],
                    "provider_version": collection["provider_version"],
                    "query_id": hit["query_id"],
                    "rank": hit["rank"],
                    "raw_manifest": manifest_ref.as_dict(),
                    "search_execution_claim": collection["search_execution_claim"],
                    "search_plan": plan_ref.as_dict(),
                },
                "source": {
                    **hit["source"],
                    "identity_keys": _source_identity_keys(
                        hit["source"], hit["version"]
                    ),
                },
                "version": hit["version"],
            }
            record_id = _record_id(record_without_id)
            if record_id in record_ids:
                continue
            record_ids.add(record_id)
            record = {"record_id": record_id, **record_without_id}
            record_refs.append(store.write("normalized-record", {"record": record}))
    sorted_refs = sorted(record_refs, key=lambda ref: ref.sha256)
    return store.write(
        "normalize",
        {
            "ingest": ingest_ref.as_dict(),
            "record_count": len(sorted_refs),
            "records": [ref.as_dict() for ref in sorted_refs],
            "search_plan": plan_ref.as_dict(),
        },
    )


def _load_records(
    store: ArtifactStore, normalize_ref: ArtifactRef
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if normalize_ref.stage != "normalize":
        raise DiscoveryError("record loading requires a normalize artifact")
    normalize_artifact = store.read(normalize_ref)
    records = []
    for value in normalize_artifact["records"]:
        ref = _ref_from_dict(
            value, expected_stage="normalized-record", label="normalize.records[]"
        )
        records.append(store.read(ref)["record"])
    records.sort(key=lambda record: record["record_id"])
    if len(records) != normalize_artifact["record_count"]:
        raise DiscoveryError("normalize record count does not match referenced records")
    return normalize_artifact, records


class _UnionFind:
    def __init__(self, values: Sequence[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def _group_id(prefix: str, members: Sequence[str]) -> str:
    digest = sha256_bytes(canonical_json_bytes(sorted(members)))
    return f"{prefix}-{digest}"


def _semantic_tokens(record: Mapping[str, Any]) -> set[str]:
    problem = record["problem"]
    text = " ".join(
        [
            record["source"]["title"],
            problem["objective"],
            problem["quantum_domain"],
            problem["situation"],
            problem["task_archetype"],
            *problem["keywords"],
        ]
    )
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return {
        token
        for token in TOKEN_RE.findall(normalized)
        if len(token) >= 2 and token not in STOPWORDS
    }


def _similarity(left: set[str], right: set[str]) -> tuple[int, int]:
    union = left | right
    if not union:
        return (1, 1)
    return (len(left & right), len(union))


def group_duplicates(store: ArtifactStore, normalize_ref: ArtifactRef) -> ArtifactRef:
    """Build deterministic exact-identity and near-semantic duplicate groups."""

    normalize_artifact, records = _load_records(store, normalize_ref)
    if not records:
        raise DiscoveryError("dedupe requires at least one normalized record")
    plan_ref = _ref_from_dict(
        normalize_artifact["search_plan"],
        expected_stage="search-plan",
        label="normalize.search_plan",
    )
    plan = store.read(plan_ref)["plan"]
    threshold = plan["deduplication"]["near_duplicate_threshold_percent"]
    records_by_id = {record["record_id"]: record for record in records}
    exact_union = _UnionFind(sorted(records_by_id))
    key_owner: dict[str, str] = {}
    for record in records:
        record_id = record["record_id"]
        for identity_key in record["source"]["identity_keys"]:
            owner = key_owner.setdefault(identity_key, record_id)
            exact_union.union(owner, record_id)
    exact_components: dict[str, list[str]] = {}
    for record_id in sorted(records_by_id):
        exact_components.setdefault(exact_union.find(record_id), []).append(record_id)
    exact_groups = []
    exact_by_id: dict[str, list[str]] = {}
    for members in sorted(exact_components.values(), key=lambda values: tuple(values)):
        group_id = _group_id("exact", members)
        identity_keys = sorted(
            {
                key
                for member in members
                for key in records_by_id[member]["source"]["identity_keys"]
            }
        )
        exact_by_id[group_id] = members
        exact_groups.append(
            {
                "exact_group_id": group_id,
                "identity_keys": identity_keys,
                "member_record_ids": members,
            }
        )

    exact_group_ids = sorted(exact_by_id)
    near_union = _UnionFind(exact_group_ids)
    tokens_by_record = {
        record_id: _semantic_tokens(record)
        for record_id, record in records_by_id.items()
    }
    near_edges = []
    for left_index, left_group in enumerate(exact_group_ids):
        for right_group in exact_group_ids[left_index + 1 :]:
            best: tuple[int, int, str, str] | None = None
            for left_record in exact_by_id[left_group]:
                for right_record in exact_by_id[right_group]:
                    numerator, denominator = _similarity(
                        tokens_by_record[left_record], tokens_by_record[right_record]
                    )
                    candidate = (numerator, denominator, left_record, right_record)
                    if best is None:
                        best = candidate
                        continue
                    left_product = numerator * best[1]
                    right_product = best[0] * denominator
                    if left_product > right_product or (
                        left_product == right_product and candidate[2:] < best[2:]
                    ):
                        best = candidate
            assert best is not None
            numerator, denominator, left_record, right_record = best
            if numerator * 100 >= denominator * threshold:
                near_union.union(left_group, right_group)
                near_edges.append(
                    {
                        "left_exact_group_id": left_group,
                        "left_record_id": left_record,
                        "right_exact_group_id": right_group,
                        "right_record_id": right_record,
                        "similarity_denominator": denominator,
                        "similarity_numerator": numerator,
                        "similarity_percent_floor": numerator * 100 // denominator,
                    }
                )
    near_components: dict[str, list[str]] = {}
    for group_id in exact_group_ids:
        near_components.setdefault(near_union.find(group_id), []).append(group_id)
    near_groups = []
    for exact_members in sorted(
        near_components.values(), key=lambda values: tuple(values)
    ):
        record_members = sorted(
            record_id
            for group_id in exact_members
            for record_id in exact_by_id[group_id]
        )
        near_groups.append(
            {
                "member_exact_group_ids": sorted(exact_members),
                "member_record_ids": record_members,
                "near_group_id": _group_id("near", record_members),
            }
        )
    return store.write(
        "dedupe",
        {
            "exact_groups": exact_groups,
            "near_duplicate_edges": sorted(
                near_edges,
                key=lambda edge: (
                    edge["left_exact_group_id"],
                    edge["right_exact_group_id"],
                ),
            ),
            "near_duplicate_threshold_percent": threshold,
            "near_groups": near_groups,
            "normalize": normalize_ref.as_dict(),
            "search_plan": plan_ref.as_dict(),
        },
    )


def _coverage_value(record: Mapping[str, Any], dimension: str) -> str:
    if dimension == "source_type":
        return record["source"]["source_type"]
    if dimension == "license_status":
        return record["license"]["status"]
    if dimension == "search_execution_claim":
        return record["provenance"]["search_execution_claim"]
    return record["problem"][dimension]


def _quality_score(record: Mapping[str, Any]) -> int:
    return sum(
        (
            4 if record["license"]["status"] == "known" else 0,
            2 if record["license"]["redistribution_allowed"] is True else 0,
            2 if record["version"]["content_sha256"] is not None else 0,
        )
    )


def sample_coverage(
    store: ArtifactStore,
    search_plan_ref: ArtifactRef,
    normalize_ref: ArtifactRef,
    dedupe_ref: ArtifactRef,
) -> ArtifactRef:
    """Select at most one record per near group using deterministic set cover."""

    if search_plan_ref.stage != "search-plan" or dedupe_ref.stage != "dedupe":
        raise DiscoveryError("coverage sampling received an incorrect artifact stage")
    plan_artifact = store.read(search_plan_ref)
    normalize_artifact, records = _load_records(store, normalize_ref)
    dedupe = store.read(dedupe_ref)
    if dedupe["normalize"] != normalize_ref.as_dict():
        raise DiscoveryError("dedupe artifact is not bound to this normalize artifact")
    if dedupe["search_plan"] != search_plan_ref.as_dict():
        raise DiscoveryError("dedupe artifact is not bound to this search plan")
    if normalize_artifact["search_plan"] != search_plan_ref.as_dict():
        raise DiscoveryError("normalize artifact is not bound to this search plan")
    plan = plan_artifact["plan"]
    coverage = plan["coverage"]
    target_pairs = {
        (dimension, target)
        for dimension, values in coverage["targets"].items()
        for target in values
    }
    records_by_id = {record["record_id"]: record for record in records}
    near_group_by_record: dict[str, str] = {}
    for group in dedupe["near_groups"]:
        for record_id in group["member_record_ids"]:
            if record_id in near_group_by_record:
                raise DiscoveryError(
                    "record appears in more than one near-duplicate group"
                )
            near_group_by_record[record_id] = group["near_group_id"]
    if set(near_group_by_record) != set(records_by_id):
        raise DiscoveryError("dedupe groups do not cover all normalized records")

    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    covered: set[tuple[str, str]] = set()
    dimensions = sorted(coverage["targets"])
    while len(selected) < coverage["sample_size"]:
        candidates = []
        for record_id, record in records_by_id.items():
            group_id = near_group_by_record[record_id]
            if group_id in selected_groups:
                continue
            record_pairs = {
                (dimension, _coverage_value(record, dimension))
                for dimension in dimensions
            }
            newly_covered = sorted(record_pairs & target_pairs - covered)
            candidates.append(
                (
                    -len(newly_covered),
                    -_quality_score(record),
                    record_id,
                    group_id,
                    newly_covered,
                )
            )
        if not candidates:
            break
        _, _, record_id, group_id, newly_covered = min(candidates)
        covered.update(newly_covered)
        selected_groups.add(group_id)
        selected.append(
            {
                "covered_new_targets": [
                    {"dimension": dimension, "value": value}
                    for dimension, value in newly_covered
                ],
                "near_group_id": group_id,
                "record": records_by_id[record_id],
                "selection_rank": len(selected) + 1,
            }
        )
    covered_sorted = sorted(covered)
    unmet_sorted = sorted(target_pairs - covered)
    return store.write(
        "coverage-sample",
        {
            "coverage_method": "deterministic_greedy_target_set_cover_one_per_near_group",
            "covered_targets": [
                {"dimension": dimension, "value": value}
                for dimension, value in covered_sorted
            ],
            "dedupe": dedupe_ref.as_dict(),
            "normalize": normalize_ref.as_dict(),
            "requested_sample_size": coverage["sample_size"],
            "search_plan": search_plan_ref.as_dict(),
            "selected": selected,
            "selected_count": len(selected),
            "unmet_targets": [
                {"dimension": dimension, "value": value}
                for dimension, value in unmet_sorted
            ],
        },
    )


def run_discovery(
    store: ArtifactStore,
    plan_path: Path | str,
    manifest_paths: Sequence[Path | str],
) -> dict[str, ArtifactRef]:
    """Run every offline stage, reusing any matching content-addressed artifacts."""

    plan_ref = freeze_search_plan(store, plan_path)
    ingest_ref = ingest_raw_manifests(store, plan_ref, manifest_paths)
    normalize_ref = normalize_ingest(store, ingest_ref)
    dedupe_ref = group_duplicates(store, normalize_ref)
    sample_ref = sample_coverage(store, plan_ref, normalize_ref, dedupe_ref)
    return {
        "search_plan": plan_ref,
        "raw_ingest": ingest_ref,
        "normalize": normalize_ref,
        "dedupe": dedupe_ref,
        "coverage_sample": sample_ref,
    }


def _artifact_output(store: ArtifactStore, ref: ArtifactRef) -> dict[str, str]:
    return {
        **ref.as_dict(),
        "relative_path": str(
            store.artifact_path(ref).relative_to(store.workspace_root)
        ),
    }


def _store_from_args(args: argparse.Namespace) -> ArtifactStore:
    return ArtifactStore(args.workspace_root, args.store_root)


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path(".artifacts/problem-discovery/autoresearch"),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic offline ORBIT-Q source discovery stages."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="freeze a query plan; execute no search")
    _add_store_arguments(plan)
    plan.add_argument("--plan", type=Path, required=True)

    ingest = subparsers.add_parser(
        "ingest", help="freeze operator-provided raw manifests"
    )
    _add_store_arguments(ingest)
    ingest.add_argument("--plan-sha256", required=True)
    ingest.add_argument("--raw-manifest", type=Path, action="append", required=True)

    normalize = subparsers.add_parser("normalize", help="normalize one ingest artifact")
    _add_store_arguments(normalize)
    normalize.add_argument("--ingest-sha256", required=True)

    dedupe = subparsers.add_parser("dedupe", help="group exact and near duplicates")
    _add_store_arguments(dedupe)
    dedupe.add_argument("--normalize-sha256", required=True)

    sample = subparsers.add_parser(
        "sample", help="take a deterministic coverage sample"
    )
    _add_store_arguments(sample)
    sample.add_argument("--plan-sha256", required=True)
    sample.add_argument("--normalize-sha256", required=True)
    sample.add_argument("--dedupe-sha256", required=True)

    run = subparsers.add_parser("run", help="run all offline stages")
    _add_store_arguments(run)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--raw-manifest", type=Path, action="append", required=True)
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point used by the thin repository script."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        store = _store_from_args(args)
        if args.command == "plan":
            ref = freeze_search_plan(store, args.plan)
            output: Any = {"search_plan": _artifact_output(store, ref)}
        elif args.command == "ingest":
            plan_ref = ArtifactRef("search-plan", args.plan_sha256)
            ref = ingest_raw_manifests(store, plan_ref, args.raw_manifest)
            output = {"raw_ingest": _artifact_output(store, ref)}
        elif args.command == "normalize":
            ref = normalize_ingest(store, ArtifactRef("raw-ingest", args.ingest_sha256))
            output = {"normalize": _artifact_output(store, ref)}
        elif args.command == "dedupe":
            ref = group_duplicates(
                store, ArtifactRef("normalize", args.normalize_sha256)
            )
            output = {"dedupe": _artifact_output(store, ref)}
        elif args.command == "sample":
            ref = sample_coverage(
                store,
                ArtifactRef("search-plan", args.plan_sha256),
                ArtifactRef("normalize", args.normalize_sha256),
                ArtifactRef("dedupe", args.dedupe_sha256),
            )
            output = {"coverage_sample": _artifact_output(store, ref)}
        else:
            refs = run_discovery(store, args.plan, args.raw_manifest)
            output = {
                "artifacts": {
                    name: _artifact_output(store, ref)
                    for name, ref in sorted(refs.items())
                },
                "pipeline_executed_web_search": False,
            }
    except DiscoveryError as exc:
        parser.error(str(exc))
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "DiscoveryError",
    "canonical_json_bytes",
    "cli",
    "freeze_search_plan",
    "group_duplicates",
    "ingest_raw_manifests",
    "normalize_ingest",
    "run_discovery",
    "sample_coverage",
]
