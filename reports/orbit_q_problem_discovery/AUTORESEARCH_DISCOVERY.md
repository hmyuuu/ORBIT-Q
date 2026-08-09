# Offline autoresearch discovery stage

This component converts a human-authored query plan plus operator-provided raw
hit manifests into normalized, duplicate-grouped, coverage-sampled quantum
problem-source records. It is deterministic and deliberately has no network
client.

## Evidence boundary

The distinctions below are part of every artifact, not just documentation:

- A `search-plan` artifact proves only that queries were planned. It always says
  `query_plan_only_not_executed` and `pipeline_executed_web_search=false`.
- Every raw manifest enters with `input_custody=operator_provided`. The manifest
  can make no execution claim (`not_claimed`) or carry an
  `operator_attested_external_execution` claim with a timestamp, executor ID,
  and raw-response digest.
- An external-execution claim is copied as provenance but remains operator
  attestation. The pipeline records
  `execution_evidence_verification=not_performed_operator_attestation_only`.
- Normalized records are research leads. They are not proof that a source
  supports every proposed task, not expert validation, and not evidence that a
  model fails a benchmark.

The raw-manifest format accepts structured metadata only. It does not accept
HTTP headers, credentials, arbitrary provider payloads, or source bodies.

## Stages and resume units

1. `search-plan` validates and freezes a query plan. It performs no search.
2. `raw-manifest` freezes each operator-provided manifest independently;
   `raw-ingest` is an order-independent index over those artifacts.
3. `normalized-record` stores each source/problem/provenance tuple;
   `normalize` indexes the records.
4. `dedupe` groups records first by exact identifiers or content hashes, then by
   deterministic token-Jaccard similarity. The plan freezes the integer
   threshold.
5. `coverage-sample` uses deterministic greedy target-set coverage and selects
   at most one record from each near-duplicate group.

Artifacts live at `<store>/<stage>/<sha256>.json`, where `sha256` is the digest
of the exact canonical JSON bytes in the file. Re-running a stage reuses matching
artifacts. Writes use a same-directory temporary file, `fsync`, and atomic
replacement; existing content is verified before reuse.

No stage records wall-clock time or an absolute host path. Observed and execution
times must come from the immutable operator input.

## Input schemas

- `autoresearch_query_plan.schema.json`
- `autoresearch_raw_hits.schema.json`
- `autoresearch_problem_source_record.schema.json`
- `autoresearch_artifact.schema.json`

Runtime validation is stricter than the descriptive JSON schemas: it rejects
duplicate JSON keys, non-finite numbers, malformed identifiers, unsupported
fields, credential-like strings, inconsistent license claims, duplicate ranks,
and manifests that are not bound to the frozen plan digest.

## Minimal workflow

Create a query plan that explicitly opts into planning-only behavior:

```json
{
  "schema_version": 1,
  "plan_id": "quantum-problem-sources-v1",
  "purpose": "Find source-grounded future benchmark leads.",
  "search_execution": {
    "pipeline_mode": "planning_only_no_network",
    "intended_result_custody": "operator_provided_raw_manifest"
  },
  "queries": [
    {
      "query_id": "q-qec-noise",
      "query_text": "coherent noise quantum error correction decoder",
      "provider_hint": "openalex",
      "requested_limit": 100,
      "source_types": ["paper"],
      "facets": {
        "quantum_domains": ["qec"],
        "situations": ["noise"],
        "evidence_roles": ["primary-method"]
      }
    }
  ],
  "deduplication": {
    "near_duplicate_threshold_percent": 80
  },
  "coverage": {
    "sample_size": 10,
    "targets": {
      "quantum_domain": ["qec"],
      "situation": ["noise"],
      "source_type": ["paper"]
    }
  }
}
```

Freeze it from the repository root:

```bash
python3 scripts/run_autoresearch_discovery.py plan \
  --plan reports/orbit_q_problem_discovery/local_inputs/query-plan.json
```

The command prints the `search-plan` SHA-256. Put that exact value in every raw
manifest as `search_plan_artifact_sha256`. A raw manifest also supplies, for
every hit, all of the following:

- `source`: title, HTTPS URL, type, authors, publisher, stable identifiers;
- `version`: version type/label/locator, observation time, optional content hash;
- `license`: known/unknown/restricted status and explicit redistribution value;
- `problem`: domain, situation, task archetype, proposed objective,
  TensorCircuit path, evidence role, and keywords;
- collection provenance: provider and adapter versions, custody, observation
  time, and the bounded search-execution claim.

Then execute all offline transforms:

```bash
python3 scripts/run_autoresearch_discovery.py run \
  --plan reports/orbit_q_problem_discovery/local_inputs/query-plan.json \
  --raw-manifest reports/orbit_q_problem_discovery/local_inputs/provider-a.json \
  --raw-manifest reports/orbit_q_problem_discovery/local_inputs/provider-b.json
```

The default store is `.artifacts/problem-discovery/autoresearch`, which is
ignored by Git.
Both inputs and outputs must resolve inside `--workspace-root`; anything in or
through `tasks/` is rejected. This prevents discovery work from mutating or
using the canonical Harbor benchmark tasks.

For stage-by-stage resumption, the CLI also exposes `ingest`, `normalize`,
`dedupe`, and `sample`. Each accepts only upstream artifact digests. Run
`python3 scripts/run_autoresearch_discovery.py <stage> --help` for the exact
arguments.

## Human-in-the-loop handoff

The `coverage-sample` artifact is the review input, not an automatic promotion
decision. Before integrating a lead with the existing ORBIT-Q candidate gates,
a reviewer still needs to verify source identity and licensing, read the actual
source, check the claim-to-source relationship, approve the problem formulation,
and authorize prototype construction. This isolated stage does not update
`review_state.json`, generate a Harbor task, or launch a solver.

An external search adapter can be added later by having it emit a schema-valid,
content-hashed raw manifest. That adapter should remain a separate acquisition
boundary so query planning can never be mistaken for executed search.
