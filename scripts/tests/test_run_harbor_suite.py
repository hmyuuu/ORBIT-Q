from __future__ import annotations

import pytest

from scripts.run_harbor_suite import (
    attempt_index,
    parse_challenges,
    summarize_result,
)


def test_parse_challenges_supports_ranges_and_deduplicates() -> None:
    assert parse_challenges("1-3, 2, 12") == [1, 2, 3, 12]


def test_parse_challenges_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="inclusive range"):
        parse_challenges("0,1")


def test_summarize_result_marks_trial_errors() -> None:
    result = {
        "finished_at": "2026-07-29T00:00:00Z",
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 1,
            "n_retries": 2,
            "evals": {
                "kimi": {
                    "exception_stats": {
                        "ApiUsageLimitError": ["challenge-07__example"]
                    },
                    "metrics": [
                        {
                            "reward": 0.0,
                            "functional_score": 0.0,
                            "static_policy_score": 0.0,
                            "llm_audit_score": 0.0,
                            "runtime_sec": -1.0,
                        }
                    ],
                }
            },
        },
    }

    assert summarize_result(result) == {
        "status": "error",
        "finished_at": "2026-07-29T00:00:00Z",
        "n_completed_trials": 1,
        "n_errored_trials": 1,
        "n_running_trials": 0,
        "n_retries": 2,
        "exception_types": ["ApiUsageLimitError"],
        "metrics": [
            {
                "reward": 0.0,
                "functional_score": 0.0,
                "static_policy_score": 0.0,
                "llm_audit_score": 0.0,
                "runtime_sec": -1.0,
            }
        ],
    }


def test_attempt_index_supports_numbered_retries() -> None:
    base = "kimi-k3-challenge-01"
    assert attempt_index(base, base) == 1
    assert attempt_index(base, f"{base}-retry-02") == 2
    assert attempt_index(base, f"{base}-other") is None
