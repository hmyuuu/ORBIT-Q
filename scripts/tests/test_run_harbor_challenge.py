from __future__ import annotations

from scripts.run_harbor_challenge import (
    default_solver_reasoning_effort,
    string_list,
    validate_solver_concurrency,
)


def test_agent_specific_default_reasoning_effort() -> None:
    assert default_solver_reasoning_effort("kimi-code") == "high"
    assert default_solver_reasoning_effort("claude-code") == "max"
    assert default_solver_reasoning_effort("codex") is None


def test_string_list_accepts_config_lists_and_comma_separated_values() -> None:
    assert string_list(["ApiOverloadedError, ApiRateLimitError", "ApiError"]) == [
        "ApiOverloadedError",
        "ApiRateLimitError",
        "ApiError",
    ]


def test_kimi_code_requires_sequential_trials() -> None:
    validate_solver_concurrency("kimi-code", 1)

    try:
        validate_solver_concurrency("kimi-code", 2)
    except ValueError as exc:
        assert "requires --n-concurrent 1" in str(exc)
    else:
        raise AssertionError("parallel Kimi Code trials should be rejected")


def test_other_agents_accept_parallel_trials() -> None:
    validate_solver_concurrency("codex", 2)
