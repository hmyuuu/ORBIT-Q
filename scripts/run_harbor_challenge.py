#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARBOR = ROOT / ".conda" / "harbor-py312" / "bin" / "harbor"
CONFIG_PATH = ROOT / "conf.toml"
LOCAL_CONFIG_PATH = ROOT / "conf.local.toml"
SOLVER_AGENT_IMPORTS = {
    "codex": "harbor.agents.installed.codex:Codex",
    "codex-para": "adapters.codex_para:CodexPara",
    "claude-code": "adapters.claude_para:ClaudePara",
    "kimi-code": "adapters.kimi_code:KimiCode",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text())


def load_config() -> dict[str, Any]:
    return deep_merge(read_toml(CONFIG_PATH), read_toml(LOCAL_CONFIG_PATH))


def config_get(config: dict[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {})
    if not isinstance(value, dict):
        return None
    return value.get(key)


def first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    items: list[str] = []
    for raw_value in raw_values:
        items.extend(
            item
            for item in (part.strip() for part in str(raw_value).split(","))
            if item
        )
    return items


def parse_bool(value: Any, *, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def resolve_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_command_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    text = str(value)
    path = Path(text).expanduser()
    if path.is_absolute() or "/" in text:
        if not path.is_absolute():
            path = ROOT / path
        return path
    return path


def default_harbor_bin() -> Path:
    return DEFAULT_HARBOR if DEFAULT_HARBOR.is_file() else Path("harbor")


def kwarg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def add_kwarg(cmd: list[str], flag: str, key: str, value: Any) -> None:
    if value is None or value == "":
        return
    cmd.extend([flag, f"{key}={kwarg_value(value)}"])


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("framework slug cannot be empty")
    return slug


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Harbor challenge with a command-line-selected quantum "
            "framework without modifying canonical tasks/challenge-* files."
        )
    )
    parser.add_argument(
        "--challenge",
        default=None,
        help="Challenge id such as 01, 1, or challenge-01.",
    )
    parser.add_argument(
        "--framework",
        default=None,
        help="Required quantum framework/prompt slug.",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Framework image tag. Defaults to challenge-benchmark-quantum-<framework>:py311.",
    )
    parser.add_argument(
        "--extra-instruction-path",
        type=Path,
        default=None,
        help="Extra framework prompt. Defaults to prompts/frameworks/<framework>.md.",
    )
    parser.add_argument(
        "--solver-agent",
        choices=sorted(SOLVER_AGENT_IMPORTS),
        default=None,
        help="Agent CLI used to solve the task. The verifier still uses the Codex audit wrapper.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Agent model passed to Harbor -m.",
    )
    parser.add_argument(
        "--audit-model",
        default=None,
        help="Verifier Codex audit model. Defaults from conf.toml.",
    )
    parser.add_argument(
        "--solver-reasoning-effort",
        default=None,
        help=(
            "Agent reasoning effort kwarg. Defaults to max for claude-code "
            "and high for kimi-code."
        ),
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Harbor job name. Defaults to challenge-<id>-<framework>-<solver-agent>.",
    )
    parser.add_argument(
        "-n",
        "--n-concurrent",
        type=int,
        default=None,
        help="Harbor concurrency.",
    )
    parser.add_argument(
        "-r",
        "--max-retries",
        type=int,
        default=None,
        help="Maximum Harbor trial retries for classified infrastructure errors.",
    )
    parser.add_argument(
        "--retry-include",
        action="append",
        default=None,
        help="Exception type eligible for retry; repeat or comma-separate values.",
    )
    parser.add_argument(
        "--retry-exclude",
        action="append",
        default=None,
        help="Exception type excluded from retry; repeat or comma-separate values.",
    )
    parser.add_argument(
        "--cpus",
        choices=("auto", "limit", "request", "guarantee", "ignore"),
        default=None,
        help="Harbor CPU enforcement policy.",
    )
    parser.add_argument(
        "--memory",
        choices=("auto", "limit", "request", "guarantee", "ignore"),
        default=None,
        help="Harbor memory enforcement policy.",
    )
    parser.add_argument(
        "--override-cpus",
        type=int,
        default=None,
        help="Override the task CPU count for this run without modifying task.toml.",
    )
    parser.add_argument(
        "--override-memory-mb",
        type=int,
        default=None,
        help="Override task memory in MB for this run without modifying task.toml.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
        help="Harbor jobs output directory.",
    )
    parser.add_argument(
        "--harbor-bin",
        type=Path,
        default=None,
        help="Harbor executable.",
    )
    parser.add_argument(
        "--codex-profile",
        default=None,
        help="Codex CLI profile for CodexPara solver/verifier. Empty means no --profile.",
    )
    parser.add_argument(
        "--codex-profile-config-path",
        type=Path,
        default=None,
        help="Optional Codex profile/base config to upload into verifier or CodexPara containers.",
    )
    parser.add_argument(
        "--codex-model-catalog-path",
        type=Path,
        default=None,
        help="Optional Codex model catalog JSON for custom providers.",
    )
    parser.add_argument(
        "--codex-force-auth-json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Upload ~/.codex/auth.json instead of relying on OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--kimi-code-home-path",
        type=Path,
        default=None,
        help=(
            "Host Kimi Code home containing config.toml and credentials/. "
            "Defaults to KIMI_CODE_HOME or ~/.kimi-code."
        ),
    )
    parser.add_argument(
        "--yes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass --yes to Harbor.",
    )
    return parser.parse_args()


def challenge_name(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("challenge-"):
        return raw
    return f"challenge-{int(raw):02d}"


def resolve_solver_model(solver_agent: str, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    if solver_agent == "kimi-code":
        return "kimi-code/k3"
    if solver_agent == "claude-code":
        model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get(
            "ANTHROPIC_DEFAULT_OPUS_MODEL"
        )
        if not model:
            raise ValueError(
                "Claude solver model is required. Pass --model or export "
                "MODEL_NAME, MODEL, ANTHROPIC_MODEL, or ANTHROPIC_DEFAULT_OPUS_MODEL."
            )
        return model
    return "gpt-5"


def default_solver_reasoning_effort(solver_agent: str) -> str | None:
    if solver_agent == "claude-code":
        return "max"
    if solver_agent == "kimi-code":
        return "high"
    return None


def validate_solver_concurrency(solver_agent: str, n_concurrent: int) -> None:
    if n_concurrent < 1:
        raise ValueError("--n-concurrent must be at least 1")
    if solver_agent == "kimi-code" and n_concurrent != 1:
        raise ValueError(
            "Kimi Code requires --n-concurrent 1 because its OAuth provider "
            "rotates refresh tokens and the adapter synchronizes refreshed "
            "credentials back to the host login."
        )


def main() -> int:
    args = parse_args()
    config = load_config()

    challenge_raw = first_value(
        args.challenge,
        env_value("CHALLENGE_ID"),
        config_get(config, "run", "challenge"),
        "01",
    )
    framework = str(
        first_value(
            args.framework,
            env_value("FRAMEWORK"),
            config_get(config, "run", "framework"),
            "tensorcircuit",
        )
    ).strip()
    framework_slug = slugify(framework)
    docker_image = first_value(
        args.docker_image,
        env_value("DOCKER_IMAGE"),
        config_get(config, "run", "docker_image"),
        f"challenge-benchmark-quantum-{framework_slug}:py311",
    )
    challenge = challenge_name(str(challenge_raw))
    task_dir = ROOT / "tasks" / challenge
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Canonical task not found: {task_dir}")

    extra_instruction_path = resolve_path(
        first_value(
            args.extra_instruction_path,
            env_value("EXTRA_INSTRUCTION_PATH"),
            config_get(config, "run", "extra_instruction_path"),
            ROOT / "prompts" / "frameworks" / f"{framework_slug}.md",
        )
    )
    assert extra_instruction_path is not None
    if not extra_instruction_path.is_file():
        raise FileNotFoundError(f"Framework prompt not found: {extra_instruction_path}")

    solver_agent = str(
        first_value(
            args.solver_agent,
            env_value("SOLVER_AGENT"),
            config_get(config, "run", "solver_agent"),
            "codex",
        )
    )
    if solver_agent not in SOLVER_AGENT_IMPORTS:
        raise ValueError(
            f"Unknown solver_agent {solver_agent!r}; expected one of "
            f"{', '.join(sorted(SOLVER_AGENT_IMPORTS))}"
        )

    configured_model = first_value(args.model, env_value("MODEL_NAME", "MODEL"))
    if configured_model is None:
        model_section = "kimi" if solver_agent == "kimi-code" else "codex"
        configured_model = config_get(config, model_section, "model")
    solver_model = resolve_solver_model(
        solver_agent, str(configured_model) if configured_model is not None else None
    )
    job_name = first_value(
        args.job_name,
        env_value("JOB_NAME"),
        config_get(config, "run", "job_name"),
        f"{challenge}-{framework_slug}-{solver_agent}",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    audit_model = first_value(
        args.audit_model,
        env_value("AUDIT_MODEL_NAME", "AUDIT_MODEL"),
        config_get(config, "codex", "audit_model"),
        solver_model,
    )
    solver_reasoning_effort = first_value(
        args.solver_reasoning_effort,
        env_value(
            "SOLVER_REASONING_EFFORT",
            "CLAUDE_CODE_EFFORT_LEVEL",
            "KIMI_CODE_REASONING_EFFORT",
        ),
        config_get(config, "run", "solver_reasoning_effort"),
    )
    if not solver_reasoning_effort:
        solver_reasoning_effort = default_solver_reasoning_effort(solver_agent)

    n_concurrent = int(
        first_value(
            args.n_concurrent,
            env_value("N_CONCURRENT"),
            config_get(config, "harbor", "n_concurrent"),
            1,
        )
    )
    validate_solver_concurrency(solver_agent, n_concurrent)
    max_retries = int(
        first_value(
            args.max_retries,
            env_value("HARBOR_MAX_RETRIES"),
            config_get(config, "harbor", "max_retries"),
            0,
        )
    )
    if max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    retry_include = string_list(
        first_value(
            args.retry_include,
            env_value("HARBOR_RETRY_INCLUDE"),
            config_get(config, "harbor", "retry_include"),
        )
    )
    retry_exclude = string_list(
        first_value(
            args.retry_exclude,
            env_value("HARBOR_RETRY_EXCLUDE"),
            config_get(config, "harbor", "retry_exclude"),
        )
    )
    override_cpus_value = first_value(
        args.override_cpus,
        env_value("HARBOR_OVERRIDE_CPUS"),
        config_get(config, "harbor", "override_cpus"),
    )
    override_cpus = (
        int(override_cpus_value) if override_cpus_value is not None else None
    )
    override_memory_value = first_value(
        args.override_memory_mb,
        env_value("HARBOR_OVERRIDE_MEMORY_MB"),
        config_get(config, "harbor", "override_memory_mb"),
    )
    override_memory_mb = (
        int(override_memory_value) if override_memory_value is not None else None
    )
    if override_cpus is not None and override_cpus < 1:
        raise ValueError("--override-cpus must be at least 1")
    if override_memory_mb is not None and override_memory_mb < 1:
        raise ValueError("--override-memory-mb must be at least 1")
    cpu_policy = first_value(
        args.cpus,
        env_value("HARBOR_CPU_POLICY"),
        config_get(config, "harbor", "cpu_policy"),
        "limit" if override_cpus is not None else None,
    )
    memory_policy = first_value(
        args.memory,
        env_value("HARBOR_MEMORY_POLICY"),
        config_get(config, "harbor", "memory_policy"),
        "limit" if override_memory_mb is not None else None,
    )
    jobs_dir = resolve_path(
        first_value(
            args.jobs_dir,
            env_value("JOBS_DIR"),
            config_get(config, "harbor", "jobs_dir"),
            ROOT / "jobs",
        )
    )
    harbor_bin = resolve_command_path(
        first_value(
            args.harbor_bin,
            env_value("HARBOR_BIN"),
            config_get(config, "harbor", "harbor_bin"),
            default_harbor_bin(),
        )
    )
    assert jobs_dir is not None
    assert harbor_bin is not None
    yes = parse_bool(
        first_value(
            args.yes,
            env_value("HARBOR_YES"),
            config_get(config, "harbor", "yes"),
            True,
        )
    )

    if args.codex_profile is not None:
        codex_profile = args.codex_profile
    elif "CODEX_PROFILE" in os.environ:
        codex_profile = os.environ.get("CODEX_PROFILE", "")
    else:
        codex_profile = config_get(config, "codex", "profile")
    codex_profile = str(codex_profile).strip() if codex_profile is not None else None
    codex_profile_config_path = resolve_path(
        first_value(
            args.codex_profile_config_path,
            env_value("CODEX_PROFILE_CONFIG_PATH"),
            config_get(config, "codex", "profile_config_path"),
        )
    )
    codex_model_catalog_path = resolve_path(
        first_value(
            args.codex_model_catalog_path,
            env_value("CODEX_MODEL_CATALOG_PATH"),
            config_get(config, "codex", "model_catalog_path"),
        )
    )
    codex_force_auth_json = parse_bool(
        first_value(
            args.codex_force_auth_json,
            env_value("CODEX_FORCE_AUTH_JSON"),
            config_get(config, "codex", "force_auth_json"),
            False,
        )
    )
    env["CODEX_FORCE_AUTH_JSON"] = "true" if codex_force_auth_json else "false"
    kimi_code_home_path = resolve_path(
        first_value(
            args.kimi_code_home_path,
            env_value("KIMI_CODE_HOME"),
            config_get(config, "kimi", "home_path"),
            Path.home() / ".kimi-code",
        )
    )
    if solver_agent == "kimi-code":
        assert kimi_code_home_path is not None
        if not (kimi_code_home_path / "config.toml").is_file():
            raise FileNotFoundError(
                "Kimi Code config not found. Run `kimi login` first or pass "
                f"--kimi-code-home-path: {kimi_code_home_path / 'config.toml'}"
            )

    cmd = [
        str(harbor_bin),
        "run",
        "-p",
        str(task_dir),
        "--extra-instruction-path",
        str(extra_instruction_path),
        "--environment-import-path",
        "adapters.framework_docker:FrameworkDockerEnvironment",
        "--environment-kwarg",
        f"framework={framework_slug}",
        "--environment-kwarg",
        f"docker_image={docker_image}",
        "--agent-import-path",
        SOLVER_AGENT_IMPORTS[solver_agent],
        "--verifier-import-path",
        "adapters.codex_para_verifier:CodexParaVerifier",
        "--verifier-kwarg",
        f"audit_model={audit_model}",
        "--verifier-env",
        f"REQUIRED_QUANTUM_FRAMEWORK={framework_slug}",
        "-m",
        solver_model,
        "-n",
        str(n_concurrent),
        "-o",
        str(jobs_dir),
        "--job-name",
        str(job_name),
    ]
    if max_retries:
        cmd.extend(["--max-retries", str(max_retries)])
    for exception_type in retry_include:
        cmd.extend(["--retry-include", exception_type])
    for exception_type in retry_exclude:
        cmd.extend(["--retry-exclude", exception_type])
    if cpu_policy:
        cmd.extend(["--cpus", str(cpu_policy)])
    if memory_policy:
        cmd.extend(["--memory", str(memory_policy)])
    if override_cpus is not None:
        cmd.extend(["--override-cpus", str(override_cpus)])
    if override_memory_mb is not None:
        cmd.extend(["--override-memory-mb", str(override_memory_mb)])
    if solver_reasoning_effort:
        cmd.extend(["--agent-kwarg", f"reasoning_effort={solver_reasoning_effort}"])

    if solver_agent == "codex-para":
        add_kwarg(cmd, "--agent-kwarg", "profile", codex_profile)
        add_kwarg(
            cmd,
            "--agent-kwarg",
            "profile_config_path",
            codex_profile_config_path,
        )
        add_kwarg(cmd, "--agent-kwarg", "force_auth_json", codex_force_auth_json)
        add_kwarg(
            cmd,
            "--agent-kwarg",
            "model_catalog_path",
            codex_model_catalog_path,
        )
    elif solver_agent == "kimi-code":
        add_kwarg(
            cmd,
            "--agent-kwarg",
            "kimi_home_path",
            kimi_code_home_path,
        )

    add_kwarg(cmd, "--verifier-kwarg", "profile", codex_profile)
    add_kwarg(
        cmd,
        "--verifier-kwarg",
        "profile_config_path",
        codex_profile_config_path,
    )
    add_kwarg(cmd, "--verifier-kwarg", "force_auth_json", codex_force_auth_json)
    add_kwarg(
        cmd,
        "--verifier-kwarg",
        "model_catalog_path",
        codex_model_catalog_path,
    )

    if yes:
        cmd.append("--yes")

    print(f"Canonical task: {task_dir.relative_to(ROOT)}")
    print(f"Framework: {framework_slug}")
    print(f"Docker image: {docker_image}")
    print(f"Extra prompt: {display_path(extra_instruction_path)}")
    print(f"Solver agent: {solver_agent}")
    print(f"Solver model: {solver_model}")
    print(f"Solver reasoning effort: {solver_reasoning_effort or '(agent default)'}")
    print(f"Audit model: {audit_model}")
    print(f"Codex profile: {codex_profile or '(default)'}")
    if max_retries:
        print(
            "Harbor retries: "
            f"{max_retries} "
            f"(include={retry_include or 'default'}, "
            f"exclude={retry_exclude or 'default'})"
        )
    if override_cpus is not None:
        print(f"CPU override: {override_cpus} ({cpu_policy})")
    if override_memory_mb is not None:
        print(f"Memory override: {override_memory_mb} MB ({memory_policy})")
    if solver_agent == "kimi-code":
        print("Kimi Code credentials: file-backed host login")
    sys.stdout.flush()
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
