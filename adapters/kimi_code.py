from __future__ import annotations

import os
import re
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import (
    AgentAuthenticationError,
    ApiConnectionClosedError,
    ApiOverloadedError,
    ApiUsageLimitError,
    BaseInstalledAgent,
    EnvVar,
    ErrorPattern,
    ModelNotFoundError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

_SOLUTION_RE = re.compile(r"\bsolution_(\d+)\.py\b")
_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


class KimiCode(BaseInstalledAgent):
    """Harbor adapter for Kimi Code CLI's non-interactive agent mode."""

    _INSTALL_URL = "https://code.kimi.com/kimi-code/install.sh"
    _OUTPUT_FILENAME = "kimi-code.jsonl"
    _REMOTE_HOME = Path("/tmp/harbor-kimi-code")
    _REMOTE_AUTH_ARCHIVE = Path("/tmp/harbor-kimi-code-auth.tar.gz")

    ENV_VARS = [
        EnvVar(
            "reasoning_effort",
            env="KIMI_MODEL_THINKING_EFFORT",
            type="enum",
            choices=["low", "medium", "high", "xhigh", "max"],
            env_fallback="KIMI_CODE_REASONING_EFFORT",
        ),
    ]
    ERROR_PATTERNS = [
        *BaseInstalledAgent.ERROR_PATTERNS,
        ErrorPattern(
            r"provider\.rate_limit: 429 The engine is currently overloaded",
            ApiOverloadedError,
        ),
        ErrorPattern(
            r"(?:reached|hit) your usage limit for this billing cycle",
            ApiUsageLimitError,
        ),
        ErrorPattern(
            r"provider\.connection_error: terminated",
            ApiConnectionClosedError,
        ),
        ErrorPattern(
            r"(not logged in|login[_ ]required|requires login|missing credentials|"
            r"provider .+ missing credentials|authentication failed|"
            r"current subscription does not have access|"
            r"upgrade to higher-tier kimi code plans)",
            AgentAuthenticationError,
        ),
        ErrorPattern(
            r"(model .+ not found|unknown model|model alias .+ is not defined)",
            ModelNotFoundError,
        ),
    ]

    def __init__(
        self,
        *args: Any,
        kimi_home_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kimi_home_path = (
            Path(kimi_home_path).expanduser() if kimi_home_path else None
        )

    @staticmethod
    @override
    def name() -> str:
        return "kimi-code"

    @override
    def get_version_command(self) -> str | None:
        return "kimi --version"

    @override
    def parse_version(self, stdout: str) -> str:
        match = _VERSION_RE.search(stdout)
        return match.group(1) if match else stdout.strip()

    async def _installed_kimi_satisfies_version(
        self, environment: BaseEnvironment
    ) -> bool:
        result = await environment.exec(command="kimi --version")
        if result.return_code != 0:
            return False
        if self._version is None:
            return True
        return self.parse_version(result.stdout or "") == self._version.lstrip("v")

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if await self._installed_kimi_satisfies_version(environment):
            self.logger.debug("Kimi Code CLI is already available")
            return

        version_arg = (
            f" --version {shlex.quote(self._version.lstrip('v'))}"
            if self._version
            else ""
        )
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -fsSL --retry 3 "
                f"{shlex.quote(self._INSTALL_URL)} "
                "-o /tmp/install-kimi-code.sh; "
                "for attempt in 1 2 3; do "
                "if KIMI_NO_MODIFY_PATH=1 bash /tmp/install-kimi-code.sh"
                f"{version_arg}; then break; fi; "
                'if [ "${attempt}" -eq 3 ]; then exit 1; fi; '
                "sleep 2; "
                "done; "
                "test -x /root/.kimi-code/bin/kimi; "
                "install -m 0755 /root/.kimi-code/bin/kimi /usr/local/bin/kimi; "
                "rm -f /tmp/install-kimi-code.sh; "
                "kimi --version"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    def _resolved_home(self) -> Path:
        if self.kimi_home_path is not None:
            home = self.kimi_home_path.resolve()
        else:
            raw_home = self._get_env("KIMI_CODE_HOME")
            home = (
                Path(raw_home).expanduser().resolve()
                if raw_home
                else (Path.home() / ".kimi-code").resolve()
            )
        if not (home / "config.toml").is_file():
            raise ValueError(
                "Kimi Code config was not found. Run `kimi login`, or pass "
                f"kimi_home_path explicitly: {home / 'config.toml'}"
            )
        return home

    @staticmethod
    def _build_auth_archive(home: Path) -> Path:
        credentials = home / "credentials"
        included = [home / "config.toml"]
        tui_config = home / "tui.toml"
        if tui_config.is_file():
            included.append(tui_config)
        if credentials.is_dir():
            included.append(credentials)

        handle = tempfile.NamedTemporaryFile(
            prefix="harbor-kimi-code-",
            suffix=".tar.gz",
            delete=False,
        )
        archive_path = Path(handle.name)
        handle.close()
        archive_path.chmod(0o600)

        try:
            with tarfile.open(archive_path, "w:gz") as archive:
                for path in included:
                    archive.add(path, arcname=path.relative_to(home))
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        return archive_path

    @staticmethod
    def _expected_solution_path(instruction: str) -> str | None:
        matches = _SOLUTION_RE.findall(instruction)
        if not matches:
            return None
        return f"/root/solution_{matches[-1]}.py"

    @staticmethod
    def _persist_refreshed_credentials(source: Path, home: Path) -> int:
        """Atomically preserve credential refreshes made by the isolated CLI."""
        if not source.is_dir():
            return 0

        target_root = home / "credentials"
        target_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        persisted = 0
        for source_path in source.rglob("*"):
            if source_path.is_symlink() or not source_path.is_file():
                continue
            relative_path = source_path.relative_to(source)
            target_path = target_root / relative_path
            target_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{target_path.name}.",
                dir=target_path.parent,
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(source_path.read_bytes())
                temp_path.chmod(0o600)
                os.replace(temp_path, target_path)
                persisted += 1
            finally:
                temp_path.unlink(missing_ok=True)
        return persisted

    async def _download_refreshed_credentials(
        self,
        environment: BaseEnvironment,
        *,
        remote_home: str,
        local_home: Path,
    ) -> None:
        staging_root = Path(tempfile.mkdtemp(prefix="harbor-kimi-code-refreshed-"))
        staging_root.chmod(0o700)
        staging_credentials = staging_root / "credentials"
        staging_credentials.mkdir(mode=0o700)
        try:
            await environment.download_dir(
                f"{remote_home}/credentials",
                staging_credentials,
            )
            persisted = self._persist_refreshed_credentials(
                staging_credentials,
                local_home,
            )
            if persisted:
                self.logger.debug(
                    "Persisted %d refreshed Kimi credential file(s)", persisted
                )
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    @with_prompt_template
    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError(
                "Kimi Code requires a configured model alias, such as kimi-code/k3"
            )

        local_home = self._resolved_home()
        local_archive = self._build_auth_archive(local_home)
        remote_home = self._REMOTE_HOME.as_posix()
        remote_archive = self._REMOTE_AUTH_ARCHIVE.as_posix()
        output_path = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        expected_solution = self._expected_solution_path(instruction)

        env = self.resolve_env_vars()
        env.update(
            {
                "CI": "true",
                "KIMI_CODE_HOME": remote_home,
                "KIMI_CODE_NO_AUTO_UPDATE": "1",
                "KIMI_DISABLE_CRON": "1",
                "KIMI_DISABLE_TELEMETRY": "1",
                "NO_COLOR": "1",
                "SHELL": "/bin/bash",
                "TERM": "dumb",
            }
        )

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    f"rm -rf {shlex.quote(remote_home)}; "
                    f"mkdir -p {shlex.quote(remote_home)} "
                    f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())}; "
                    f"chmod 700 {shlex.quote(remote_home)}"
                ),
                env=env,
            )
            await environment.upload_file(local_archive, remote_archive)
            await self.exec_as_agent(
                environment,
                command=(
                    f"python -m tarfile -e {shlex.quote(remote_archive)} "
                    f"{shlex.quote(remote_home)}; "
                    f"rm -f {shlex.quote(remote_archive)}; "
                    f"chmod -R go-rwx {shlex.quote(remote_home)}"
                ),
                env=env,
            )

            kimi_command = (
                "kimi "
                f"--model {shlex.quote(self.model_name)} "
                f"--prompt {shlex.quote(instruction)} "
                "--output-format stream-json"
            )
            shell_command = (
                "set -o pipefail; "
                f"cd /root; {kimi_command} "
                f"2>&1 | stdbuf -oL tee {shlex.quote(output_path)}; "
                "kimi_status=${PIPESTATUS[0]}; "
            )
            if expected_solution is not None:
                shell_command += (
                    f"if test -s {shlex.quote(expected_solution)}; then "
                    "exit 0; else exit ${kimi_status}; fi"
                )
            else:
                shell_command += "exit ${kimi_status}"

            await self.exec_as_agent(
                environment,
                command=f"bash -lc {shlex.quote(shell_command)}",
                env=env,
            )
        finally:
            local_archive.unlink(missing_ok=True)
            try:
                await self._download_refreshed_credentials(
                    environment,
                    remote_home=remote_home,
                    local_home=local_home,
                )
            except Exception as error:
                self.logger.warning(
                    "Could not persist refreshed Kimi credentials: %s", error
                )
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"rm -rf {shlex.quote(remote_home)} "
                        f"{shlex.quote(remote_archive)}"
                    ),
                    env=env,
                )
            except Exception:
                pass
