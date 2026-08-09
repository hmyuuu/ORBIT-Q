from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

FRAMEWORK_IMPORTS = {
    "mindquantum": {"mindquantum"},
    "pennylane": {"pennylane"},
    "pennylane-lane": {"pennylane"},
    "torchquantum": {"torchquantum"},
    "tensorcircuit": {"tensorcircuit"},
    "tensorcircuit-ng": {"tensorcircuit"},
    "tc": {"tensorcircuit"},
}
QUANTUM_FRAMEWORK_IMPORTS = {
    "braket",
    "cirq",
    "mindquantum",
    "paddle_quantum",
    "pennylane",
    "pyqpanda",
    "qiskit",
    "qutip",
    "tensorcircuit",
    "torchquantum",
}

CHEATING_PATTERNS = [
    r"open\(['\"]/(solution|tests|logs)\b",
    r"Path\(['\"]/(solution|tests|logs)\b",
    r"reward\.(txt|json).*write",
    r"write_text\(.*reward\.(txt|json)",
    r"solution_\d+\.py.*read_text",
    r"\bos\.(environ|getenv|putenv)\b",
    r"\b(sys\._getframe|inspect\.(currentframe|stack|getouterframes)|f_back)\b",
    r"['\"](?:/proc/|/tests(?:/|['\"])|/logs(?:/|['\"])|ORBIT_Q_CANDIDATE_SEED)",
]

RAW_SIMULATOR_HINTS = [
    r"np\.kron",
    r"numpy\.kron",
    r"eigvalsh",
    r"eigh",
    r"statevector",
    r"2\s*\*\*\s*n_qubits",
    r"1\s*<<\s*n_qubits",
]

MAX_SOURCE_BYTES = 96 * 1024
MAX_PHYSICAL_LINE_LENGTH = 320
DYNAMIC_CODE_CALLS = {"__import__", "compile", "eval", "exec"}
DYNAMIC_CODE_IMPORTS = {"ctypes", "gc", "importlib", "inspect", "marshal", "runpy"}


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _effective_code_lines(source: str) -> int:
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def _semicolon_count(source: str) -> int:
    """Count real statement separators, excluding strings and comments."""

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return sum(
            token.type == tokenize.OP and token.string == ";" for token in tokens
        )
    except (IndentationError, tokenize.TokenError):
        return 0


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dynamic_code_hits(tree: ast.AST, imports: set[str]) -> list[str]:
    hits = {f"import:{name}" for name in imports & DYNAMIC_CODE_IMPORTS}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in DYNAMIC_CODE_CALLS:
                hits.add(f"call:{name}")
    return sorted(hits)


def check_source(path: Path, framework: str, max_lines: int = 200) -> dict:
    source = path.read_text()
    source_lines = source.splitlines()
    result = {
        "path": str(path),
        "framework": framework,
        "exists": path.exists(),
        "parse_ok": 0.0,
        "line_count": _effective_code_lines(source),
        "logical_statement_count": 0,
        "max_lines": max_lines,
        "max_logical_statements": max_lines,
        "source_size_bytes": len(source.encode("utf-8")),
        "max_source_bytes": MAX_SOURCE_BYTES,
        "max_physical_line_length": max(
            (len(line) for line in source_lines), default=0
        ),
        "max_allowed_line_length": MAX_PHYSICAL_LINE_LENGTH,
        "semicolon_count": _semicolon_count(source),
        "dynamic_code_hits": [],
        "line_count_score": 0.0,
        "framework_score": 0.0,
        "no_forbidden_framework_score": 1.0,
        "no_static_cheating_score": 0.0,
        "no_raw_simulator_bypass_score": 1.0,
        "imports": [],
        "forbidden_imports": [],
        "cheating_hits": [],
        "raw_simulator_hits": [],
    }

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result["syntax_error"] = str(exc)
        return result

    result["parse_ok"] = 1.0
    imports = _import_roots(tree)
    result["imports"] = sorted(imports)
    required = FRAMEWORK_IMPORTS.get(framework.lower(), {framework.lower()})

    result["logical_statement_count"] = sum(
        isinstance(node, ast.stmt) for node in ast.walk(tree)
    )
    result["dynamic_code_hits"] = _dynamic_code_hits(tree, imports)
    source_budget_ok = all(
        (
            result["line_count"] <= max_lines,
            result["logical_statement_count"] <= max_lines,
            result["source_size_bytes"] <= MAX_SOURCE_BYTES,
            result["max_physical_line_length"] <= MAX_PHYSICAL_LINE_LENGTH,
            result["semicolon_count"] == 0,
        )
    )
    result["line_count_score"] = 1.0 if source_budget_ok else 0.0
    result["framework_score"] = 1.0 if imports & required else 0.0
    forbidden_imports = sorted((imports & QUANTUM_FRAMEWORK_IMPORTS) - required)
    result["forbidden_imports"] = forbidden_imports
    result["no_forbidden_framework_score"] = 0.0 if forbidden_imports else 1.0

    cheating_hits = [
        pattern for pattern in CHEATING_PATTERNS if re.search(pattern, source, re.I)
    ]
    if result["semicolon_count"]:
        cheating_hits.append("semicolon_statement_packing")
    cheating_hits.extend(result["dynamic_code_hits"])
    result["cheating_hits"] = cheating_hits
    result["no_static_cheating_score"] = 0.0 if cheating_hits else 1.0

    raw_hits = [
        pattern for pattern in RAW_SIMULATOR_HINTS if re.search(pattern, source)
    ]
    result["raw_simulator_hits"] = raw_hits
    if raw_hits and not imports & required:
        result["no_raw_simulator_bypass_score"] = 0.0
    elif len(raw_hits) >= 4:
        result["no_raw_simulator_bypass_score"] = 0.7

    components = [
        result["parse_ok"],
        result["line_count_score"],
        result["framework_score"],
        result["no_forbidden_framework_score"],
        result["no_static_cheating_score"],
        result["no_raw_simulator_bypass_score"],
    ]
    result["static_policy_score"] = float(min(components))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--framework", default="tensorcircuit")
    parser.add_argument("--max-lines", type=int, default=200)
    args = parser.parse_args()

    print(
        json.dumps(check_source(args.source, args.framework, args.max_lines), indent=2)
    )


if __name__ == "__main__":
    main()
