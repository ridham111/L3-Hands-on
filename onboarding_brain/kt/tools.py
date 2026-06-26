"""Agent tool definitions and executor.

The LLM receives TOOL_DEFINITIONS and decides which tools to call. Python
executes exactly what is requested — it makes no decisions about what to
retrieve or when to stop. This is the core of Level-3 agent behaviour.

Tools
─────
  search_code        — semantic/TF-IDF search across indexed chunks
  read_file          — full file content from the index
  find_files         — glob/substring file discovery
  get_file_structure — list every indexed path
  grep_code          — exact string / regex across all chunks
  list_symbols       — exported functions, classes, types in a file (no full read)
  get_dependencies   — structured dependency list from package.json / requirements.txt / etc.
  call_graph         — callers + callees of a function (grep-based, depth 1-2)
  run_grep_ast       — find structural nodes by type: class, function, interface,
                       component, decorator, route, export
"""
from __future__ import annotations

import fnmatch
import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import VectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Claude-format tool schemas — passed verbatim to the API
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "search_code",
        "description": (
            "Semantically search the indexed codebase and return matching code snippets "
            "with file paths and line numbers. Use this first for any concept, feature "
            "name, function name, or keyword. If results are poor, try alternate queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — concept, feature, function name, etc.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default 8, max 20).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the complete indexed content of a specific file. "
            "Accepts the full path or a unique path suffix — 'auth.ts' matches "
            "'src/auth/auth.ts'. Use when you need to understand a file fully."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (or unique path suffix) relative to repo root.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_files",
        "description": (
            "List files whose path or name matches a pattern. "
            "Use to discover what files exist for an area ('auth', 'routes', 'models') "
            "or when you know a filename but not its full path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Substring or glob pattern to match against file paths.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "get_file_structure",
        "description": (
            "List every indexed file in the repository. "
            "Use at the start to understand project structure before diving in, "
            "or when you need to know what areas of code exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "grep_code",
        "description": (
            "Search for an exact string or regex pattern across all indexed files. "
            "Use to find specific function calls, import statements, class names, "
            "configuration keys, or error messages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "String or regex to find across all files.",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Optional: limit to files matching this glob (e.g. '*.ts', 'src/**').",
                },
            },
            "required": ["pattern"],
        },
    },
    # ── New tools ─────────────────────────────────────────────────────────────
    {
        "name": "list_symbols",
        "description": (
            "List all exported/defined functions, classes, interfaces, types, and constants "
            "in a file with their line numbers — without reading the full file body. "
            "Use this instead of read_file when you only need to know a file's API surface "
            "or want to find which line a specific symbol is on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path or unique path suffix (same as read_file).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_dependencies",
        "description": (
            "Return a structured list of project dependencies from package.json, "
            "requirements.txt, pyproject.toml, go.mod, Cargo.toml, or pom.xml. "
            "Use to answer 'what libraries are used for X?', 'what version of Y?', "
            "or 'what do I need to install?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Optional: only show dependencies whose name contains this string.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "call_graph",
        "description": (
            "Show the call graph for a function or class: where it is defined, "
            "every place it is called from (callers), and what functions it calls internally (callees). "
            "Use to answer 'where is X called?', 'what does Y call?', or 'what breaks if I change Z?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "The function, method, or class name to trace.",
                },
                "depth": {
                    "type": "integer",
                    "description": "How many levels of callees to trace (1 or 2, default 1).",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "run_grep_ast",
        "description": (
            "Find all structural definitions of a given node type across the repo. "
            "Supported types: class, function, interface, component, decorator, route, export. "
            "Use to answer 'show me all Angular services', 'list all API routes', "
            "'what components exist?', 'find all decorators used'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_type": {
                    "type": "string",
                    "description": "One of: class, function, interface, component, decorator, route, export.",
                },
                "name_filter": {
                    "type": "string",
                    "description": "Optional: only return nodes whose name contains this string.",
                },
            },
            "required": ["node_type"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Symbol extraction helpers  (used by list_symbols + run_grep_ast)
# ─────────────────────────────────────────────────────────────────────────────

_SYM_RULES: list[tuple[str, str, str]] = [
    # (pattern, kind, applies_to)
    # TypeScript / JavaScript
    (r"(?m)^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",           "class",        "ts"),
    (r"(?m)^\s*(?:export\s+)?(?:async\s+)?function[*]?\s+(\w+)",       "function",     "ts"),
    (r"(?m)^\s*(?:export\s+)?interface\s+(\w+)",                        "interface",    "ts"),
    (r"(?m)^\s*(?:export\s+)?type\s+(\w+)\s*=",                         "type",         "ts"),
    (r"(?m)^\s*(?:export\s+)?enum\s+(\w+)",                             "enum",         "ts"),
    (r"(?m)^\s*export\s+(?:default\s+)?(?:const|let|var)\s+(\w+)",     "export const", "ts"),
    # Python
    (r"(?m)^class\s+(\w+)",                                             "class",        "py"),
    (r"(?m)^(?:async\s+)?def\s+([a-zA-Z_]\w*)",                         "def",          "py"),
    # Go
    (r"(?m)^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",               "func",         "go"),
    (r"(?m)^type\s+(\w+)\s+struct",                                     "struct",       "go"),
    (r"(?m)^type\s+(\w+)\s+interface",                                  "interface",    "go"),
    # Java / Kotlin
    (r"(?m)(?:public\s+)?(?:abstract\s+)?class\s+(\w+)",               "class",        "java"),
    (r"(?m)(?:public\s+)?interface\s+(\w+)",                            "interface",    "java"),
    (r"(?m)(?:public|private|protected)\s+\w+\s+(\w+)\s*\(",           "method",       "java"),
    # Rust
    (r"(?m)^pub\s+(?:async\s+)?fn\s+(\w+)",                            "fn",           "rs"),
    (r"(?m)^(?:pub\s+)?struct\s+(\w+)",                                 "struct",       "rs"),
    (r"(?m)^(?:pub\s+)?trait\s+(\w+)",                                  "trait",        "rs"),
]

_EXT_FAMILY: dict[str, str] = {
    "ts": "ts", "tsx": "ts", "js": "ts", "jsx": "ts", "mjs": "ts", "cjs": "ts",
    "py": "py",
    "go": "go",
    "java": "java", "kt": "java", "scala": "java",
    "rs": "rs",
}

_CALLEE_SKIP = {
    "if", "for", "while", "switch", "catch", "return", "typeof", "instanceof",
    "new", "delete", "void", "throw", "yield", "await", "print", "len", "range",
    "super", "require", "import", "export", "const", "let", "var", "type",
    "class", "function", "async", "static", "public", "private", "protected",
    "True", "False", "None", "true", "false", "null", "undefined",
}

# AST node type → list of (regex_pattern, language_hint)
_AST_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "class": [
        (r"(?m)^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", "ts/js"),
        (r"(?m)^class\s+(\w+)", "python"),
        (r"(?m)(?:public\s+)?(?:abstract\s+)?class\s+(\w+)", "java/kt"),
        (r"(?m)^type\s+(\w+)\s+struct", "go"),
        (r"(?m)^(?:pub\s+)?struct\s+(\w+)", "rust"),
    ],
    "function": [
        (r"(?m)^\s*(?:export\s+)?(?:async\s+)?function[*]?\s+(\w+)", "ts/js"),
        (r"(?m)^(?:async\s+)?def\s+([a-z_]\w*)", "python"),
        (r"(?m)^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", "go"),
        (r"(?m)^pub\s+(?:async\s+)?fn\s+(\w+)", "rust"),
    ],
    "interface": [
        (r"(?m)^\s*(?:export\s+)?interface\s+(\w+)", "ts"),
        (r"(?m)^type\s+(\w+)\s+interface", "go"),
        (r"(?m)(?:public\s+)?interface\s+(\w+)", "java"),
        (r"(?m)^(?:pub\s+)?trait\s+(\w+)", "rust"),
    ],
    "component": [
        (r"(?m)^\s*(?:export\s+(?:default\s+)?)?(?:function|class|const)\s+([A-Z]\w+)", "react"),
        (r"@Component\s*\(", "angular"),
        (r"@Directive\s*\(", "angular"),
        (r"@Pipe\s*\(", "angular"),
    ],
    "decorator": [
        (r"(?m)^\s*@(\w+)(?:\s*[\(\n])", "ts/py"),
    ],
    "route": [
        (r"""(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*['"]([^'"]+)['"]""", "express/fastapi"),
        (r"""@(Get|Post|Put|Delete|Patch)\s*\(\s*['"]([^'"]*)['"]""", "nestjs/spring"),
        (r"""@app\.(get|post|put|delete|patch)\s*\(\s*['"]([^'"]+)['"]""", "flask"),
        (r"""@router\.(get|post|put|delete|patch)\s*\(\s*['"]([^'"]+)['"]""", "fastapi"),
        (r"""path\s*\(\s*['"]([^'"]+)['"]\s*,""", "django"),
    ],
    "export": [
        (r"(?m)^\s*export\s+(?:default\s+)?(?:const|class|function|interface|type|enum)\s+(\w+)", "ts/js"),
        (r"(?m)^module\.exports\s*=", "commonjs"),
        (r"(?m)^__all__\s*=", "python"),
    ],
}


def _extract_symbols(content: str, path: str) -> list[dict]:
    """Extract function/class/interface definitions from source content."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    family = _EXT_FAMILY.get(ext, "ts")  # default to ts-style patterns

    symbols: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for pattern, kind, applies in _SYM_RULES:
        if applies != family:
            continue
        try:
            for m in re.finditer(pattern, content):
                name = m.group(1)
                key = (kind, name)
                if key in seen:
                    continue
                seen.add(key)
                line_no = content[: m.start()].count("\n") + 1
                symbols.append({"kind": kind, "name": name, "line": line_no})
        except re.error:
            continue

    symbols.sort(key=lambda x: x["line"])
    return symbols


def _parse_dep_file(filename: str, content: str, filter_str: str) -> str:
    """Parse a dependency manifest and return a human-readable summary."""
    lines: list[str] = []
    fil = filter_str.lower()

    if filename == "package.json":
        try:
            data = json.loads(content)
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                deps = data.get(section) or {}
                if not deps:
                    continue
                section_lines = []
                for name, ver in sorted(deps.items()):
                    if fil and fil not in name.lower():
                        continue
                    section_lines.append(f"  {name}  {ver}")
                if section_lines:
                    lines.append(f"\n{section} ({len(section_lines)}):")
                    lines.extend(section_lines)
        except (json.JSONDecodeError, KeyError):
            for m in re.finditer(r'"([@\w/.-]+)"\s*:\s*"([^"]+)"', content):
                name, ver = m.group(1), m.group(2)
                if fil and fil not in name.lower():
                    continue
                lines.append(f"  {name}: {ver}")

    elif filename == "requirements.txt":
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if fil and fil not in stripped.lower():
                continue
            lines.append(f"  {stripped}")

    elif filename == "pyproject.toml":
        in_section = False
        for line in content.splitlines():
            if re.match(r'\[(?:tool\.poetry\.)?dependencies\]', line):
                in_section = True
                lines.append(f"\n{line}")
            elif line.startswith("[") and in_section:
                in_section = False
            elif in_section and "=" in line and not line.strip().startswith("#"):
                if fil and fil not in line.lower():
                    continue
                lines.append(f"  {line.strip()}")

    elif filename == "go.mod":
        in_require = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("require ("):
                in_require = True
            elif in_require and stripped == ")":
                in_require = False
            elif in_require or stripped.startswith("require "):
                pkg = stripped.replace("require ", "").strip()
                if fil and fil not in pkg.lower():
                    continue
                lines.append(f"  {pkg}")

    elif filename == "cargo.toml":
        in_deps = False
        for line in content.splitlines():
            if line.strip() in ("[dependencies]", "[dev-dependencies]"):
                in_deps = True
                lines.append(f"\n{line.strip()}")
            elif line.startswith("[") and in_deps:
                in_deps = False
            elif in_deps and "=" in line:
                if fil and fil not in line.lower():
                    continue
                lines.append(f"  {line.strip()}")

    return "\n".join(lines)


def _extract_callees(content: str, function_name: str) -> list[str]:
    """Extract unique function calls made within the named function's body."""
    patterns = [
        rf"(?:def|async def)\s+{re.escape(function_name)}\s*\(",
        rf"(?:async\s+)?function[*]?\s+{re.escape(function_name)}\s*\(",
        rf"(?:const|let|var)\s+{re.escape(function_name)}\s*=",
        rf"(?:class)\s+{re.escape(function_name)}\s*[{{\(]",
    ]
    start = -1
    for pat in patterns:
        m = re.search(pat, content)
        if m:
            start = m.start()
            break
    if start == -1:
        return []

    # take up to 3000 chars of the function body
    body = content[start: start + 3000]
    callees: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(', body):
        name = m.group(1)
        if (name not in _CALLEE_SKIP
                and name != function_name
                and name not in seen
                and not name[0].isupper()):   # skip constructor-style calls
            seen.add(name)
            callees.append(name)
    return callees[:15]


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor — runs one tool call, returns a string the LLM reads next
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes tool calls issued by the LLM. Tracks all files accessed."""

    def __init__(self, namespace: str, store: "VectorStore") -> None:
        self.namespace = namespace
        self.store = store
        self.used_paths: set[str] = set()
        self.call_log: list[dict[str, Any]] = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        self.call_log.append({"tool": tool_name, "input": tool_input})
        try:
            if tool_name == "search_code":
                return self._search_code(**tool_input)
            if tool_name == "read_file":
                return self._read_file(**tool_input)
            if tool_name == "find_files":
                return self._find_files(**tool_input)
            if tool_name == "get_file_structure":
                return self._get_file_structure()
            if tool_name == "grep_code":
                return self._grep_code(**tool_input)
            if tool_name == "list_symbols":
                return self._list_symbols(**tool_input)
            if tool_name == "get_dependencies":
                return self._get_dependencies(**tool_input)
            if tool_name == "call_graph":
                return self._call_graph(**tool_input)
            if tool_name == "run_grep_ast":
                return self._run_grep_ast(**tool_input)
            return f"Unknown tool: {tool_name!r}"
        except Exception as exc:  # noqa: BLE001
            return f"Tool error ({tool_name}): {exc}"

    # ── Original tools ────────────────────────────────────────────────────────

    def _search_code(self, query: str, top_k: int = 8) -> str:
        k = max(1, min(int(top_k), 20))
        results = self.store.search(self.namespace, query, k)
        if not results:
            return (
                f"No results for '{query}'. "
                "Try a more specific term, grep_code for exact strings, "
                "or find_files if you know the filename."
            )
        parts = []
        for r in results:
            m = r.get("metadata", {})
            path = m.get("path", "unknown")
            self.used_paths.add(path)
            l0, l1 = m.get("line_start", 0), m.get("line_end", 0)
            parts.append(
                f"[{path}:{l0}-{l1}] (relevance {r.get('score', 0):.3f})\n"
                f"{r.get('text', '').strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def _read_file(self, path: str) -> str:
        result = self.store.full_file(self.namespace, path)
        matched_path = path
        if not result:
            known = self.store.known_paths(self.namespace)
            norm = path.lower().replace("\\", "/")
            candidates = sorted(
                [p for p in known if norm in p.lower().replace("\\", "/")],
                key=len,
            )
            if candidates:
                matched_path = candidates[0]
                result = self.store.full_file(self.namespace, matched_path)
        if not result:
            return (
                f"File '{path}' not found in the index. "
                f"Try find_files('{path.split('/')[-1]}') to locate the correct path."
            )
        self.used_paths.add(matched_path)
        content = result.get("content", result.get("text", "")).strip()
        if result.get("truncated"):
            content += "\n\n[... file truncated at ingest limit ...]"
        header = f"[{matched_path}]" + (" (truncated)" if result.get("truncated") else "")
        return f"{header}\n{content}"

    def _find_files(self, pattern: str) -> str:
        known = sorted(self.store.known_paths(self.namespace))
        pat = pattern.lower().replace("\\", "/")
        matches = [
            p for p in known
            if pat in p.lower() or fnmatch.fnmatch(p.lower(), f"*{pat}*")
        ]
        if not matches:
            return f"No files found matching '{pattern}'."
        header = f"Found {len(matches)} file(s) matching '{pattern}':"
        listing = "\n".join(matches[:80])
        suffix = f"\n... and {len(matches) - 80} more" if len(matches) > 80 else ""
        return f"{header}\n{listing}{suffix}"

    def _get_file_structure(self) -> str:
        known = sorted(self.store.known_paths(self.namespace))
        if not known:
            return "No files indexed."
        return f"Repository — {len(known)} indexed files:\n" + "\n".join(known[:400])

    def _grep_code(self, pattern: str, file_glob: str = "") -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        chunks = self.store.grep_chunks(self.namespace, pattern)
        if file_glob:
            chunks = [
                c for c in chunks
                if fnmatch.fnmatch(c.get("metadata", {}).get("path", ""), file_glob)
            ]
        if not chunks:
            return f"Pattern '{pattern}' not found in any indexed file."

        parts = []
        for c in chunks[:15]:
            m = c.get("metadata", {})
            path = m.get("path", "unknown")
            self.used_paths.add(path)
            l0, l1 = m.get("line_start", 0), m.get("line_end", 0)
            text = c.get("text", "")
            matching_lines = [ln for ln in text.splitlines() if regex.search(ln)]
            if matching_lines:
                parts.append(
                    f"[{path}:{l0}-{l1}]\n" + "\n".join(matching_lines[:12])
                )
        return (
            "\n\n---\n\n".join(parts)
            if parts
            else f"Pattern found in chunks but no individual lines matched."
        )

    # ── New tools ─────────────────────────────────────────────────────────────

    def _list_symbols(self, path: str) -> str:
        """Return all exported/defined symbols in a file without reading the body."""
        result = self.store.full_file(self.namespace, path)
        matched_path = path
        if not result:
            known = self.store.known_paths(self.namespace)
            norm = path.lower().replace("\\", "/")
            candidates = sorted(
                [p for p in known if norm in p.lower().replace("\\", "/")], key=len
            )
            if candidates:
                matched_path = candidates[0]
                result = self.store.full_file(self.namespace, matched_path)
        if not result:
            return (
                f"File '{path}' not found. "
                f"Try find_files('{path.split('/')[-1]}') to locate the correct path."
            )
        self.used_paths.add(matched_path)
        content = result.get("content", result.get("text", ""))
        symbols = _extract_symbols(content, matched_path)
        if not symbols:
            return f"[{matched_path}]\nNo exported symbols or definitions found."
        lines = [f"[{matched_path}] — {len(symbols)} symbol(s):"]
        for s in symbols:
            lines.append(f"  {s['kind']:<14} {s['name']}  (line {s['line']})")
        return "\n".join(lines)

    def _get_dependencies(self, filter: str = "") -> str:
        """Return parsed dependency list from the project's manifest files."""
        PKG_NAMES = {
            "package.json", "requirements.txt", "pyproject.toml",
            "go.mod", "cargo.toml", "pom.xml", "gemfile", "composer.json",
        }
        known = self.store.known_paths(self.namespace)
        pkg_paths = [
            p for p in known
            if p.rsplit("/", 1)[-1].lower() in PKG_NAMES
        ]
        if not pkg_paths:
            return (
                "No dependency files (package.json, requirements.txt, go.mod, etc.) "
                "found in the indexed repo."
            )

        results: list[str] = []
        for pkg_path in pkg_paths[:4]:
            result = self.store.full_file(self.namespace, pkg_path)
            if not result:
                continue
            self.used_paths.add(pkg_path)
            content = result.get("content", result.get("text", ""))
            filename = pkg_path.rsplit("/", 1)[-1].lower()
            parsed = _parse_dep_file(filename, content, filter)
            if parsed.strip():
                results.append(f"[{pkg_path}]{parsed}")

        if not results:
            suffix = f" matching '{filter}'" if filter else ""
            return f"No dependencies found{suffix}."
        return "\n\n".join(results)

    def _call_graph(self, function_name: str, depth: int = 1) -> str:
        """Grep-based call graph: definition location, callers, and callees."""
        depth = max(1, min(int(depth), 2))

        # ── Find definition ────────────────────────────────────────────────
        def_patterns = [
            rf"(?:def|async def)\s+{re.escape(function_name)}\s*\(",
            rf"(?:async\s+)?function[*]?\s+{re.escape(function_name)}\s*\(",
            rf"(?:const|let|var)\s+{re.escape(function_name)}\s*=\s*(?:async\s*)?\(",
            rf"(?:const|let|var)\s+{re.escape(function_name)}\s*=\s*(?:async\s*)?(?:\w+\s*)?=>",
            rf"class\s+{re.escape(function_name)}\s*[{{\(]",
        ]
        def_hits: list[tuple[str, int]] = []   # (path, line)
        def_path = ""
        for pat in def_patterns:
            chunks = self.store.grep_chunks(self.namespace, pat)
            for c in chunks:
                m = c.get("metadata", {})
                path = m.get("path", "")
                text = c.get("text", "")
                try:
                    match = re.search(pat, text)
                    if match:
                        offset = text[: match.start()].count("\n")
                        line_no = int(m.get("line_start", 0)) + offset
                        if (path, line_no) not in def_hits:
                            def_hits.append((path, line_no))
                            if not def_path:
                                def_path = path
                            self.used_paths.add(path)
                except re.error:
                    pass

        # ── Find callers ───────────────────────────────────────────────────
        call_pattern = rf"\b{re.escape(function_name)}\s*\("
        caller_chunks = self.store.grep_chunks(self.namespace, call_pattern)
        callers: list[tuple[str, int, str]] = []   # (path, line, snippet)
        seen_caller: set[str] = set()
        def_path_set = {p for p, _ in def_hits}

        for c in caller_chunks:
            m = c.get("metadata", {})
            path = m.get("path", "")
            text = c.get("text", "")
            try:
                rx = re.compile(call_pattern)
                for match in rx.finditer(text):
                    offset = text[: match.start()].count("\n")
                    line_no = int(m.get("line_start", 0)) + offset
                    key = f"{path}:{line_no}"
                    if key in seen_caller:
                        continue
                    seen_caller.add(key)
                    # skip the definition file's own line
                    if path in def_path_set and any(abs(line_no - dl) < 3 for _, dl in def_hits if _ == path):
                        continue
                    snippet = text.splitlines()[offset].strip()[:80] if text.splitlines() else ""
                    self.used_paths.add(path)
                    callers.append((path, line_no, snippet))
            except re.error:
                pass

        callers.sort(key=lambda x: (x[0], x[1]))

        # ── Find callees (what does the function call) ─────────────────────
        callees: list[str] = []
        if def_path:
            full = self.store.full_file(self.namespace, def_path)
            if full:
                content = full.get("content", full.get("text", ""))
                callees = _extract_callees(content, function_name)

        # ── Format output ──────────────────────────────────────────────────
        out: list[str] = [f"Call graph: {function_name}"]

        if def_hits:
            out.append("\nDEFINED AT:")
            for path, ln in def_hits[:3]:
                out.append(f"  {path}:{ln}")
        else:
            out.append("\nDEFINED AT: not found (may be dynamically defined or from an external library)")

        if callers:
            out.append(f"\nCALLED FROM ({len(callers)} call site(s)):")
            for path, ln, snippet in callers[:20]:
                out.append(f"  {path}:{ln}  →  {snippet}")
            if len(callers) > 20:
                out.append(f"  ... and {len(callers) - 20} more call sites")
        else:
            out.append("\nCALLED FROM: no call sites found — may be an entry point or externally invoked")

        if callees:
            out.append(f"\n{function_name} INTERNALLY CALLS:")
            for name in callees:
                out.append(f"  {name}()")
        else:
            out.append(f"\n{function_name} INTERNALLY CALLS: none detected in body")

        return "\n".join(out)

    def _run_grep_ast(self, node_type: str, name_filter: str = "") -> str:
        """Find all structural definitions of a given AST node type across the repo."""
        nt = node_type.lower().strip()
        patterns_list = _AST_PATTERNS.get(nt)
        if not patterns_list:
            available = ", ".join(sorted(_AST_PATTERNS.keys()))
            return f"Unknown node type '{node_type}'. Available: {available}"

        results: list[tuple[str, int, str, str]] = []  # (path, line, name, hint)
        seen: set[str] = set()
        fil = name_filter.lower()

        for pattern, lang_hint in patterns_list:
            try:
                chunks = self.store.grep_chunks(self.namespace, pattern)
            except Exception:
                continue
            for c in chunks:
                m = c.get("metadata", {})
                path = m.get("path", "")
                text = c.get("text", "")
                try:
                    for match in re.finditer(pattern, text, re.MULTILINE):
                        # routes have two groups (method + path), decorators one group
                        if nt == "route":
                            method = match.group(1).upper()
                            route_path = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                            name = f"{method} {route_path}"
                        elif nt == "decorator":
                            name = f"@{match.group(1)}"
                        elif match.lastindex and match.lastindex >= 1:
                            name = match.group(1)
                        else:
                            name = match.group(0)[:40]

                        if fil and fil not in name.lower():
                            continue
                        dedup_key = f"{path}:{name}"
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        offset = text[: match.start()].count("\n")
                        line_no = int(m.get("line_start", 0)) + offset
                        self.used_paths.add(path)
                        results.append((path, line_no, name, lang_hint))
                except (re.error, IndexError):
                    continue

        if not results:
            suffix = f" matching '{name_filter}'" if name_filter else ""
            return f"No {nt} definitions found{suffix} in the indexed repo."

        results.sort(key=lambda x: (x[0], x[1]))

        label = f"matching '{name_filter}'" if name_filter else "in repo"
        lines = [f"Found {len(results)} {nt}(s) {label}:"]
        for path, ln, name, _ in results[:50]:
            lines.append(f"  {path}:{ln}  {name}")
        if len(results) > 50:
            lines.append(f"  ... and {len(results) - 50} more")
        return "\n".join(lines)
