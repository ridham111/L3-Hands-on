"""Agent tool definitions and executor.

The LLM receives TOOL_DEFINITIONS and decides which tools to call. Python
executes exactly what is requested — it makes no decisions about what to
retrieve or when to stop. This is the core of Level-3 agent behaviour.
"""
from __future__ import annotations

import fnmatch
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
]


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
            return f"Unknown tool: {tool_name!r}"
        except Exception as exc:  # noqa: BLE001
            return f"Tool error ({tool_name}): {exc}"

    # ── tool implementations ──────────────────────────────────────────────────

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
            # suffix / substring fallback
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
