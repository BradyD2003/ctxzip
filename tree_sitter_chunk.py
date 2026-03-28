"""
AST-based chunking and enclosing-symbol lookup using Tree-sitter.

Used for Python, JavaScript, and TypeScript/TSX when grammar wheels and
tree-sitter bindings are available. Falls back to regex chunkers on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chunker import Chunk, _make_chunk, detect_language

# ── Optional Tree-sitter imports ───────────────────────────────────────────

try:
    from tree_sitter import Language, Parser, Point
except ImportError:
    Language = Parser = Point = None  # type: ignore

_TS_AVAILABLE = Language is not None


def _load_language_for_path(path: str | Path) -> tuple[Language, str] | None:
    """Return (Language, kind) where kind is python|javascript|typescript|tsx."""
    if not _TS_AVAILABLE:
        return None
    suf = Path(path).suffix.lower()
    if suf == ".py":
        try:
            import tree_sitter_python as tsp

            return Language(tsp.language()), "python"
        except ImportError:
            return None
    if suf in (".js", ".mjs", ".cjs", ".jsx"):
        try:
            import tree_sitter_javascript as tsj

            return Language(tsj.language()), "javascript"
        except ImportError:
            return None
    if suf == ".ts":
        try:
            import tree_sitter_typescript as tst

            return Language(tst.language_typescript()), "typescript"
        except ImportError:
            return None
    if suf == ".tsx":
        try:
            import tree_sitter_typescript as tst

            return Language(tst.language_tsx()), "tsx"
        except ImportError:
            return None
    return None


def _parser_for_path(path: str | Path) -> tuple[Parser, str] | None:
    loaded = _load_language_for_path(path)
    if not loaded:
        return None
    lang, kind = loaded
    p = Parser(lang)
    return p, kind


# ── Nesting rules (match prior regex behavior: no nested funcs as own chunks) ─


def _py_nested_inside_function(node) -> bool:
    p = node.parent
    while p is not None:
        if p.type == "module":
            return False
        if p.type == "function_definition":
            return True
        p = p.parent
    return False


def _js_nested_inside_function(node) -> bool:
    p = node.parent
    while p is not None:
        if p.type in ("program", "module"):
            return False
        if p.type in (
            "function_declaration",
            "function_expression",
            "method_definition",
            "arrow_function",
        ):
            return True
        p = p.parent
    return False


# Node types to emit as chunks (language-specific)
_PY_EMIT_TYPES = frozenset({"function_definition", "class_definition", "decorated_definition"})

_JS_TS_EMIT_TYPES = frozenset(
    {
        "function_declaration",
        "function_expression",
        "generator_function",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "abstract_method_signature",
    }
)


def _py_should_emit(node) -> bool:
    if node.type not in _PY_EMIT_TYPES:
        return False
    if node.type == "function_definition" and node.parent and node.parent.type == "decorated_definition":
        return False
    if node.type == "class_definition" and node.parent and node.parent.type == "decorated_definition":
        return False
    if node.type in ("function_definition", "class_definition") and _py_nested_inside_function(node):
        return False
    if node.type == "decorated_definition":
        inner = next(
            (c for c in node.children if c.type in ("function_definition", "class_definition")),
            None,
        )
        if inner is not None and _py_nested_inside_function(inner):
            return False
    return True


def _js_should_emit(node) -> bool:
    if node.type not in _JS_TS_EMIT_TYPES:
        return False
    if _js_nested_inside_function(node):
        return False
    return True


def _walk(node, visit):
    visit(node)
    for c in node.children:
        _walk(c, visit)


def _py_extract_name_sig(raw: str, node) -> tuple[str, str]:
    line0 = raw.splitlines()[0].strip() if raw.strip() else "anonymous"
    if node.type == "decorated_definition":
        inner = next(
            (c for c in node.children if c.type in ("function_definition", "class_definition")),
            None,
        )
        if inner:
            return _py_extract_name_sig(raw[inner.start_byte - node.start_byte :], inner)
    if node.type == "class_definition":
        for c in node.children:
            if c.type == "identifier":
                return c.text.decode("utf-8", errors="replace"), f"class {c.text.decode('utf-8', errors='replace')}"
        return line0[:40], line0[:80]
    if node.type == "function_definition":
        name = "lambda"
        for c in node.children:
            if c.type == "identifier":
                name = c.text.decode("utf-8", errors="replace")
                break
        inner = raw
        paren = inner.find("(")
        paren2 = inner.find(")", paren + 1) if paren >= 0 else -1
        params = ""
        if paren >= 0 and paren2 > paren:
            params = inner[paren + 1 : paren2].replace("\n", " ")
            params = " ".join(params.split())
        return name, f"{name}({params})"
    return line0[:40], line0[:80]


def _js_extract_name_sig(raw: str, node) -> tuple[str, str]:
    line0 = raw.splitlines()[0].strip() if raw.strip() else "anonymous"
    t = node.type
    if t == "class_declaration":
        for c in node.children:
            if c.type in ("identifier", "type_identifier"):
                n = c.text.decode("utf-8", errors="replace")
                return n, f"class {n}"
        return line0[:40], line0[:80]
    if t == "method_definition":
        for c in node.children:
            if c.type in ("property_identifier", "identifier"):
                n = c.text.decode("utf-8", errors="replace")
                pm = __import__("re").search(r"\(([^)]{0,200})\)", raw.splitlines()[0])
                params = pm.group(1).strip() if pm else ""
                return n, f"{n}({params})"
        return line0[:40], line0[:80]
    if t in ("function_declaration", "function_expression", "generator_function", "generator_function_declaration"):
        for c in node.children:
            if c.type == "identifier":
                n = c.text.decode("utf-8", errors="replace")
                pm = __import__("re").search(r"\(([^)]{0,200})\)", raw.splitlines()[0])
                params = pm.group(1).strip() if pm else ""
                return n, f"{n}({params})"
        return "anonymous", line0[:80]
    if t == "arrow_function":
        pm = __import__("re").search(r"\(([^)]*)\)\s*=>", raw[:300])
        if pm:
            return "arrow", f"({pm.group(1).strip()}) =>"
        return "arrow", line0[:80]
    return line0[:40], line0[:80]


def _chunk_type_for_js(node) -> str:
    t = node.type
    if t == "class_declaration":
        return "class"
    if t == "method_definition":
        return "method"
    return "function"


def chunk_with_tree_sitter(source: str, filepath: str, lang: str) -> list[Chunk] | None:
    """
    Build chunks from Tree-sitter AST. Returns None if parsing is unavailable
    or produced no chunks (caller should fall back to regex).
    """
    p = _parser_for_path(filepath)
    if not p:
        return None
    parser, kind = p
    src_bytes = source.encode("utf-8")
    tree = parser.parse(src_bytes)
    root = tree.root_node
    if root.has_error:
        # Still try — many files parse with partial errors
        pass

    chunks: list[Chunk] = []
    lang_label = lang

    def visit(node):
        if kind == "python":
            if not _py_should_emit(node):
                return
            raw = src_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            if len(raw.strip()) < 8:
                return
            name, sig = _py_extract_name_sig(raw, node)
            ctype = "class" if node.type == "class_definition" else "function"
            if node.type == "decorated_definition":
                inner = next(
                    (c for c in node.children if c.type in ("function_definition", "class_definition")),
                    None,
                )
                if inner and inner.type == "class_definition":
                    ctype = "class"
            sl = node.start_point.row + 1
            el = node.end_point.row + 1
            chunks.append(_make_chunk(raw, sig, name, filepath, sl, el, lang_label, ctype))
        else:
            if not _js_should_emit(node):
                return
            raw = src_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            if len(raw.strip()) < 8:
                return
            name, sig = _js_extract_name_sig(raw, node)
            ctype = _chunk_type_for_js(node)
            sl = node.start_point.row + 1
            el = node.end_point.row + 1
            chunks.append(_make_chunk(raw, sig, name, filepath, sl, el, lang_label, ctype))

    _walk(root, visit)
    return chunks if chunks else None


# ── Enclosing symbol (for ctxzip_get_function) ─────────────────────────────

_ENCLOSING_PY = frozenset({"function_definition", "class_definition", "decorated_definition"})
_ENCLOSING_JS = frozenset(
    {
        "function_declaration",
        "function_expression",
        "generator_function",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    }
)


@dataclass
class EnclosingSpan:
    raw: str
    start_line: int
    end_line: int
    name: str
    signature: str
    chunk_type: str
    language: str


def _first_enclosing(node, kind: str):
    cur = node
    while cur is not None:
        if kind == "python":
            if cur.type in _ENCLOSING_PY:
                return cur
        else:
            if cur.type in _ENCLOSING_JS:
                return cur
        cur = cur.parent
    return None


def _point_first_code_on_line(source: str, line_1_indexed: int) -> Point:
    """Row/column for tree-sitter lookup; column 0 often maps to the wrong node."""
    lines = source.splitlines(keepends=True)
    row = line_1_indexed - 1
    if not lines:
        return Point(0, 0)
    if row < 0:
        row = 0
    elif row >= len(lines):
        row = len(lines) - 1
    line = lines[row]
    col = 0
    for i, ch in enumerate(line):
        if ch not in " \t\r\n":
            col = i
            break
    return Point(row, col)


def get_enclosing_symbol(source: str, filepath: str, line_1_indexed: int) -> EnclosingSpan | None:
    """
    Return the smallest AST node (function/class/method/decorated def) that
    encloses the given 1-based line, using Tree-sitter.
    """
    p = _parser_for_path(filepath)
    if not p:
        return None
    parser, kind = p
    src_bytes = source.encode("utf-8")
    tree = parser.parse(src_bytes)
    root = tree.root_node
    pt = _point_first_code_on_line(source, line_1_indexed)
    leaf = root.descendant_for_point_range(pt, pt)
    if leaf is None:
        return None
    enc = _first_enclosing(leaf, kind)
    if enc is None:
        return None
    raw = src_bytes[enc.start_byte : enc.end_byte].decode("utf-8", errors="replace")
    sl = enc.start_point.row + 1
    el = enc.end_point.row + 1
    lang_label = detect_language(filepath)

    if kind == "python":
        name, sig = _py_extract_name_sig(raw, enc)
        ctype = "class" if enc.type == "class_definition" else "function"
        if enc.type == "decorated_definition":
            inner = next(
                (c for c in enc.children if c.type in ("function_definition", "class_definition")),
                None,
            )
            if inner and inner.type == "class_definition":
                ctype = "class"
    else:
        name, sig = _js_extract_name_sig(raw, enc)
        ctype = _chunk_type_for_js(enc)

    return EnclosingSpan(
        raw=raw,
        start_line=sl,
        end_line=el,
        name=name,
        signature=sig,
        chunk_type=ctype,
        language=lang_label,
    )


def merge_chunks_for_range(
    chunks: list[Chunk],
    filepath: str,
    start_line: int,
    end_line: int,
) -> str | None:
    """
    Fallback: concatenate indexed chunks in the same file that overlap or touch
    [start_line, end_line], sorted by start_line.
    """
    path = str(Path(filepath).resolve())
    related = [
        c
        for c in chunks
        if str(Path(c.file).resolve()) == path
        and not (c.end_line < start_line or c.start_line > end_line)
    ]
    if not related:
        return None
    related.sort(key=lambda c: (c.start_line, c.end_line))
    merged_start = min(c.start_line for c in related)
    merged_end = max(c.end_line for c in related)
    raw_parts = [c.raw for c in related]
    return "\n\n".join(raw_parts)
