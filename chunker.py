"""
chunker.py — universal multi-language code chunker

For Python, JavaScript, and TypeScript/TSX, indexing prefers Tree-sitter
(AST-accurate spans) when `tree-sitter` and grammar wheels are installed;
otherwise regex-based chunkers are used.

Supported languages:
  Python      .py
  JavaScript  .js .mjs .cjs .jsx
  TypeScript  .ts .tsx
  Go          .go
  Rust        .rs
  Java        .java
  C / C++     .c .cpp .cc .h .hpp
  Ruby        .rb
  PHP         .php
  Swift       .swift
  Kotlin      .kt
  Scala       .scala
  Markdown    .md   (indexed; Tier-1 text = raw chunk, no Haiku)
  SQL         .sql  (indexed; Tier-1 text = raw chunk, no Haiku)

  Skipped (not indexed): .json .yaml .yml — usually config, low semantic value.

Falls back to line-window chunking for unknown file types.
"""
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    signature: str
    name: str
    raw: str
    compressed: str
    file: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    raw_tokens: int = 0
    compressed_tokens: int = 0
    docstring: str = ""
    docstring_generated: bool = False
    embedding: list[float] = field(default_factory=list)

    def token_savings(self) -> float:
        if self.raw_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.raw_tokens)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if not d.get('embedding'):
            d.pop('embedding', None)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        d.setdefault('language', 'python')
        d.setdefault('chunk_type', 'function')
        d.setdefault('docstring', '')
        d.setdefault('docstring_generated', False)
        d.setdefault('embedding', [])
        return Chunk(**d)


def _hash(text: str) -> str:
    return "cx_" + hashlib.md5(text.encode()).hexdigest()[:10]

def _tokens(text: str) -> int:
    return max(1, len(text) // 4)

def _make_chunk(raw, sig, name, file, start, end, lang, ctype):
    compressed = _compress(raw, lang)
    return Chunk(
        id=_hash(file + raw), signature=sig, name=name,
        raw=raw, compressed=compressed, file=file,
        start_line=start, end_line=end, language=lang, chunk_type=ctype,
        raw_tokens=_tokens(raw), compressed_tokens=_tokens(compressed),
    )


EXTENSION_MAP = {
    '.py': 'python',
    '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.go': 'go', '.rs': 'rust', '.java': 'java',
    '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp',
    '.rb': 'ruby', '.php': 'php', '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
    '.md': 'markdown',
    '.sql': 'sql',
}

# Not indexed — config-heavy, rarely useful for code semantic search
SKIP_EXTENSIONS = frozenset({'.json', '.yaml', '.yml'})

# Indexed; docstrings.py uses chunk raw as Tier-1 summary (no Haiku generation)
LIGHTWEIGHT_EXTENSIONS = frozenset({'.md', '.sql'})
LIGHTWEIGHT_DOCSTRING_LANGUAGES = frozenset(
    EXTENSION_MAP[ext] for ext in LIGHTWEIGHT_EXTENSIONS
)

def detect_language(path: str | Path) -> str:
    return EXTENSION_MAP.get(Path(path).suffix.lower(), 'unknown')


def _compress(raw: str, lang: str) -> str:
    lines = raw.splitlines()
    out = []

    if lang == 'python':
        in_doc = False
        doc_char = None
        for line in lines:
            s = line.strip()
            if not in_doc:
                if s.startswith(('"""', "'''")):
                    doc_char = s[:3]
                    if s.count(doc_char) >= 2 and len(s) > 3:
                        continue
                    in_doc = True
                    continue
            else:
                if doc_char and doc_char in s:
                    in_doc = False
                continue
            if not s or s.startswith('#'):
                continue
            if re.match(r'^(import |from )\S', s) and '(' not in s:
                continue
            indent = min(len(line) - len(line.lstrip()), 8)
            out.append(' ' * indent + re.sub(r'\s{2,}', ' ', s))

    elif lang in ('javascript', 'typescript'):
        in_block = False
        for line in lines:
            s = line.strip()
            if not in_block:
                if s.startswith(('/*', '/**')):
                    if '*/' not in s:
                        in_block = True
                    continue
                if s.startswith('//'):
                    continue
                if re.match(r'^(import |export \{|require\()', s) and ';' in s and len(s) < 80:
                    continue
            else:
                if '*/' in s:
                    in_block = False
                continue
            if not s:
                continue
            indent = min(len(line) - len(line.lstrip()), 8)
            out.append(' ' * indent + re.sub(r'\s{2,}', ' ', s))

    else:
        in_block = False
        for line in lines:
            s = line.strip()
            if not in_block:
                if '/*' in s:
                    before = s[:s.index('/*')]
                    if '*/' in s[s.index('/*'):]:
                        s = (before + s[s.rindex('*/') + 2:]).strip()
                        if not s:
                            continue
                    else:
                        in_block = True
                        if before.strip():
                            out.append(before.strip())
                        continue
                s = re.sub(r'//.*$', '', s).strip()
                if not s:
                    continue
            else:
                if '*/' in s:
                    in_block = False
                    after = s[s.rindex('*/') + 2:].strip()
                    if after:
                        out.append(after)
                continue
            indent = min(len(line) - len(line.lstrip()), 8)
            out.append(' ' * indent + re.sub(r'\s{2,}', ' ', s))

    return '\n'.join(out)


def _find_block_end(lines: list[str], start: int) -> int:
    """Find the closing brace for a block, aware of string literals and comments."""
    depth = 0
    found = False
    in_block_comment = False

    for i in range(start, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if in_block_comment:
            if '*/' in stripped:
                in_block_comment = False
            continue
        if stripped.startswith('/*') and '*/' not in stripped:
            in_block_comment = True
            continue

        in_str = False
        str_char = None
        escaped = False
        for ch in line:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if in_str:
                if ch == str_char:
                    in_str = False
                continue
            if ch in ('"', "'", '`'):
                in_str = True
                str_char = ch
                continue
            if ch == '{':
                depth += 1
                found = True
            elif ch == '}':
                depth -= 1
                if found and depth == 0:
                    return i + 1
    return len(lines)


def _chunk_python(source: str, fp: str) -> list[Chunk]:
    """
    Chunk Python source into individual functions, methods, and class headers.
    Classes are captured as header-only (up to the first method), and each
    method is captured individually. Decorators are included with their target.
    Matches any indentation level.
    """
    lines = source.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\s*)(async\s+def |def |class )', line)
        if not m:
            i += 1
            continue

        base_indent = len(m.group(1))
        is_class = m.group(2).strip() == 'class'

        # Walk backwards to capture decorators
        dec_start = i
        while dec_start > 0 and lines[dec_start - 1].strip().startswith('@'):
            dec_start -= 1

        if is_class:
            # Capture class header only: everything up to the first def inside it
            j = i + 1
            while j < len(lines):
                nl = lines[j]
                if nl.strip() == '':
                    j += 1
                    continue
                nl_indent = len(nl) - len(nl.lstrip())
                if nl_indent <= base_indent and nl.strip():
                    break
                if nl_indent > base_indent and re.match(r'^\s*(async\s+)?def\s', nl):
                    break
                j += 1
            raw = '\n'.join(lines[dec_start:j])
            cm = re.search(r'class\s+(\w+)', line)
            name = cm.group(1) if cm else line.strip()[:30]
            sig = f"class {name}"
            if len(raw.strip()) > 10:
                chunks.append(_make_chunk(raw, sig, name, fp, dec_start + 1, j, 'python', 'class'))
            i = j
        else:
            j = i + 1
            while j < len(lines):
                nl = lines[j]
                if nl.strip() == '':
                    j += 1
                    continue
                if len(nl) - len(nl.lstrip()) <= base_indent and nl.strip():
                    break
                j += 1
            raw = '\n'.join(lines[dec_start:j])
            s = line.strip()
            fm = re.search(r'(async\s+)?def\s+(\w+)\s*\(([^)]*)\)', s)
            if fm:
                name = fm.group(2)
                sig = f"{name}({re.sub(r'\\s+', ' ', fm.group(3)).strip()})"
            else:
                name = s[:30]
                sig = s[:60]
            if len(raw.strip()) > 10:
                chunks.append(_make_chunk(raw, sig, name, fp, dec_start + 1, j, 'python', 'function'))
            i = j

    return chunks


JS_PATS = [
    (r'^(?:export\s+)?(?:default\s+)?async\s+function\s+(\w+)\s*\(', 'function'),
    (r'^(?:export\s+)?(?:default\s+)?function\s+(\w+)\s*\(', 'function'),
    (r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>', 'function'),
    (r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function', 'function'),
    (r'^(?:export\s+)?(?:default\s+)?class\s+(\w+)', 'class'),
    (r'^  (?:async\s+)?(?:static\s+)?(?:get\s+|set\s+)?(\w+)\s*\(', 'method'),
]

def _chunk_js_ts(source: str, fp: str, lang: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    used = set()
    for i, line in enumerate(lines):
        if i in used:
            continue
        s = line.strip()
        for pattern, ctype in JS_PATS:
            m = re.match(pattern, s)
            if m:
                name = m.group(1) if m.lastindex and m.lastindex >= 1 else 'anonymous'
                end = _find_block_end(lines, i)
                raw = '\n'.join(lines[i:end])
                if len(raw.strip()) < 15:
                    break
                pm = re.search(r'\(([^)]{0,120})\)', lines[i])
                params = pm.group(1).strip() if pm else ''
                sig = f"class {name}" if ctype == 'class' else f"{name}({params})"
                chunks.append(_make_chunk(raw, sig, name, fp, i+1, end, lang, ctype))
                for j in range(i, end):
                    used.add(j)
                break
    return chunks


def _brace_pats(lang: str):
    if lang == 'go':
        return [(r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(', 'function'),
                (r'^type\s+(\w+)\s+struct\s*\{', 'class')]
    elif lang == 'rust':
        return [(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', 'function'),
                (r'^(?:pub\s+)?(?:struct|enum|impl|trait)\s+(\w+)', 'class')]
    elif lang == 'java':
        return [(r'(?:public|private|protected|static|final|\s)+\w[\w<>\[\]]*\s+(\w+)\s*\(', 'function'),
                (r'(?:public|private|protected)?\s*(?:abstract\s+)?class\s+(\w+)', 'class')]
    elif lang in ('c', 'cpp'):
        return [(r'^[\w:*&<>\s]+\s+(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{', 'function'),
                (r'^(?:class|struct)\s+(\w+)', 'class')]
    elif lang == 'swift':
        return [(r'^(?:\w+\s+)*func\s+(\w+)', 'function'),
                (r'^(?:\w+\s+)*(?:class|struct)\s+(\w+)', 'class')]
    elif lang == 'kotlin':
        return [(r'^(?:\w+\s+)*fun\s+(\w+)', 'function'),
                (r'^(?:\w+\s+)*class\s+(\w+)', 'class')]
    elif lang == 'scala':
        return [(r'^(?:def)\s+(\w+)', 'function'),
                (r'^(?:(?:case\s+)?class|object|trait)\s+(\w+)', 'class')]
    return []

def _chunk_brace(source: str, fp: str, lang: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    used = set()
    for i, line in enumerate(lines):
        if i in used:
            continue
        s = line.strip()
        for pattern, ctype in _brace_pats(lang):
            m = re.search(pattern, s)
            if m:
                name = m.group(1)
                end = _find_block_end(lines, i)
                raw = '\n'.join(lines[i:end])
                if len(raw.strip()) < 10:
                    break
                pm = re.search(r'\(([^)]{0,120})\)', lines[i])
                params = pm.group(1).strip() if pm else ''
                sig = f"{ctype} {name}" if ctype != 'function' else f"{name}({params})"
                chunks.append(_make_chunk(raw, sig, name, fp, i+1, end, lang, ctype))
                for j in range(i, end):
                    used.add(j)
                break
    return chunks


def _chunk_ruby(source: str, fp: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r'^def\s+(\w+[\?!]?)\s*(\([^)]*\))?', s)
        cm = re.match(r'^class\s+(\w+)', s)
        if m or cm:
            name = (m or cm).group(1)
            ctype = 'function' if m else 'class'
            j = i + 1
            depth = 1
            while j < len(lines) and depth > 0:
                ls = lines[j].strip()
                if re.match(r'^(def |class |module |if |unless |while |for |do\b|begin\b)', ls):
                    depth += 1
                if ls == 'end':
                    depth -= 1
                j += 1
            raw = '\n'.join(lines[i:j])
            pm = re.search(r'\(([^)]*)\)', lines[i])
            params = pm.group(1) if pm else ''
            sig = f"{name}({params})" if ctype == 'function' else f"class {name}"
            if len(raw.strip()) > 10:
                chunks.append(_make_chunk(raw, sig, name, fp, i+1, j, 'ruby', ctype))
            i = j
        else:
            i += 1
    return chunks


def _chunk_php(source: str, fp: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    used = set()
    pats = [(r'(?:public|private|protected|static|\s)*function\s+(\w+)\s*\(', 'function'),
            (r'(?:abstract\s+)?class\s+(\w+)', 'class')]
    for i, line in enumerate(lines):
        if i in used:
            continue
        for pattern, ctype in pats:
            m = re.search(pattern, line.strip())
            if m:
                name = m.group(1)
                end = _find_block_end(lines, i)
                raw = '\n'.join(lines[i:end])
                if len(raw.strip()) < 10:
                    break
                pm = re.search(r'\(([^)]{0,120})\)', lines[i])
                params = pm.group(1).strip() if pm else ''
                sig = f"class {name}" if ctype == 'class' else f"{name}({params})"
                chunks.append(_make_chunk(raw, sig, name, fp, i+1, end, 'php', ctype))
                for j in range(i, end):
                    used.add(j)
                break
    return chunks


def _chunk_generic(source: str, fp: str, lang: str, window: int = 40) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    for i in range(0, len(lines), window):
        raw = '\n'.join(lines[i:i+window])
        if not raw.strip():
            continue
        sig = f"lines {i+1}–{min(i+window, len(lines))}"
        chunks.append(_make_chunk(raw, sig, sig, fp, i+1, i+window, lang, 'block'))
    return chunks


def chunk_source(source: str, filepath: str) -> list[Chunk]:
    lang = detect_language(filepath)
    fp = str(filepath)
    # Prefer Tree-sitter AST chunking for supported languages when installed.
    if lang in ('python', 'javascript', 'typescript'):
        try:
            from tree_sitter_chunk import chunk_with_tree_sitter

            ast_chunks = chunk_with_tree_sitter(source, fp, lang)
            if ast_chunks:
                return ast_chunks
        except Exception:
            pass

    if lang == 'python':
        chunks = _chunk_python(source, fp)
    elif lang in ('javascript', 'typescript'):
        chunks = _chunk_js_ts(source, fp, lang)
    elif lang == 'ruby':
        chunks = _chunk_ruby(source, fp)
    elif lang == 'php':
        chunks = _chunk_php(source, fp)
    elif lang in ('go', 'rust', 'java', 'c', 'cpp', 'swift', 'kotlin', 'scala'):
        chunks = _chunk_brace(source, fp, lang)
    else:
        chunks = _chunk_generic(source, fp, lang)
    if not chunks and lang != 'unknown':
        chunks = _chunk_generic(source, fp, lang)
    return chunks


def chunk_file(path: str | Path) -> list[Chunk]:
    path = Path(path)
    source = path.read_text(encoding='utf-8', errors='replace')
    return chunk_source(source, str(path))


SKIP_DIRS = {'__pycache__', '.git', 'node_modules', 'venv', '.venv',
             'dist', 'build', '.next', 'target', '.cargo', 'vendor'}

def chunk_directory(root: str | Path, extensions: tuple | None = None) -> list[Chunk]:
    root = Path(root)
    supported = set(EXTENSION_MAP.keys())
    if extensions:
        supported = {e for e in extensions if e in supported}
    all_chunks = []
    for filepath in root.rglob('*'):
        if not filepath.is_file():
            continue
        if any(skip in filepath.parts for skip in SKIP_DIRS):
            continue
        suffix = filepath.suffix.lower()
        if suffix in SKIP_EXTENSIONS:
            continue
        if suffix not in supported:
            continue
        try:
            all_chunks.extend(chunk_file(filepath))
        except Exception:
            pass
    return all_chunks
