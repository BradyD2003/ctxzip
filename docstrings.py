"""
docstrings.py — docstring extraction and auto-generation

At index time, every chunk goes through this module:
  1. Markdown / SQL (LIGHTWEIGHT_DOCSTRING_LANGUAGES): Tier-1 text is chunk.raw; no Haiku.
  2. Else: try to extract an existing docstring from the raw source
  3. If none found, call Haiku to generate a concise docstring
  4. Store the docstring on the chunk as chunk.docstring

The docstring becomes Tier 1 in the three-tier context system:
  Tier 0: signature only     (~8 tokens)   — always sent
  Tier 1: docstring summary  (~25 tokens)  — sent for relevant chunks
  Tier 2: full raw source    (~130 tokens) — pulled on demand for edit targets
"""

import re
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from chunker import LIGHTWEIGHT_DOCSTRING_LANGUAGES

# Parallel Anthropic calls during index (Haiku docstring generation)
DOCSTRING_API_CONCURRENCY = 10


# ── Extraction ─────────────────────────────────────────────────────────────

def extract_docstring(raw: str, language: str) -> Optional[str]:
    if language == 'python':
        return _extract_python_docstring(raw)
    elif language in ('javascript', 'typescript'):
        return _extract_js_docstring(raw)
    elif language in ('go', 'rust', 'java', 'c', 'cpp', 'swift', 'kotlin', 'scala', 'php'):
        return _extract_block_comment(raw)
    elif language == 'ruby':
        return _extract_ruby_comment(raw)
    return None


def _extract_python_docstring(raw: str) -> Optional[str]:
    lines = raw.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(r'\s*(def |async def |class )', line):
            start = i + 1
            break

    in_doc = False
    doc_lines = []
    doc_char = None

    for line in lines[start:]:
        s = line.strip()
        if not in_doc:
            if s.startswith('"""') or s.startswith("'''"):
                doc_char = s[:3]
                content = s[3:]
                if content.endswith(doc_char) and len(content) > len(doc_char):
                    return content[:-3].strip()
                in_doc = True
                if content.strip():
                    doc_lines.append(content.strip())
            elif s and not s.startswith('#'):
                break
        else:
            if doc_char and doc_char in s:
                final = s[:s.index(doc_char)].strip()
                if final:
                    doc_lines.append(final)
                break
            doc_lines.append(s)

    return '\n'.join(doc_lines).strip() if doc_lines else None


def _extract_js_docstring(raw: str) -> Optional[str]:
    """Extract JSDoc /** ... */ comment before the function."""
    lines = raw.splitlines()
    in_block = False
    doc_lines = []

    for line in lines:
        s = line.strip()
        if not in_block:
            if s.startswith('/**'):
                in_block = True
                content = s[3:].strip()
                if content and not content.startswith('*'):
                    doc_lines.append(content)
            elif re.match(r'(export|async|function|const|class|public|private)', s):
                break
        else:
            if s.endswith('*/'):
                content = s[:-2].lstrip('* ').strip()
                if content:
                    doc_lines.append(content)
                break
            content = s.lstrip('* ')
            if content:
                doc_lines.append(content)

    return '\n'.join(doc_lines).strip() if doc_lines else None


def _extract_block_comment(raw: str) -> Optional[str]:
    """Extract doc comments (///, /** */) immediately before the function signature."""
    lines = raw.splitlines()
    comment_lines = []
    hit_code = False

    for line in lines:
        s = line.strip()
        if s.startswith('///') or s.startswith('//!'):
            comment_lines.append(s[3:].strip())
        elif s.startswith('//') and not hit_code:
            comment_lines.append(s[2:].strip())
        elif s.startswith('/**') or s.startswith('/*'):
            in_block = '*/' not in s
            content = re.sub(r'^/?\*+/?', '', s).strip()
            if content and content != '/':
                comment_lines.append(content)
            if in_block:
                continue
        elif s.startswith('*'):
            content = s.lstrip('* ').rstrip('*/').strip()
            if content:
                comment_lines.append(content)
        elif comment_lines:
            break
        else:
            hit_code = True

    return '\n'.join(comment_lines).strip() if comment_lines else None


def _extract_ruby_comment(raw: str) -> Optional[str]:
    lines = raw.splitlines()
    comment_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith('#'):
            comment_lines.append(s[1:].strip())
        elif comment_lines:
            break
    return '\n'.join(comment_lines).strip() if comment_lines else None


# ── Auto-generation via Haiku ──────────────────────────────────────────────

DOCSTRING_SYSTEM = """You are a precise technical writer generating docstrings for code functions.
Given a function's source code and its file path, generate a concise docstring:
1. One-sentence description of what the function does
2. Key parameters and their purpose (skip self/cls)
3. What it returns

Use the File path to disambiguate context (e.g. service, route, edge function, or feature area).
Use the language's conventional doc comment style.

Rules:
- Max 6 lines total.
- No fluff, no restating the function name.
- Focus on WHY and WHAT, not HOW.
- Return ONLY the docstring text, no wrapping code or explanation."""


def generate_docstring(
    chunk_raw: str,
    signature: str,
    language: str,
    filepath: str = "",
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return f"Function: {signature}"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        file_line = f"File: {filepath}\n\n" if filepath else ""
        prompt = f"""Generate a docstring for this {language} function.

{file_line}```{language}
{chunk_raw[:2000]}
```

Signature: {signature}"""

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=DOCSTRING_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()

    except Exception:
        return f"Function: {signature}"


def docstring_tokens(docstring: str) -> int:
    return max(1, len(docstring) // 4)


def ensure_docstring(
    chunk_raw: str,
    signature: str,
    language: str,
    existing: Optional[str] = None,
    generate: bool = True,
    filepath: str = "",
) -> tuple[str, bool]:
    if existing:
        return existing, False
    extracted = extract_docstring(chunk_raw, language)
    if extracted:
        return extracted, False
    if generate:
        generated = generate_docstring(chunk_raw, signature, language, filepath)
        return generated, True
    return f"Function: {signature}", False


def enrich_with_docstrings(chunks, force_generate: bool = False):
    """
    For each chunk: extract existing docstring, or generate one via Haiku.
    Returns (chunks, extracted_count, generated_count).
    """
    extracted = 0
    generated = 0
    needs_generation = []

    for chunk in chunks:
        if chunk.docstring and not force_generate:
            continue
        if chunk.language in LIGHTWEIGHT_DOCSTRING_LANGUAGES:
            chunk.docstring = chunk.raw
            chunk.docstring_generated = False
            extracted += 1
            continue
        doc = extract_docstring(chunk.raw, chunk.language)
        if doc:
            chunk.docstring = doc
            chunk.docstring_generated = False
            extracted += 1
        else:
            needs_generation.append(chunk)

    if needs_generation:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:

            def _gen_one(c):
                return generate_docstring(c.raw, c.signature, c.language, c.file)

            with ThreadPoolExecutor(max_workers=DOCSTRING_API_CONCURRENCY) as pool:
                docs = list(pool.map(_gen_one, needs_generation))

            for chunk, doc in zip(needs_generation, docs):
                chunk.docstring = doc
                chunk.docstring_generated = True
                generated += 1

    return chunks, extracted, generated
