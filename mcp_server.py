"""
mcp_server.py — ctxzip as an MCP tool server

Exposes four tools to any MCP-compatible client (Cursor, Claude Code, Claude Desktop):
  ctxzip_index        Index a file or directory (generates docstrings + embeddings)
  ctxzip_query        Get three-tier context payload for a coding task
  ctxzip_get_source   Pull full raw source for any chunk by ID
  ctxzip_stats        Show what's currently indexed

Setup:
  pip install mcp anthropic openai numpy rich

Register in Cursor (.cursor/mcp.json):
  {
    "mcpServers": {
      "ctxzip": {
        "command": "python",
        "args": ["/absolute/path/to/ctxzip/mcp_server.py"],
        "env": {
          "ANTHROPIC_API_KEY": "sk-ant-...",
          "OPENAI_API_KEY": "sk-..."
        }
      }
    }
  }
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load .env file from the same directory as the server
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            if value and not os.environ.get(key.strip()):
                os.environ[key.strip()] = value

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from chunker import chunk_file, chunk_directory, Chunk, _tokens
from docstrings import enrich_with_docstrings
from retriever import (
    Retriever, tier0_repr, tier1_repr,
    embed_texts, _chunk_embed_text,
)
from payload import build_payload


# ── Index persistence ──────────────────────────────────────────────────────


def _index_path() -> Path:
    raw = os.environ.get("CTXZIP_INDEX")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / ".ctxzip_index.json"


def _save_index(chunks: list[Chunk]):
    data = [c.to_dict() for c in chunks]
    _index_path().write_text(json.dumps(data, indent=2))


def _load_index() -> list[Chunk]:
    path = _index_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [Chunk.from_dict(d) for d in data]


def _merge(existing: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    """
    Replace chunks from re-indexed files instead of accumulating stale entries.
    Any existing chunk whose file matches a file in the new set is removed,
    then all new chunks are added.
    """
    new_files = {c.file for c in new}
    kept = [c for c in existing if c.file not in new_files]
    return kept + new


def _generate_embeddings(chunks: list[Chunk]) -> int:
    """
    Generate embeddings for chunks that don't have them yet.
    Returns the number of newly embedded chunks.
    """
    needs_embed = [c for c in chunks if not c.embedding]
    if not needs_embed:
        return 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return 0

    try:
        texts = [_chunk_embed_text(c) for c in needs_embed]
        vectors = embed_texts(texts, api_key)
        embedded = 0
        for chunk, vec in zip(needs_embed, vectors):
            if vec:
                chunk.embedding = [round(x, 5) for x in vec]
                embedded += 1
        return embedded
    except ImportError:
        return 0
    except Exception as e:
        print(f"[ctxzip] Embedding generation failed: {e}")
        return 0


# ── MCP server ─────────────────────────────────────────────────────────────

app = Server("ctxzip")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ctxzip_index",
            description=(
                "Index a source file or directory into ctxzip. "
                "Chunks code into functions/classes, extracts or AI-generates docstrings, "
                "and computes semantic embeddings for every function. "
                "Run this once per project before using ctxzip_query. "
                "Re-indexing a path replaces stale chunks from changed files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to a file or directory to index"
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file extensions to include, e.g. [\".py\", \".ts\"]. Defaults to all supported languages.",
                    }
                },
                "required": ["path"]
            }
        ),

        Tool(
            name="ctxzip_query",
            description=(
                "Get a three-tier compressed context payload for a coding task. "
                "Uses semantic search (OpenAI embeddings) when available, falls back to TF-IDF. "
                "Tier 0: signatures of all indexed functions (full codebase map). "
                "Tier 1: docstring summaries for the most relevant functions. "
                "Tier 2: full raw source for the edit target (edit intent only). "
                "Returns the payload you should use as context for your next action."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The coding task or question, e.g. 'fix the seam artifact in stitch_tiles' or 'how does token refresh work'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many Tier 1 context summaries to include (default: 4)",
                        "default": 4
                    }
                },
                "required": ["task"]
            }
        ),

        Tool(
            name="ctxzip_get_source",
            description=(
                "Retrieve the full raw source code for any indexed function or class by its chunk ID. "
                "Chunk IDs (cx_xxxxxxxx) appear in the Tier 0 directory returned by ctxzip_query. "
                "Call this when you need to inspect a function's full implementation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID from the codebase directory, e.g. cx_68be5bb6f8"
                    }
                },
                "required": ["chunk_id"]
            }
        ),

        Tool(
            name="ctxzip_stats",
            description=(
                "Show what's currently indexed in ctxzip: chunk count, file count, "
                "embedding coverage, docstring coverage, and token breakdown."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    # ── ctxzip_index ───────────────────────────────────────────────────────
    if name == "ctxzip_index":
        path_str = arguments.get("path", "")
        extensions = arguments.get("extensions")
        p = Path(path_str)

        if not p.exists():
            return [TextContent(type="text", text=f"Error: path not found: {path_str}")]

        try:
            if p.is_file():
                new_chunks = chunk_file(p)
            else:
                ext_tuple = tuple(extensions) if extensions else None
                new_chunks = chunk_directory(p, extensions=ext_tuple)

            if not new_chunks:
                return [TextContent(type="text", text=f"No supported source files found at: {path_str}")]

            new_chunks, extracted, generated = enrich_with_docstrings(new_chunks)

            embedded = _generate_embeddings(new_chunks)

            existing = _load_index()
            merged = _merge(existing, new_chunks)

            # Re-embed any existing chunks that lost embeddings after merge
            _generate_embeddings(merged)
            _save_index(merged)

            new_files = {c.file for c in new_chunks}
            stale_removed = sum(1 for c in existing if c.file in new_files)

            total_raw = sum(c.raw_tokens for c in merged)
            tier0_tok = sum(_tokens(tier0_repr(c)) for c in merged)
            has_doc = sum(1 for c in merged if c.docstring)
            has_embed = sum(1 for c in merged if c.embedding)
            files = len(set(c.file for c in merged))

            lines = [
                f"✓ Indexed {path_str}",
                f"  New chunks:            {len(new_chunks)}",
                f"  Stale chunks replaced: {stale_removed}",
                f"  Total chunks:          {len(merged)} across {files} files",
                f"  Docstrings extracted:  {extracted}",
                f"  Docstrings generated:  {generated}",
                f"  Embeddings computed:   {embedded}",
                f"  Embedding coverage:    {has_embed}/{len(merged)}",
                f"",
                f"  Full codebase tokens:  {total_raw:,}",
                f"  Tier 0 (all sigs):     {tier0_tok:,} ({tier0_tok/total_raw*100:.0f}% of full)" if total_raw else "  Tier 0: (empty)",
                f"  Docstring coverage:    {has_doc}/{len(merged)} functions",
                f"",
                f"  Search mode: {'semantic (embeddings)' if has_embed else 'TF-IDF (set OPENAI_API_KEY for semantic)'}",
                f"  Ready. Use ctxzip_query to get context for a task.",
            ]
            return [TextContent(type="text", text='\n'.join(lines))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error indexing {path_str}: {e}")]


    # ── ctxzip_query ───────────────────────────────────────────────────────
    elif name == "ctxzip_query":
        task = arguments.get("task", "")
        top_k = arguments.get("top_k", 4)

        chunks = _load_index()
        if not chunks:
            return [TextContent(type="text", text=(
                "No index found. Run ctxzip_index first:\n"
                "  ctxzip_index(path='/path/to/your/project')"
            ))]

        try:
            retriever = Retriever(chunks, top_k_tier1=top_k)
            result = retriever.retrieve(task)
            payload = build_payload(result)

            savings = result.savings_pct()
            full_tok = result.full_codebase_tokens()
            payload_tok = result.total_payload_tokens()

            header = '\n'.join([
                f"── ctxzip context for: {task}",
                f"   Intent:  {result.intent.value.upper()}",
                f"   Search:  {result.search_mode}",
                f"   Tier 0:  {len(result.tier0_chunks)} functions (directory)",
                f"   Tier 1:  {len(result.tier1_chunks)} summaries (docstrings)",
                f"   Tier 2:  {len(result.tier2_chunks)} edit target (full source)",
                f"   Tokens:  {payload_tok:,} vs {full_tok:,} full  (−{savings:.0f}% saved)",
                f"   Tip:     Call ctxzip_get_source(chunk_id) for any cx_xxxxxxxx in the directory",
                f"",
            ])

            return [TextContent(type="text", text=header + payload.user)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error retrieving context: {e}")]


    # ── ctxzip_get_source ──────────────────────────────────────────────────
    elif name == "ctxzip_get_source":
        chunk_id = arguments.get("chunk_id", "")
        chunks = _load_index()

        if not chunks:
            return [TextContent(type="text", text="No index found. Run ctxzip_index first.")]

        chunk = next((c for c in chunks if c.id == chunk_id), None)
        if not chunk:
            chunk = next((c for c in chunks if c.id.startswith(chunk_id[:8])), None)

        if not chunk:
            available = '\n'.join(f"  {c.id}  {c.signature[:50]}" for c in chunks[:10])
            return [TextContent(type="text", text=(
                f"Chunk {chunk_id!r} not found.\n"
                f"Available chunk IDs (first 10):\n{available}"
            ))]

        lines = [
            f"── Full source: {chunk.signature}",
            f"   File: {chunk.file}  L{chunk.start_line}–{chunk.end_line}",
            f"   Language: {chunk.language}  |  {chunk.raw_tokens} tokens",
            f"",
            chunk.raw,
        ]
        return [TextContent(type="text", text='\n'.join(lines))]


    # ── ctxzip_stats ───────────────────────────────────────────────────────
    elif name == "ctxzip_stats":
        chunks = _load_index()

        if not chunks:
            return [TextContent(type="text", text=(
                "No index found.\n"
                "Run ctxzip_index(path='/path/to/project') to get started."
            ))]

        total_raw = sum(c.raw_tokens for c in chunks)
        files = len(set(c.file for c in chunks))
        has_doc = sum(1 for c in chunks if c.docstring)
        has_embed = sum(1 for c in chunks if c.embedding)
        generated = sum(1 for c in chunks if c.docstring_generated)
        extracted = has_doc - generated
        tier0_tok = sum(_tokens(tier0_repr(c)) for c in chunks)
        tier1_tok = sum(_tokens(tier1_repr(c)) for c in chunks if c.docstring)

        by_lang: dict[str, int] = {}
        for c in chunks:
            by_lang[c.language] = by_lang.get(c.language, 0) + 1

        lang_lines = '\n'.join(
            f"    {lang:<15} {count} chunks"
            for lang, count in sorted(by_lang.items(), key=lambda x: -x[1])
        )

        missing = [c for c in chunks if not c.docstring]
        missing_lines = ""
        if missing:
            missing_lines = "\n  Functions missing docstrings:\n" + '\n'.join(
                f"    {c.id}  {c.signature[:55]}  [{Path(c.file).name}]"
                for c in missing[:10]
            )
            if len(missing) > 10:
                missing_lines += f"\n    ... and {len(missing)-10} more"

        lines = [
            f"── ctxzip index stats",
            f"",
            f"  Chunks:      {len(chunks)} across {files} files",
            f"  Languages:",
            lang_lines,
            f"",
            f"  Docstrings:  {has_doc}/{len(chunks)} ({has_doc/len(chunks)*100:.0f}% coverage)",
            f"    Extracted from source:  {extracted}",
            f"    AI-generated (Haiku):   {generated}",
            f"",
            f"  Embeddings:  {has_embed}/{len(chunks)} ({has_embed/len(chunks)*100:.0f}% coverage)",
            f"  Search mode: {'semantic (embeddings)' if has_embed > 0 else 'TF-IDF (set OPENAI_API_KEY for semantic)'}",
            f"",
            f"  Token breakdown:",
            f"    Full codebase:          {total_raw:,}",
            f"    Tier 0 all sigs:        {tier0_tok:,}  ({tier0_tok/total_raw*100:.0f}% of full)" if total_raw else "",
            f"    Tier 1 all docstrings:  {tier1_tok:,}  ({tier1_tok/total_raw*100:.0f}% of full)" if total_raw else "",
            missing_lines,
        ]
        return [TextContent(type="text", text='\n'.join(lines))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Entry point ────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
