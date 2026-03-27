# ctxzip — MCP codebase context (semantic search)

**ctxzip** is **open source** ([MIT License](LICENSE)): a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that indexes a project’s source files, builds a compressed “directory” of all symbols, and retrieves **semantic** (embedding) or **TF‑IDF** context for coding tasks—so your agent gets relevant code without stuffing the full repo into the prompt.

Use it from any public GitHub clone: install dependencies, point your MCP client at `mcp_server.py`, and index your repo. Issues and pull requests are welcome.

## Features

- **Four tools:** `ctxzip_index`, `ctxzip_query`, `ctxzip_get_source`, `ctxzip_stats`
- **Tiered context:** signatures for everything (Tier 0), docstring summaries for top matches (Tier 1), full source for an edit target when intent is “edit” (Tier 2)
- **Semantic search** when `OPENAI_API_KEY` is set (`text-embedding-3-small`); otherwise **TF‑IDF** fallback
- **Optional** Haiku docstring generation with `ANTHROPIC_API_KEY`
- **Languages:** Python, JS/TS, Go, Rust, Java, C/C++, Ruby, PHP, Swift, Kotlin, Scala (plus line-based fallback)
- **Persistent index:** `.ctxzip_index.json` in this directory (or set `CTXZIP_INDEX`; see below)

## Requirements

- Python **3.11+**
- Network access for OpenAI (if using embeddings) / Anthropic (if using generated docstrings)

## Quick start (local / Cursor / Claude Desktop)

1. Clone this repository (or copy the `ctxzip` folder into your own project).

2. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate      # macOS / Linux
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add at least `OPENAI_API_KEY` for semantic search.

4. Register the server (stdio transport).

### Cursor

**Project** config: `.cursor/mcp.json` next to your app:

```json
{
  "mcpServers": {
    "ctxzip": {
      "command": "/absolute/path/to/ctxzip/.venv/bin/python",
      "args": ["/absolute/path/to/ctxzip/mcp_server.py"]
    }
  }
}
```

On Windows, use `\\.venv\\Scripts\\python.exe` and `mcp_server.py` paths.

If you use `.env` beside `mcp_server.py`, you do **not** need to duplicate keys in `mcp.json`.

### Claude Desktop

`claude_desktop_config.json` (see [Claude docs](https://docs.anthropic.com)) — same `command` / `args` pattern as above. Optional `env` block for API keys.

## Tools (for agents)

| Tool | Purpose |
|------|---------|
| `ctxzip_index` | Index a file or directory; refresh embeddings; merge/replace stale chunks |
| `ctxzip_query` | Given a natural-language task, return Tier 0 + Tier 1 (+ Tier 2 for edits) |
| `ctxzip_get_source` | Fetch full source for a chunk id (`cx_…`) from the directory |
| `ctxzip_stats` | Chunks, languages, embedding/doc coverage, token breakdown |

Typical flow: **`ctxzip_index`** once per project (or after large changes), then **`ctxzip_query`** for each task.

## Environment variables

| Variable | Role |
|----------|------|
| `OPENAI_API_KEY` | Semantic embeddings (recommended) |
| `ANTHROPIC_API_KEY` | Generate missing docstrings at index time (optional) |
| `CTXZIP_INDEX` | Optional absolute path to the index JSON file (default: `.ctxzip_index.json` next to `mcp_server.py`) |

Indexing sends code excerpts to OpenAI when embeddings are enabled. Use only on code you are allowed to process.

## CLI (optional)

Same folder, same venv:

```bash
python ctxzip.py index /path/to/project
python ctxzip.py query "how does authentication work?"
python ctxzip.py stats
```

## License

Released under the [MIT License](LICENSE).

## More detail

See [MCP_SETUP.md](MCP_SETUP.md) for duplicate examples and search-mode notes.
