# ctxzip MCP Setup

For install, tools overview, and license, see **[README.md](README.md)**.

## Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install mcp anthropic openai numpy rich
```

- `mcp` — MCP protocol (required)
- `anthropic` — docstring generation via Haiku (optional but recommended)
- `openai` — semantic search embeddings (optional, falls back to TF-IDF)
- `numpy` — fast cosine similarity (optional, falls back to pure Python)
- `rich` — pretty CLI output (optional)

Optional: set **`CTXZIP_INDEX`** to an absolute path for the index JSON (defaults: next to `mcp_server.py` for MCP; current working directory for CLI).

## Cursor

File: `.cursor/mcp.json` (in your project root or home dir)

Prefer a **venv interpreter** so dependencies do not clash with other projects:

```json
{
  "mcpServers": {
    "ctxzip": {
      "command": "/absolute/path/to/ctxzip/.venv/bin/python",
      "args": ["/absolute/path/to/ctxzip/mcp_server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-YOUR_KEY_HERE",
        "OPENAI_API_KEY": "sk-YOUR_KEY_HERE"
      }
    }
  }
}
```

## Claude Desktop

macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ctxzip": {
      "command": "python",
      "args": ["/absolute/path/to/ctxzip/mcp_server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-YOUR_KEY_HERE",
        "OPENAI_API_KEY": "sk-YOUR_KEY_HERE"
      }
    }
  }
}
```

## Claude Code (terminal)

```bash
claude mcp add ctxzip -- python /absolute/path/to/ctxzip/mcp_server.py
export ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
export OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

## Tools

Once registered, the model can call these tools:

| Tool | What it does |
|------|-------------|
| `ctxzip_index(path="/path/to/project")` | Index all code, generate docstrings + embeddings |
| `ctxzip_query(task="fix the seam artifact in stitch_tiles")` | Three-tier context: directory + summaries + edit target |
| `ctxzip_get_source(chunk_id="cx_68be5bb6f8")` | Pull full raw source for any function by ID |
| `ctxzip_stats()` | Show coverage, token breakdown, embedding status |

## Search modes

- **Semantic** (with `OPENAI_API_KEY`): Embeddings computed at index time via `text-embedding-3-small`. Queries are embedded and matched by cosine similarity. Finds conceptually related code even without exact keyword matches.
- **TF-IDF** (without `OPENAI_API_KEY`): Term frequency–inverse document frequency with cosine similarity. Solid keyword matching without any external API. Function name mentions still get a strong boost.

Both modes include a name-matching heuristic that boosts chunks when the query directly mentions a function or class name.

## Tips

1. Index once per project, re-index when files change (stale chunks are automatically replaced)
2. The index persists at `.ctxzip_index.json` (or `CTXZIP_INDEX`) between sessions
3. `ANTHROPIC_API_KEY` is needed for docstring generation; `OPENAI_API_KEY` for semantic search
4. Both keys are optional — the tool degrades gracefully without them
5. Works with any MCP-compatible client
6. Copy `.env.example` to `.env` beside `mcp_server.py` to load keys without putting them in `mcp.json`
