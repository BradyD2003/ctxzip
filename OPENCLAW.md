# Using ctxzip with OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is an agent stack that speaks **MCP** (Model Context Protocol). ctxzip is a normal **stdio** MCP server—OpenClaw runs it as a subprocess and exposes its four tools (`ctxzip_index`, `ctxzip_query`, `ctxzip_get_source`, `ctxzip_stats`) to your agents.

This file explains how to wire ctxzip into OpenClaw’s config. Exact CLI names (`openclaw gateway restart`, etc.) may vary slightly by OpenClaw version; check your install’s docs if a command differs.

## 1. Install ctxzip on the same machine as OpenClaw

The gateway must be able to run Python and read your clone of this repo.

```bash
git clone https://github.com/BradyD2003/ctxzip.git
cd ctxzip
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` in the `ctxzip` folder and set `OPENAI_API_KEY` (recommended for semantic search). You can also pass keys in `openclaw.json` (see below).

## 2. Add ctxzip to `mcpServers`

OpenClaw typically reads **`~/.openclaw/openclaw.json`** (Windows: under your user profile in `.openclaw\openclaw.json`). Edit that file and **merge** a `ctxzip` entry under `mcpServers`.

- If you **already** have other MCP servers, add only the `"ctxzip": { ... }` block inside your existing `"mcpServers"` object—do not duplicate the outer `"mcpServers"` key twice.
- If the file is new, you can start from [openclaw.json.example](openclaw.json.example) and expand it.

Each local MCP server needs:

| Field | Purpose |
|--------|--------|
| `command` | Absolute path to the **Python interpreter** (use the **venv** inside your ctxzip clone so dependencies exist). |
| `args` | Absolute path to **`mcp_server.py`** in that clone. |
| `transport` | **`stdio`** for local process MCP (standard for ctxzip). |

### Path placeholders

Replace `/path/to/ctxzip` with the real directory where you cloned ctxzip.

**Linux / macOS**

- Python: `/path/to/ctxzip/.venv/bin/python`
- Script: `/path/to/ctxzip/mcp_server.py`

**Windows**

- Python: `C:\\path\\to\\ctxzip\\.venv\\Scripts\\python.exe`
- Script: `C:\\path\\to\\ctxzip\\mcp_server.py`

Use double backslashes in JSON, or forward slashes if your OpenClaw build accepts them on Windows.

### Optional: API keys in config

If you do **not** use a `.env` file next to `mcp_server.py`, add an `env` object so the process sees your keys:

```json
"env": {
  "OPENAI_API_KEY": "sk-...",
  "ANTHROPIC_API_KEY": ""
}
```

Never commit real keys. Prefer `.env` on disk (gitignored) or your host’s secret store.

## 3. Restart the gateway

After saving `openclaw.json`, restart OpenClaw’s gateway so it spawns the new MCP server (e.g. `openclaw gateway restart` or your platform’s equivalent).

## 4. Verify

Use your OpenClaw CLI or UI to list MCP tools (e.g. `openclaw mcp list` if available). You should see ctxzip’s tools. Then run **`ctxzip_index`** once against a project path, and use **`ctxzip_query`** for tasks—same behavior as in Cursor.

## Summary

| Step | Action |
|------|--------|
| Install | `venv` + `pip install -r requirements.txt` in the ctxzip repo |
| Configure | Add `mcpServers.ctxzip` with `command`, `args`, `transport: "stdio"` |
| Keys | `.env` beside `mcp_server.py` and/or `env` in JSON |
| Reload | Restart OpenClaw gateway |

ctxzip does not require HTTP or SSE; it is **stdio-only**, which matches OpenClaw’s usual local MCP model.
