"""
ctxzip.py — three-tier context compression CLI

Usage:
  python ctxzip.py index <path>       Index a file or directory
  python ctxzip.py query "<task>"     Dry run — show payload without API call
  python ctxzip.py ask "<task>"       Send to Claude with full tool loop
  python ctxzip.py stats              Show index stats

Examples:
  python ctxzip.py index ./my_project
  python ctxzip.py query "fix the seam artifact in stitch_tiles"
  python ctxzip.py ask "how does the token refresh work"
"""

import sys
import json
import os
from pathlib import Path

# Load .env file from the same directory
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            if value and not os.environ.get(key.strip()):
                os.environ[key.strip()] = value

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from chunker import chunk_file, chunk_directory, Chunk, _tokens
from docstrings import extract_docstring, generate_docstring, docstring_tokens
from retriever import (
    Retriever, RetrievalResult, tier0_repr, tier1_repr,
    embed_texts, _chunk_embed_text,
)
from payload import build_payload, build_anthropic_messages

console = Console() if HAS_RICH else None


def _cli_index_path() -> Path:
    raw = os.environ.get("CTXZIP_INDEX")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / ".ctxzip_index.json").resolve()


# ── Index persistence ──────────────────────────────────────────────────────

def save_index(chunks: list[Chunk]):
    data = [c.to_dict() for c in chunks]
    _cli_index_path().write_text(json.dumps(data, indent=2))


def load_index() -> list[Chunk]:
    path = _cli_index_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [Chunk.from_dict(d) for d in data]


def merge_index(existing: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    new_files = {c.file for c in new}
    kept = [c for c in existing if c.file not in new_files]
    return kept + new


# ── Docstring enrichment ───────────────────────────────────────────────────

def enrich_with_docstrings(chunks: list[Chunk], force_generate: bool = False) -> tuple[list[Chunk], int, int]:
    extracted = 0
    generated = 0
    needs_generation = []

    for chunk in chunks:
        if chunk.docstring and not force_generate:
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
        if not api_key:
            if HAS_RICH:
                console.print(f"[yellow]⚠ No ANTHROPIC_API_KEY — skipping generation for {len(needs_generation)} chunks[/yellow]")
            else:
                print(f"Warning: No ANTHROPIC_API_KEY — {len(needs_generation)} chunks have no docstring")
        else:
            if HAS_RICH:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Generating docstrings for {len(needs_generation)} functions...",
                        total=len(needs_generation)
                    )
                    for chunk in needs_generation:
                        doc = generate_docstring(chunk.raw, chunk.signature, chunk.language)
                        chunk.docstring = doc
                        chunk.docstring_generated = True
                        generated += 1
                        progress.advance(task)
            else:
                for i, chunk in enumerate(needs_generation):
                    print(f"Generating docstring {i+1}/{len(needs_generation)}: {chunk.signature[:40]}")
                    doc = generate_docstring(chunk.raw, chunk.signature, chunk.language)
                    chunk.docstring = doc
                    chunk.docstring_generated = True
                    generated += 1

    return chunks, extracted, generated


# ── Embedding generation ───────────────────────────────────────────────────

def generate_embeddings(chunks: list[Chunk]) -> int:
    needs = [c for c in chunks if not c.embedding]
    if not needs:
        return 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        if HAS_RICH:
            console.print(f"[yellow]⚠ No OPENAI_API_KEY — skipping embeddings for {len(needs)} chunks (TF-IDF fallback active)[/yellow]")
        else:
            print(f"Warning: No OPENAI_API_KEY — {len(needs)} chunks without embeddings")
        return 0

    try:
        texts = [_chunk_embed_text(c) for c in needs]
        if HAS_RICH:
            console.print(f"[dim]Computing embeddings for {len(needs)} chunks...[/dim]")
        vectors = embed_texts(texts, api_key)
        embedded = 0
        for chunk, vec in zip(needs, vectors):
            if vec:
                chunk.embedding = [round(x, 5) for x in vec]
                embedded += 1
        return embedded
    except ImportError:
        if HAS_RICH:
            console.print("[yellow]⚠ openai package not installed — pip install openai[/yellow]")
        return 0
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]Embedding error: {e}[/red]")
        return 0


# ── Display ────────────────────────────────────────────────────────────────

def print_query_result(result: RetrievalResult, payload_text: str, payload):
    if not HAS_RICH:
        print(payload_text)
        return

    intent_color = "red" if result.intent.value == "edit" else "blue"
    console.print(f"\n[bold {intent_color}]Intent: {result.intent.value.upper()}[/]  Search: {result.search_mode}")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0,2))
    table.add_column("Tier", style="dim")
    table.add_column("Chunks", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Format", style="dim")

    t0_tok = result.tier0_tokens()
    t1_tok = result.tier1_tokens()
    t2_tok = result.tier2_tokens()
    full_tok = result.full_codebase_tokens()
    total_tok = result.total_payload_tokens()
    savings = result.savings_pct()

    table.add_row("0 · Directory (all signatures)", str(len(result.tier0_chunks)), str(t0_tok), "sig + location")
    table.add_row("1 · Context summaries", str(len(result.tier1_chunks)), str(t1_tok), "sig + docstring")
    table.add_row("2 · Edit target", str(len(result.tier2_chunks)), str(t2_tok), "full raw source")
    table.add_row("─" * 25, "", "─────", "")
    table.add_row(
        "[bold]Total payload[/bold]",
        "",
        f"[bold]{total_tok}[/bold]",
        f"[bold green]vs {full_tok} full (−{savings:.0f}%)[/bold green]"
    )
    console.print(table)

    console.print("\n[bold]Payload preview:[/bold]")
    console.print(Panel(
        payload_text[:3000] + ("..." if len(payload_text) > 3000 else ""),
        border_style="dim"
    ))


def print_stats(chunks: list[Chunk]):
    if not chunks:
        print("No chunks indexed. Run: python ctxzip.py index <path>")
        return

    total_raw = sum(c.raw_tokens for c in chunks)
    files = len(set(c.file for c in chunks))
    has_doc = sum(1 for c in chunks if c.docstring)
    has_embed = sum(1 for c in chunks if c.embedding)
    generated = sum(1 for c in chunks if c.docstring_generated)
    extracted = has_doc - generated

    tier0_total = sum(_tokens(tier0_repr(c)) for c in chunks)
    tier1_total = sum(_tokens(tier1_repr(c)) for c in chunks if c.docstring)

    if HAS_RICH:
        table = Table(title="ctxzip index", show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        table.add_row("Chunks indexed", str(len(chunks)))
        table.add_row("Files", str(files))
        table.add_row("", "")
        table.add_row("Docstrings extracted", str(extracted))
        table.add_row("Docstrings AI-generated", str(generated))
        table.add_row("Missing docstrings", str(len(chunks) - has_doc))
        table.add_row("", "")
        table.add_row("Embeddings", f"{has_embed}/{len(chunks)} ({'semantic' if has_embed else 'TF-IDF'})")
        table.add_row("", "")
        table.add_row("Full codebase tokens", f"{total_raw:,}")
        table.add_row("Tier 0 (all sigs)", f"{tier0_total:,}  ({tier0_total/total_raw*100:.0f}% of full)" if total_raw else "0")
        table.add_row("Tier 1 (all docstrings)", f"{tier1_total:,}  ({tier1_total/total_raw*100:.0f}% of full)" if total_raw else "0")
        console.print(table)

        t2 = Table(show_header=True, header_style="bold", box=None)
        t2.add_column("Function", no_wrap=True)
        t2.add_column("Lang", style="dim")
        t2.add_column("Raw", justify="right")
        t2.add_column("T0", justify="right")
        t2.add_column("T1", justify="right")
        t2.add_column("Doc?", justify="center")
        t2.add_column("Emb?", justify="center")
        for c in sorted(chunks, key=lambda x: x.raw_tokens, reverse=True):
            t0 = _tokens(tier0_repr(c))
            t1 = _tokens(tier1_repr(c)) if c.docstring else 0
            has = "[green]✓[/green]" if c.docstring else "[red]✗[/red]"
            gen = " [dim](gen)[/dim]" if c.docstring_generated else ""
            emb = "[green]✓[/green]" if c.embedding else "[red]✗[/red]"
            t2.add_row(c.signature[:45], c.language, str(c.raw_tokens), str(t0), str(t1), has + gen, emb)
        console.print(t2)
    else:
        print(f"Chunks: {len(chunks)} | Files: {files} | Raw: {total_raw:,} | Tier0: {tier0_total:,} | Docs: {has_doc}/{len(chunks)} | Embeds: {has_embed}/{len(chunks)}")


# ── Claude call with tool loop ─────────────────────────────────────────────

def call_claude_with_tools(payload_dict: dict, retriever: Retriever,
                            model: str = "claude-haiku-4-5-20251001") -> str:
    try:
        import anthropic
    except ImportError:
        return "Error: anthropic package not installed"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set"

    client = anthropic.Anthropic(api_key=api_key)
    messages = list(payload_dict["messages"])

    if HAS_RICH:
        console.print(f"\n[dim]→ Sending to {model}...[/dim]")

    for turn in range(5):
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=payload_dict["system"],
            tools=payload_dict["tools"],
            messages=messages,
        )

        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            text_blocks = [b.text for b in resp.content if b.type == "text"]
            return '\n'.join(text_blocks)

        tool_results = []
        for tool_use in tool_uses:
            if tool_use.name == "get_full_source":
                chunk_id = tool_use.input.get("chunk_id", "")
                reason = tool_use.input.get("reason", "")
                source = retriever.get_full_source(chunk_id)

                if HAS_RICH:
                    console.print(f"  [dim]→ get_full_source({chunk_id})  {reason}[/dim]")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": source if source else f"Chunk {chunk_id} not found in index"
                })

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_results})

    return "Max tool call rounds reached"


# ── CLI commands ───────────────────────────────────────────────────────────

def cmd_index(path: str):
    p = Path(path)
    if not p.exists():
        print(f"Path not found: {path}")
        sys.exit(1)

    if HAS_RICH:
        console.print(f"[bold]Indexing[/bold] {path}...")

    chunks = chunk_file(p) if p.is_file() else chunk_directory(p)
    chunks, extracted, generated = enrich_with_docstrings(chunks)
    embedded = generate_embeddings(chunks)

    existing = load_index()
    merged = merge_index(existing, chunks)
    generate_embeddings(merged)
    save_index(merged)

    added = len(merged) - len(existing)
    stale = sum(1 for c in existing if c.file in {ch.file for ch in chunks})
    if HAS_RICH:
        console.print(f"[green]✓[/green] {len(chunks)} chunks indexed | {stale} stale replaced | {extracted} docs extracted | {generated} docs generated | {embedded} embedded → {_cli_index_path()}")
    else:
        print(f"Indexed {len(chunks)} chunks | {stale} stale replaced | {extracted} extracted | {generated} generated | {embedded} embedded")

    print_stats(merged)


def cmd_query(query: str, send: bool = False):
    chunks = load_index()
    if not chunks:
        print("No index found. Run: python ctxzip.py index <path>")
        sys.exit(1)

    retriever = Retriever(chunks, top_k_tier1=4)
    result = retriever.retrieve(query)
    payload = build_payload(result)
    anthropic_dict = build_anthropic_messages(payload)

    print_query_result(result, payload.user, payload)

    if send:
        if HAS_RICH:
            console.print("\n[bold]Claude response:[/bold]")
        response = call_claude_with_tools(anthropic_dict, retriever)
        if HAS_RICH:
            console.print(Panel(
                Syntax(response, "python", theme="monokai") if '```' not in response else response,
                title="Response",
                border_style="green"
            ))
        else:
            print("\n--- RESPONSE ---")
            print(response)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    if cmd == "index" and len(args) >= 2:
        cmd_index(args[1])
    elif cmd == "query" and len(args) >= 2:
        cmd_query(' '.join(args[1:]), send=False)
    elif cmd == "ask" and len(args) >= 2:
        cmd_query(' '.join(args[1:]), send=True)
    elif cmd == "stats":
        print_stats(load_index())
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
