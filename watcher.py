"""
ctxzip file watcher — re-index changed source files into .ctxzip_index.json

Run alongside your editor (separate terminal):
  pip install -r requirements.txt
  python watcher.py c:\\path\\to\\project

Uses the same chunking, docstrings, embeddings, and merge rules as ctxzip_index.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Same directory as mcp_server / chunker
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_env = _ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            if value and not os.environ.get(key.strip()):
                os.environ[key.strip()] = value

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from chunker import EXTENSION_MAP, SKIP_DIRS, chunk_file
from docstrings import enrich_with_docstrings
from mcp_server import _generate_embeddings, _load_index, _save_index

SUPPORTED_SUFFIXES = {ext.lower() for ext in EXTENSION_MAP}


def _norm_file(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path


def _should_watch(path: Path) -> bool:
    if not path.is_file() and path.exists():
        return False
    try:
        rp = path.resolve()
    except OSError:
        return False
    if any(skip in rp.parts for skip in SKIP_DIRS):
        return False
    return rp.suffix.lower() in SUPPORTED_SUFFIXES


def _remove_chunks_for_file(path: str) -> int:
    key = _norm_file(path)
    existing = _load_index()
    kept = [c for c in existing if _norm_file(c.file) != key]
    removed = len(existing) - len(kept)
    if removed:
        _save_index(kept)
    return removed


def _reindex_file(filepath: str) -> None:
    fp = Path(filepath)
    try:
        rp = fp.resolve()
    except OSError:
        return

    if not rp.is_file():
        return
    if not _should_watch(rp):
        return

    print(f"[ctxzip] Re-indexing: {rp}", flush=True)
    try:
        new_chunks = chunk_file(rp)
        new_chunks, extracted, generated = enrich_with_docstrings(new_chunks)
        existing = _load_index()
        key = _norm_file(str(rp))
        kept = [c for c in existing if _norm_file(c.file) != key]
        merged = kept + new_chunks
        embedded = _generate_embeddings(merged)
        _save_index(merged)
        print(
            f"  {len(new_chunks)} chunks ({extracted} extracted, {generated} generated), "
            f"embeddings +{embedded}, total index {len(merged)}",
            flush=True,
        )
    except Exception as e:
        print(f"  Error re-indexing {rp}: {e}", flush=True)


class CodeWatcher(FileSystemEventHandler):
    def __init__(self) -> None:
        self.pending: set[str] = set()

    def _queue(self, path: str) -> None:
        p = Path(path)
        if not _should_watch(p):
            return
        self.pending.add(_norm_file(str(p)))

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        self._queue(event.src_path)

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self._queue(event.src_path)

    def on_moved(self, event) -> None:
        if getattr(event, "is_directory", False):
            return
        old = getattr(event, "src_path", None)
        dest = getattr(event, "dest_path", None)
        if old:
            removed = _remove_chunks_for_file(old)
            if removed:
                print(f"[ctxzip] Removed {removed} chunks (moved away): {old}", flush=True)
        if dest:
            self._queue(dest)

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        removed = _remove_chunks_for_file(event.src_path)
        if removed:
            print(f"[ctxzip] Removed {removed} chunks (deleted): {event.src_path}", flush=True)


def watch(project_path: str | Path, interval: float = 2.0) -> None:
    project_path = Path(project_path).resolve()
    if not project_path.is_dir():
        print(f"Not a directory: {project_path}", file=sys.stderr)
        sys.exit(1)

    handler = CodeWatcher()
    observer = Observer()
    observer.schedule(handler, str(project_path), recursive=True)
    observer.start()
    print(f"[ctxzip] Watching {project_path} (debounce {interval}s)...", flush=True)
    print("[ctxzip] Ctrl+C to stop.", flush=True)

    try:
        while True:
            time.sleep(interval)
            if not handler.pending:
                continue
            batch = list(handler.pending)
            handler.pending.clear()
            for fp in batch:
                p = Path(fp)
                if p.is_file():
                    _reindex_file(fp)
    except KeyboardInterrupt:
        print("\n[ctxzip] Stopping watcher.", flush=True)
        observer.stop()
    observer.join()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Watch a project and update ctxzip index on save.")
    ap.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project root to watch (default: current directory)",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Debounce interval in seconds (default: 2)",
    )
    args = ap.parse_args()
    watch(args.path, interval=args.interval)


if __name__ == "__main__":
    main()
