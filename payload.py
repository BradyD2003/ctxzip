"""
payload.py — builds three-tier context payload

Licensed under BUSL-1.1 — see LICENSE in the repository (not MIT).

Structure sent to the model:
  [SYSTEM]  Instructions + tool definition for get_full_source()
  [USER]    TASK
            CODEBASE DIRECTORY (Tier 0 — all signatures)
            CONTEXT SUMMARIES  (Tier 1 — docstrings for relevant chunks)
            EDIT TARGET        (Tier 2 — full source, edit intent only)
"""

from dataclasses import dataclass
from retriever import RetrievalResult, Intent, tier0_repr, tier1_repr, tier2_repr
from chunker import _tokens


SYSTEM_PROMPT = """You are a precise code editor and analyst with access to a codebase.

You are given three tiers of context:
  TIER 0 — DIRECTORY: Every function/class in the codebase (signature + location only)
  TIER 1 — SUMMARIES: Docstring descriptions for the most relevant functions
  TIER 2 — FULL SOURCE: Complete source code for your edit target (if applicable)

You also have a tool: get_full_source(chunk_id)
  Call this when you need to inspect the full implementation of any function in the directory.
  chunk_id is the cx_xxxxxxxx identifier shown next to each entry in the directory.

Rules:
  - Only modify code shown in TIER 2 (EDIT TARGET)
  - Use TIER 0 + TIER 1 to understand the broader codebase
  - Call get_full_source() for any function you need to fully understand before editing
  - Return ONLY the corrected code for the edit target, no explanation unless asked
  - If this is a READ query, answer from TIER 0 + TIER 1 and call get_full_source() as needed"""


GET_FULL_SOURCE_TOOL = {
    "name": "get_full_source",
    "description": "Retrieve the complete source code for any function or class in the codebase by its chunk ID. Use when you need to inspect an implementation in full before editing or to answer a detailed question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_id": {
                "type": "string",
                "description": "The chunk ID (cx_xxxxxxxx) from the codebase directory"
            },
            "reason": {
                "type": "string",
                "description": "Why you need the full source"
            }
        },
        "required": ["chunk_id"]
    }
}


@dataclass
class Payload:
    system: str
    user: str
    tools: list
    intent: str
    tier2_chunk_ids: list[str]
    estimated_tokens: int
    tier0_count: int
    tier1_count: int
    tier2_count: int


def build_payload(result: RetrievalResult) -> Payload:
    lines = []

    lines.append(f"TASK: {result.query}")
    lines.append(f"INTENT: {result.intent.value.upper()}")
    lines.append(f"SEARCH: {result.search_mode}")
    lines.append("")

    lines.append("── TIER 0: CODEBASE DIRECTORY ─────────────────────────────")
    lines.append("(All functions/classes — call get_full_source(chunk_id) to inspect any)")
    lines.append("")

    t1_ids = {c.id for c in result.tier1_chunks}
    t2_ids = {c.id for c in result.tier2_chunks}

    for chunk in result.tier0_chunks:
        tier = ""
        if chunk.id in t2_ids:
            tier = "  ← EDIT TARGET"
        elif chunk.id in t1_ids:
            tier = "  ← see summary below"
        lines.append(f"  {tier0_repr(chunk)}{tier}")

    lines.append("")

    if result.tier1_chunks:
        lines.append("── TIER 1: CONTEXT SUMMARIES ───────────────────────────────")
        lines.append("(Relevant functions — descriptions to understand without full source)")
        lines.append("")
        for chunk in result.tier1_chunks:
            lines.append(tier1_repr(chunk))
            lines.append("")

    if result.tier2_chunks:
        lines.append("── TIER 2: EDIT TARGET ─────────────────────────────────────")
        lines.append("(Full source — ONLY modify this block)")
        lines.append("")
        for chunk in result.tier2_chunks:
            lines.append(f"// {chunk.id}  ·  {chunk.signature}")
            lines.append(f"// {chunk.file}  L{chunk.start_line}–{chunk.end_line}")
            lines.append("")
            lines.append(tier2_repr(chunk))
            lines.append("")

    lines.append("────────────────────────────────────────────────────────────")
    if result.tier2_chunks:
        lines.append("Return ONLY the corrected TIER 2 block. Call get_full_source() if you need more context.")
    else:
        lines.append("Answer from the summaries above. Call get_full_source() for any function you need to inspect fully.")

    user_msg = '\n'.join(lines)

    return Payload(
        system=SYSTEM_PROMPT,
        user=user_msg,
        tools=[GET_FULL_SOURCE_TOOL],
        intent=result.intent.value,
        tier2_chunk_ids=[c.id for c in result.tier2_chunks],
        estimated_tokens=_tokens(SYSTEM_PROMPT) + _tokens(user_msg),
        tier0_count=len(result.tier0_chunks),
        tier1_count=len(result.tier1_chunks),
        tier2_count=len(result.tier2_chunks),
    )


def build_anthropic_messages(payload: Payload) -> dict:
    return {
        "system": [
            {
                "type": "text",
                "text": payload.system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": payload.tools,
        "messages": [
            {"role": "user", "content": payload.user}
        ],
    }
