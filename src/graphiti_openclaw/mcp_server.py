#!/usr/bin/env python3
"""Graphiti-OpenClaw MCP Server — lightweight MCP wrapper around graphiti-openclaw.

Exposes context graph operations as MCP tools:
- cg_search: Search the context graph
- cg_status: Get graph statistics
- cg_ingest: Ingest a file into the graph
- cg_ingest_today: Ingest today's daily note (with dedup)
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── Bootstrap ────────────────────────────────────────────────────────────────
# Add graphiti-openclaw to path if running standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_SITE = PROJECT_ROOT / ".venv" / "lib"
for p in VENV_SITE.glob("python*/site-packages"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Lazy init ────────────────────────────────────────────────────────────────
_graphiti = None
_embedder = None

WORKSPACE = os.environ.get("CG_WORKSPACE", os.path.expanduser("~/.openclaw/workspace"))
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
STATE_FILE = os.path.join(WORKSPACE, "secrets", "cg_ingest_state.json")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "testpassword123")

LLM_MODEL = os.environ.get("CG_LLM_MODEL", "cc/claude-sonnet-4-5-20250929")
SMALL_MODEL = os.environ.get("CG_SMALL_MODEL", "cc/claude-haiku-4-5-20251001")
NROUTER_BASE = os.environ.get("NROUTER_BASE_URL", "")
NROUTER_KEY = os.environ.get("NROUTER_API_KEY", "")

# Auto-detect 9router credentials from OpenClaw config if not set via env
if not NROUTER_BASE or not NROUTER_KEY:
    try:
        import json as _json
        _oc_path = Path.home() / ".openclaw" / "openclaw.json"
        if _oc_path.exists():
            _oc = _json.loads(_oc_path.read_text())
            _9r = _oc.get("models", {}).get("providers", {}).get("9router", {})
            if not NROUTER_BASE:
                NROUTER_BASE = _9r.get("baseUrl", _9r.get("baseURL", ""))
            if not NROUTER_KEY:
                NROUTER_KEY = _9r.get("apiKey", "")
    except Exception:
        pass

GROUP_ID = os.environ.get("CG_GROUP_ID", "openclaw")


async def _get_graphiti():
    """Lazy-init Graphiti client."""
    global _graphiti, _embedder
    if _graphiti is not None:
        return _graphiti

    from graphiti_core import Graphiti
    from graphiti_core.llm_client import LLMConfig
    from graphiti_openclaw.compat_client import CompatOpenAIClient

    llm_config = LLMConfig(
        api_key=NROUTER_KEY,
        base_url=NROUTER_BASE,
        model=LLM_MODEL,
        small_model=SMALL_MODEL,
    )
    llm_client = CompatOpenAIClient(config=llm_config)

    # Try local embedder, fall back to None
    _embedder = None
    try:
        from graphiti_openclaw.local_embedder import LocalEmbedder
        _embedder = LocalEmbedder()
    except ImportError:
        pass

    _graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASS,
        llm_client=llm_client,
        embedder=_embedder,
    )
    return _graphiti


def _load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── MCP Server ───────────────────────────────────────────────────────────────
app = Server("graphiti-openclaw")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="cg_search",
            description="Search the context graph for entities, facts, and relations matching a query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (semantic search over the knowledge graph)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="cg_status",
            description="Get context graph statistics: node count, relationships, episodes, entities.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="cg_ingest",
            description="Ingest a text file into the context graph as an episode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to ingest",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="cg_ingest_today",
            description="Ingest today's daily note (memory/YYYY-MM-DD.md) into the context graph. Skips if already ingested (dedup by hash).",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "cg_search":
            return await _handle_search(arguments)
        elif name == "cg_status":
            return await _handle_status()
        elif name == "cg_ingest":
            return await _handle_ingest(arguments)
        elif name == "cg_ingest_today":
            return await _handle_ingest_today()
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def _handle_search(args: dict) -> list[TextContent]:
    query = args["query"]
    limit = args.get("limit", 10)

    g = await _get_graphiti()
    results = await g.search(query, num_results=limit)

    if not results:
        return [TextContent(type="text", text="No results found.")]

    lines = []
    for i, edge in enumerate(results, 1):
        fact = edge.fact if hasattr(edge, "fact") else str(edge)
        lines.append(f"{i}. {fact}")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_status() -> list[TextContent]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        episodes = session.run(
            "MATCH (n:Episodic) RETURN count(n) AS c"
        ).single()["c"]
        entities = session.run(
            "MATCH (n) WHERE NOT n:Episodic RETURN count(n) AS c"
        ).single()["c"]
    driver.close()

    status = {
        "neo4j": NEO4J_URI,
        "total_nodes": nodes,
        "total_relationships": rels,
        "episodes": episodes,
        "entities": entities,
    }
    return [TextContent(type="text", text=json.dumps(status, indent=2))]


async def _handle_ingest(args: dict) -> list[TextContent]:
    file_path = args["file_path"]
    path = Path(file_path).expanduser()
    if not path.exists():
        return [TextContent(type="text", text=f"File not found: {file_path}")]

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return [TextContent(type="text", text=f"File is empty: {file_path}")]

    g = await _get_graphiti()

    # Chunk if large (>8000 chars)
    MAX_CHUNK = 8000
    chunks = []
    if len(content) > MAX_CHUNK:
        lines = content.split("\n")
        chunk = []
        chunk_len = 0
        for line in lines:
            if chunk_len + len(line) > MAX_CHUNK and chunk:
                chunks.append("\n".join(chunk))
                chunk = [line]
                chunk_len = len(line)
            else:
                chunk.append(line)
                chunk_len += len(line) + 1
        if chunk:
            chunks.append("\n".join(chunk))
    else:
        chunks = [content]

    from graphiti_core.nodes import EpisodeType

    ref_time = datetime.now()

    for i, chunk in enumerate(chunks):
        source_desc = f"{path.name}" + (f" (part {i+1}/{len(chunks)})" if len(chunks) > 1 else "")
        await g.add_episode(
            name=source_desc,
            episode_body=chunk,
            source=EpisodeType.text,
            source_description=f"Daily note: {path.name}",
            group_id=GROUP_ID,
            reference_time=ref_time,
        )

    return [TextContent(
        type="text",
        text=f"Ingested {path.name}: {len(chunks)} chunk(s), {len(content)} chars",
    )]


async def _handle_ingest_today() -> list[TextContent]:
    import hashlib

    today = datetime.now().strftime("%Y-%m-%d")
    daily_path = Path(MEMORY_DIR) / f"{today}.md"

    if not daily_path.exists():
        return [TextContent(type="text", text=f"No daily note for {today}")]

    content = daily_path.read_text(encoding="utf-8")
    if not content.strip():
        return [TextContent(type="text", text=f"Daily note {today} is empty")]

    content_hash = hashlib.sha256(content.encode()).hexdigest()
    state = _load_state()

    if state.get(today) == content_hash:
        return [TextContent(type="text", text=f"Daily note {today} already ingested (hash match)")]

    # Ingest
    result = await _handle_ingest({"file_path": str(daily_path)})

    # Update state
    state[today] = content_hash
    _save_state(state)

    return [TextContent(
        type="text",
        text=f"Ingested daily note {today} ({len(content)} chars). State updated.",
    )]


# ── Entry point ──────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
