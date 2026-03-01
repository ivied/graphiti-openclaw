#!/usr/bin/env python3
"""CLI for graphiti-openclaw — add episodes and search the context graph.

Usage:
    graphiti-oc add "Some context or decision"
    graphiti-oc add --file memory/2026-03-01.md
    graphiti-oc search "What decisions were made?"
    graphiti-oc status
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_config():
    """Load config from environment or openclaw.json."""
    config = {
        "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ.get("NEO4J_PASSWORD", ""),
        "llm_api_key": os.environ.get("LLM_API_KEY", ""),
        "llm_base_url": os.environ.get("LLM_BASE_URL", ""),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o"),
        "llm_small_model": os.environ.get("LLM_SMALL_MODEL", "gpt-4o-mini"),
        "embedder_model": os.environ.get("EMBEDDER_MODEL", "all-MiniLM-L6-v2"),
        "group_id": os.environ.get("GRAPHITI_GROUP", "default"),
    }

    # Try loading from openclaw.json
    oc_path = Path.home() / ".openclaw" / "openclaw.json"
    if oc_path.exists():
        try:
            with open(oc_path) as f:
                oc = json.load(f)
            if not config["llm_api_key"]:
                provider = oc.get("models", {}).get("providers", {}).get("9router", {})
                if provider:
                    config["llm_api_key"] = provider.get("apiKey", "")
                    config["llm_base_url"] = provider.get("baseUrl", "")
                    config["llm_model"] = "cc/claude-sonnet-4-5-20250929"
                    config["llm_small_model"] = "cc/claude-haiku-4-5-20251001"
        except Exception:
            pass

    # Try loading Neo4j config from graphiti config file
    graphiti_config = Path.home() / "projects" / "graphiti-mcp" / "mcp_server" / "config" / "config-openclaw.yaml"
    if graphiti_config.exists() and not config["neo4j_password"]:
        try:
            import yaml
            with open(graphiti_config) as f:
                gc = yaml.safe_load(f)
            db = gc.get("database", {}).get("providers", {}).get("neo4j", {})
            if db:
                config["neo4j_uri"] = db.get("uri", config["neo4j_uri"])
                config["neo4j_user"] = db.get("username", config["neo4j_user"])
                config["neo4j_password"] = db.get("password", config["neo4j_password"])
        except ImportError:
            # No PyYAML — try simple regex parsing
            try:
                text = graphiti_config.read_text()
                import re
                pw = re.search(r'password:\s*["\']?(\S+)["\']?', text)
                if pw:
                    config["neo4j_password"] = pw.group(1).strip('"\'')
                uri = re.search(r'uri:\s*["\']?(\S+)["\']?', text)
                if uri:
                    config["neo4j_uri"] = uri.group(1).strip('"\'')
                user = re.search(r'username:\s*["\']?(\S+)["\']?', text)
                if user:
                    config["neo4j_user"] = user.group(1).strip('"\'')
            except Exception:
                pass

    return config


async def create_client(config):
    """Create and initialize a Graphiti client."""
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import LLMConfig
    from graphiti_openclaw.compat_client import CompatOpenAIClient

    llm_config = LLMConfig(
        api_key=config["llm_api_key"],
        base_url=config["llm_base_url"],
        model=config["llm_model"],
        small_model=config["llm_small_model"],
    )
    llm_client = CompatOpenAIClient(config=llm_config)

    # Try local embedder, fall back to None (Graphiti default)
    embedder = None
    try:
        from graphiti_openclaw.local_embedder import LocalEmbedder
        embedder = LocalEmbedder(model_name=config["embedder_model"])
    except ImportError:
        pass

    client = Graphiti(
        config["neo4j_uri"],
        config["neo4j_user"],
        config["neo4j_password"],
        llm_client=llm_client,
        embedder=embedder,
    )
    await client.build_indices_and_constraints()
    return client


async def cmd_add(args, config):
    """Add an episode to the graph."""
    from graphiti_core.nodes import EpisodeType

    if args.file:
        content = Path(args.file).read_text()
        name = Path(args.file).stem
        source_desc = f"File: {args.file}"
    else:
        content = " ".join(args.text)
        name = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        source_desc = "CLI input"

    client = await create_client(config)
    try:
        await client.add_episode(
            name=name,
            episode_body=content,
            source_description=source_desc,
            reference_time=datetime.now(timezone.utc),
            source=EpisodeType.text,
        )
        print(f"✅ Added episode: {name} ({len(content)} chars)")
    finally:
        await client.close()


async def cmd_search(args, config):
    """Search the context graph."""
    query = " ".join(args.query)
    client = await create_client(config)
    try:
        results = await client.search(query)
        if not results:
            print("No results found.")
            return

        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results[:args.limit], 1):
            fact = r.fact if hasattr(r, "fact") else str(r)
            print(f"  {i}. {fact}")
        print()
    finally:
        await client.close()


async def cmd_status(args, config):
    """Show graph status."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config["neo4j_uri"],
        auth=(config["neo4j_user"], config["neo4j_password"]),
    )
    try:
        with driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) as cnt").single()["cnt"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) as cnt").single()["cnt"]
            episodes = session.run(
                "MATCH (n:Episodic) RETURN count(n) as cnt"
            ).single()["cnt"]
            entities = session.run(
                "MATCH (n:Entity) RETURN count(n) as cnt"
            ).single()["cnt"]

        print(f"📊 Context Graph Status")
        print(f"   Neo4j: {config['neo4j_uri']}")
        print(f"   Total nodes: {nodes}")
        print(f"   Total relationships: {rels}")
        print(f"   Episodes: {episodes}")
        print(f"   Entities: {entities}")
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(
        prog="graphiti-oc",
        description="CLI for graphiti-openclaw context graph",
    )
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add an episode")
    p_add.add_argument("text", nargs="*", help="Text to add as episode")
    p_add.add_argument("--file", "-f", help="File to add as episode")

    # search
    p_search = sub.add_parser("search", help="Search the graph")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--limit", "-n", type=int, default=5, help="Max results")

    # status
    sub.add_parser("status", help="Show graph status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = get_config()

    if args.command == "add":
        if not args.text and not args.file:
            print("Error: provide text or --file")
            sys.exit(1)
        asyncio.run(cmd_add(args, config))
    elif args.command == "search":
        asyncio.run(cmd_search(args, config))
    elif args.command == "status":
        asyncio.run(cmd_status(args, config))


if __name__ == "__main__":
    main()
