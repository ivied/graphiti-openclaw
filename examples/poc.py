#!/usr/bin/env python3
"""Graphiti PoC — test basic entity/relation operations with Neo4j.

Uses 9router (OpenAI-compatible proxy for Claude models) as LLM backend.
"""

import asyncio
import os
import json
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.nodes import EpisodeType
from compat_client import CompatOpenAIClient
from local_embedder import LocalEmbedder


def get_llm_config():
    """Load LLM config from openclaw.json — use 9router (Claude via OpenAI-compat API)."""
    with open(os.path.expanduser("~/.openclaw/openclaw.json")) as f:
        cfg = json.load(f)
    
    provider = cfg["models"]["providers"]["9router"]
    api_key = provider["apiKey"]
    base_url = provider["baseUrl"]  # http://ec2-...:20128/v1
    
    # Use haiku for cost efficiency (entity extraction is high-volume)
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model="cc/claude-sonnet-4-5-20250929",       # main model
        small_model="cc/claude-haiku-4-5-20251001",   # small model for simple tasks
    )


async def main():
    print("=== Graphiti PoC ===\n")
    
    llm_config = get_llm_config()
    print(f"LLM: {llm_config.model} (small: {llm_config.small_model})")
    print(f"Base URL: {llm_config.base_url}")
    
    # Create LLM client (compat wrapper for non-OpenAI providers)
    llm_client = CompatOpenAIClient(config=llm_config)
    
    # Create local embedder (sentence-transformers, no API key needed)
    print("\nLoading local embedder (all-MiniLM-L6-v2)...")
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    # Warm up the model (first load downloads ~80MB)
    test_emb = await embedder.create("test")
    print(f"  ✓ Embedder loaded, dim={len(test_emb)}")
    
    # Connect to Neo4j (running in Docker on default ports)
    print("\n1. Connecting to Neo4j...")
    client = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "testpassword123",
        llm_client=llm_client,
        embedder=embedder,
    )
    
    # Build indices
    print("2. Building indices...")
    try:
        await client.build_indices_and_constraints()
        print("   ✓ Indices built")
    except Exception as e:
        print(f"   ✗ Error building indices: {e}")
    
    # Add some episodes (context entries)
    print("\n3. Adding test episodes...")
    
    episodes = [
        {
            "name": "project_decision_1",
            "body": "We decided to use Graphiti as the graph engine for our context graph project. The alternatives were TrustGraph (too enterprise-heavy) and building from scratch (too much work). Graphiti is Apache 2.0, Python-based, and has MCP support.",
            "source": EpisodeType.text,
            "time": datetime(2026, 2, 28, tzinfo=timezone.utc),
        },
        {
            "name": "project_decision_2", 
            "body": "Sergey wants to build an open-source context graph tool. The idea is to store decision traces as first-class citizens — not just WHAT was decided, but WHY, what alternatives were considered, and what conditions led to the decision.",
            "source": EpisodeType.text,
            "time": datetime(2026, 2, 27, tzinfo=timezone.utc),
        },
        {
            "name": "team_context",
            "body": "Ivan is an Android and KMP developer. He's not a pure iOS dev, so iOS-specific outsourcing requests from Extyl don't fit him well. Sergey runs a development agency and is looking for leads in DataScience and mobile development.",
            "source": EpisodeType.text,
            "time": datetime(2026, 2, 27, tzinfo=timezone.utc),
        },
    ]
    
    for ep in episodes:
        try:
            await client.add_episode(
                name=ep["name"],
                episode_body=ep["body"],
                source_description="OpenClaw PoC test",
                reference_time=ep["time"],
                source=ep["source"],
            )
            print(f"   ✓ Added: {ep['name']}")
        except Exception as e:
            print(f"   ✗ Error adding {ep['name']}: {e}")
    
    # Search the graph
    print("\n4. Searching graph...")
    
    queries = [
        "What graph engine are we using?",
        "Who is Ivan?",
        "What is Sergey building?",
    ]
    
    for q in queries:
        try:
            results = await client.search(q)
            print(f"\n   Q: {q}")
            if results:
                for r in results[:3]:
                    fact_text = r.fact if hasattr(r, 'fact') else str(r)
                    print(f"   → {fact_text[:150]}")
            else:
                print("   → No results")
        except Exception as e:
            print(f"   ✗ Error searching '{q}': {e}")
    
    # Close
    await client.close()
    print("\n=== PoC Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
