#!/usr/bin/env python3
"""Test Graphiti with real daily notes from OpenClaw workspace.

Ingests memory/2026-03-01.md as episodes and tests search quality.
"""

import asyncio
import os
import json
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.nodes import EpisodeType
from compat_client import CompatOpenAIClient
from local_embedder import LocalEmbedder


WORKSPACE = Path.home() / ".openclaw" / "workspace"


def get_llm_config():
    with open(os.path.expanduser("~/.openclaw/openclaw.json")) as f:
        cfg = json.load(f)
    provider = cfg["models"]["providers"]["9router"]
    return LLMConfig(
        api_key=provider["apiKey"],
        base_url=provider["baseUrl"],
        model="cc/claude-sonnet-4-5-20250929",
        small_model="cc/claude-haiku-4-5-20251001",
    )


def parse_daily_notes(filepath: Path) -> list[dict]:
    """Split a daily notes file into sections as separate episodes."""
    content = filepath.read_text()
    sections = []
    current_section = ""
    current_title = "intro"
    
    for line in content.split("\n"):
        if line.startswith("## "):
            if current_section.strip():
                sections.append({
                    "title": current_title,
                    "body": current_section.strip(),
                })
            current_title = line[3:].strip()
            current_section = ""
        else:
            current_section += line + "\n"
    
    if current_section.strip():
        sections.append({
            "title": current_title,
            "body": current_section.strip(),
        })
    
    return sections


async def main():
    print("=== Real Data Test ===\n")
    
    llm_config = get_llm_config()
    llm_client = CompatOpenAIClient(config=llm_config)
    
    print("Loading embedder...")
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    test_emb = await embedder.create("test")
    print(f"  ✓ dim={len(test_emb)}")
    
    client = Graphiti(
        "bolt://localhost:7687", "neo4j", "testpassword123",
        llm_client=llm_client, embedder=embedder,
    )
    await client.build_indices_and_constraints()
    
    # Parse today's daily notes
    daily_file = WORKSPACE / "memory" / "2026-03-01.md"
    if not daily_file.exists():
        print(f"File not found: {daily_file}")
        return
    
    sections = parse_daily_notes(daily_file)
    print(f"\nParsed {len(sections)} sections from {daily_file.name}\n")
    
    # Ingest each section as an episode
    ref_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    
    for i, section in enumerate(sections):
        name = f"daily_2026-03-01_{i}_{section['title'][:30].replace(' ', '_')}"
        print(f"  Adding: [{section['title']}] ({len(section['body'])} chars)...")
        try:
            await client.add_episode(
                name=name,
                episode_body=f"Daily note — {section['title']}:\n\n{section['body']}",
                source_description=f"OpenClaw daily notes 2026-03-01",
                reference_time=ref_time,
                source=EpisodeType.text,
            )
            print(f"    ✓ OK")
        except Exception as e:
            print(f"    ✗ {e}")
    
    # Test searches
    print("\n=== Search Tests ===\n")
    
    test_queries = [
        "What work was done on the $100 challenge?",
        "What is the context graph project about?",
        "What tasks are blocked and why?",
        "What massage courses were researched?",
        "What improvements were made to the algorithm?",
    ]
    
    for q in test_queries:
        print(f"Q: {q}")
        try:
            results = await client.search(q)
            if results:
                for r in results[:2]:
                    fact = r.fact if hasattr(r, 'fact') else str(r)
                    print(f"  → {fact[:200]}")
            else:
                print("  → No results")
        except Exception as e:
            print(f"  ✗ {e}")
        print()
    
    await client.close()
    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
