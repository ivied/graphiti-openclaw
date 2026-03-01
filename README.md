# graphiti-openclaw

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI - graphiti-core](https://img.shields.io/badge/graphiti--core-0.28+-purple.svg)](https://pypi.org/project/graphiti-core/)

**OpenAI-compatible adapters for [Graphiti](https://github.com/getzep/graphiti) context graph engine.**

## The Problem

Graphiti 0.28+ switched to the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) (`client.responses.parse`), which is:
- **Not supported** by most OpenAI-compatible proxies (OpenRouter, LiteLLM, vLLM, Ollama, etc.)
- **Not available** if you're using Anthropic, Mistral, or other LLMs through a proxy
- **A blocker** for self-hosted and custom LLM setups

This package provides drop-in replacements that use the classic **Chat Completions API** instead, making Graphiti work with any OpenAI-compatible endpoint.

## What's Inside

### `CompatOpenAIClient` — LLM Client

Works with **any OpenAI-compatible proxy**: [9router](https://github.com/9router), [OpenRouter](https://openrouter.ai), [LiteLLM](https://litellm.ai), [vLLM](https://vllm.ai), [Ollama](https://ollama.ai), etc.

**What it handles:**
- Translates structured completion requests to classic `chat.completions.create` with JSON schema in prompt
- Explicit `stream=False` (some proxies default to streaming)
- JSON extraction from markdown code blocks (Claude wraps JSON in ````json ... ````)
- Omits `temperature=None` (some providers reject null values)

### `LocalEmbedder` — Embedding Client

Runs entirely on your machine — no API keys needed.

- Uses `sentence-transformers/all-MiniLM-L6-v2` by default (384 dims)
- Async-safe: runs encoding in thread pool
- Supports single and batch embeddings

## Quick Start

```bash
pip install graphiti-openclaw[all]
```

Or minimal (without local embeddings):
```bash
pip install graphiti-openclaw
```

### Usage

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_openclaw import CompatOpenAIClient, LocalEmbedder

# Configure LLM (any OpenAI-compatible endpoint)
llm_config = LLMConfig(
    api_key="your-api-key",
    base_url="https://your-proxy.com/v1",
    model="claude-sonnet-4-20250514",
    small_model="claude-haiku-4-20250414",
)

llm_client = CompatOpenAIClient(config=llm_config)
embedder = LocalEmbedder()  # or pass model_name="your-model"

# Initialize Graphiti with Neo4j
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password",
    llm_client=llm_client,
    embedder=embedder,
)
```

See [examples/poc.py](examples/poc.py) for a full working example.

### Proxy Configuration Examples

**OpenRouter:**
```python
llm_config = LLMConfig(
    api_key="sk-or-v1-...",
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4-20250514",
    small_model="anthropic/claude-haiku-4-20250414",
)
```

**LiteLLM:**
```python
llm_config = LLMConfig(
    api_key="sk-...",
    base_url="http://localhost:4000/v1",
    model="claude-sonnet-4-20250514",
    small_model="claude-haiku-4-20250414",
)
```

**Ollama (local models):**
```python
llm_config = LLMConfig(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="llama3.1:70b",
    small_model="llama3.1:8b",
)
```

## CLI Tool

The package includes a CLI for quick interaction with your context graph:

```bash
# Check graph status
graphiti-oc status
# 📊 Context Graph Status
#    Neo4j: bolt://localhost:7687
#    Total nodes: 101
#    Episodes: 13
#    Entities: 88

# Add context from text
graphiti-oc add "We decided to use Postgres instead of MongoDB for better JOIN support"

# Add context from a file
graphiti-oc add --file notes/2026-03-01.md

# Search the graph
graphiti-oc search "What database did we choose?"
# → We decided to use Postgres instead of MongoDB for better JOIN support
```

The CLI auto-loads configuration from `~/.openclaw/openclaw.json` (LLM settings) and the Graphiti config YAML (Neo4j credentials).

## Requirements

- Python 3.11+
- Neo4j 5.x (for Graphiti backend)
- Any OpenAI-compatible LLM proxy

## MCP Server

This package also works with the [Graphiti MCP server](https://github.com/getzep/graphiti/tree/main/mcp_server). See the `configs/` directory for example configurations.

## Related

- [Graphiti](https://github.com/getzep/graphiti) — The context graph engine
- [OpenClaw](https://openclaw.dev) — AI agent framework

## License

MIT
