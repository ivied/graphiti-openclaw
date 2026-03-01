# graphiti-openclaw

**OpenAI-compatible adapters for [Graphiti](https://github.com/getzep/graphiti) context graph engine.**

Graphiti 0.28+ uses the OpenAI Responses API (`client.responses.parse`) which isn't supported by most OpenAI-compatible proxies. This package provides drop-in replacements that use the classic Chat Completions API instead.

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
