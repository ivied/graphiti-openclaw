"""
graphiti-openclaw — OpenAI-compatible adapters for Graphiti context graph engine.

Provides:
- CompatOpenAIClient: LLM client that uses classic chat completions API
  (works with any OpenAI-compatible proxy: 9router, OpenRouter, LiteLLM, vLLM, etc.)
- LocalEmbedder: Local embedding client using sentence-transformers
  (no API keys needed — runs entirely on your machine)
"""

from graphiti_openclaw.compat_client import CompatOpenAIClient

__version__ = "0.1.0"
__all__ = ["CompatOpenAIClient"]

try:
    from graphiti_openclaw.local_embedder import LocalEmbedder
    __all__.append("LocalEmbedder")
except ImportError:
    # sentence-transformers not installed — LocalEmbedder unavailable
    # Install with: pip install graphiti-openclaw[local-embedder]
    pass
