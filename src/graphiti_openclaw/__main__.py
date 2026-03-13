"""Allow running as `python -m graphiti_openclaw.mcp_server`."""
from graphiti_openclaw.mcp_server import main
import asyncio
asyncio.run(main())
