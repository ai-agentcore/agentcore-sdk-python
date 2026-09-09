"""Run in AgentCore with a named MCP Header credential allowed for test-mcp."""

import asyncio

from agentcore import AsyncAgentCore


async def main():
    async with AsyncAgentCore.auto() as core:
        mcp = await core.mcp(
            "test-mcp",
            credential_name="my-mcp-credential",
            headers={"x-business-id": "example-app"},
        )
        print("MCP tools:", [tool.name for tool in await mcp.list_tools()])
        # Framework adapters reuse this client's fixed headers for tool calls.

        # Explicit retrieval is also available. Do not log credential.value or headers.
        credential = await core.credentials.get("my-mcp-credential")
        headers = credential.as_headers()  # Only mcpHeader credentials support this method.
        print("Header names:", list(headers))
        # For apiKey credentials, use credential.value; it remains a string.


if __name__ == "__main__":
    asyncio.run(main())
