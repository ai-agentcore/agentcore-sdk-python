"""Synchronous and asynchronous calls to an OpenAI/v1 managed model."""

import asyncio

from agentcore import AgentCore, AsyncAgentCore


def synchronous() -> None:
    with AgentCore.auto() as core:
        model = core.model("test-mc", model="qwen3.8-max")
        response = model.invoke([{"role": "user", "content": "请简短介绍你自己。"}])
        print(response["choices"][0]["message"]["content"])


async def streaming() -> None:
    async with AsyncAgentCore.auto() as core:
        model = await core.model("test-mc", model="qwen3.8-max")
        async for chunk in model.stream([{"role": "user", "content": "用一句话介绍杭州。"}]):
            # Usage-only chunks can have no choices.
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    print(text, end="", flush=True)
        print()


if __name__ == "__main__":
    synchronous()
    asyncio.run(streaming())
