"""Call native Context Memory with trusted scope and persisted event JSON."""

import argparse
import asyncio
import json
from pathlib import Path

from agentcore import AsyncAgentCore
from agentcore.memory import ContextEvent, MemoryScope, MemoryType


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store")
    parser.add_argument("--endpoint")
    parser.add_argument("--user-id")
    parser.add_argument("--team-id")
    parser.add_argument("--agent-id")
    parser.add_argument("--session-id")
    commands = parser.add_subparsers(dest="operation", required=True)
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--type", action="append", choices=[kind.value for kind in MemoryType])
    search.add_argument("--top-k", type=int)
    listing = commands.add_parser("list-events")
    listing.add_argument("--next-token")
    commands.add_parser("create-events").add_argument("events", type=Path)
    args = parser.parse_args()
    scope = MemoryScope(args.agent_id, args.session_id, args.user_id, args.team_id)
    async with AsyncAgentCore.auto() as core:
        memory = core.context_memory(args.store, endpoint=args.endpoint)
        if args.operation == "search":
            result = await memory.search_memories(
                args.query,
                scope=scope,
                types=[MemoryType(kind) for kind in args.type] if args.type else None,
                top_k=args.top_k,
            )
            for hit in result.memories:
                print(hit)
        elif args.operation == "list-events":
            page = await memory.list_events(scope=scope, next_token=args.next_token)
            for event in page.items:
                print(event)
            print("next_token", page.next_token)
        else:
            records = json.loads(args.events.read_text(encoding="utf-8"))
            created = await memory.create_events(
                [
                    ContextEvent.from_source(row["type"], row["timestamp"], row["data"])
                    for row in records
                ],
                scope=scope,
            )
            for receipt in created.results:
                print(receipt)
            print("request_id", created.request_id)


if __name__ == "__main__":
    asyncio.run(main())
