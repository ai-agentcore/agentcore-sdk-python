"""Explicit gateway operations using trusted identity and persisted source events."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from agentcore_memory_plugin import MemoryBinding, MemoryPlugin, SourceEvent, SourceRecord


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity", type=Path, help="Trusted binding and session JSON")
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("recall").add_argument("query")
    commands.add_parser("capture").add_argument("events", type=Path)
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    binding = MemoryBinding(
        identity["memory_store_name"],
        identity["user_id"],
        identity["agent_id"],
        identity.get("team_id"),
    )
    async with MemoryPlugin.from_env(
        os.environ["MEMORY_GATEWAY_ENDPOINT"],
        token_provider=lambda: os.environ["MEMORY_TOKEN"],
    ) as plugin:
        session = plugin.session(binding, identity["session_id"])
        if args.operation == "recall":
            result = await session.recall(args.query)
            print(result.context)
        else:
            records = [
                SourceRecord(row["source_id"], SourceEvent(**row["event"]))
                for row in json.loads(args.events.read_text())
            ]
            receipts = await session.capture(records)
            print(json.dumps([{"source_id": r.source_id, "state": r.state} for r in receipts]))


if __name__ == "__main__":
    asyncio.run(main())
