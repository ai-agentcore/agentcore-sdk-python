"""Use an AgentCore business model as a LangChain tool's argument schema."""

from langchain_core.tools import tool
from pydantic import ConfigDict

from agentcore import BaseModel


class CharacterArgs(BaseModel):
    # LangChain passes model fields to the Python function as keyword arguments.
    model_config = ConfigDict(serialize_by_alias=False)

    character_id: str


@tool(args_schema=CharacterArgs)
def describe_character(character_id: str) -> str:
    """Return the requested character identifier."""
    return f"Character: {character_id}"


if __name__ == "__main__":
    print(describe_character.invoke({"characterId": "alice"}))
    print(describe_character.invoke({"character_id": "bob"}))
