"""Parse business parameters and serialize camelCase JSON without cloud resources."""

from pydantic import Field

from agentcore import BaseModel


class Character(BaseModel):
    character_id: str
    display_name: str = Field(alias="name")


class CharacterRequest(BaseModel):
    main_character: Character
    schema_version: int = 1
    metadata: dict[str, str]


def main() -> None:
    # Parse the business portion of an incoming request explicitly.
    request = CharacterRequest.model_validate(
        {
            "mainCharacter": {"characterId": "alice", "name": "Alice"},
            "metadata": {"source_id": "demo"},
        }
    )
    print(request.main_character.character_id)
    # Declared model fields use aliases; metadata keys stay unchanged.
    print(request.model_dump_json(indent=2))

    # Python field names are also accepted; explicit aliases still take precedence.
    character = Character(character_id="bob", display_name="Bob")
    print(character.model_dump())
    print(character.model_dump(by_alias=False))


if __name__ == "__main__":
    main()
