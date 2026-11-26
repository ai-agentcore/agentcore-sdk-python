from __future__ import annotations

import json

import pytest
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, ValidationError

from agentcore import BaseModel


class CharacterRequest(BaseModel):
    character_id: str
    schema_version: int = 1


@pytest.mark.parametrize("field", ["character_id", "characterId"])
def test_accepts_field_names_and_camel_case_aliases(field: str) -> None:
    payload = {field: "alice"}

    assert CharacterRequest(**payload).character_id == "alice"
    assert CharacterRequest.model_validate(payload).character_id == "alice"
    assert CharacterRequest.model_validate_json(json.dumps(payload)).character_id == "alice"


def test_serializes_aliases_by_default_and_allows_field_names() -> None:
    request = CharacterRequest(character_id="alice")

    assert request.model_dump() == {"characterId": "alice", "schemaVersion": 1}
    assert json.loads(request.model_dump_json()) == request.model_dump()
    expected = {"character_id": "alice", "schema_version": 1}
    assert request.model_dump(by_alias=False) == expected
    assert json.loads(request.model_dump_json(by_alias=False)) == expected


def test_explicit_alias_takes_precedence() -> None:
    class Request(BaseModel):
        character_id: str = Field(alias="actorID")

    request = Request.model_validate({"actorID": "alice"})

    assert request.character_id == "alice"
    assert request.model_dump() == {"actorID": "alice"}
    assert Request(character_id="alice").model_dump() == request.model_dump()


def test_nested_models_convert_fields_without_rewriting_dictionary_keys() -> None:
    class Request(BaseModel):
        main_character: CharacterRequest
        other_characters: list[CharacterRequest]
        custom_data: dict[str, str]

    payload = {
        "mainCharacter": {"characterId": "alice", "schemaVersion": 1},
        "otherCharacters": [{"characterId": "bob", "schemaVersion": 1}],
        "customData": {"character_id": "untouched", "schemaVersion": "custom"},
    }
    request = Request.model_validate(payload)

    assert request.main_character.character_id == "alice"
    assert request.other_characters[0].character_id == "bob"
    assert request.model_dump() == payload
    assert request.model_dump(by_alias=False)["main_character"]["character_id"] == "alice"
    assert payload["customData"]["character_id"] == "untouched"


def test_plain_pydantic_models_keep_their_own_configuration() -> None:
    class PlainRequest(PydanticBaseModel):
        character_id: str

    class Request(BaseModel):
        plain_request: PlainRequest

    plain = PlainRequest(character_id="alice")

    assert plain.model_dump() == {"character_id": "alice"}
    assert Request(plain_request=plain).model_dump() == {
        "plainRequest": {"character_id": "alice"},
    }


def test_json_schema_uses_aliases() -> None:
    schema = CharacterRequest.model_json_schema()

    assert set(schema["properties"]) == {"characterId", "schemaVersion"}
    assert schema["required"] == ["characterId"]
    assert "character_id" in CharacterRequest.model_json_schema(by_alias=False)["properties"]


def test_missing_and_invalid_fields_still_fail_validation() -> None:
    with pytest.raises(ValidationError) as missing:
        CharacterRequest.model_validate({})
    assert missing.value.errors()[0]["loc"] == ("characterId",)

    with pytest.raises(ValidationError) as invalid:
        CharacterRequest.model_validate({"characterId": "alice", "schemaVersion": "invalid"})
    assert invalid.value.errors()[0]["loc"] == ("schemaVersion",)


def test_alias_wins_when_both_names_are_present() -> None:
    request = CharacterRequest.model_validate({"characterId": "alice", "character_id": "bob"})

    assert request.character_id == "alice"


def test_subclasses_can_override_pydantic_configuration() -> None:
    class Request(BaseModel):
        model_config = ConfigDict(serialize_by_alias=False, extra="forbid")

        character_id: str

    request = Request.model_validate({"characterId": "alice"})

    assert request.model_dump() == {"character_id": "alice"}
    assert request.model_dump(by_alias=True) == {"characterId": "alice"}
    with pytest.raises(ValidationError):
        Request.model_validate({"characterId": "alice", "unknown": "value"})
