"""Optional Pydantic base class for business models with camelCase JSON fields."""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel


class BaseModel(PydanticBaseModel):
    """Accept field names and aliases; serialize declared fields as camelCase."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )
