from typing import Literal

from pydantic.fields import FieldInfo


def get_aletheia_field_type(field: FieldInfo) -> Literal["input", "output"]:
    field_type = field.json_schema_extra.get("__aletheia_field_type")
    if field_type is None:
        raise ValueError(f"Field {field} does not have a __aletheia_field_type")
    return field_type
