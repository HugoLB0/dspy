from unittest import mock

import pydantic
import pytest
from pydantic import create_model

import aletheia


def test_json_adapter_passes_structured_output_when_supported_by_model():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1", ge=0, le=10)
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(aletheia.Signature):
        input1: str = aletheia.InputField()
        output1: str = aletheia.OutputField()  # Description intentionally left blank
        output2: bool = aletheia.OutputField(desc="Boolean output field")
        output3: OutputField3 = aletheia.OutputField(desc="Nested output field")
        output4_unannotated = aletheia.OutputField(desc="Unannotated output field")

    program = aletheia.Predict(TestSignature)

    # Configure aletheia to use an OpenAI LM that supports structured outputs
    aletheia.configure(lm=aletheia.LM(model="openai/gpt4o"), adapter=aletheia.JSONAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        program(input1="Test input")

    def clean_schema_extra(field_name, field_info):
        attrs = dict(field_info.__repr_args__())
        if "json_schema_extra" in attrs:
            attrs["json_schema_extra"] = {
                k: v
                for k, v in attrs["json_schema_extra"].items()
                if k != "__aletheia_field_type" and not (k == "desc" and v == f"${{{field_name}}}")
            }
        return attrs

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    response_format = call_kwargs.get("response_format")
    assert response_format is not None
    assert issubclass(response_format, pydantic.BaseModel)
    assert response_format.model_fields.keys() == {"output1", "output2", "output3", "output4_unannotated"}
    for field_name in response_format.model_fields:
        assert dict(response_format.model_fields[field_name].__repr_args__()) == clean_schema_extra(
            field_name=field_name,
            field_info=TestSignature.output_fields[field_name],
        )

    # Configure aletheia to use a model from a fake provider that doesn't support structured outputs
    aletheia.configure(lm=aletheia.LM(model="fakeprovider/fakemodel"), adapter=aletheia.JSONAdapter())
    with mock.patch("litellm.completion") as mock_completion:
        program(input1="Test input")

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert response_format not in call_kwargs


def test_json_adapter_falls_back_when_structured_outputs_fails():
    class TestSignature(aletheia.Signature):
        input1: str = aletheia.InputField()
        output1: str = aletheia.OutputField(desc="String output field")

    aletheia.configure(lm=aletheia.LM(model="openai/gpt4o"), adapter=aletheia.JSONAdapter())
    program = aletheia.Predict(TestSignature)
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.side_effect = [Exception("Bad structured outputs!"), mock_completion.return_value]
        program(input1="Test input")
        assert mock_completion.call_count == 2
        _, first_call_kwargs = mock_completion.call_args_list[0]
        assert issubclass(first_call_kwargs.get("response_format"), pydantic.BaseModel)
        _, second_call_kwargs = mock_completion.call_args_list[1]
        assert second_call_kwargs.get("response_format") == {"type": "json_object"}


def test_json_adapter_with_structured_outputs_does_not_mutate_original_signature():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1")
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(aletheia.Signature):
        input1: str = aletheia.InputField()
        output1: str = aletheia.OutputField()  # Description intentionally left blank
        output2: bool = aletheia.OutputField(desc="Boolean output field")
        output3: OutputField3 = aletheia.OutputField(desc="Nested output field")
        output4_unannotated = aletheia.OutputField(desc="Unannotated output field")

    aletheia.configure(lm=aletheia.LM(model="openai/gpt4o"), adapter=aletheia.JSONAdapter())
    program = aletheia.Predict(TestSignature)
    with mock.patch("litellm.completion"):
        program(input1="Test input")

    assert program.signature.output_fields == TestSignature.output_fields
