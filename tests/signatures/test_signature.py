import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union

import pydantic
import pytest

import aletheia
from aletheia import InputField, OutputField, Signature, infer_prefix
from aletheia.utils.dummies import DummyLM


def test_field_types_and_custom_attributes():
    class TestSignature(Signature):
        """Instructions"""

        input1: str = InputField()
        input2: int = InputField()
        output1: List[str] = OutputField()
        output2 = OutputField()

    assert TestSignature.instructions == "Instructions"
    assert TestSignature.input_fields["input1"].annotation == str
    assert TestSignature.input_fields["input2"].annotation == int
    assert TestSignature.output_fields["output1"].annotation == List[str]
    assert TestSignature.output_fields["output2"].annotation == str


def test_no_input_output():
    with pytest.raises(TypeError):

        class TestSignature(Signature):
            input1: str


def test_no_input_output2():
    with pytest.raises(TypeError):

        class TestSignature(Signature):
            input1: str = pydantic.Field()


def test_all_fields_have_prefix():
    class TestSignature(Signature):
        input = InputField(prefix="Modified:")
        output = OutputField()

    assert TestSignature.input_fields["input"].json_schema_extra["prefix"] == "Modified:"
    assert TestSignature.output_fields["output"].json_schema_extra["prefix"] == "Output:"


def test_signature_parsing():
    signature = Signature("input1, input2 -> output")
    assert "input1" in signature.input_fields
    assert "input2" in signature.input_fields
    assert "output" in signature.output_fields


def test_with_signature():
    signature1 = Signature("input1, input2 -> output")
    signature2 = signature1.with_instructions("This is a test")
    assert signature2.instructions == "This is a test"
    assert signature1 is not signature2, "The type should be immutable"


def test_with_updated_field():
    signature1 = Signature("input1, input2 -> output")
    signature2 = signature1.with_updated_fields("input1", prefix="Modified:")
    assert signature2.input_fields["input1"].json_schema_extra["prefix"] == "Modified:"
    assert signature1.input_fields["input1"].json_schema_extra["prefix"] == "Input 1:"
    assert signature1 is not signature2, "The type should be immutable"
    for key in signature1.fields.keys():
        if key != "input1":
            assert signature1.fields[key].json_schema_extra == signature2.fields[key].json_schema_extra
    assert signature1.instructions == signature2.instructions


def test_empty_signature():
    with pytest.raises(ValueError):
        Signature("")


def test_instructions_signature():
    with pytest.raises(ValueError):
        Signature("")


def test_signature_instructions():
    sig1 = Signature("input1 -> output1", instructions="This is a test")
    assert sig1.instructions == "This is a test"
    sig2 = Signature("input1 -> output1", "This is a test")
    assert sig2.instructions == "This is a test"


def test_signature_instructions_none():
    sig1 = Signature("a, b -> c")
    assert sig1.instructions == f"Given the fields `a`, `b`, produce the fields `c`."


def test_signature_from_dict():
    signature = Signature({"input1": InputField(), "input2": InputField(), "output": OutputField()})
    for k in ["input1", "input2", "output"]:
        assert k in signature.fields
        assert signature.fields[k].annotation == str


def test_signature_from_dict():
    signature = Signature({"input1": InputField(), "input2": InputField(), "output": OutputField()})
    assert "input1" in signature.input_fields
    assert "input2" in signature.input_fields
    assert "output" in signature.output_fields


def test_signature_equality():
    sig1 = Signature("input1 -> output1")
    sig2 = Signature("input1 -> output1")
    assert sig1.equals(sig2)


def test_signature_inequality():
    sig1 = Signature("input1 -> output1")
    sig2 = Signature("input2 -> output2")
    assert not sig1.equals(sig2)


def test_equality_format():
    class TestSignature(Signature):
        input = InputField(format=lambda x: x)
        output = OutputField()

    assert TestSignature.equals(TestSignature)


def test_signature_reverse():
    sig = Signature("input1 -> output1")
    assert sig.signature == "input1 -> output1"


def test_insert_field_at_various_positions():
    class InitialSignature(Signature):
        input1: str = InputField()
        output1: int = OutputField()

    S1 = InitialSignature.prepend("new_input_start", InputField(), str)
    S2 = InitialSignature.append("new_input_end", InputField(), str)
    assert "new_input_start" == list(S1.input_fields.keys())[0]
    assert "new_input_end" == list(S2.input_fields.keys())[-1]

    S3 = InitialSignature.prepend("new_output_start", OutputField(), str)
    S4 = InitialSignature.append("new_output_end", OutputField(), str)
    assert "new_output_start" == list(S3.output_fields.keys())[0]
    assert "new_output_end" == list(S4.output_fields.keys())[-1]


def test_order_preserved_with_mixed_annotations():
    class ExampleSignature(aletheia.Signature):
        text: str = aletheia.InputField()
        output = aletheia.OutputField()
        pass_evaluation: bool = aletheia.OutputField()

    expected_order = ["text", "output", "pass_evaluation"]
    actual_order = list(ExampleSignature.fields.keys())
    assert actual_order == expected_order


def test_infer_prefix():
    assert infer_prefix("someAttributeName42IsCool") == "Some Attribute Name 42 Is Cool"
    assert infer_prefix("version2Update") == "Version 2 Update"
    assert infer_prefix("modelT45Enhanced") == "Model T 45 Enhanced"
    assert infer_prefix("someAttributeName") == "Some Attribute Name"
    assert infer_prefix("some_attribute_name") == "Some Attribute Name"
    assert infer_prefix("URLAddress") == "URL Address"
    assert infer_prefix("isHTTPSecure") == "Is HTTP Secure"
    assert infer_prefix("isHTTPSSecure123") == "Is HTTPS Secure 123"


def test_insantiating():
    sig = Signature("input -> output")
    assert issubclass(sig, Signature)
    assert sig.__name__ == "StringSignature"
    value = sig(input="test", output="test")
    assert isinstance(value, sig)


def test_insantiating2():
    class SubSignature(Signature):
        input = InputField()
        output = OutputField()

    assert issubclass(SubSignature, Signature)
    assert SubSignature.__name__ == "SubSignature"
    value = SubSignature(input="test", output="test")
    assert isinstance(value, SubSignature)


def test_multiline_instructions():
    lm = DummyLM([{"output": "short answer"}])
    aletheia.settings.configure(lm=lm)

    class MySignature(Signature):
        """First line
        Second line
            Third line"""

        output = OutputField()

    predictor = aletheia.Predict(MySignature)
    assert predictor().output == "short answer"


def test_dump_and_load_state():
    class CustomSignature(aletheia.Signature):
        """I am just an instruction."""

        sentence = aletheia.InputField(desc="I am an innocent input!")
        sentiment = aletheia.OutputField()

    state = CustomSignature.dump_state()
    expected = {
        "instructions": "I am just an instruction.",
        "fields": [
            {
                "prefix": "Sentence:",
                "description": "I am an innocent input!",
            },
            {
                "prefix": "Sentiment:",
                "description": "${sentiment}",
            },
        ],
    }
    assert state == expected

    class CustomSignature2(aletheia.Signature):
        """I am a malicious instruction."""

        sentence = aletheia.InputField(desc="I am an malicious input!")
        sentiment = aletheia.OutputField()

    assert CustomSignature2.dump_state() != expected
    # Overwrite the state with the state of CustomSignature.
    loaded_signature = CustomSignature2.load_state(state)
    assert loaded_signature.instructions == "I am just an instruction."
    # After `load_state`, the state should be the same as CustomSignature.
    assert loaded_signature.dump_state() == expected
    # CustomSignature2 should not have been modified.
    assert CustomSignature2.instructions == "I am a malicious instruction."
    assert CustomSignature2.fields["sentence"].json_schema_extra["desc"] == "I am an malicious input!"
    assert CustomSignature2.fields["sentiment"].json_schema_extra["prefix"] == "Sentiment:"


def test_typed_signatures_basic_types():
    # Simple built-in types
    sig = Signature("input1: int, input2: str -> output: float")
    assert "input1" in sig.input_fields
    assert sig.input_fields["input1"].annotation == int
    assert "input2" in sig.input_fields
    assert sig.input_fields["input2"].annotation == str
    assert "output" in sig.output_fields
    assert sig.output_fields["output"].annotation == float


def test_typed_signatures_generics():
    # More complex generic types
    sig = Signature("input_list: List[int], input_dict: Dict[str, float] -> output_tuple: Tuple[str, int]")
    assert "input_list" in sig.input_fields
    assert sig.input_fields["input_list"].annotation == List[int]
    assert "input_dict" in sig.input_fields
    assert sig.input_fields["input_dict"].annotation == Dict[str, float]
    assert "output_tuple" in sig.output_fields
    assert sig.output_fields["output_tuple"].annotation == Tuple[str, int]


def test_typed_signatures_unions_and_optionals():
    sig = Signature("input_opt: Optional[str], input_union: Union[int, None] -> output_union: Union[int, str]")
    assert "input_opt" in sig.input_fields
    # Optional[str] is actually Union[str, None]
    # Depending on the environment, it might resolve to Union[str, None] or Optional[str], either is correct.
    # We'll just check for a Union containing str and NoneType:
    input_opt_annotation = sig.input_fields["input_opt"].annotation
    assert input_opt_annotation == Optional[str] or (
        getattr(input_opt_annotation, "__origin__", None) is Union
        and str in input_opt_annotation.__args__
        and type(None) in input_opt_annotation.__args__
    )

    assert "input_union" in sig.input_fields
    input_union_annotation = sig.input_fields["input_union"].annotation
    assert (
        getattr(input_union_annotation, "__origin__", None) is Union
        and int in input_union_annotation.__args__
        and type(None) in input_union_annotation.__args__
    )

    assert "output_union" in sig.output_fields
    output_union_annotation = sig.output_fields["output_union"].annotation
    assert (
        getattr(output_union_annotation, "__origin__", None) is Union
        and int in output_union_annotation.__args__
        and str in output_union_annotation.__args__
    )


def test_typed_signatures_any():
    sig = Signature("input_any: Any -> output_any: Any")
    assert "input_any" in sig.input_fields
    assert sig.input_fields["input_any"].annotation == Any
    assert "output_any" in sig.output_fields
    assert sig.output_fields["output_any"].annotation == Any


def test_typed_signatures_nested():
    # Nested generics and unions
    sig = Signature("input_nested: List[Union[str, int]] -> output_nested: Tuple[int, Optional[float], List[str]]")
    input_nested_ann = sig.input_fields["input_nested"].annotation
    assert getattr(input_nested_ann, "__origin__", None) is list
    assert len(input_nested_ann.__args__) == 1
    union_arg = input_nested_ann.__args__[0]
    assert getattr(union_arg, "__origin__", None) is Union
    assert str in union_arg.__args__ and int in union_arg.__args__

    output_nested_ann = sig.output_fields["output_nested"].annotation
    assert getattr(output_nested_ann, "__origin__", None) is tuple
    assert output_nested_ann.__args__[0] == int
    # The second arg is Optional[float], which is Union[float, None]
    second_arg = output_nested_ann.__args__[1]
    assert getattr(second_arg, "__origin__", None) is Union
    assert float in second_arg.__args__ and type(None) in second_arg.__args__
    # The third arg is List[str]
    third_arg = output_nested_ann.__args__[2]
    assert getattr(third_arg, "__origin__", None) is list
    assert third_arg.__args__[0] == str


def test_typed_signatures_from_dict():
    # Creating a Signature directly from a dictionary with types
    fields = {
        "input_str_list": (List[str], InputField()),
        "input_dict_int": (Dict[str, int], InputField()),
        "output_tup": (Tuple[int, float], OutputField()),
    }
    sig = Signature(fields)
    assert "input_str_list" in sig.input_fields
    assert sig.input_fields["input_str_list"].annotation == List[str]
    assert "input_dict_int" in sig.input_fields
    assert sig.input_fields["input_dict_int"].annotation == Dict[str, int]
    assert "output_tup" in sig.output_fields
    assert sig.output_fields["output_tup"].annotation == Tuple[int, float]


def test_typed_signatures_complex_combinations():
    # Test a very complex signature with multiple nested constructs
    # input_complex: Dict[str, List[Optional[Tuple[int, str]]]] -> output_complex: Union[List[str], Dict[str, Any]]
    sig = Signature(
        "input_complex: Dict[str, List[Optional[Tuple[int, str]]]] -> output_complex: Union[List[str], Dict[str, Any]]"
    )
    input_complex_ann = sig.input_fields["input_complex"].annotation
    assert getattr(input_complex_ann, "__origin__", None) is dict
    key_arg, value_arg = input_complex_ann.__args__
    assert key_arg == str
    # value_arg: List[Optional[Tuple[int, str]]]
    assert getattr(value_arg, "__origin__", None) is list
    inner_union = value_arg.__args__[0]
    # inner_union should be Optional[Tuple[int, str]]
    # which is Union[Tuple[int, str], None]
    assert getattr(inner_union, "__origin__", None) is Union
    tuple_type = [t for t in inner_union.__args__ if t != type(None)][0]
    assert getattr(tuple_type, "__origin__", None) is tuple
    assert tuple_type.__args__ == (int, str)

    output_complex_ann = sig.output_fields["output_complex"].annotation
    assert getattr(output_complex_ann, "__origin__", None) is Union
    assert len(output_complex_ann.__args__) == 2
    possible_args = set(output_complex_ann.__args__)
    # Expecting List[str] and Dict[str, Any]
    # Because sets don't preserve order, just check membership.
    # Find the List[str] arg
    list_arg = next(a for a in possible_args if getattr(a, "__origin__", None) is list)
    dict_arg = next(a for a in possible_args if getattr(a, "__origin__", None) is dict)
    assert list_arg.__args__ == (str,)
    k, v = dict_arg.__args__
    assert k == str and v == Any


def test_make_signature_from_string():
    sig = Signature("input1: int, input2: Dict[str, int] -> output1: List[str], output2: Union[int, str]")
    assert "input1" in sig.input_fields
    assert sig.input_fields["input1"].annotation == int
    assert "input2" in sig.input_fields
    assert sig.input_fields["input2"].annotation == Dict[str, int]
    assert "output1" in sig.output_fields
    assert sig.output_fields["output1"].annotation == List[str]
    assert "output2" in sig.output_fields
    assert sig.output_fields["output2"].annotation == Union[int, str]


def test_signature_field_with_constraints():
    class MySignature(Signature):
        inputs: str = InputField()
        outputs1: str = OutputField(min_length=5, max_length=10)
        outputs2: int = OutputField(ge=5, le=10)

    assert "outputs1" in MySignature.output_fields
    output1_constraints = MySignature.output_fields["outputs1"].json_schema_extra["constraints"]

    assert "minimum length: 5" in output1_constraints
    assert "maximum length: 10" in output1_constraints

    assert "outputs2" in MySignature.output_fields
    output2_constraints = MySignature.output_fields["outputs2"].json_schema_extra["constraints"]
    assert "greater than or equal to: 5" in output2_constraints
    assert "less than or equal to: 10" in output2_constraints
