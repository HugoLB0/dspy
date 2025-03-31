from dataclasses import dataclass

from pydantic import BaseModel

import aletheia
from aletheia.predict import react
from aletheia.utils.dummies import DummyLM, dummy_rm
import litellm

# def test_example_no_tools():
#     # Create a simple dataset which the model will use with the Retrieve tool.
#     lm = DummyLM(
#         [
#             {"Thought_1": "Initial thoughts", "Action_1": "Finish[blue]"},
#         ]
#     )
#     aletheia.settings.configure(lm=lm, rm=dummy_rm())

#     program = aletheia.ReAct("question -> answer")

#     # Check default tools
#     assert isinstance(program.tools["Finish"], aletheia.Example)

#     # Call the ReAct module on a particular input
#     question = "What is the color of the sky?"
#     result = program(question=question)
#     assert result.answer == "blue"


# def test_example_search():
#     # Create a simple dataset which the model will use with the Retrieve tool.
#     lm = DummyLM(
#         [
#             {"Thought_1": "Initial thoughts", "Action_1": "Search[the color of the sky]"},
#             {"Thought_2": "More thoughts", "Action_2": "Finish[blue]\n\n"},
#         ]
#     )
#     rm = dummy_rm(
#         [
#             "We all know the color of the sky is blue.",
#             "Something about the sky colors",
#             "This sentence is completely irellevant to answer the question.",
#             "Let's add some more sentences to act as summy passages.",
#             "Let's add some more sentences to act as summy passages.",
#             "Let's add some more sentences to act as summy passages.",
#         ]
#     )
#     aletheia.settings.configure(lm=lm, rm=rm)

#     program = aletheia.ReAct("question -> answer")

#     # Check default tools
#     assert len(program.tools) == 2
#     assert isinstance(program.tools["Search"], aletheia.Retrieve)
#     assert isinstance(program.tools["Finish"], aletheia.Example)

#     # Call the ReAct module on a particular input
#     question = "What is the color of the sky?"
#     result = program(question=question)
#     assert result.answer == "blue"


# class DummyTool1:
#     name = "Tool1"
#     input_variable = "query"
#     desc = ""
#     num_calls = 0

#     def __call__(self, *args, **kwargs):
#         # test case with no passages attribute
#         assert args[0] == "foo"
#         self.num_calls += 1
#         return "tool 1 output"


# @dataclass
# class DummyOutput:
#     passages: str


# class DummyTool2:
#     name = "Tool2"
#     input_variable = "query"
#     desc = ""
#     num_calls = 0

#     def __call__(self, *args, **kwargs):
#         # test case with passages attribute
#         assert args[0] == "bar"
#         self.num_calls += 1
#         return DummyOutput(passages="tool 2 output")


# def test_custom_tools():
#     lm = DummyLM(
#         [
#             {"Thought_1": "Initial thoughts", "Action_1": "Tool1[foo]"},
#             {"Thought_2": "More thoughts", "Action_2": "Tool2[bar]"},
#             {"Thought_3": "Even more thoughts", "Action_3": "Finish[baz]"},
#         ]
#     )
#     aletheia.settings.configure(lm=lm)

#     tool1 = DummyTool1()
#     tool2 = DummyTool2()
#     program = aletheia.ReAct("question -> answer", tools=[tool1, tool2])

#     question = "What is the color of the sky?"
#     result = program(question=question)
#     assert result.answer == "baz"

#     # each tool should be called only once
#     assert tool1.num_calls == 1
#     assert tool2.num_calls == 1


# def test_signature_instructions():
#     class ExampleSignature(aletheia.Signature):
#         """You are going to generate output based on input."""

#         input = aletheia.InputField()
#         output = aletheia.OutputField()

#     react = aletheia.ReAct(ExampleSignature)

#     assert react.react[0].signature.instructions is not None
#     assert react.react[0].signature.instructions.startswith("You are going to generate output based on input.")


def test_tool_from_function():
    def foo(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = react.Tool(foo)
    assert tool.name == "foo"
    assert tool.desc == "Add two numbers."
    assert tool.args == {"a": {"type": "integer"}, "b": {"type": "integer"}}


def test_tool_from_class():
    class Foo:
        def __init__(self, user_id: str):
            self.user_id = user_id

        def foo(self, a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

    tool = react.Tool(Foo("123").foo)
    assert tool.name == "foo"
    assert tool.desc == "Add two numbers."
    assert tool.args == {"a": {"type": "integer"}, "b": {"type": "integer"}}


def test_tool_calling_with_pydantic_args():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    class InvitationSignature(aletheia.Signature):
        participant_name: str = aletheia.InputField(desc="The name of the participant to invite")
        event_info: CalendarEvent = aletheia.InputField(desc="The information about the event")
        invitation_letter: str = aletheia.OutputField(desc="The invitation letter to be sent to the participant")

    react = aletheia.ReAct(InvitationSignature, tools=[write_invitation_letter])

    lm = DummyLM(
        [
            {
                "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
                "next_tool_name": "write_invitation_letter",
                "next_tool_args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            },
            {
                "next_thought": (
                    "I have successfully written the invitation letter for Alice to the Science Fair. Now "
                    "I can finish the task."
                ),
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "This is a very rigorous reasoning process, trust me bro!",
                "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
            },
        ]
    )
    aletheia.settings.configure(lm=lm)

    outputs = react(
        participant_name="Alice",
        event_info=CalendarEvent(
            name="Science Fair",
            date="Friday",
            participants={"Alice": "female", "Bob": "male"},
        ),
    )
    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."

    expected_trajectory = {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert outputs.trajectory == expected_trajectory


def test_tool_calling_without_typehint():
    def foo(a, b):
        """Add two numbers."""
        return a + b

    react = aletheia.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"next_thought": "I have the sum, now I can finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    aletheia.settings.configure(lm=lm)
    outputs = react(a=1, b=2)

    expected_trajectory = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {
            "a": 1,
            "b": 2,
        },
        "observation_0": 3,
        "thought_1": "I have the sum, now I can finish.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert outputs.trajectory == expected_trajectory


def test_trajectory_truncation():
    # Create a simple tool for testing
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    # Create ReAct instance with our echo tool
    react = aletheia.ReAct("input_text -> output_text", tools=[echo])

    # Mock react.react to simulate multiple tool calls
    call_count = 0

    def mock_react(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            # First 2 calls use the echo tool
            return aletheia.Prediction(
                next_thought=f"Thought {call_count}",
                next_tool_name="echo",
                next_tool_args={"text": f"Text {call_count}"},
            )
        elif call_count == 3:
            # The 3rd call raises context window exceeded error
            raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")
        else:
            # The 4th call finishes
            return aletheia.Prediction(next_thought="Final thought", next_tool_name="finish", next_tool_args={})

    react.react = mock_react
    react.extract = lambda **kwargs: aletheia.Prediction(output_text="Final output")

    # Call forward and get the result
    result = react(input_text="test input")

    # Verify that older entries in the trajectory were truncated
    assert "thought_0" not in result.trajectory
    assert "thought_2" in result.trajectory
    assert result.output_text == "Final output"
