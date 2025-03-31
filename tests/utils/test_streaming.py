import pytest

import aletheia
from aletheia.utils.streaming import StatusMessage, StatusMessageProvider, streaming_response
from ..test_utils.server import litellm_test_server


@pytest.mark.anyio
async def test_streamify_yields_expected_response_chunks(litellm_test_server):
    api_base, _ = litellm_test_server
    lm = aletheia.LM(
        model="openai/aletheia-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    with aletheia.context(lm=lm):

        class TestSignature(aletheia.Signature):
            input_text: str = aletheia.InputField()
            output_text: str = aletheia.OutputField()

        program = aletheia.streamify(aletheia.Predict(TestSignature))
        output_stream1 = program(input_text="Test")
        output_chunks1 = [chunk async for chunk in output_stream1]
        assert len(output_chunks1) > 1
        last_chunk1 = output_chunks1[-1]
        assert isinstance(last_chunk1, aletheia.Prediction)
        assert last_chunk1.output_text == "Hello!"

        output_stream2 = program(input_text="Test")
        output_chunks2 = [chunk async for chunk in output_stream2]
        # Since the input is cached, only one chunk should be
        # yielded containing the prediction
        assert len(output_chunks2) == 1
        last_chunk2 = output_chunks2[-1]
        assert isinstance(last_chunk2, aletheia.Prediction)
        assert last_chunk2.output_text == "Hello!"


@pytest.mark.anyio
async def test_streaming_response_yields_expected_response_chunks(litellm_test_server):
    api_base, _ = litellm_test_server
    lm = aletheia.LM(
        model="openai/aletheia-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    with aletheia.context(lm=lm):

        class TestSignature(aletheia.Signature):
            input_text: str = aletheia.InputField()
            output_text: str = aletheia.OutputField()

        program = aletheia.streamify(aletheia.Predict(TestSignature))
        output_stream_from_program = streaming_response(program(input_text="Test"))
        output_stream_for_server_response = streaming_response(output_stream_from_program)
        output_chunks = [chunk async for chunk in output_stream_for_server_response]
        assert all(chunk.startswith("data: ") for chunk in output_chunks)
        assert 'data: {"prediction":{"output_text":"Hello!"}}\n\n' in output_chunks
        assert output_chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_default_status_streaming():
    class MyProgram(aletheia.Module):
        def __init__(self):
            self.generate_question = aletheia.Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = aletheia.Predict("question->answer")

        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    lm = aletheia.utils.DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with aletheia.context(lm=lm):
        program = aletheia.streamify(MyProgram())
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)

    assert len(status_messages) == 2
    assert status_messages[0].message == "Calling tool generate_question..."
    assert status_messages[1].message == "Tool calling finished! Querying the LLM with tool calling results..."


@pytest.mark.anyio
async def test_custom_status_streaming():
    class MyProgram(aletheia.Module):
        def __init__(self):
            self.generate_question = aletheia.Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = aletheia.Predict("question->answer")

        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    class MyStatusMessageProvider(StatusMessageProvider):
        def tool_start_status_message(self, instance, inputs):
            return f"Tool starting!"

        def tool_end_status_message(self, outputs):
            return "Tool finished!"

        def module_start_status_message(self, instance, inputs):
            if isinstance(instance, aletheia.Predict):
                return "Predict starting!"

    lm = aletheia.utils.DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with aletheia.context(lm=lm):
        program = aletheia.streamify(MyProgram(), status_message_provider=MyStatusMessageProvider())
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)

        assert len(status_messages) == 3
        assert status_messages[0].message == "Tool starting!"
        assert status_messages[1].message == "Tool finished!"
        assert status_messages[2].message == "Predict starting!"
