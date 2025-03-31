from typing import Any

import pydantic


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated signature.
    For example, if you have the following signature:

    ```
    class MySignature(aletheia.Signature):
        question: str = aletheia.InputField()
        history: aletheia.History = aletheia.InputField()
        answer: str = aletheia.OutputField()
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Example:
        ```
        import aletheia

        aletheia.settings.configure(lm=aletheia.LM("openai/gpt-4o-mini"))

        class MySignature(aletheia.Signature):
            question: str = aletheia.InputField()
            history: aletheia.History = aletheia.InputField()
            answer: str = aletheia.OutputField()

        history = aletheia.History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = aletheia.Predict(MySignature)
        outputs = predict(question="What is the capital of France?", history=history)
        ```

    Example of capturing the conversation history:
        ```
        import aletheia

        aletheia.settings.configure(lm=aletheia.LM("openai/gpt-4o-mini"))

        class MySignature(aletheia.Signature):
            question: str = aletheia.InputField()
            history: aletheia.History = aletheia.InputField()
            answer: str = aletheia.OutputField()

        predict = aletheia.Predict(MySignature)
        outputs = predict(question="What is the capital of France?")
        history = aletheia.History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = predict(question="Are you sure?", history=history)
        ```
    """

    messages: list[dict[str, Any]]

    model_config = {
        "frozen": True,
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
    }
