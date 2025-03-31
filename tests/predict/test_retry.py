# import functools

# import pydantic

# import aletheia
# from aletheia.primitives.assertions import assert_transform_module, backtrack_handler
# from aletheia.utils import DummyLM


# def test_retry_simple():
#     predict = aletheia.Predict("question -> answer")
#     retry_module = aletheia.Retry(predict)

#     # Test Retry has created the correct new signature
#     for field in predict.signature.output_fields:
#         assert f"past_{field}" in retry_module.new_signature.input_fields
#     assert "feedback" in retry_module.new_signature.input_fields

#     lm = DummyLM([{"answer": "blue"}])
#     aletheia.settings.configure(lm=lm)
#     result = retry_module.forward(
#         question="What color is the sky?",
#         past_outputs={"answer": "red"},
#         feedback="Try harder",
#     )
#     assert result.answer == "blue"


# def test_retry_forward_with_feedback():
#     # First we make a mistake, then we fix it
#     lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
#     aletheia.settings.configure(lm=lm, trace=[])

#     class SimpleModule(aletheia.Module):
#         def __init__(self):
#             super().__init__()
#             self.predictor = aletheia.Predict("question -> answer")

#         def forward(self, **kwargs):
#             result = self.predictor(**kwargs)
#             print(f"SimpleModule got {result.answer=}")
#             aletheia.Suggest(result.answer == "blue", "Please think harder")
#             return result

#     program = SimpleModule()
#     program = assert_transform_module(
#         program.map_named_predictors(aletheia.Retry),
#         functools.partial(backtrack_handler, max_backtracks=1),
#     )

#     result = program(question="What color is the sky?")

#     assert result.answer == "blue"


# # def test_retry_forward_with_typed_predictor():
# #     # First we make a mistake, then we fix it
# #     lm = DummyLM([{"output": '{"answer":"red"}'}, {"output": '{"answer":"blue"}'}])
# #     aletheia.settings.configure(lm=lm, trace=[])

# #     class AnswerQuestion(aletheia.Signature):
# #         """Answer questions with succinct responses."""

# #         class Input(pydantic.BaseModel):
# #             question: str

# #         class Output(pydantic.BaseModel):
# #             answer: str

# #         input: Input = aletheia.InputField()
# #         output: Output = aletheia.OutputField()

# #     class QuestionAnswerer(aletheia.Module):
# #         def __init__(self):
# #             super().__init__()
# #             self.answer_question = aletheia.TypedPredictor(AnswerQuestion)

# #         def forward(self, **kwargs):
# #             result = self.answer_question(input=AnswerQuestion.Input(**kwargs)).output
# #             aletheia.Suggest(result.answer == "blue", "Please think harder")
# #             return result

# #     program = QuestionAnswerer()
# #     program = assert_transform_module(
# #         program.map_named_predictors(aletheia.Retry),
# #         functools.partial(backtrack_handler, max_backtracks=1),
# #     )

# #     result = program(question="What color is the sky?")

# #     assert result.answer == "blue"
