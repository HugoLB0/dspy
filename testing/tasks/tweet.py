import os
from functools import lru_cache

import openai
from dotenv import load_dotenv

import aletheia
from aletheia.datasets import HotPotQA

from .base_task import BaseTask


class TweetSignature(aletheia.Signature):
    ("""Given context and a question, answer with a tweet""")

    context = aletheia.InputField()
    question = aletheia.InputField()
    answer = aletheia.OutputField(desc="Yes or No")


class TweetCoT(aletheia.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = aletheia.ChainOfThought(TweetSignature)

    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)


class MultiHopTweet(aletheia.Module):
    def __init__(self, passages_per_hop):
        super().__init__()
        self.retrieve = aletheia.Retrieve(k=passages_per_hop)
        self.generate_query = aletheia.ChainOfThought("context ,question->search_query")
        self.generate_answer = TweetCoT()

    def forward(self, question):
        context = []
        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages
        return aletheia.Prediction(
            context=context,
            answer=self.generate_answer(context=context, question=question).answer,
        )


# Define the signature for automatic assessments.
class Assess(aletheia.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = aletheia.InputField(desc="ignore if N/A")
    assessed_text = aletheia.InputField()
    assessment_question = aletheia.InputField()
    assessment_answer = aletheia.OutputField(desc="Yes or No")


@lru_cache
def load_models():
    load_dotenv()  # This will load the .env file's variables

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai.api_base = os.environ.get("OPENAI_API_BASE")
    gpt4T = aletheia.OpenAI(model="gpt-3.5-turbo-1106", max_tokens=1000, model_type="chat")
    retrieve = aletheia.Retrieve(k=5)
    return gpt4T, retrieve


METRIC = None


def metric(gold, pred, trace=None):

    gpt4T, retrieve = load_models()

    question, answer, tweet = gold.question, gold.answer, pred.answer
    context = retrieve(question).passages

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    correct = (
        f"The text above is should answer `{question}`. The gold answer is `{answer}`."
    )
    correct = f"{correct} Does the assessed text above contain the gold answer?"

    with aletheia.context(lm=gpt4T):
        faithful = aletheia.Predict(Assess)(
            context=context, assessed_text=tweet, assessment_question=faithful
        )
        correct = aletheia.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=correct
        )
        engaging = aletheia.Predict(Assess)(
            context="N/A", assessed_text=tweet, assessment_question=engaging
        )

    correct, engaging, faithful = (
        m.assessment_answer.split()[0].lower() == "yes"
        for m in [correct, engaging, faithful]
    )
    score = (correct + engaging + faithful) if correct and (len(tweet) <= 280) else 0

    if METRIC is not None:
        if METRIC == "correct":
            return correct
        if METRIC == "engaging":
            return engaging
        if METRIC == "faithful":
            return faithful

    if trace is not None:
        return score >= 3
    return score / 3.0


class TweetTask(BaseTask):
    def __init__(self):

        # Load the dataset.
        dataset = HotPotQA(
            train_seed=1,
            train_size=500,
            eval_seed=2023,
            dev_size=200,
            test_size=0,
            keep_details=True,
        )

        # Tell aletheia that the 'question' field is the input. Any other fields are labels and/or metadata.
        self.trainset = [
            x.without("id", "type").with_inputs("question") for x in dataset.train
        ]
        self.testset = [
            x.without("id", "type").with_inputs("question") for x in dataset.dev
        ]

        self.trainset = [x.with_inputs("question") for x in dataset.train]
        self.testset = [x.with_inputs("question") for x in dataset.dev]

        self.metric = metric

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, TEST_NUM=100)

    def get_program(self):
        return MultiHopTweet(
            passages_per_hop=3
        )  # TODO: make it so we can specify # of passages

    def get_metric(self):
        return self.metric
