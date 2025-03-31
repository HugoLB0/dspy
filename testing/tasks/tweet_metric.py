import os
import uuid
from functools import lru_cache

import openai
from dotenv import load_dotenv
from tqdm import tqdm

import aletheia
from aletheia import Example
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
    gpt3T = aletheia.OpenAI(model="gpt-3.5-turbo-1106", max_tokens=1000, model_type="chat")
    gpt4T = aletheia.OpenAI(model="gpt-4-1106-preview", max_tokens=1000, model_type="chat")
    retrieve = aletheia.Retrieve(k=5)
    return gpt3T, gpt4T, retrieve


METRIC = None


def metric(gold, pred, trace=None):
    gpt3T, gpt4T, retrieve = load_models()

    question, answer, tweet, context = (
        gold.question,
        gold.answer,
        gold.tweet,
        gold.context,
    )
    score_pred = pred.score

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    correct = (
        f"The text above is should answer `{question}`. The gold answer is `{answer}`."
    )
    correct = f"{correct} Does the assessed text above contain the gold answer?"

    with aletheia.context(lm=gpt3T):  # TODO Update to GPT4
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

    return 1 - abs(
        score - score_pred
    )  # We want a score we can maximize, so take the negative L1 norm and add 1


class TweetMetric(aletheia.Module):
    def __init__(self):
        super().__init__()
        self.engaging = aletheia.Predict(Assess)
        self.faithful = aletheia.Predict(Assess)
        self.correct = aletheia.Predict(Assess)

    def forward(self, tweet, context, question, answer):
        engaging = "Does the assessed text make for a self-contained, engaging tweet?"
        faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
        correct = f"The text above is should answer `{question}`. The gold answer is `{answer}`."
        correct = f"{correct} Does the assessed text above contain the gold answer?"

        faithful = self.faithful(
            context=context, assessed_text=tweet, assessment_question=faithful
        )
        correct = self.correct(
            context="N/A", assessed_text=tweet, assessment_question=correct
        )
        engaging = self.engaging(
            context="N/A", assessed_text=tweet, assessment_question=engaging
        )

        correct, engaging, faithful = (
            m.assessment_answer.split()[0].lower() == "yes"
            for m in [correct, engaging, faithful]
        )
        score = (
            (correct + engaging + faithful) if correct and (len(tweet) <= 280) else 0
        )

        return aletheia.Prediction(score=score / 3.0)


class TweetMetricTask(BaseTask):
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
        trainset_temp = [
            x.without("id", "type").with_inputs("question") for x in dataset.train
        ]
        devset_temp = [
            x.without("id", "type").with_inputs("question") for x in dataset.dev
        ]
        self.trainset = []
        self.testset = []

        gpt3T, gpt4T, retrieve = load_models()

        with aletheia.context(lm=gpt3T):
            for ex in tqdm(trainset_temp, desc="Preprocessing Trainset"):
                context = retrieve(ex.question).passages
                question = ex.question
                answer = ex.answer
                tweet = MultiHopTweet(passages_per_hop=3)(ex.question).answer
                example = {
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "tweet": tweet,
                }

                self.trainset.append(
                    Example(
                        **example, aletheia_uuid=str(uuid.uuid4()), aletheia_split="train"
                    ).with_inputs("context", "question", "answer", "tweet")
                )

            for ex in tqdm(devset_temp, desc="Preprocessing Devset"):
                context = retrieve(ex.question).passages
                question = ex.question
                answer = ex.answer
                tweet = MultiHopTweet(passages_per_hop=3)(ex.question).answer
                example = {
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "tweet": tweet,
                }

                self.testset.append(
                    Example(
                        **example, aletheia_uuid=str(uuid.uuid4()), aletheia_split="dev"
                    ).with_inputs("context", "question", "answer", "tweet")
                )

        self.metric = metric

        self.set_splits(TRAIN_NUM=100, DEV_NUM=100, TEST_NUM=100)

    def get_program(self):
        return TweetMetric()

    def get_metric(self):
        return self.metric
