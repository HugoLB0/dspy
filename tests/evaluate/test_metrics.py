# FILEPATH: /Users/ahle/repos/aletheia/tests/evaluate/test_metrics.py

import aletheia
from aletheia.evaluate.metrics import answer_exact_match
from aletheia.predict import Predict

def test_answer_exact_match_string():
    example = aletheia.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)

def test_answer_exact_match_list():
    example = aletheia.Example(
        question="What is 1+1?",
        answer=["2", "two"],
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)

def test_answer_exact_match_no_match():
    example = aletheia.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)