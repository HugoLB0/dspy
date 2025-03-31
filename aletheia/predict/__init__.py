from aletheia.predict.aggregation import majority
from aletheia.predict.best_of_n import BestOfN
from aletheia.predict.chain_of_thought import ChainOfThought
from aletheia.predict.chain_of_thought_with_hint import ChainOfThoughtWithHint
from aletheia.predict.knn import KNN
from aletheia.predict.multi_chain_comparison import MultiChainComparison
from aletheia.predict.predict import Predict
from aletheia.predict.program_of_thought import ProgramOfThought
from aletheia.predict.react import ReAct, Tool
from aletheia.predict.refine import Refine
from aletheia.predict.parallel import Parallel

__all__ = [
    "majority",
    "BestOfN",
    "ChainOfThought",
    "ChainOfThoughtWithHint",
    "KNN",
    "MultiChainComparison",
    "Predict",
    "ProgramOfThought",
    "ReAct",
    "Refine",
    "Tool",
    "Parallel",
]
