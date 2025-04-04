import unittest
import uuid

import pandas as pd

from aletheia import Example
from aletheia.datasets.dataset import Dataset

dummy_data = """content,question,answer
"This is content 1","What is this?","This is answer 1"
"This is content 2","What is that?","This is answer 2"
"""

with open("dummy.csv", "w") as file:
    file.write(dummy_data)


class CSVDataset(Dataset):
    def __init__(self, file_path, input_keys=None, *args, **kwargs) -> None:
        super().__init__(input_keys=input_keys, *args, **kwargs)
        df = pd.read_csv(file_path)
        data = df.to_dict(orient="records")
        self._train = [
            Example(**record, aletheia_uuid=str(uuid.uuid4()), aletheia_split="train").with_inputs(*input_keys)
            for record in data[:1]
        ]
        self._dev = [
            Example(**record, aletheia_uuid=str(uuid.uuid4()), aletheia_split="dev").with_inputs(*input_keys)
            for record in data[1:2]
        ]


class TestCSVDataset(unittest.TestCase):
    def test_input_keys(self):
        dataset = CSVDataset("dummy.csv", input_keys=["content", "question"])
        self.assertIsNotNone(dataset.train)

        for example in dataset.train:
            print(example)
            inputs = example.inputs()
            print(f"Example inputs: {inputs}")
            self.assertIsNotNone(inputs)
            self.assertIn("content", inputs)
            self.assertIn("question", inputs)
            self.assertEqual(set(example._input_keys), {"content", "question"})


if __name__ == "__main__":
    unittest.main()
