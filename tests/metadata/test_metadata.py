import aletheia
import re


def test_metadata():
    assert aletheia.__name__ == "aletheia"
    assert re.match(r"\d+\.\d+\.\d+", aletheia.__version__)
    assert aletheia.__author__ == "Omar Khattab"
    assert aletheia.__author_email__ == "okhattab@stanford.edu"
    assert aletheia.__url__ == "https://github.com/stanfordnlp/aletheia"
    assert aletheia.__description__ == "aletheia"
