import pytest

from preprocessor import SpacyPreprocessor


@pytest.mark.parametrize(
    "batch,expected",
    [
        (
            ["My name is Peter"],
            [[2, 3, 4, 5]],
        ),
        (
            ["my name is peter"],
            [[2, 3, 4, 5]],
        ),
        (
            ["My name is Peter", "Is my name David?"],
            [[2, 3, 4, 5, 0], [4, 2, 3, 1, 1]],
        ),
    ],
)
def test_spacy_preprocessor(batch, expected):
    sp = SpacyPreprocessor(["<pad>", "<unk>", "my", "name", "is", "peter"])
    assert sp.preprocess(batch) == expected
