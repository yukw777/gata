import pytest
import torch

from preprocessor import SpacyPreprocessor


@pytest.mark.parametrize(
    "batch,expected_preprocessed,expected_mask",
    [
        (
            ["My name is Peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["my name is peter"],
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1, 1, 1, 1]]).float(),
        ),
        (
            ["My name is Peter", "Is my name David?"],
            torch.tensor([[2, 3, 4, 5, 0], [4, 2, 3, 1, 1]]),
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]).float(),
        ),
    ],
)
def test_spacy_preprocessor(batch, expected_preprocessed, expected_mask):
    sp = SpacyPreprocessor(["<pad>", "<unk>", "my", "name", "is", "peter"])
    preprocessed, mask = sp.preprocess(batch)
    assert preprocessed.equal(expected_preprocessed)
    assert mask.equal(expected_mask)


def test_spacy_preprocessor_load_from_file():
    sp = SpacyPreprocessor.load_from_file("vocabs/word_vocab.txt")
    assert len(sp.word_to_id_dict) == 772
