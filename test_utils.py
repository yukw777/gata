import torch
import pytest

from preprocessor import SpacyPreprocessor, PAD, UNK
from utils import load_fasttext, masked_mean, generate_square_subsequent_mask


def test_load_fasttext():
    preprocessor = SpacyPreprocessor([PAD, UNK, "my", "name", "is", "peter"])
    emb = load_fasttext("test-data/test-fasttext.vec", preprocessor)
    word_ids, _ = preprocessor.preprocess_tokenized(
        [
            ["hi", "there", "what's", "your", "name"],
            ["my", "name", "is", "peter"],
        ]
    )
    embedded = emb(word_ids)
    # OOVs
    assert embedded[0, :4].equal(
        emb(torch.tensor(preprocessor.unk_id)).unsqueeze(0).expand(4, -1)
    )
    # name
    assert embedded[0, 4].equal(emb(torch.tensor(3)))
    # my name is peter
    assert embedded[1, :4].equal(emb(torch.tensor([2, 3, 4, 5])))
    # pad, should be zero
    assert embedded[1, 4].equal(torch.zeros(300))


def test_masked_mean():
    batched_input = torch.tensor(
        [
            [
                [1, 2, 300],
                [300, 100, 200],
                [3, 4, 100],
            ],
            [
                [300, 100, 200],
                [6, 2, 300],
                [10, 4, 100],
            ],
        ]
    ).float()
    batched_mask = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
        ]
    ).float()
    assert masked_mean(batched_input, batched_mask).equal(
        torch.tensor(
            [
                [2, 3, 200],
                [8, 3, 200],
            ]
        ).float()
    )


@pytest.mark.parametrize("size", [1, 3, 5, 7])
def test_generate_subsequent_mask(size):
    mask = generate_square_subsequent_mask(size)
    # assert that the sum of tril and triu is the original mask
    assert mask.equal(torch.tril(mask) + torch.triu(mask, diagonal=1))
