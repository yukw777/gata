import torch

from preprocessor import SpacyPreprocessor, PAD, UNK
from utils import load_fasttext


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
