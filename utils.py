import torch
import torch.nn as nn

from preprocessor import SpacyPreprocessor


def load_fasttext(fname: str, preprocessor: SpacyPreprocessor) -> nn.Embedding:
    with open(fname, "r") as f:
        _, emb_dim = map(int, f.readline().split())

        data = {}
        for line in f:
            tokens = line.rstrip().split(" ")
            data[tokens[0]] = map(float, tokens[1:])
    # embedding for pad is initalized to 0
    # embeddings for OOVs are randomly initialized from N(0, 1)
    emb = nn.Embedding(
        len(preprocessor.word_to_id_dict), emb_dim, padding_idx=preprocessor.pad_id
    )
    for word, i in preprocessor.word_to_id_dict.items():
        if word in data:
            emb.weight[i] = torch.tensor(list(data[word]))
    return emb


def masked_mean(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    input: (batch, seq_len, hidden_dim)
    mask: (batch, seq_len)
    output: (batch, hidden_dim)
    """
    return (input * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
