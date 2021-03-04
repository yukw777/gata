import torch

from typing import List, Tuple
from spacy.lang.en import English

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"


class SpacyPreprocessor:
    def __init__(self, word_vocab: List[str]) -> None:
        super().__init__()
        self.tokenizer = English().tokenizer
        self.word_vocab = word_vocab
        self.word_to_id_dict = {w: i for i, w in enumerate(word_vocab)}
        self.pad_id = self.word_to_id_dict[PAD]
        self.unk_id = self.word_to_id_dict[UNK]

    def id_to_word(self, word_id: int) -> str:
        return self.word_vocab[word_id]

    def word_to_id(self, word: str) -> int:
        return self.word_to_id_dict.get(word, self.unk_id)

    def tokenize(self, s: str) -> List[str]:
        return [t.text.lower() for t in self.tokenizer(s)]

    def pad(self, unpadded_batch: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # return padded tensor and corresponding mask
        max_len = max(len(word_ids) for word_ids in unpadded_batch)
        return (
            torch.tensor(
                [
                    word_ids + [0] * (max_len - len(word_ids))
                    for word_ids in unpadded_batch
                ]
            ),
            torch.tensor(
                [
                    [1] * len(word_ids) + [0] * (max_len - len(word_ids))
                    for word_ids in unpadded_batch
                ]
            ).float(),
        )

    def preprocess_tokenized(
        self, tokenized_batch: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pad(
            [
                [self.word_to_id(token) for token in tokenized]
                for tokenized in tokenized_batch
            ]
        )

    def preprocess(self, batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess_tokenized([self.tokenize(s) for s in batch])

    def decode(self, batch: List[List[int]]) -> List[str]:
        return [
            " ".join(
                [
                    self.id_to_word(word_id)
                    for word_id in word_ids
                    if word_id != self.pad_id
                ]
            )
            for word_ids in batch
        ]

    @classmethod
    def load_from_file(cls, word_vocab_path: str) -> "SpacyPreprocessor":
        with open(word_vocab_path, "r") as f:
            word_vocab = [word.strip() for word in f]
        return cls(word_vocab)
