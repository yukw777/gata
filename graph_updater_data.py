import json
import torch
import pytorch_lightning as pl

from typing import Optional, Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from hydra.utils import to_absolute_path

from preprocessor import SpacyPreprocessor, PAD, BOS, EOS


class GraphUpdaterDataset(Dataset):
    """
    No preprocessing, raw data.
    Each data point is a list of dictionaries of the following form:
    [{
        'game': 'game_id',
        'step': [0, 1],
        'observation': 'text observation...',
        'previous_action': 'previous action...',
    }, ...]
    """

    def __init__(self, filename: str) -> None:
        super().__init__()
        with open(filename, "r") as f:
            self.data: List[List[Dict[str, Any]]] = json.load(f)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class GraphUpdaterObsGenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        train_batch_size: int,
        train_num_workers: int,
        val_path: str,
        val_batch_size: int,
        val_num_workers: int,
        test_path: str,
        test_batch_size: int,
        test_num_workers: int,
        word_vocab_file: str,
    ) -> None:
        super().__init__()
        self.train_path = to_absolute_path(train_path)
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.val_path = to_absolute_path(val_path)
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers
        self.test_path = to_absolute_path(test_path)
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers

        with open(to_absolute_path(word_vocab_file), "r") as f:
            word_vocab = [word.strip() for word in f.readlines()]
        self.preprocessor = SpacyPreprocessor(word_vocab)

    def prepare_data(self) -> None:  # type: ignore
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train = GraphUpdaterDataset(self.train_path)
            self.valid = GraphUpdaterDataset(self.val_path)

        if stage == "test" or stage is None:
            self.test = GraphUpdaterDataset(self.test_path)

    def prepare_batch(
        self, batch: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        This is a bit tricky, b/c we have to pad the episodes as well as the
        observation and previous action strings within each episode. The original
        GATA code padded episodes with '<pad>' strings, and that's what we do here.

        Following the original GATA code, '<bos>' is prepended to observation strings,
        and '<eos>' is appeneded to ground-truth observation strings.

        returns: [
            {
                'obs_word_ids': tensor of shape (batch, obs_len),
                'obs_mask': tensor of shape (batch, obs_len),
                'prev_action_word_ids': tensor of shape (batch, prev_action_len),
                'prev_action_mask': tensor of shape (batch, prev_action_len),
                'groundtruth_obs_word_ids': tensor of shape (batch, obs_len),
                'step_mask': tensor of shape (batch),
            },
            ...
        ]
        """
        episode_lens = [len(episode) for episode in batch]
        max_episode_len = max(episode_lens)

        prepared_batch: List[Dict[str, torch.Tensor]] = []
        for i in range(max_episode_len):
            # calculate step_mask first
            step_mask = torch.tensor(
                [1 if i < episode_len else 0 for episode_len in episode_lens]
            ).float()

            # Collect the observations and prev action of the i'th episode
            # and batchify them.
            # If the length of an episode is shorter than max_episode_len,
            # just add an empty string.
            # They're already tokenized, so split() is sufficient.
            episode_padded_obs = [
                episode[i]["observation"] if i < len(episode) else ""
                for episode in batch
            ]

            # we add BOS and EOS even if the observation should be masked
            # to prevent nan from multiheaded attention which happens due to a bug
            # https://github.com/pytorch/pytorch/issues/41508
            obs_word_ids, obs_mask = self.preprocessor.preprocess_tokenized(
                [[BOS] + padded_obs.split() for padded_obs in episode_padded_obs]
            )
            groundtruth_obs_word_ids, _ = self.preprocessor.preprocess_tokenized(
                [padded_obs.split() + [EOS] for padded_obs in episode_padded_obs]
            )

            episode_padded_prev_actions = [
                episode[i]["previous_action"] if i < len(episode) else PAD
                for episode in batch
            ]
            (
                prev_action_word_ids,
                prev_action_mask,
            ) = self.preprocessor.preprocess_tokenized(
                [
                    padded_prev_action.split()
                    for padded_prev_action in episode_padded_prev_actions
                ]
            )

            prepared_batch.append(
                {
                    "obs_word_ids": obs_word_ids,
                    "obs_mask": obs_mask,
                    "prev_action_word_ids": prev_action_word_ids,
                    "prev_action_mask": prev_action_mask,
                    "groundtruth_obs_word_ids": groundtruth_obs_word_ids,
                    "step_mask": step_mask,
                }
            )

        return prepared_batch

    def train_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.prepare_batch,
            pin_memory=True,
            num_workers=self.train_num_workers,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.valid,
            batch_size=self.val_batch_size,
            collate_fn=self.prepare_batch,
            pin_memory=True,
            num_workers=self.val_num_workers,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore
        return DataLoader(
            self.test,
            batch_size=self.val_batch_size,
            collate_fn=self.prepare_batch,
            pin_memory=True,
            num_workers=self.val_num_workers,
        )
