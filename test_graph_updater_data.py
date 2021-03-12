import torch

from graph_updater_data import GraphUpdaterDataset, GraphUpdaterObsGenDataModule
from preprocessor import BOS, EOS


def test_graph_updater_dataset():
    dataset = GraphUpdaterDataset("test-data/test-data.json")
    assert len(dataset) == 3
    for datapoint in dataset:
        assert type(datapoint) == list
        for step in datapoint:
            assert type(step["game"]) == str
            assert type(step["step"]) == list
            assert type(step["observation"]) == str
            assert type(step["previous_action"]) == str


def test_graph_updater_obs_gen_data_module_prepare_batch():
    data_module = GraphUpdaterObsGenDataModule(
        "test-data/test-data.json",
        3,
        1,
        "test-data/test-data.json",
        3,
        1,
        "test-data/test-data.json",
        3,
        1,
        "vocabs/word_vocab.txt",
    )
    data_module.setup()
    prepared_batch = data_module.prepare_batch(list(data_module.train))

    # expected step masks
    expected_step_masks = [
        torch.tensor([1, 1, 1]).float(),
        torch.tensor([0, 1, 1]).float(),
        torch.tensor([0, 1, 1]).float(),
        torch.tensor([0, 1, 1]).float(),
        torch.tensor([0, 0, 1]).float(),
        torch.tensor([0, 0, 1]).float(),
        torch.tensor([0, 0, 1]).float(),
    ]
    assert len(prepared_batch) == 7
    for episode, expected_step_mask in zip(prepared_batch, expected_step_masks):
        assert episode["step_mask"].equal(expected_step_mask)
        # if step_mask == 0, observations should be [<bos>, <pad>, ...]
        assert (
            episode["obs_word_ids"][episode["step_mask"] == 0][:, 0]
            == data_module.preprocessor.word_to_id(BOS)
        ).all()
        assert (episode["prev_action_word_ids"].sum(dim=1) == 0).equal(
            episode["step_mask"] == 0
        )
        # if episode_mask == 0, ground truth observations should be [<eos>, <pad>, ...]
        assert (
            episode["groundtruth_obs_word_ids"][episode["step_mask"] == 0][:, 0]
            == data_module.preprocessor.word_to_id(EOS)
        ).all()

        # make sure the masks have the right dimensions
        assert episode["obs_mask"].size() == episode["obs_word_ids"].size()
        assert (
            episode["prev_action_mask"].size() == episode["prev_action_word_ids"].size()
        )

        # the sum of mask should equal the number of nonzero word ids
        assert (
            torch.sum(episode["obs_word_ids"] != 0, dim=1)
            .float()
            .equal(episode["obs_mask"].sum(dim=1))
        )
