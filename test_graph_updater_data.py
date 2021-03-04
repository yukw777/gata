import torch

from graph_updater_data import GraphUpdaterDataset, GraphUpdaterObsGenDataModule


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
    prepared_batch, episode_mask = data_module.prepare_batch(list(data_module.train))

    # check episode_mask
    assert episode_mask.equal(
        torch.Tensor(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        ).float()
    )
    assert len(prepared_batch) == 7
    for i, episode in enumerate(prepared_batch):
        # if episode_mask == 0, all the word ids should be 0 too. (pad_id)
        assert (episode["obs_word_ids"].sum(dim=1) == 0).equal(episode_mask[:, i] == 0)
        assert (episode["prev_action_word_ids"].sum(dim=1) == 0).equal(
            episode_mask[:, i] == 0
        )
        assert (episode["groundtruth_obs_word_ids"].sum(dim=1) == 0).equal(
            episode_mask[:, i] == 0
        )

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
