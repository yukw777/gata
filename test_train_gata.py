import pytest
import torch
import torch.nn as nn

from train_gata import (
    request_infos_for_train,
    request_infos_for_eval,
    get_game_dir,
    GATADoubleDQN,
)
from preprocessor import PAD, UNK, BOS, EOS
from utils import increasing_mask


def test_request_infos_for_train():
    infos = request_infos_for_train()
    assert infos.admissible_commands is True
    assert infos.description is False
    assert infos.location is False
    assert infos.facts is False
    assert infos.last_action is False
    assert infos.game is True


def test_request_infos_for_eval():
    infos = request_infos_for_eval()
    assert infos.admissible_commands is True
    assert infos.description is True
    assert infos.location is True
    assert infos.facts is True
    assert infos.last_action is True
    assert infos.game is True


@pytest.mark.parametrize(
    "base_dir_path,dataset,difficulty_level,training_size,expected_game_dir",
    [
        ("base_dir", "train", 1, 1, "base_dir/train_1/difficulty_level_1"),
        ("base_dir", "train", 2, 10, "base_dir/train_10/difficulty_level_2"),
        ("base_dir", "valid", 10, None, "base_dir/valid/difficulty_level_10"),
        ("base_dir", "test", 20, None, "base_dir/test/difficulty_level_20"),
    ],
)
def test_get_game_dir(
    base_dir_path, dataset, difficulty_level, training_size, expected_game_dir
):
    assert (
        get_game_dir(
            base_dir_path, dataset, difficulty_level, training_size=training_size
        )
        == expected_game_dir
    )


def test_gata_double_dqn_default_init():
    gata_ddqn = GATADoubleDQN()

    # train_env is initialized with the test games
    assert len(gata_ddqn.train_env.gamefiles) == 2
    assert gata_ddqn.train_env.request_infos == request_infos_for_train()
    assert gata_ddqn.train_env.batch_size == gata_ddqn.hparams.game_batch_size
    assert gata_ddqn.train_env.spec.id.split("-")[1] == "train"

    # val_env is initialized with the test games
    assert len(gata_ddqn.val_env.gamefiles) == 2
    assert gata_ddqn.val_env.request_infos == request_infos_for_eval()
    assert gata_ddqn.val_env.batch_size == gata_ddqn.hparams.game_batch_size
    assert gata_ddqn.val_env.spec.id.split("-")[1] == "val"

    # test_env is initialized with the test games
    assert len(gata_ddqn.test_env.gamefiles) == 2
    assert gata_ddqn.test_env.request_infos == request_infos_for_eval()
    assert gata_ddqn.test_env.batch_size == gata_ddqn.hparams.game_batch_size
    assert gata_ddqn.test_env.spec.id.split("-")[1] == "test"

    # default words
    default_word_vocab = [PAD, UNK, BOS, EOS]
    assert gata_ddqn.preprocessor.word_vocab == default_word_vocab
    assert gata_ddqn.graph_updater.word_embeddings[0].weight.size() == (
        len(default_word_vocab),
        gata_ddqn.hparams.word_emb_dim,
    )

    # default node_vocab = ['node']
    assert gata_ddqn.graph_updater.node_name_word_ids.size() == (
        len(gata_ddqn.node_vocab),
        1,
    )
    assert gata_ddqn.graph_updater.node_name_mask.size() == (
        len(gata_ddqn.node_vocab),
        1,
    )

    # default relation_vocab = ['relation', 'relation reverse']
    assert gata_ddqn.graph_updater.rel_name_word_ids.size() == (
        len(gata_ddqn.relation_vocab),
        2,
    )
    assert gata_ddqn.graph_updater.rel_name_mask.size() == (
        len(gata_ddqn.relation_vocab),
        2,
    )

    # main action selector is train mode
    assert gata_ddqn.action_selector.training

    # target action selector is in train mode
    assert gata_ddqn.target_action_selector.training
    # and frozen
    for param in gata_ddqn.target_action_selector.parameters():
        assert param.requires_grad is False

    # main and target action selectors should be initialized to be the same
    for main, target in zip(
        gata_ddqn.action_selector.parameters(),
        gata_ddqn.target_action_selector.parameters(),
    ):
        assert main.equal(target)

    # graph updater is in eval mode
    assert not gata_ddqn.graph_updater.training
    # and frozen
    for param in gata_ddqn.graph_updater.parameters():
        assert param.requires_grad is False


def test_gata_double_dqn_update_target_action_selector():
    gata_ddqn = GATADoubleDQN()
    # scramble layers in the main action selector and update
    gata_ddqn.action_selector.node_name_word_ids.fill_(42)
    gata_ddqn.action_selector.node_embeddings = nn.Embedding(
        gata_ddqn.num_nodes, gata_ddqn.hparams.node_emb_dim
    )

    # make sure the weights are the same after updating
    gata_ddqn.update_target_action_selector()
    for main, target in zip(
        gata_ddqn.action_selector.parameters(),
        gata_ddqn.target_action_selector.parameters(),
    ):
        assert main.equal(target)


@pytest.mark.parametrize(
    "batch_size,obs_len,num_action_cands,action_cand_len",
    [(1, 5, 4, 10), (3, 6, 5, 12)],
)
def test_gata_double_dqn_forward(
    batch_size,
    obs_len,
    num_action_cands,
    action_cand_len,
):
    gata_ddqn = GATADoubleDQN()
    action_scores, action_mask = gata_ddqn(
        torch.randint(gata_ddqn.num_words, (batch_size, obs_len)),
        increasing_mask(batch_size, obs_len),
        torch.rand(
            batch_size,
            gata_ddqn.num_relations,
            gata_ddqn.num_nodes,
            gata_ddqn.num_nodes,
        ),
        torch.randint(
            gata_ddqn.num_words, (batch_size, num_action_cands, action_cand_len)
        ),
        increasing_mask(batch_size * num_action_cands, action_cand_len).view(
            batch_size, num_action_cands, action_cand_len
        ),
    )
    assert action_scores.size() == (batch_size, num_action_cands)
    assert action_mask.size() == (batch_size, num_action_cands)


@pytest.mark.parametrize(
    "action_scores,action_mask,actions_idx,expected",
    [
        (
            torch.tensor([[1.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([0]),
            torch.tensor([1.0]),
        ),
        (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([1]),
            torch.tensor([2.0]),
        ),
        (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[1.0, 2.0], [2.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            torch.tensor([1, 0]),
            torch.tensor([0.0, 2.0]),
        ),
    ],
)
def test_gata_double_dqn_get_q_values(
    action_scores, action_mask, actions_idx, expected
):
    assert GATADoubleDQN.get_q_values(action_scores, action_mask, actions_idx).equal(
        expected
    )


@pytest.mark.parametrize(
    "batch_size,obs_len,num_action_cands,action_cand_len",
    [
        (1, 8, 5, 3),
        (3, 12, 8, 4),
    ],
)
def test_gata_double_dqn_training_step(
    batch_size, obs_len, num_action_cands, action_cand_len
):
    gata_ddqn = GATADoubleDQN()
    # Note: batch_idx is not used
    assert (
        gata_ddqn.training_step(
            {
                "obs_word_ids": torch.randint(4, (batch_size, obs_len)),
                "obs_mask": torch.randint(2, (batch_size, obs_len), dtype=torch.float),
                "current_graph": torch.rand(
                    batch_size,
                    gata_ddqn.num_relations,
                    gata_ddqn.num_nodes,
                    gata_ddqn.num_nodes,
                ),
                "action_cand_word_ids": torch.randint(
                    4, (batch_size, num_action_cands, action_cand_len)
                ),
                "action_cand_mask": torch.randint(
                    2,
                    (batch_size, num_action_cands, action_cand_len),
                    dtype=torch.float,
                ),
                "actions_idx": torch.randint(num_action_cands, (batch_size,)),
                "rewards": torch.rand(batch_size),
                "next_obs_word_ids": torch.randint(4, (batch_size, obs_len)),
                "next_obs_mask": torch.randint(
                    2, (batch_size, obs_len), dtype=torch.float
                ),
                "next_graph": torch.rand(
                    batch_size,
                    gata_ddqn.num_relations,
                    gata_ddqn.num_nodes,
                    gata_ddqn.num_nodes,
                ),
                "next_action_cand_word_ids": torch.randint(
                    4, (batch_size, num_action_cands, action_cand_len)
                ),
                "next_action_cand_mask": torch.randint(
                    2,
                    (batch_size, num_action_cands, action_cand_len),
                    dtype=torch.float,
                ),
            },
            0,
        ).ndimension()
        == 0
    )
