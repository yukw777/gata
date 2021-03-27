import pytest
import torch
import torch.nn as nn
import random

from collections import deque

from train_gata import (
    request_infos_for_train,
    request_infos_for_eval,
    get_game_dir,
    GATADoubleDQN,
    TransitionCache,
    Transition,
)
from agent import EpsilonGreedyAgent
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
    assert gata_ddqn.train_env.batch_size == gata_ddqn.hparams.train_game_batch_size
    assert gata_ddqn.train_env.spec.id.split("-")[1] == "train"

    # val_env is initialized with the test games
    assert len(gata_ddqn.val_env.gamefiles) == 2
    assert gata_ddqn.val_env.request_infos == request_infos_for_eval()
    assert gata_ddqn.val_env.batch_size == gata_ddqn.hparams.eval_game_batch_size
    assert gata_ddqn.val_env.spec.id.split("-")[1] == "val"

    # test_env is initialized with the test games
    assert len(gata_ddqn.test_env.gamefiles) == 2
    assert gata_ddqn.test_env.request_infos == request_infos_for_eval()
    assert gata_ddqn.test_env.batch_size == gata_ddqn.hparams.eval_game_batch_size
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


def test_transition_cache_batch_add():
    t_cache = TransitionCache(3)

    def generate_batch(step, dones):
        return {
            "obs": [f"{i}: step {step} obs" for i in range(3)],
            "batch_action_cands": [
                [
                    f"{i}: step {step} act 1",
                    f"{i}: step {step} act 2",
                    f"{i}: step {step} act 3",
                ]
                for i in range(3)
            ],
            "current_graphs": torch.rand(3, 6, 4, 4),
            "actions_idx": [random.randint(0, 2) for _ in range(3)],
            "rewards": [random.random() for _ in range(3)],
            "dones": dones,
            "next_obs": [f"{i}: step {step} next obs" for i in range(3)],
            "batch_next_action_cands": [
                [
                    f"{i}: step {step} next act 1",
                    f"{i}: step {step} next act 2",
                    f"{i}: step {step} next act 3",
                ]
                for i in range(3)
            ],
            "next_graphs": torch.rand(3, 6, 4, 4),
        }

    def compare_batch_transition(batch, batch_num, transition):
        assert transition.ob == batch["obs"][batch_num]
        assert transition.action_cands == batch["batch_action_cands"][batch_num]
        assert transition.current_graph.equal(batch["current_graphs"][batch_num])
        assert transition.action_id == batch["actions_idx"][batch_num]
        assert transition.reward == batch["rewards"][batch_num]
        assert transition.done == batch["dones"][batch_num]
        assert transition.next_ob == batch["next_obs"][batch_num]
        assert (
            transition.next_action_cands == batch["batch_next_action_cands"][batch_num]
        )
        assert transition.next_graph.equal(batch["next_graphs"][batch_num])

    # add a not done step
    batch_0 = generate_batch(0, [False] * 3)
    t_cache.batch_add(**batch_0)
    for i in range(3):
        assert len(t_cache.cache[i]) == 1
        compare_batch_transition(batch_0, i, t_cache.cache[i][-1])

    # add a done game
    batch_1 = generate_batch(1, [False, True, False])
    t_cache.batch_add(**batch_1)
    for i in range(3):
        assert len(t_cache.cache[i]) == 2
        compare_batch_transition(batch_1, i, t_cache.cache[i][-1])

    # add another done step
    batch_2 = generate_batch(2, [False, True, False])
    t_cache.batch_add(**batch_2)
    for i in range(3):
        if batch_2["dones"][i]:
            # it shouldn't have been added
            assert len(t_cache.cache[i]) == 2
            compare_batch_transition(batch_1, i, t_cache.cache[i][-1])
        else:
            assert len(t_cache.cache[i]) == 3
            compare_batch_transition(batch_2, i, t_cache.cache[i][-1])


@pytest.mark.parametrize("step_size", [1, 3, 5])
@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_transition_cache_get_avg_rewards(batch_size, step_size):
    t_cache = TransitionCache(batch_size)
    batch_rewards = torch.rand(batch_size, step_size)
    for i, rewards in enumerate(batch_rewards.tolist()):
        for reward in rewards:
            t_cache.cache[i].append(Transition(reward=reward))
    for avg, expected in zip(
        t_cache.get_avg_rewards(), batch_rewards.mean(dim=1).tolist()
    ):
        assert pytest.approx(avg) == expected


@pytest.fixture
def eps_greedy_agent():
    gata_double_dqn = GATADoubleDQN(word_vocab_path="vocabs/word_vocab.txt")
    return EpsilonGreedyAgent(
        gata_double_dqn.graph_updater,
        gata_double_dqn.action_selector,
        gata_double_dqn.preprocessor,
        0.1,
        1.0,
        20,
    )


@pytest.fixture
def replay_buffer_gata_double_dqn():
    return GATADoubleDQN(
        max_episodes=30,
        train_game_batch_size=2,
        train_max_episode_steps=5,
        episodes_before_learning=10,
        yield_step_freq=10,
        replay_buffer_capacity=20,
        train_sample_batch_size=4,
    )


@pytest.mark.parametrize(
    "initial_buffer,batch_transitions,expected_buffer",
    [
        (
            deque(),
            [[Transition(reward=1.0)], [Transition(reward=1.0)]],
            deque([Transition(reward=1.0), Transition(reward=1.0)]),
        ),
        (
            deque([Transition(reward=1.0), Transition(reward=0.5)]),
            [
                [Transition(reward=2.0), Transition(reward=1.0)],
                [Transition(reward=0.05)],
            ],
            deque(
                [
                    Transition(reward=1.0),
                    Transition(reward=0.5),
                    Transition(reward=1.0),
                    Transition(reward=0.5),
                ]
            ),
        ),
    ],
)
def test_gata_double_dqn_push_to_buffer(
    replay_buffer_gata_double_dqn, initial_buffer, batch_transitions, expected_buffer
):
    replay_buffer_gata_double_dqn.buffer = initial_buffer
    t_cache = TransitionCache(0)
    t_cache.cache = batch_transitions
    replay_buffer_gata_double_dqn.push_to_buffer(t_cache)
    assert replay_buffer_gata_double_dqn.buffer == expected_buffer


def test_gata_double_dqn_sample(replay_buffer_gata_double_dqn):
    replay_buffer_gata_double_dqn.buffer = deque(
        [
            Transition(
                ob=f"{i} o",
                action_cands=[f"{i} a1", f"{i} a2"],
                current_graph=torch.rand(2, 1, 1),
                action_id=random.randint(0, 1),
                reward=random.random(),
                next_ob=f"{i} next o",
                next_action_cands=[f"{i} next a1", f"{i} next a2"],
                next_graph=torch.rand(2, 1, 1),
            )
            for i in range(10)
        ]
    )
    for transition in replay_buffer_gata_double_dqn.sample():
        assert transition in replay_buffer_gata_double_dqn.buffer


def test_gata_double_dqn_prepare_batch(replay_buffer_gata_double_dqn):
    transitions = [
        Transition(
            ob=f"{i} o",
            action_cands=[f"{i} a1", f"{i} a2"],
            current_graph=torch.rand(2, 1, 1),
            action_id=random.randint(0, 1),
            reward=random.random(),
            next_ob=f"{i} next o",
            next_action_cands=[f"{i} next a1", f"{i} next a2"],
            next_graph=torch.rand(2, 1, 1),
        )
        for i in range(10)
    ]
    batch_size = len(transitions)
    batch = replay_buffer_gata_double_dqn.prepare_batch(transitions)
    assert batch["obs_word_ids"].size() == (batch_size, 2)
    assert batch["obs_mask"].size() == (batch_size, 2)
    assert batch["current_graph"].size() == (batch_size, 2, 1, 1)
    assert batch["action_cand_word_ids"].size() == (batch_size, 2, 2)
    assert batch["action_cand_mask"].size() == (batch_size, 2, 2)
    assert batch["actions_idx"].size() == (batch_size,)
    assert batch["rewards"].size() == (batch_size,)
    assert batch["next_obs_word_ids"].size() == (batch_size, 3)
    assert batch["next_obs_mask"].size() == (batch_size, 3)
    assert batch["next_graph"].size() == (batch_size, 2, 1, 1)
    assert batch["next_action_cand_word_ids"].size() == (batch_size, 2, 3)
    assert batch["next_action_cand_mask"].size() == (batch_size, 2, 3)


def test_get_double_dqn_train_dataloader(replay_buffer_gata_double_dqn):
    batch_size = replay_buffer_gata_double_dqn.hparams.train_sample_batch_size
    for batch in replay_buffer_gata_double_dqn.train_dataloader():
        assert batch["obs_word_ids"].size(0) == batch_size
        assert batch["obs_mask"].size() == batch["obs_word_ids"].size()
        assert batch["current_graph"].size() == (batch_size, 2, 1, 1)
        assert batch["action_cand_word_ids"].size(0) == batch_size
        assert batch["action_cand_mask"].size() == batch["action_cand_word_ids"].size()
        assert batch["actions_idx"].size() == (batch_size,)
        assert batch["rewards"].size() == (batch_size,)
        assert batch["next_obs_word_ids"].size(0) == batch_size
        assert batch["next_obs_mask"].size() == batch["next_obs_word_ids"].size()
        assert batch["next_graph"].size() == (batch_size, 2, 1, 1)
        assert batch["next_action_cand_word_ids"].size(0) == batch_size
        assert (
            batch["next_action_cand_mask"].size()
            == batch["next_action_cand_word_ids"].size()
        )
