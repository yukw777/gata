import pytest
import torch
import torch.nn as nn
import random
import itertools

from collections import deque
from hydra.experimental import initialize, compose
from pytorch_lightning import Trainer

from train_gata import (
    request_infos_for_train,
    request_infos_for_eval,
    get_game_dir,
    GATADoubleDQN,
    TransitionCache,
    Transition,
    RLEarlyStopping,
    main,
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

    # online action selector is train mode
    assert gata_ddqn.action_selector.training

    # target action selector is in train mode
    assert gata_ddqn.target_action_selector.training
    # and frozen
    for param in gata_ddqn.target_action_selector.parameters():
        assert param.requires_grad is False

    # online and target action selectors should be initialized to be the same
    for online, target in zip(
        gata_ddqn.action_selector.parameters(),
        gata_ddqn.target_action_selector.parameters(),
    ):
        assert online.equal(target)

    # graph updater is in eval mode
    assert not gata_ddqn.graph_updater.training
    # and frozen
    for param in gata_ddqn.graph_updater.parameters():
        assert param.requires_grad is False


def test_gata_double_dqn_update_target_action_selector():
    gata_ddqn = GATADoubleDQN()
    # scramble layers in the online action selector and update
    gata_ddqn.action_selector.node_name_word_ids.fill_(42)
    gata_ddqn.action_selector.node_embeddings = nn.Embedding(
        gata_ddqn.num_nodes, gata_ddqn.hparams.node_emb_dim
    )

    # make sure the weights are the same after updating
    gata_ddqn.update_target_action_selector()
    for online, target in zip(
        gata_ddqn.action_selector.parameters(),
        gata_ddqn.target_action_selector.parameters(),
    ):
        assert online.equal(target)


@pytest.mark.parametrize(
    "batch_size,obs_len,prev_action_len,num_action_cands,action_cand_len",
    [(1, 5, 3, 4, 10), (3, 6, 4, 5, 12)],
)
def test_gata_double_dqn_forward(
    batch_size,
    obs_len,
    prev_action_len,
    num_action_cands,
    action_cand_len,
):
    gata_ddqn = GATADoubleDQN()
    results = gata_ddqn(
        torch.randint(gata_ddqn.num_words, (batch_size, obs_len)),
        increasing_mask(batch_size, obs_len),
        torch.randint(gata_ddqn.num_words, (batch_size, prev_action_len)),
        increasing_mask(batch_size, prev_action_len),
        torch.rand(batch_size, gata_ddqn.hparams.hidden_dim),
        torch.randint(
            gata_ddqn.num_words, (batch_size, num_action_cands, action_cand_len)
        ),
        increasing_mask(batch_size * num_action_cands, action_cand_len).view(
            batch_size, num_action_cands, action_cand_len
        ),
        increasing_mask(batch_size, num_action_cands),
    )
    assert results["action_scores"].size() == (batch_size, num_action_cands)
    assert results["rnn_curr_hidden"].size() == (
        batch_size,
        gata_ddqn.hparams.hidden_dim,
    )
    assert results["current_graph"].size() == (
        batch_size,
        gata_ddqn.num_relations,
        gata_ddqn.num_nodes,
        gata_ddqn.num_nodes,
    )


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
    "batch_size,obs_len,prev_action_len,curr_action_len,"
    "num_action_cands,action_cand_len",
    [
        (1, 8, 6, 4, 5, 3),
        (3, 12, 10, 9, 8, 4),
    ],
)
def test_gata_double_dqn_training_step(
    batch_size,
    obs_len,
    prev_action_len,
    curr_action_len,
    num_action_cands,
    action_cand_len,
):
    gata_ddqn = GATADoubleDQN()
    # Note: batch_idx is not used
    assert (
        gata_ddqn.training_step(
            {
                "obs_word_ids": torch.randint(4, (batch_size, obs_len)),
                "obs_mask": torch.randint(2, (batch_size, obs_len), dtype=torch.float),
                "prev_action_word_ids": torch.randint(4, (batch_size, prev_action_len)),
                "prev_action_mask": torch.randint(
                    2, (batch_size, prev_action_len), dtype=torch.float
                ),
                "rnn_prev_hidden": torch.rand(batch_size, gata_ddqn.hparams.hidden_dim),
                "action_cand_word_ids": torch.randint(
                    4, (batch_size, num_action_cands, action_cand_len)
                ),
                "action_cand_mask": torch.randint(
                    2,
                    (batch_size, num_action_cands, action_cand_len),
                    dtype=torch.float,
                ),
                "action_mask": torch.randint(
                    2, (batch_size, num_action_cands), dtype=torch.float
                ),
                "actions_idx": torch.randint(num_action_cands, (batch_size,)),
                "rewards": torch.rand(batch_size),
                "next_obs_word_ids": torch.randint(4, (batch_size, obs_len)),
                "next_obs_mask": torch.randint(
                    2, (batch_size, obs_len), dtype=torch.float
                ),
                "curr_action_word_ids": torch.randint(4, (batch_size, curr_action_len)),
                "curr_action_mask": torch.randint(
                    2, (batch_size, curr_action_len), dtype=torch.float
                ),
                "next_action_cand_word_ids": torch.randint(
                    4, (batch_size, num_action_cands, action_cand_len)
                ),
                "next_action_cand_mask": torch.randint(
                    2,
                    (batch_size, num_action_cands, action_cand_len),
                    dtype=torch.float,
                ),
                "next_action_mask": torch.randint(
                    2, (batch_size, num_action_cands), dtype=torch.float
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
            "prev_actions": [f"{i}: step {step} prev act" for i in range(3)],
            "rnn_prev_hiddens": torch.rand(3, 12),
            "batch_action_cands": [
                [
                    f"{i}: step {step} act 1",
                    f"{i}: step {step} act 2",
                    f"{i}: step {step} act 3",
                ]
                for i in range(3)
            ],
            "actions_idx": [random.randint(0, 2) for _ in range(3)],
            "cum_rewards": [random.random() for _ in range(3)],
            "step_rewards": [random.random() for _ in range(3)],
            "next_obs": [f"{i}: step {step} next obs" for i in range(3)],
            "batch_next_action_cands": [
                [
                    f"{i}: step {step} next act 1",
                    f"{i}: step {step} next act 2",
                    f"{i}: step {step} next act 3",
                ]
                for i in range(3)
            ],
            "dones": dones,
        }

    def compare_batch_transition(batch, batch_num, transition):
        assert transition.ob == batch["obs"][batch_num]
        assert transition.prev_action == batch["prev_actions"][batch_num]
        assert transition.rnn_prev_hidden.equal(batch["rnn_prev_hiddens"][batch_num])
        assert transition.action_cands == batch["batch_action_cands"][batch_num]
        assert transition.action_id == batch["actions_idx"][batch_num]
        assert transition.cum_reward == batch["cum_rewards"][batch_num]
        assert transition.step_reward == batch["step_rewards"][batch_num]
        assert transition.next_ob == batch["next_obs"][batch_num]
        assert (
            transition.next_action_cands == batch["batch_next_action_cands"][batch_num]
        )
        assert transition.done == batch["dones"][batch_num]

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


@pytest.mark.parametrize(
    "batch_cum_rewards,batch_expected",
    [
        (
            [
                [1, 3, 3, 4],
            ],
            [1],
        ),
        (
            [
                [1, 3, 3, 4],
                [1, 3, 3, 4, 6],
            ],
            [1.0, 1.2],
        ),
    ],
)
def test_transition_cache_get_avg_rewards(batch_cum_rewards, batch_expected):
    t_cache = TransitionCache(len(batch_cum_rewards))
    for i, cum_rewards in enumerate(batch_cum_rewards):
        for cum_reward in cum_rewards:
            t_cache.cache[i].append(Transition(cum_reward=cum_reward))
    for avg, expected in zip(t_cache.get_avg_rewards(), batch_expected):
        assert pytest.approx(avg) == expected


@pytest.mark.parametrize(
    "batch_cum_rewards,batch_expected",
    [
        (
            [
                [1, 3, 3, 4],
            ],
            [4],
        ),
        (
            [
                [1, 3, 3, 4],
                [1, 3, 3, 4, 6],
            ],
            [4, 6],
        ),
    ],
)
def test_transition_cache_get_rewards(batch_cum_rewards, batch_expected):
    t_cache = TransitionCache(len(batch_cum_rewards))
    for i, cum_rewards in enumerate(batch_cum_rewards):
        for cum_reward in cum_rewards:
            t_cache.cache[i].append(Transition(cum_reward=cum_reward))
    for rewards, expected in zip(t_cache.get_game_rewards(), batch_expected):
        assert rewards == expected


@pytest.mark.parametrize("expected_steps", [[1], [2, 3, 1, 5]])
def test_transition_cache_get_game_steps(expected_steps):
    t_cache = TransitionCache(len(expected_steps))
    for i, steps in enumerate(expected_steps):
        t_cache.cache[i].extend([Transition()] * steps)
    for steps, expected in zip(t_cache.get_game_steps(), expected_steps):
        assert steps == expected


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
        train_game_batch_size=2,
        train_max_episode_steps=5,
        replay_buffer_populate_episodes=10,
        yield_step_freq=10,
        replay_buffer_capacity=20,
        train_sample_batch_size=4,
    )


@pytest.mark.parametrize(
    "initial_buffer,batch_transitions,expected_buffer",
    [
        (
            deque(),
            [[Transition(cum_reward=1)], [Transition(cum_reward=1)]],
            deque([Transition(cum_reward=1), Transition(cum_reward=1)]),
        ),
        (
            deque([Transition(step_reward=2), Transition(step_reward=1)]),
            [
                [Transition(cum_reward=2), Transition(cum_reward=3)],
                [Transition(cum_reward=0)],
            ],
            deque(
                [
                    Transition(step_reward=2),
                    Transition(step_reward=1),
                    Transition(cum_reward=2),
                    Transition(cum_reward=3),
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


def test_gata_double_dqn_extend_limited_list(replay_buffer_gata_double_dqn):
    replay_buffer_gata_double_dqn.extend_limited_list(
        [Transition(ob=str(i)) for i in range(10)]
    )
    assert len(replay_buffer_gata_double_dqn.buffer) == 10
    assert replay_buffer_gata_double_dqn.buffer_next_id == 10
    for i in range(10):
        assert replay_buffer_gata_double_dqn.buffer[i].ob == str(i)

    replay_buffer_gata_double_dqn.extend_limited_list(
        [Transition(ob=str(i)) for i in range(10, 20)]
    )
    assert len(replay_buffer_gata_double_dqn.buffer) == 20
    assert replay_buffer_gata_double_dqn.buffer_next_id == 0
    for i in range(10, 20):
        assert replay_buffer_gata_double_dqn.buffer[i].ob == str(i)

    replay_buffer_gata_double_dqn.extend_limited_list(
        [Transition(ob=str(i)) for i in range(20, 30)]
    )
    assert len(replay_buffer_gata_double_dqn.buffer) == 20
    assert replay_buffer_gata_double_dqn.buffer_next_id == 10
    for i in range(20, 30):
        assert replay_buffer_gata_double_dqn.buffer[i % 20].ob == str(i)


def test_gata_double_dqn_sample(replay_buffer_gata_double_dqn):
    replay_buffer_gata_double_dqn.buffer = deque(
        [
            Transition(
                ob=f"{i} o",
                prev_action=f"{i} p a",
                rnn_prev_hidden=torch.rand(
                    replay_buffer_gata_double_dqn.hparams.hidden_dim
                ),
                action_cands=[f"{i} a1", f"{i} a2"],
                action_id=random.randint(0, 1),
                cum_reward=random.random(),
                step_reward=random.random(),
                next_ob=f"{i} next o",
                next_action_cands=[f"{i} next a1", f"{i} next a2"],
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
            prev_action=f"{i} p a",
            rnn_prev_hidden=torch.rand(
                replay_buffer_gata_double_dqn.hparams.hidden_dim
            ),
            action_cands=[f"{i} a1", f"{i} a2"],
            action_id=random.randint(0, 1),
            cum_reward=random.randint(0, 10),
            step_reward=random.randint(0, 1),
            next_ob=f"{i} next o",
            next_action_cands=[f"{i} next a1", f"{i} next a2"],
            done=False,
        )
        for i in range(10)
    ]
    batch_size = len(transitions)
    batch = replay_buffer_gata_double_dqn.prepare_batch(transitions)
    assert batch["obs_word_ids"].size() == (batch_size, 2)
    assert batch["obs_mask"].size() == (batch_size, 2)
    assert batch["prev_action_word_ids"].size() == (batch_size, 3)
    assert batch["prev_action_mask"].size() == (batch_size, 3)
    assert batch["rnn_prev_hidden"].size() == (
        batch_size,
        replay_buffer_gata_double_dqn.hparams.hidden_dim,
    )
    assert batch["rnn_prev_hidden"].equal(
        torch.stack([t.rnn_prev_hidden for t in transitions])
    )
    assert batch["action_cand_word_ids"].size() == (batch_size, 2, 2)
    assert batch["action_cand_mask"].size() == (batch_size, 2, 2)
    assert batch["action_mask"].size() == (batch_size, 2)
    assert batch["actions_idx"].equal(torch.tensor([t.action_id for t in transitions]))
    assert batch["rewards"].equal(torch.tensor([t.step_reward for t in transitions]))
    assert batch["curr_action_word_ids"].size() == (batch_size, 2)
    assert batch["curr_action_mask"].size() == (batch_size, 2)
    assert batch["next_obs_word_ids"].size() == (batch_size, 3)
    assert batch["next_obs_mask"].size() == (batch_size, 3)
    assert batch["next_action_cand_word_ids"].size() == (batch_size, 2, 3)
    assert batch["next_action_cand_mask"].size() == (batch_size, 2, 3)
    assert batch["next_action_mask"].size() == (batch_size, 2)


def test_gata_double_dqn_train_dataloader(replay_buffer_gata_double_dqn):
    batch_size = replay_buffer_gata_double_dqn.hparams.train_sample_batch_size
    for batch in replay_buffer_gata_double_dqn.train_dataloader():
        assert batch["obs_word_ids"].size(0) == batch_size
        assert batch["obs_mask"].size() == batch["obs_word_ids"].size()
        assert batch["prev_action_word_ids"].size(0) == batch_size
        assert batch["prev_action_mask"].size() == batch["prev_action_word_ids"].size()
        assert batch["rnn_prev_hidden"].size() == (
            batch_size,
            replay_buffer_gata_double_dqn.hparams.hidden_dim,
        )
        assert batch["action_cand_word_ids"].size(0) == batch_size
        assert batch["action_cand_mask"].size() == batch["action_cand_word_ids"].size()
        assert batch["action_mask"].size(0) == batch_size
        assert batch["action_mask"].size(1) == batch["action_cand_mask"].size(1)
        assert batch["actions_idx"].size() == (batch_size,)
        assert batch["rewards"].size() == (batch_size,)
        assert batch["curr_action_word_ids"].size(0) == batch_size
        assert batch["curr_action_mask"].size() == batch["curr_action_word_ids"].size()
        assert batch["next_obs_word_ids"].size(0) == batch_size
        assert batch["next_obs_mask"].size() == batch["next_obs_word_ids"].size()
        assert batch["next_action_cand_word_ids"].size(0) == batch_size
        assert (
            batch["next_action_cand_mask"].size()
            == batch["next_action_cand_word_ids"].size()
        )
        assert batch["next_action_mask"].size(0) == batch_size
        assert batch["next_action_mask"].size(1) == batch["next_action_cand_mask"].size(
            1
        )


def test_gata_double_dqn_populate_replay_buffer(replay_buffer_gata_double_dqn):
    assert len(replay_buffer_gata_double_dqn.buffer) == 0
    replay_buffer_gata_double_dqn.populate_replay_buffer()
    assert len(replay_buffer_gata_double_dqn.buffer) > 0
    # make sure everyhing is sequential
    a, b = itertools.tee(replay_buffer_gata_double_dqn.buffer)
    next(b, None)
    for prev_t, curr_t in zip(a, b):
        if prev_t.done:
            # different game started, so skip
            continue
        # ob should be the same as previous next_ob
        assert curr_t.ob == prev_t.next_ob

        # prev_action should be the same as the selected action
        # from previous transition
        assert curr_t.prev_action == prev_t.action_cands[prev_t.action_id]

        # rnn_prev_hidden should be the right size
        assert prev_t.rnn_prev_hidden.size() == (
            replay_buffer_gata_double_dqn.hparams.hidden_dim,
        )
        assert curr_t.rnn_prev_hidden.size() == (
            replay_buffer_gata_double_dqn.hparams.hidden_dim,
        )

        # action_cands should be same as the previous next_action_cands
        assert curr_t.action_cands == prev_t.next_action_cands

        # cum_reward should be previous cum_reward + current step_reward
        assert curr_t.cum_reward == prev_t.cum_reward + curr_t.step_reward


def test_rl_early_stopping(replay_buffer_gata_double_dqn):
    trainer = Trainer()
    es = RLEarlyStopping("val_monitor", "train_monitor", 0.95, patience=3)

    # if val score and train score are all below the threshold 0.95, don't stop
    trainer.callback_metrics = {"val_monitor": 0.1, "train_monitor": 0.1}
    es._run_early_stopping_check(trainer, replay_buffer_gata_double_dqn)
    assert not trainer.should_stop

    # if val score is 1.0 and train score is above the threshold, stop
    trainer.callback_metrics = {"val_monitor": 1.0, "train_monitor": 0.95}
    trainer.current_epoch = 1
    es._run_early_stopping_check(trainer, replay_buffer_gata_double_dqn)
    assert trainer.should_stop
    assert es.stopped_epoch == 1

    # if train score is above the threshold for `patience` times,
    # but val score is not 1.0, stop
    trainer.should_stop = False
    es.wait_count = 0
    es.stopped_epoch = 0
    for i in range(3):
        trainer.current_epoch = i
        trainer.callback_metrics = {"val_monitor": 0.9, "train_monitor": 0.95}
        es._run_early_stopping_check(trainer, replay_buffer_gata_double_dqn)
        if i == 2:
            assert trainer.should_stop
            assert es.stopped_epoch == 2
        else:
            assert not trainer.should_stop
            assert es.stopped_epoch == 0


def test_main(tmp_path):
    with initialize(config_path="train_gata_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data.train_data_size=1",
                "data.train_game_batch_size=3",
                "data.train_max_episode_steps=5",
                "data.train_sample_batch_size=4",
                "data.replay_buffer_populate_episodes=3",
                "data.eval_max_episode_steps=5",
                "data.eval_game_batch_size=3",
                "train.training_step_freq=4",
                "train.target_net_update_frequency=3",
                "pl_trainer.max_epochs=2",
                f"+pl_trainer.default_root_dir={tmp_path}",
            ],
        )
        main(cfg)


def test_main_test_only(tmp_path):
    with initialize(config_path="train_gata_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data.eval_max_episode_steps=5",
                "data.eval_game_batch_size=3",
                "eval.test_only=true",
                "eval.checkpoint_path=test-data/test-gata.ckpt",
                f"+pl_trainer.default_root_dir={tmp_path}",
                "+pl_trainer.limit_test_batches=1",
            ],
        )
        main(cfg)
