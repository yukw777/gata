import pytest
import torch

from agent import Agent
from train_graph_updater import GraphUpdaterObsGen
from train_gata import GATADoubleDQN
from preprocessor import SpacyPreprocessor, PAD, UNK


@pytest.fixture
def agent():
    graph_updater_obs_gen = GraphUpdaterObsGen(word_vocab_path="vocabs/word_vocab.txt")
    return Agent(
        graph_updater_obs_gen.graph_updater,
        GATADoubleDQN(word_vocab_path="vocabs/word_vocab.txt").action_selector,
        graph_updater_obs_gen.preprocessor,
    )


@pytest.fixture
def agent_simple_words():
    preprocessor = SpacyPreprocessor(
        [PAD, UNK, "action", "1", "2", "3", "examine", "cookbook", "table"]
    )
    return Agent(
        GraphUpdaterObsGen().graph_updater,
        GATADoubleDQN().action_selector,
        preprocessor,
    )


@pytest.mark.parametrize(
    "batch_action_cands,expected_filtered,"
    "expected_action_cand_word_ids,expected_action_cand_mask",
    [
        (
            [
                ["action 1", "action 2"],
                ["action 1", "action 2"],
                ["action 1", "action 2"],
            ],
            [
                ["action 1", "action 2"],
                ["action 1", "action 2"],
                ["action 1", "action 2"],
            ],
            torch.tensor(
                [
                    [[2, 3], [2, 4]],
                    [[2, 3], [2, 4]],
                    [[2, 3], [2, 4]],
                ]
            ),
            torch.ones(3, 2, 2, dtype=torch.float),
        ),
        (
            [
                ["action 1 2 3", "action 1"],
                ["action 3"],
                ["action 1"],
            ],
            [
                ["action 1 2 3", "action 1"],
                ["action 3"],
                ["action 1"],
            ],
            torch.tensor(
                [
                    [[2, 3, 4, 5], [2, 3, 0, 0]],
                    [[2, 5, 0, 0], [0, 0, 0, 0]],
                    [[2, 3, 0, 0], [0, 0, 0, 0]],
                ]
            ),
            torch.tensor(
                [
                    [[1, 1, 1, 1], [1, 1, 0, 0]],
                    [[1, 1, 0, 0], [0, 0, 0, 0]],
                    [[1, 1, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=torch.float,
            ),
        ),
        (
            [
                ["action 1 2 3", "action 1"],
                ["action 3", "examine cookbook"],
                ["action 1", "examine table"],
                ["look cookbook", "action 2"],
            ],
            [
                ["action 1 2 3", "action 1"],
                ["action 3", "examine cookbook"],
                ["action 1"],
                ["action 2"],
            ],
            torch.tensor(
                [
                    [[2, 3, 4, 5], [2, 3, 0, 0]],
                    [[2, 5, 0, 0], [6, 7, 0, 0]],
                    [[2, 3, 0, 0], [0, 0, 0, 0]],
                    [[2, 4, 0, 0], [0, 0, 0, 0]],
                ]
            ),
            torch.tensor(
                [
                    [[1, 1, 1, 1], [1, 1, 0, 0]],
                    [[1, 1, 0, 0], [1, 1, 0, 0]],
                    [[1, 1, 0, 0], [0, 0, 0, 0]],
                    [[1, 1, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=torch.float,
            ),
        ),
    ],
)
def test_agent_preprocess_action_cands(
    agent_simple_words,
    batch_action_cands,
    expected_filtered,
    expected_action_cand_word_ids,
    expected_action_cand_mask,
):
    (
        filtered_batch_action_cands,
        action_cand_word_ids,
        action_cand_mask,
    ) = agent_simple_words.preprocess_action_cands(batch_action_cands)
    assert filtered_batch_action_cands == expected_filtered
    assert action_cand_word_ids.equal(expected_action_cand_word_ids)
    assert action_cand_mask.equal(expected_action_cand_mask)


@pytest.mark.parametrize(
    "action_cands,actions_idx,expected_decoded",
    [
        ([["action 1", "action 2", "action 3"]], [2], ["action 3"]),
        (
            [
                ["action 1", "action 2", "action 3"],
                ["action 1", "action 2"],
            ],
            [0, 1],
            ["action 1", "action 2"],
        ),
    ],
)
def test_agent_decode_actions(agent, action_cands, actions_idx, expected_decoded):
    assert agent.decode_actions(action_cands, actions_idx) == expected_decoded


@pytest.mark.parametrize(
    "batch_action_cands,expected_filtered",
    [
        (
            [["action 1", "action 2", "action 3"]],
            [["action 1", "action 2", "action 3"]],
        ),
        (
            [
                ["action 1", "action 2", "action 3"],
                ["action 1", "action 2", "action 3"],
            ],
            [
                ["action 1", "action 2", "action 3"],
                ["action 1", "action 2", "action 3"],
            ],
        ),
        (
            [
                ["examine cookbook", "examine table", "look potato"],
                ["action 1", "examine table", "action 3"],
            ],
            [
                ["examine cookbook"],
                ["action 1", "action 3"],
            ],
        ),
    ],
)
def test_agent_filter_action_cands(agent, batch_action_cands, expected_filtered):
    assert agent.filter_action_cands(batch_action_cands) == expected_filtered


@pytest.mark.parametrize(
    "obs,action_cands,prev_actions,rnn_prev_hidden,batch,"
    "num_action_cands,expected_filtered",
    [
        (
            ["observation for batch 0"],
            [["action 1", "action 2", "action 3"]],
            None,
            None,
            1,
            3,
            [["action 1", "action 2", "action 3"]],
        ),
        (
            ["observation for batch 0", "observation for batch 1"],
            [
                ["action 1", "action 2", "action 3"],
                ["examine cookbook", "examine table", "look potato"],
            ],
            None,
            None,
            2,
            3,
            [
                ["action 1", "action 2", "action 3"],
                ["examine cookbook"],
            ],
        ),
        (
            ["observation for batch 0", "observation for batch 1"],
            [
                ["action 1", "action 2", "action 3"],
                ["examine cookbook", "examine table", "look potato"],
            ],
            ["examine cookbook", "action 2"],
            torch.rand(2, 8),
            2,
            3,
            [
                ["action 1", "action 2", "action 3"],
                ["examine cookbook"],
            ],
        ),
    ],
)
def test_agent_calculate_action_scores(
    agent,
    obs,
    action_cands,
    prev_actions,
    rnn_prev_hidden,
    batch,
    num_action_cands,
    expected_filtered,
):
    action_scores, action_mask, filtered = agent.calculate_action_scores(
        obs, action_cands, prev_actions=prev_actions, rnn_prev_hidden=rnn_prev_hidden
    )
    assert action_scores.size() == (batch, num_action_cands)
    assert action_mask.size() == (batch, num_action_cands)
    assert filtered == expected_filtered