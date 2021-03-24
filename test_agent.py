import pytest
import torch

from agent import Agent
from train_graph_updater import GraphUpdaterObsGen
from train_gata import GATADoubleDQN
from preprocessor import SpacyPreprocessor, PAD, UNK


@pytest.fixture
def agent():
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
    agent,
    batch_action_cands,
    expected_filtered,
    expected_action_cand_word_ids,
    expected_action_cand_mask,
):
    (
        filtered_batch_action_cands,
        action_cand_word_ids,
        action_cand_mask,
    ) = agent.preprocess_action_cands(batch_action_cands)
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
def test_filter_action_cands(agent, batch_action_cands, expected_filtered):
    assert agent.filter_action_cands(batch_action_cands) == expected_filtered
