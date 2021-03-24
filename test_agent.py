import pytest
import torch

from agent import Agent
from train_graph_updater import GraphUpdaterObsGen
from train_gata import GATADoubleDQN
from preprocessor import SpacyPreprocessor, PAD, UNK


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
    batch_action_cands,
    expected_filtered,
    expected_action_cand_word_ids,
    expected_action_cand_mask,
):
    preprocessor = SpacyPreprocessor(
        [PAD, UNK, "action", "1", "2", "3", "examine", "cookbook", "table"]
    )
    agent = Agent(
        GraphUpdaterObsGen().graph_updater,
        GATADoubleDQN().action_selector,
        preprocessor,
    )
    (
        filtered_batch_action_cands,
        action_cand_word_ids,
        action_cand_mask,
    ) = agent.preprocess_action_cands(batch_action_cands)
    assert filtered_batch_action_cands == expected_filtered
    assert action_cand_word_ids.equal(expected_action_cand_word_ids)
    assert action_cand_mask.equal(expected_action_cand_mask)
