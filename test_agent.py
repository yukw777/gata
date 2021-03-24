import pytest
import torch

from agent import Agent, EpsilonGreedyAgent
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


@pytest.fixture
def eps_greedy_agent():
    graph_updater_obs_gen = GraphUpdaterObsGen(word_vocab_path="vocabs/word_vocab.txt")
    return EpsilonGreedyAgent(
        graph_updater_obs_gen.graph_updater,
        GATADoubleDQN(word_vocab_path="vocabs/word_vocab.txt").action_selector,
        graph_updater_obs_gen.preprocessor,
        1.0,
        0.1,
        20,
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


@pytest.mark.parametrize(
    "obs,action_cands,prev_actions,rnn_prev_hidden,filtered_action_cands",
    [
        (
            ["observation for batch 0"],
            [["action 1", "action 2", "action 3"]],
            None,
            None,
            [{"action 1", "action 2", "action 3"}],
        ),
        (
            ["observation for batch 0", "observation for batch 1"],
            [
                ["action 1", "action 2", "action 3"],
                ["examine cookbook", "examine table", "look potato"],
            ],
            None,
            None,
            [
                {"action 1", "action 2", "action 3"},
                {"examine cookbook"},
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
            [
                {"action 1", "action 2", "action 3"},
                {"examine cookbook"},
            ],
        ),
    ],
)
def test_agent_act(
    agent,
    obs,
    action_cands,
    prev_actions,
    rnn_prev_hidden,
    filtered_action_cands,
):
    chosen_actions = agent.act(
        obs, action_cands, prev_actions=prev_actions, rnn_prev_hidden=rnn_prev_hidden
    )
    for action, cands in zip(chosen_actions, filtered_action_cands):
        # make sure the chosen action is within the filtered action candidates
        assert action in cands


def test_eps_greedy_agent_select_epsilon_greedy(eps_greedy_agent):
    max_q_actions_idx = torch.randint(5, (3,))
    random_actions_idx = torch.randint(5, (3,))

    # if epsilon = 0, this degrades to max q greedy
    eps_greedy_agent.epsilon = 0.0
    assert eps_greedy_agent.select_epsilon_greedy(
        max_q_actions_idx, random_actions_idx
    ).equal(max_q_actions_idx)

    # if epsilon = 1, this degrades to random
    eps_greedy_agent.epsilon = 1.0
    assert eps_greedy_agent.select_epsilon_greedy(
        max_q_actions_idx, random_actions_idx
    ).equal(random_actions_idx)


@pytest.mark.parametrize(
    "obs,action_cands,prev_actions,rnn_prev_hidden,filtered_action_cands",
    [
        (
            ["observation for batch 0"],
            [["action 1", "action 2", "action 3"]],
            None,
            None,
            [{"action 1", "action 2", "action 3"}],
        ),
        (
            ["observation for batch 0", "observation for batch 1"],
            [
                ["action 1", "action 2", "action 3"],
                ["examine cookbook", "examine table", "look potato"],
            ],
            None,
            None,
            [
                {"action 1", "action 2", "action 3"},
                {"examine cookbook"},
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
            [
                {"action 1", "action 2", "action 3"},
                {"examine cookbook"},
            ],
        ),
    ],
)
def test_eps_greedy_agent_act(
    eps_greedy_agent,
    obs,
    action_cands,
    prev_actions,
    rnn_prev_hidden,
    filtered_action_cands,
):
    chosen_actions = eps_greedy_agent.act(
        obs, action_cands, prev_actions=prev_actions, rnn_prev_hidden=rnn_prev_hidden
    )
    for action, cands in zip(chosen_actions, filtered_action_cands):
        # make sure the chosen action is within the filtered action candidates
        assert action in cands


@pytest.mark.parametrize(
    "epsilon_anneal_from,epsilon_anneal_to,epsilon_anneal_episodes",
    [(0.1, 1.0, 20), (0.0, 1.0, 20), (1.0, 1.0, 20), (0.1, 1.0, 20000)],
)
def test_update_epsilon(
    eps_greedy_agent,
    epsilon_anneal_from,
    epsilon_anneal_to,
    epsilon_anneal_episodes,
):
    eps_greedy_agent.epsilon_anneal_from = epsilon_anneal_from
    eps_greedy_agent.epsilon_anneal_to = epsilon_anneal_to
    eps_greedy_agent.epsilon_anneal_episodes = epsilon_anneal_episodes

    # if step is 0, epsilon should equal epsilon_anneal_from
    eps_greedy_agent.update_epsilon(0)
    assert eps_greedy_agent.epsilon == epsilon_anneal_from

    # if step is bigger than epsilon_anneal_episodes,
    # epsilon should equal epsilon_anneal_to
    eps_greedy_agent.update_epsilon(epsilon_anneal_episodes)
    assert pytest.approx(eps_greedy_agent.epsilon) == epsilon_anneal_to
    eps_greedy_agent.update_epsilon(epsilon_anneal_episodes + 10)
    assert eps_greedy_agent.epsilon == epsilon_anneal_to

    # if step is in the middle, epsilon should be the mean of from and to
    eps_greedy_agent.update_epsilon(epsilon_anneal_episodes // 2)
    assert eps_greedy_agent.epsilon == pytest.approx(
        (epsilon_anneal_from + epsilon_anneal_to) / 2
    )
