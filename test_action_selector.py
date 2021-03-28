import pytest
import torch

from action_selector import ActionScorer, ActionSelector
from utils import increasing_mask


@pytest.mark.parametrize(
    "hidden_dim,num_heads,batch_size,num_action_cands,action_cand_len,num_node,obs_len",
    [
        (10, 1, 1, 2, 2, 6, 5),
        (24, 2, 3, 5, 4, 12, 24),
        (64, 2, 5, 8, 6, 24, 36),
    ],
)
def test_action_scorer(
    hidden_dim,
    num_heads,
    batch_size,
    num_action_cands,
    action_cand_len,
    num_node,
    obs_len,
):
    action_scorer = ActionScorer(hidden_dim, num_heads)

    assert (
        action_scorer(
            torch.rand(batch_size, num_action_cands, action_cand_len, hidden_dim),
            increasing_mask(num_action_cands, action_cand_len, start_with_zero=True)
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
            torch.randint(2, (batch_size, num_action_cands)),
            torch.rand(batch_size, obs_len, hidden_dim),
            torch.rand(batch_size, num_node, hidden_dim),
            increasing_mask(batch_size, obs_len),
        ).size()
        == (batch_size, num_action_cands)
    )


@pytest.mark.parametrize(
    "hidden_dim,num_words,word_emb_dim,num_nodes,node_emb_dim,num_relations,"
    "relation_emb_dim,text_encoder_num_blocks,text_encoder_num_conv_layers,"
    "text_encoder_kernel_size,text_encoder_num_heads,graph_encoder_num_cov_layers,"
    "graph_encoder_num_bases,action_scorer_num_heads,batch_size,obs_len,"
    "num_action_cands,action_cand_len",
    [
        (12, 100, 24, 5, 24, 10, 36, 1, 1, 3, 1, 1, 3, 1, 1, 5, 3, 4),
        (12, 100, 24, 5, 24, 10, 36, 1, 1, 3, 1, 1, 3, 1, 3, 5, 3, 4),
        (12, 100, 24, 5, 24, 10, 36, 1, 1, 3, 1, 1, 3, 1, 3, 5, 3, 4),
    ],
)
def test_action_selector(
    hidden_dim,
    num_words,
    word_emb_dim,
    num_nodes,
    node_emb_dim,
    num_relations,
    relation_emb_dim,
    text_encoder_num_blocks,
    text_encoder_num_conv_layers,
    text_encoder_kernel_size,
    text_encoder_num_heads,
    graph_encoder_num_cov_layers,
    graph_encoder_num_bases,
    action_scorer_num_heads,
    batch_size,
    obs_len,
    num_action_cands,
    action_cand_len,
):
    action_selector = ActionSelector(
        hidden_dim,
        num_words,
        word_emb_dim,
        num_nodes,
        node_emb_dim,
        num_relations,
        relation_emb_dim,
        text_encoder_num_blocks,
        text_encoder_num_conv_layers,
        text_encoder_kernel_size,
        text_encoder_num_heads,
        graph_encoder_num_cov_layers,
        graph_encoder_num_bases,
        action_scorer_num_heads,
        torch.randint(num_words, (num_nodes, 3)),
        increasing_mask(num_nodes, 3),
        torch.randint(num_words, (num_relations, 3)),
        increasing_mask(num_relations, 3),
    )
    assert (
        action_selector(
            torch.randint(num_words, (batch_size, obs_len)),
            increasing_mask(batch_size, obs_len),
            torch.rand(batch_size, num_relations, num_nodes, num_nodes),
            torch.randint(num_words, (batch_size, num_action_cands, action_cand_len)),
            increasing_mask(batch_size * num_action_cands, action_cand_len).view(
                batch_size, num_action_cands, action_cand_len
            ),
            increasing_mask(batch_size, num_action_cands),
        ).size()
        == (batch_size, num_action_cands)
    )


@pytest.mark.parametrize(
    "action_scores,action_mask,max_q_actions",
    [
        (
            torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.float),
            torch.ones(2, 3),
            torch.tensor([2, 0], dtype=torch.long),
        ),
        (
            torch.tensor([[1, 2, -3], [-3, 2, 1]], dtype=torch.float),
            torch.ones(2, 3),
            torch.tensor([1, 1], dtype=torch.long),
        ),
        (
            torch.tensor([[-1, -2, 3], [-3, 2, 1]], dtype=torch.float),
            torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float),
            torch.tensor([0, 1], dtype=torch.long),
        ),
    ],
)
def test_action_selector_choose_max_q(action_scores, action_mask, max_q_actions):
    num_words = 10
    num_nodes = 5
    num_relations = 10
    action_selector = ActionSelector(
        12,
        num_words,
        16,
        num_nodes,
        12,
        num_relations,
        12,
        1,
        1,
        3,
        1,
        1,
        1,
        1,
        torch.randint(num_words, (num_nodes, 3)),
        increasing_mask(num_nodes, 3),
        torch.randint(num_words, (num_relations, 3)),
        increasing_mask(num_relations, 3),
    )
    assert action_selector.select_max_q(action_scores, action_mask).equal(max_q_actions)
