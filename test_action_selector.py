import pytest
import torch

from action_selector import ActionScorer
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

    action_scores, action_mask = action_scorer(
        torch.rand(batch_size, num_action_cands, action_cand_len, hidden_dim),
        increasing_mask(num_action_cands, action_cand_len, start_with_zero=True)
        .unsqueeze(0)
        .expand(batch_size, -1, -1),
        torch.rand(batch_size, num_node, hidden_dim),
        torch.rand(batch_size, obs_len, hidden_dim),
        increasing_mask(batch_size, obs_len),
    )
    assert action_scores.size() == (batch_size, num_action_cands)
    # b/c we're using an increasing mask, only the first action of
    # each batch should be masked
    assert action_mask.equal(
        torch.tensor(([0] + [1] * (num_action_cands - 1)) * batch_size)
        .float()
        .view(batch_size, num_action_cands)
    )
