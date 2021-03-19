import torch
import torch.nn as nn

from typing import Tuple

from utils import masked_mean


class ActionScorer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # self attention layers
        self.self_attn_text = nn.MultiheadAttention(hidden_dim, num_heads)
        self.self_attn_graph = nn.MultiheadAttention(hidden_dim, num_heads)

        # linear layers
        self.linear1 = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU())
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        enc_action_cands: torch.Tensor,
        enc_action_cand_mask: torch.Tensor,
        h_go: torch.Tensor,
        h_og: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        enc_action_cands: encoded action candidates produced by TextEncoder.
            (batch, num_action_cands, action_cand_len, hidden_dim)
        enc_action_cand_mask: mask for encoded action candidates.
            (batch, num_action_cands, action_cand_len)
        h_go: aggregated node representation of the current graph with the observation.
            (batch, num_node, hidden_dim)
        h_og: aggregated representation of the observation with the current graph.
            (batch, obs_len, hidden_dim)
        obs_mask: mask for the observations. (batch, obs_len)

        output:
            action scores: (batch, num_action_cands)
            action mask: (batch, num_action_cands)
        """
        # get the action candidate representation
        # perform masked mean pooling over the action_cand_len dim
        # we first flatten the encoded action candidates,
        # calculate the masked mean and then restore the original dims
        flat_enc_action_cands = enc_action_cands.flatten(end_dim=1)
        # (batch * num_action_cands, action_cand_len, hidden_dim)
        flat_enc_action_cand_mask = enc_action_cand_mask.flatten(end_dim=1)
        # (batch * num_action_cands, action_cand_len)
        action_cand_repr = masked_mean(flat_enc_action_cands, flat_enc_action_cand_mask)
        # (batch * num_action_cands, hidden_dim)
        batch_size = enc_action_cands.size(0)
        action_cand_repr = action_cand_repr.view(batch_size, -1, self.hidden_dim)
        # (batch, num_action_cands, hidden_dim)

        # get the graph representation
        h_go_t = h_go.transpose(0, 1)
        graph_repr, _ = self.self_attn_graph(h_go_t, h_go_t, h_go_t)
        # (num_node, batch, hidden_dim)
        graph_repr = graph_repr.transpose(0, 1)
        # (batch, num_node, hidden_dim)
        # mean pooling. no masks necessary as we use all the nodes
        graph_repr = graph_repr.mean(dim=1)
        # (batch, hidden_dim)

        # get the obs representation
        h_og_t = h_og.transpose(0, 1)
        obs_repr, _ = self.self_attn_text(
            h_og_t, h_og_t, h_og_t, key_padding_mask=obs_mask == 0
        )
        # (obs_len, batch, hidden_dim)
        obs_repr = obs_repr.transpose(0, 1)
        # (batch, obs_len, hidden_dim)
        # masked mean pooling.
        obs_repr = masked_mean(obs_repr, obs_mask)
        # (batch, hidden_dim)

        # expand the graph and obs representations
        num_action_cands = enc_action_cands.size(1)
        expanded_graph_repr = graph_repr.unsqueeze(1).expand(-1, num_action_cands, -1)
        # (batch, num_action_cands, hidden_dim)
        expanded_obs_repr = obs_repr.unsqueeze(1).expand(-1, num_action_cands, -1)
        # (batch, num_action_cands, hidden_dim)

        # calculate the action mask by selecting any action candidate
        # that had unmasked tokens
        action_mask = enc_action_cand_mask.bool().any(-1).float()
        # (batch, num_action_cands)

        # concatenate them with encoded action candidates and
        # send them through the final linear layers
        output = torch.cat(
            [action_cand_repr, expanded_obs_repr, expanded_graph_repr], dim=-1
        )
        # (batch, num_action_cands, 3 * hidden_dim)
        output = self.linear1(output)
        # (batch, num_action_cands, hidden_dim)
        output *= action_mask.unsqueeze(-1)
        # (batch, num_action_cands, hidden_dim)
        output = self.linear2(output).squeeze(dim=-1)
        # (batch, num_action_cands)
        output *= action_mask
        # (batch, num_action_cands)

        return output, action_mask
