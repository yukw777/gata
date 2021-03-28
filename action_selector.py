import torch
import torch.nn as nn

from layers import TextEncoder, GraphEncoder, ReprAggregator, EncoderMixin
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
        action_cand_mask: torch.Tensor,
        action_mask: torch.Tensor,
        h_og: torch.Tensor,
        h_go: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        enc_action_cands: encoded action candidates produced by TextEncoder.
            (batch, num_action_cands, action_cand_len, hidden_dim)
        action_cand_mask: mask for encoded action candidates.
            (batch, num_action_cands, action_cand_len)
        action_mask: mask for action candidates. (batch, num_action_cands)
        h_og: aggregated representation of the observation with the current graph.
            (batch, obs_len, hidden_dim)
        h_go: aggregated node representation of the current graph with the observation.
            (batch, num_node, hidden_dim)
        obs_mask: mask for the observations. (batch, obs_len)

        output: action scores of shape (batch, num_action_cands)
        """
        # get the action candidate representation
        # perform masked mean pooling over the action_cand_len dim
        # we first flatten the encoded action candidates,
        # calculate the masked mean and then restore the original dims
        flat_enc_action_cands = enc_action_cands.flatten(end_dim=1)
        # (batch * num_action_cands, action_cand_len, hidden_dim)
        flat_action_cand_mask = action_cand_mask.flatten(end_dim=1)
        # (batch * num_action_cands, action_cand_len)
        action_cand_repr = masked_mean(flat_enc_action_cands, flat_action_cand_mask)
        # (batch * num_action_cands, hidden_dim)
        batch_size = enc_action_cands.size(0)
        action_cand_repr = action_cand_repr.view(batch_size, -1, self.hidden_dim)
        # (batch, num_action_cands, hidden_dim)

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

        # get the graph representation
        h_go_t = h_go.transpose(0, 1)
        graph_repr, _ = self.self_attn_graph(h_go_t, h_go_t, h_go_t)
        # (num_node, batch, hidden_dim)
        graph_repr = graph_repr.transpose(0, 1)
        # (batch, num_node, hidden_dim)
        # mean pooling. no masks necessary as we use all the nodes
        graph_repr = graph_repr.mean(dim=1)
        # (batch, hidden_dim)

        # expand the graph and obs representations
        num_action_cands = enc_action_cands.size(1)
        expanded_graph_repr = graph_repr.unsqueeze(1).expand(-1, num_action_cands, -1)
        # (batch, num_action_cands, hidden_dim)
        expanded_obs_repr = obs_repr.unsqueeze(1).expand(-1, num_action_cands, -1)
        # (batch, num_action_cands, hidden_dim)

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

        return output


class ActionSelector(EncoderMixin, nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_words: int,
        word_emb_dim: int,
        num_nodes: int,
        node_emb_dim: int,
        num_relations: int,
        relation_emb_dim: int,
        text_encoder_num_blocks: int,
        text_encoder_num_conv_layers: int,
        text_encoder_kernel_size: int,
        text_encoder_num_heads: int,
        graph_encoder_num_cov_layers: int,
        graph_encoder_num_bases: int,
        action_scorer_num_heads: int,
        node_name_word_ids: torch.Tensor,
        node_name_mask: torch.Tensor,
        rel_name_word_ids: torch.Tensor,
        rel_name_mask: torch.Tensor,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations

        # word embeddings
        self.word_embeddings = nn.Sequential(
            nn.Embedding(num_words, word_emb_dim),
            nn.Linear(word_emb_dim, hidden_dim, bias=False),
        )

        # text encoder
        self.text_encoder = TextEncoder(
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            hidden_dim,
            text_encoder_num_heads,
        )

        # node and relation embeddings
        self.node_embeddings = nn.Embedding(num_nodes, node_emb_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_emb_dim)

        # save the node and relation name word ids and masks as buffers.
        # GATA used the mean word embeddings of the node and relation name words.
        # These are static as we have a fixed set of node and relation names.
        assert node_name_word_ids.dtype == torch.int64
        assert node_name_mask.dtype == torch.float
        assert node_name_word_ids.size() == node_name_mask.size()
        assert node_name_word_ids.size(0) == self.num_nodes
        assert node_name_mask.size(0) == self.num_nodes
        assert rel_name_word_ids.dtype == torch.int64
        assert rel_name_mask.dtype == torch.float
        assert rel_name_word_ids.size() == rel_name_mask.size()
        assert rel_name_word_ids.size(0) == self.num_relations
        assert rel_name_mask.size(0) == self.num_relations
        self.register_buffer("node_name_word_ids", node_name_word_ids)
        self.register_buffer("node_name_mask", node_name_mask)
        self.register_buffer("rel_name_word_ids", rel_name_word_ids)
        self.register_buffer("rel_name_mask", rel_name_mask)

        # graph encoder
        self.graph_encoder = GraphEncoder(
            hidden_dim + node_emb_dim,
            hidden_dim + relation_emb_dim,
            num_relations,
            [hidden_dim] * graph_encoder_num_cov_layers,
            graph_encoder_num_bases,
        )

        # representation aggregator
        self.repr_aggr = ReprAggregator(hidden_dim)

        # action scorer
        self.action_scorer = ActionScorer(hidden_dim, action_scorer_num_heads)

    def forward(
        self,
        obs_word_ids: torch.Tensor,
        obs_mask: torch.Tensor,
        current_graph: torch.Tensor,
        action_cand_word_ids: torch.Tensor,
        action_cand_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        obs_word_ids: (batch, obs_len)
        obs_mask: (batch, obs_len)
        current_graph: (batch, num_relation, num_node, num_node)
        action_cand_word_ids: (batch, num_action_cands, action_cand_len)
        action_cand_mask: (batch, num_action_cands, action_cand_len)
        action_mask: (batch, num_action_cands)

        output:
            action scores of shape (batch, num_action_cands)
        """
        # encode text observations
        encoded_obs = self.encode_text(obs_word_ids, obs_mask)
        # (batch, obs_len, hidden_dim)

        # encode the current graph
        encoded_curr_graph = self.encode_graph(current_graph)
        # (batch, num_node, hidden_dim)

        # aggregate obs and current graph representations
        batch_size = obs_word_ids.size(0)
        h_og, h_go = self.repr_aggr(
            encoded_obs,
            encoded_curr_graph,
            obs_mask,
            # no masks necessary for the graph
            torch.ones(batch_size, self.num_nodes, device=encoded_obs.device),
        )
        # h_og: (batch, obs_len, hidden_dim)
        # h_go: (batch, num_node, hidden_dim)

        # encode candidate actions
        _, num_action_cands, action_cand_len = action_cand_word_ids.size()
        enc_action_cands = self.encode_text(
            action_cand_word_ids.flatten(end_dim=1), action_cand_mask.flatten(end_dim=1)
        ).view(batch_size, num_action_cands, action_cand_len, -1)
        # (batch, num_action_cands, action_cand_len, hidden_dim)

        return self.action_scorer(
            enc_action_cands, action_cand_mask, action_mask, h_og, h_go, obs_mask
        )

    @staticmethod
    def select_max_q(
        action_scores: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        action_scores of shape (batch, num_action_cands)
        action_mask of shape (batch, num_action_cands)

        output: indices of max q actions of shape (batch)
        """
        # we want to select only from the unmasked actions
        # if we naively take the argmax, masked actions would
        # be picked over actions with negative scores if there are no
        # actions with positive scores. So we first subtract the minimum score
        # from all actions and add a small epsilon so that actions with negative
        # scores would have small positive scores and they would get chosen
        # over masked actions
        shifted_action_scores = (
            action_scores - action_scores.min(dim=1, keepdim=True)[0] + 1e-5
        ) * action_mask
        return shifted_action_scores.argmax(dim=1)
