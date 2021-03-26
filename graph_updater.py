import torch
import torch.nn as nn

from typing import Optional, Dict

from layers import GraphEncoder, TextEncoder, ReprAggregator, EncoderMixin
from utils import masked_mean


class GraphUpdater(EncoderMixin, nn.Module):
    def __init__(
        self,
        hidden_dim: int,
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
        pretrained_word_embeddings: nn.Embedding,
        node_name_word_ids: torch.Tensor,
        node_name_mask: torch.Tensor,
        rel_name_word_ids: torch.Tensor,
        rel_name_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        # constants
        self.hidden_dim = hidden_dim
        # b/c we add inverse relations, num_relations has to be even
        assert num_relations % 2 == 0
        self.num_nodes = num_nodes
        self.num_relations = num_relations

        # word embeddings
        assert word_emb_dim == pretrained_word_embeddings.embedding_dim
        self.word_embeddings = nn.Sequential(
            pretrained_word_embeddings, nn.Linear(word_emb_dim, hidden_dim, bias=False)
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

        # encoders
        self.text_encoder = TextEncoder(
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            hidden_dim,
            text_encoder_num_heads,
        )
        self.graph_encoder = GraphEncoder(
            hidden_dim + node_emb_dim,
            hidden_dim + relation_emb_dim,
            num_relations,
            [hidden_dim] * graph_encoder_num_cov_layers,
            graph_encoder_num_bases,
        )

        # other layers
        self.repr_aggr = ReprAggregator(hidden_dim)
        self.rnncell_input_prj = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Tanh()
        )
        self.rnncell = nn.GRUCell(hidden_dim, hidden_dim)
        self.f_d_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations // 2 * num_nodes * num_nodes),
            nn.Tanh(),
        )

        # pretraining flag
        self.pretraining = False

    def f_delta(
        self,
        prev_node_hidden: torch.Tensor,
        obs_hidden: torch.Tensor,
        prev_action_hidden: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        prev_node_hidden: (batch, num_node, hidden_dim)
        obs_hidden: (batch, obs_len, hidden_dim)
        prev_action_hidden: (batch, prev_action_len, hidden_dim)
        obs_mask: (batch, obs_len)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, 4 * hidden_dim)
        """
        batch_size = prev_node_hidden.size(0)
        # no masks necessary for prev_node_hidden, so just create a fake one
        prev_node_mask = torch.ones(
            batch_size, self.num_nodes, device=prev_node_hidden.device
        )

        # h_og: (batch, obs_len, hidden_dim)
        # h_go: (batch, num_node, hidden_dim)
        h_og, h_go = self.repr_aggr(
            obs_hidden, prev_node_hidden, obs_mask, prev_node_mask
        )
        # h_ag: (batch, prev_action_len, hidden_dim)
        # h_ga: (batch, num_node, hidden_dim)
        h_ag, h_ga = self.repr_aggr(
            prev_action_hidden, prev_node_hidden, prev_action_mask, prev_node_mask
        )

        mean_h_og = masked_mean(h_og, obs_mask)
        mean_h_go = masked_mean(h_go, prev_node_mask)
        mean_h_ag = masked_mean(h_ag, prev_action_mask)
        mean_h_ga = masked_mean(h_go, prev_node_mask)

        return torch.cat([mean_h_og, mean_h_go, mean_h_ag, mean_h_ga], dim=1)

    def f_d(self, rnn_hidden: torch.Tensor) -> torch.Tensor:
        """
        rnn_hidden: (batch, hidden_dim)
        output: (batch, num_relation, num_node, num_node)
        """
        h = self.f_d_layers(rnn_hidden).view(
            -1, self.num_relations // 2, self.num_nodes, self.num_nodes
        )
        # (batch, num_relation // 2, num_node, num_node)
        return torch.cat([h, h.transpose(2, 3)], dim=1)
        # (batch, num_relation, num_node, num_node)

    def forward(
        self,
        obs_word_ids: torch.Tensor,
        prev_action_word_ids: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_mask: torch.Tensor,
        rnn_prev_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        obs_word_ids: (batch, obs_len)
        prev_action_word_ids: (batch, prev_action_len)
        obs_mask: (batch, obs_len)
        prev_action_mask: (batch, prev_action_len)
        rnn_prev_hidden: (batch, hidden_dim)

        output:
        {
            'h_t': hidden state of the rnn cell at time t; (batch, hidden_dim)
            'g_t': decoded graph at time t; (batch, num_relation, num_node, num_node)
            'h_ag': aggregated representation of the previous action
                with the current graph. Used for pretraining.
                (batch, prev_action_len, hidden_dim)
            'h_ga': aggregated node representation of the current graph
                with the previous action. Used for pretraining.
                (batch, num_node, hidden_dim)
            'prj_obs': projected input obs word embeddings. Used for pretraining.
                (batch, obs_len, hidden_dim)
        }
        """
        batch_size = obs_word_ids.size(0)

        # encode previous actions
        encoded_prev_action = self.encode_text(prev_action_word_ids, prev_action_mask)
        # (batch, prev_action_len, hidden_dim)

        # decode the previous graph
        if rnn_prev_hidden is None:
            prev_graph = torch.zeros(
                batch_size,
                self.num_relations,
                self.num_nodes,
                self.num_nodes,
                device=obs_word_ids.device,
            )
            # (batch, num_relation, num_node, num_node)
        else:
            prev_graph = self.f_d(rnn_prev_hidden)
            # (batch, num_relation, num_node, num_node)

        if self.pretraining:
            # encode text observations
            # we don't use encode_text here
            # b/c we want to return obs_word_embs for pretraining
            obs_word_embs = self.word_embeddings(obs_word_ids)
            # (batch, obs_len, hidden_dim)
            encoded_obs = self.text_encoder(obs_word_embs, obs_mask)
            # encoded_obs: (batch, obs_len, hidden_dim)
            # prj_obs: (batch, obs_len, hidden_dim)

            # encode the previous graph
            # we don't want to use encode_graph here
            # b/c we're going to use node_features and relation_features
            # for the current graph later
            node_features = (
                self.get_node_features().unsqueeze(0).expand(batch_size, -1, -1)
            )
            # (batch, num_node, hidden_dim + node_emb_dim)
            relation_features = (
                self.get_relation_features().unsqueeze(0).expand(batch_size, -1, -1)
            )
            # (batch, num_relations, hidden_dim + relation_emb_dim)
            encoded_prev_graph = self.graph_encoder(
                node_features, relation_features, prev_graph
            )
            # (batch, num_node, hidden_dim)
        else:
            # encode text observations
            encoded_obs = self.encode_text(obs_word_ids, obs_mask)
            # encoded_obs: (batch, obs_len, hidden_dim)

            # encode the previous graph
            encoded_prev_graph = self.encode_graph(prev_graph)
            # (batch, num_node, hidden_dim)

        delta_g = self.f_delta(
            encoded_prev_graph,
            encoded_obs,
            encoded_prev_action,
            obs_mask,
            prev_action_mask,
        )
        # (batch, 4 * hidden_dim)

        rnn_input = self.rnncell_input_prj(delta_g)
        # (batch, hidden_dim)
        h_t = self.rnncell(rnn_input, hx=rnn_prev_hidden)
        # (batch, hidden_dim)

        # (batch, num_node, hidden_dim)
        curr_graph = self.f_d(h_t)
        # (batch, num_relation, num_node, num_node)

        results: Dict[str, torch.Tensor] = {"h_t": h_t, "g_t": curr_graph}
        if not self.pretraining:
            return results

        # pretraining, so calculate the aggregated representations of
        # the current graph and previous action
        # no masks necessary for encoded_curr_graph, so just create a fake one
        encoded_curr_graph = self.graph_encoder(
            node_features, relation_features, curr_graph
        )
        # (batch, num_node, hidden_dim)
        h_ag, h_ga = self.repr_aggr(
            encoded_prev_action,
            encoded_curr_graph,
            prev_action_mask,
            torch.ones(batch_size, self.num_nodes, device=encoded_curr_graph.device),
        )
        # h_ag: (batch, prev_action_len, hidden_dim)
        # h_ga: (batch, num_node, hidden_dim)
        results["h_ag"] = h_ag
        results["h_ga"] = h_ga

        # finally include prj_obs
        results["prj_obs"] = obs_word_embs

        return results
