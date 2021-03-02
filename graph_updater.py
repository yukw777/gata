import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math

from typing import List, Optional, Tuple


class RelationalGraphConvolution(nn.Module):
    """
    Taken from the original GATA code (https://github.com/xingdi-eric-yuan/GATA-public),
    and simplified.
    """

    def __init__(
        self,
        node_input_dim: int,
        relation_input_dim: int,
        num_relations: int,
        out_dim: int,
        num_bases: int,
    ) -> None:
        super().__init__()
        self.node_input_dim = node_input_dim
        self.relation_input_dim = relation_input_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        assert self.num_bases > 0
        self.bottleneck_layer = torch.nn.Linear(
            (self.node_input_dim + self.relation_input_dim) * self.num_relations,
            self.num_bases,
            bias=False,
        )
        self.weight = torch.nn.Linear(self.num_bases, self.out_dim, bias=False)
        self.bias = torch.nn.Parameter(torch.tensor(self.out_dim).float())
        self.activation = nn.Sigmoid()

        # initialize layers
        self.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.weight.weight.data)

    def forward(
        self,
        node_features: torch.Tensor,
        relation_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_features: (batch, num_node, node_input_dim)
        relation_features: (batch, num_relation, relation_input_dim)
        adj: (batch, num_relations, num_node, num_node)

        output: (batch, num_node, out_dim)
        """
        support_list: List[torch.Tensor] = []
        # TODO: see if we can vectorize this loop
        # for each relation
        for relation_idx in range(self.num_relations):
            # get the features for the current relation (relation_idx)
            # (batch, 1, relation_input_dim)
            _r_features = relation_features[:, relation_idx].unsqueeze(1)

            # concatenate each node feature and the current relation feature
            # then sum over neighbors by matrix multiplying with the adjacency matrix
            # (batch, num_node, relation_input_dim)
            _r_features = _r_features.repeat(1, node_features.size(1), 1)
            # (batch, num_node, node_input_dim + relation_input_dim)
            support_list.append(
                torch.bmm(
                    adj[:, relation_idx],
                    torch.cat([node_features, _r_features], dim=-1),
                )
            )
        # (batch, num_node, (node_input_dim+relation_input_dim)*num_relations)
        supports = torch.cat(support_list, dim=-1)
        # (batch, num_node, num_bases)
        supports = self.bottleneck_layer(supports)
        # (batch, num_node, out_dim)
        output = self.weight(supports)

        return self.activation(output + self.bias)


class RGCNHighwayConnections(RelationalGraphConvolution):
    def __init__(
        self,
        node_input_dim: int,
        relation_input_dim: int,
        num_relations: int,
        out_dim: int,
        num_bases: int,
    ) -> None:
        super().__init__(
            node_input_dim, relation_input_dim, num_relations, out_dim, num_bases
        )
        if self.node_input_dim != self.out_dim:
            self.input_linear = nn.Linear(self.node_input_dim, self.out_dim)
        self.highway = nn.Linear(self.out_dim, self.out_dim)
        self.highway_sigmoid = nn.Sigmoid()

    def forward(
        self,
        node_features: torch.Tensor,
        relation_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_features: (batch, num_node, node_input_dim)
        relation_features: (batch, num_relation, relation_input_dim)
        adj: (batch, num_relations, num_node, num_node)

        output: (batch, num_node, out_dim)
        """
        if self.node_input_dim != self.out_dim:
            prev = self.input_linear(node_features)
        else:
            prev = node_features
        x = super().forward(node_features, relation_features, adj)
        gate = self.highway_sigmoid(self.highway(x))
        return gate * x + (1 - gate) * prev


class GraphEncoder(nn.Module):
    """
    Taken from the original GATA code
    (StackedRelationalGraphConvolution,
     https://github.com/xingdi-eric-yuan/GATA-public), and simplified.
    """

    def __init__(
        self,
        node_input_dim: int,
        relation_input_dim: int,
        num_relations: int,
        hidden_dims: List[int],
        num_bases: int,
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.relation_input_dim = relation_input_dim
        self.num_relations = num_relations
        self.hidden_dims = hidden_dims
        self.num_bases = num_bases

        # cool trick to iterate through a list pairwise
        # https://stackoverflow.com/questions/5434891/iterate-a-list-as-pair-current-next-in-python
        a, b = itertools.tee([self.node_input_dim] + self.hidden_dims)
        next(b, None)
        dims = zip(a, b)

        # R-GCNs
        # Sequential doesn't quite work b/c its forward() can only accept one argument
        self.rgcns = nn.ModuleList(
            RelationalGraphConvolution(
                input_dim,
                self.relation_input_dim,
                self.num_relations,
                output_dim,
                self.num_bases,
            )
            for input_dim, output_dim in dims
        )

    def forward(
        self,
        node_features: torch.Tensor,
        relation_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        node features: (batch, num_node, node_input_dim)
        relation features: (batch, num_relations, relation_input_dim)
        adjacency matrix: (batch, num_relations, num_node, num_node)

        output: (batch, num_node, hidden_dims[-1])
        """
        x = node_features
        for rgcn in self.rgcns:
            x = rgcn(x, relation_features, adj)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise, separable 1d convolution to save computation in exchange for
    a bit of accuracy.
    https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            padding=kernel_size // 2,
            bias=False,
        )
        self.pointwise_conv = torch.nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, in_channels, seq_len_in)
        output: (batch, out_channels, seq_len_out)

        seq_len_out = (seq_len_in + 2 * (kernel_size // 2) - (kernel_size - 1) - 1) + 1
        """
        return self.pointwise_conv(self.depthwise_conv(input))


class TextEncoderConvBlock(nn.Module):
    """
    Convolutional blocks used in QANet.
    A layer norm followed by a depthwise, separable convolutional layer
    with a residual connection.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        assert (
            kernel_size % 2 == 1
        ), "kernel_size has to be odd in order to preserve the sequence length"
        self.layer_norm = nn.LayerNorm(channels)
        self.relu = nn.ReLU()
        self.conv = DepthwiseSeparableConv1d(channels, channels, kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, channels)
        output: (batch, seq_len, channels)
        """
        residual = input
        output = self.layer_norm(input)
        output = self.relu(self.conv(output.transpose(1, 2))).transpose(1, 2)
        return output + residual


class PositionalEncoder(nn.Module):
    """
    The positional encoding from the original Transformer paper.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, d_model)
        output: (batch, seq_len, d_model)
        """
        # add positional encodings to the input using broadcast
        return input + self.pe[: input.size(1), :]  # type: ignore


class PositionalEncoderTensor2Tensor(nn.Module):
    """
    Add positional encodings to the given input. This is the tensor2tensor
    implementation of the positional encoding, which is slightly different
    from the one used by the original Transformer paper.
    Specifically, there are 2 key differences:
    1. Sine and cosine values are concatenated rather than interweaved.
    2. The divisor is calculated differently
        ((d_model (or channels) // 2) -1 vs. d_model)

    There are no material differences between positional encoding implementations.
    The important point is that you use the same implementation throughout. The
    original GATA code uses this version. I've cleaned up the implementation a bit,
    including a small optimization that caches all the positional encodings, which
    was shown in the PyTorch Transformer tutorial
    (https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    """

    def __init__(
        self,
        channels: int,
        max_len: int,
        min_timescale: float = 1.0,
        max_timescale: float = 1e4,
    ) -> None:
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        num_timescales = channels // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / (
            num_timescales - 1
        )
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).float() * -log_timescale_increment
        ).unsqueeze(0)
        scaled_time = position * inv_timescales
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).view(
            max_len, channels
        )
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (batch, seq_len, channels)
        output: (batch, seq_len, channels)
        """
        # add positional encodings to the input using broadcast
        return input + self.pe[: input.size(1)]  # type: ignore


class TextEncoderBlock(nn.Module):
    """
    Based on QANet (https://arxiv.org/abs/1804.09541)
    """

    def __init__(
        self,
        num_conv_layers: int,
        kernel_size: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim has to be even for positional encoding"
        self.pos_encoder = PositionalEncoderTensor2Tensor(hidden_dim, 512)
        self.conv_layers = nn.Sequential(
            *[
                TextEncoderConvBlock(hidden_dim, kernel_size)
                for _ in range(num_conv_layers)
            ]
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.linear_layer_norm = nn.LayerNorm(hidden_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, input: torch.Tensor, key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        input: (batch, seq_len, hidden_dim)
        output: (batch, seq_len, hidden_dim)
        """
        # add the positional encodings
        output = self.pos_encoder(input)

        # conv layers
        output = self.conv_layers(output)

        # self attention layer
        residual = output
        # MultiheadAttention expects batch dim to be 1 for q, k, v
        # but 0 for key_padding_mask, so we need to transpose
        output = output.transpose(0, 1)
        output = self.self_attn_layer_norm(output)
        output, _ = self.self_attn(
            output, output, output, key_padding_mask=key_padding_mask
        )
        output = output.transpose(0, 1)
        output += residual

        # linear layer
        residual = output
        output = self.linear_layer_norm(output)
        output = self.linear_layers(output)
        output += residual

        return output


class TextEncoder(nn.Module):
    def __init__(
        self,
        word_emb_dim: int,
        num_enc_blocks: int,
        enc_block_num_conv_layers: int,
        enc_block_kernel_size: int,
        enc_block_hidden_dim: int,
        enc_block_num_heads: int,
    ) -> None:
        super().__init__()
        self.word_emb_prj = nn.Linear(word_emb_dim, enc_block_hidden_dim, bias=False)
        self.enc_blocks = nn.ModuleList(
            TextEncoderBlock(
                enc_block_num_conv_layers,
                enc_block_kernel_size,
                enc_block_hidden_dim,
                enc_block_num_heads,
            )
            for _ in range(num_enc_blocks)
        )

    def forward(
        self, input_word_embs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        input_word_embs: (batch_size, seq_len, word_emb_dim)
        mask: (batch_size, seq_len)

        output: (batch_size, seq_len, enc_block_hidden_dim)
        """
        # (batch_size, seq_len, word_emb_dim)
        output = self.word_emb_prj(input_word_embs)
        # (batch_size, seq_len, enc_block_hidden_dim)
        for enc_block in self.enc_blocks:
            output = enc_block(output, key_padding_mask=mask == 0)
        # (batch_size, seq_len, enc_block_hidden_dim)

        return output


def masked_softmax(
    input: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    input, mask and output all have the same dimensions
    """
    # replace the values to be ignored with negative infinity
    return F.softmax(input.masked_fill(mask == 0, float("-inf")), dim=dim)


class ContextQueryAttention(nn.Module):
    """
    Based on Context-Query Attention Layer from QANet, which is in turn
    based on Attention Flow Layer from https://arxiv.org/abs/1611.01603
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        w_C = torch.empty(hidden_dim, 1)
        w_Q = torch.empty(hidden_dim, 1)
        w_CQ = torch.empty(hidden_dim, 1)
        torch.nn.init.xavier_uniform_(w_C)
        torch.nn.init.xavier_uniform_(w_Q)
        torch.nn.init.xavier_uniform_(w_CQ)
        self.w_C = torch.nn.Parameter(w_C)
        self.w_Q = torch.nn.Parameter(w_Q)
        self.w_CQ = torch.nn.Parameter(w_CQ)

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(
        self,
        ctx: torch.Tensor,
        query: torch.Tensor,
        ctx_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        ctx: (batch, ctx_seq_len, hidden_dim)
        query: (batch, query_seq_len, hidden_dim)
        ctx_mask: (batch, ctx_seq_len)
        query_mask: (batch, query_seq_len)

        output: (batch, ctx_seq_len, 4 * hidden_dim)
        """
        ctx_seq_len = ctx.size(1)
        query_seq_len = query.size(1)

        # (batch, ctx_seq_len, query_seq_len)
        similarity = self.trilinear_for_attention(ctx, query)
        # (batch, ctx_seq_len, query_seq_len)
        s_ctx = masked_softmax(
            similarity, ctx_mask.unsqueeze(2).expand(-1, -1, query_seq_len), dim=1
        )
        # (batch, ctx_seq_len, query_seq_len)
        s_query = masked_softmax(
            similarity, query_mask.unsqueeze(1).expand(-1, ctx_seq_len, -1), dim=2
        )
        # (batch, ctx_seq_len, hidden_dim)
        P = torch.bmm(s_query, query)
        # (batch, ctx_seq_len, hidden_dim)
        Q = torch.bmm(torch.bmm(s_query, s_ctx.transpose(1, 2)), ctx)

        # (batch, ctx_seq_len, 4 * hidden_dim)
        return torch.cat([ctx, P, ctx * P, ctx * Q], dim=2)

    def trilinear_for_attention(
        self, ctx: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        ctx: (batch, ctx_seq_len, hidden_dim), context C
        query: (batch, query_seq_len, hidden_dim), query Q
        output: (batch, ctx_seq_len, query_seq_len), similarity matrix S

        This is an optimized implementation. The number of multiplications of the
        original equation S_ij = w^T[C_i; Q_j; C_i * Q_j] is
        O(ctx_seq_len * query_seq_len * 3 * hidden_dim)
        = O(ctx_seq_len * query_seq_len * hidden_dim)

        We can reduce this number by splitting the weight matrix w into three parts,
        one for each part of the concatenated vector [C_i; Q_j; C_i * Q_j].
        Specifically,
        S_ij = w^T[C_i; Q_j; C_i * Q_j]
        = w^1C^1_i + ... + w^dC^d_i + w^{d+1}Q^1_j + ... +w^{2d}Q^d_j
          + w^{2d+1}C^1_iQ^1_j + ... + w^{3d}C^d_iQ^d_j
        = w_CC_i + w_QQ_j + w_{C * Q}C_iQ_j
        where d = hidden_dim, and the superscript i denotes the i'th element of a
        vector. The number of multiplications of this formulation is
        O(hidden_dim + hidden_dim + hidden_dim + ctx_seq_len * query_seq_len)
        = O(hidden_dim + ctx_seq_len * query_seq_len)
        """
        ctx_seq_len = ctx.size(1)
        query_seq_len = query.size(1)

        # (batch, ctx_seq_len, query_seq_len)
        res_C = torch.matmul(ctx, self.w_C).expand(-1, -1, query_seq_len)
        # (batch, query_seq_len, ctx_seq_len)
        res_Q = torch.matmul(query, self.w_Q).expand(-1, -1, ctx_seq_len)
        # (batch, ctx_seq_len, query_seq_len)
        res_CQ = torch.matmul(self.w_CQ.squeeze() * ctx, query.transpose(1, 2))

        return res_C + res_Q.transpose(1, 2) + res_CQ + self.bias


class ReprAggregator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cqattn = ContextQueryAttention(hidden_dim)
        self.linear = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(
        self,
        repr1: torch.Tensor,
        repr2: torch.Tensor,
        repr1_mask: torch.Tensor,
        repr2_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        repr1: (batch, repr1_seq_len, hidden_dim)
        repr2: (batch, repr2_seq_len, hidden_dim)
        repr1_mask: (batch, repr1_seq_len)
        repr2_mask: (batch, repr2_seq_len)

        output: (batch, repr1_seq_len, hidden_dim), (batch, repr2_seq_len, hidden_dim)
        """
        return (
            self.linear(self.cqattn(repr1, repr2, repr1_mask, repr2_mask)),
            self.linear(self.cqattn(repr2, repr1, repr2_mask, repr1_mask)),
        )


def masked_mean(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    input: (batch, seq_len, hidden_dim)
    mask: (batch, seq_len)
    output: (batch, hidden_dim)
    """
    return (input * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


class GraphUpdater(nn.Module):
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
        node_name_embeddings: torch.Tensor,
        relation_name_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        # constants
        # b/c we add inverse relations, num_relations has to be even
        assert num_relations % 2 == 0
        self.num_nodes = num_nodes
        self.num_relations = num_relations

        # untrainable word embeddings
        assert word_emb_dim == pretrained_word_embeddings.embedding_dim
        assert not pretrained_word_embeddings.weight.requires_grad
        self.word_embeddings = pretrained_word_embeddings

        # trainable node and relation embeddings
        self.node_embeddings = nn.Embedding(num_nodes, node_emb_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_emb_dim)

        # save the node and relation name embeddings as buffers. GATA used the
        # mean word embeddings of the node and relation name words.
        # these are static as we have a fixed set of node and relation names
        # and word embeddings are frozen.
        assert node_name_embeddings.size() == (self.num_nodes, word_emb_dim)
        assert relation_name_embeddings.size() == (self.num_relations, word_emb_dim)
        self.register_buffer("node_name_embeddings", node_name_embeddings)
        self.register_buffer("relation_name_embeddings", relation_name_embeddings)

        # encoders
        self.text_encoder = TextEncoder(
            word_emb_dim,
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            hidden_dim,
            text_encoder_num_heads,
        )
        self.graph_encoder = GraphEncoder(
            word_emb_dim + node_emb_dim,
            word_emb_dim + relation_emb_dim,
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
        batch, num_node, _ = prev_node_hidden.size()
        # no masks necessary for prev_node_hidden, so just create a fake one
        prev_node_mask = torch.ones(batch, num_node, device=prev_node_hidden.device)

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
    ) -> torch.Tensor:
        """
        obs_word_ids: (batch, obs_len)
        prev_action_word_ids: (batch, prev_action_len)
        obs_mask: (batch, obs_len)
        prev_action_mask: (batch, prev_action_len)
        rnn_prev_hidden: (batch, hidden_dim)

        output: (batch, hidden_dim), hidden state of the rnn cell
        """
        batch_size = obs_word_ids.size(0)

        # encode text observations and previous actions
        obs_word_embs = self.word_embeddings(obs_word_ids)
        # (batch, obs_len, word_emb_dim)
        encoded_obs = self.text_encoder(obs_word_embs, obs_mask)
        # (batch, obs_len, hidden_dim)
        prev_action_embs = self.word_embeddings(prev_action_word_ids)
        # (batch, prev_action_len, word_emb_dim)
        encoded_prev_action = self.text_encoder(prev_action_embs, prev_action_mask)
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

        # get the initial graph node embeddings by concatenating the node name
        # embeddings and node embeddings
        node_embs = self.node_embeddings(
            torch.arange(self.num_nodes, device=self.node_embeddings.weight.device)
        )
        # (num_node, node_emb_dim)
        node_features = (
            torch.cat([self.node_name_embeddings, node_embs], dim=1)  # type: ignore
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        # (batch, num_node, word_emb_dim + node_emb_dim)
        relation_embs = self.relation_embeddings(
            torch.arange(
                self.num_relations, device=self.relation_embeddings.weight.device
            )
        )
        # (num_node, relation_emb_dim)
        relation_features = (
            torch.cat(
                [self.relation_name_embeddings, relation_embs], dim=1  # type: ignore
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        # (batch, num_node, word_emb_dim + relation_emb_dim)

        # encode the previous graph
        encoded_prev_graph = self.graph_encoder(
            node_features, relation_features, prev_graph
        )
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
        return self.rnncell(rnn_input, hx=rnn_prev_hidden)
        # (batch, hidden_dim)
