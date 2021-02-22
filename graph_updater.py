import torch
import torch.nn as nn
import itertools
import math

from typing import List


class RelationalGraphConvolution(nn.Module):
    """
    Taken from the original GATA code (https://github.com/xingdi-eric-yuan/GATA-public),
    and simplified.
    """

    def __init__(
        self,
        entity_input_dim: int,
        relation_input_dim: int,
        num_relations: int,
        out_dim: int,
        num_bases: int,
    ) -> None:
        super().__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        assert self.num_bases > 0
        self.bottleneck_layer = torch.nn.Linear(
            (self.entity_input_dim + self.relation_input_dim) * self.num_relations,
            self.num_bases,
            bias=False,
        )
        self.weight = torch.nn.Linear(self.num_bases, self.out_dim, bias=False)
        self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim))
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
        node_features: (batch, num_entity, entity_input_dim)
        relation_features: (batch, num_relation, relation_input_dim)
        adj: (batch, num_relations, num_entity, num_entity)
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
            # (batch, num_entity, relation_input_dim)
            _r_features = _r_features.repeat(1, node_features.size(1), 1)
            # (batch, num_entity, entity_input_dim + relation_input_dim)
            support_list.append(
                torch.bmm(
                    adj[:, relation_idx],
                    torch.cat([node_features, _r_features], dim=-1),
                )
            )
        # (batch, num_entity, (entity_input_dim+relation_input_dim)*num_relations)
        supports = torch.cat(support_list, dim=-1)
        # (batch, num_entity, num_bases)
        supports = self.bottleneck_layer(supports)
        # (batch, num_entity, out_dim)
        output = self.weight(supports)

        return self.activation(output + self.bias)


class RGCNHighwayConnections(RelationalGraphConvolution):
    def __init__(
        self,
        entity_input_dim: int,
        relation_input_dim: int,
        num_relations: int,
        out_dim: int,
        num_bases: int,
    ) -> None:
        super().__init__(
            entity_input_dim, relation_input_dim, num_relations, out_dim, num_bases
        )
        if self.entity_input_dim != self.out_dim:
            self.input_linear = nn.Linear(self.entity_input_dim, self.out_dim)
        self.highway = nn.Linear(self.out_dim, self.out_dim)
        self.highway_sigmoid = nn.Sigmoid()

    def forward(
        self,
        node_features: torch.Tensor,
        relation_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        if self.entity_input_dim != self.out_dim:
            prev = self.input_linear(node_features)
        else:
            prev = node_features.clone()
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
        entity_input_dim: int,
        relation_input_dim: int,
        num_relations: int,
        hidden_dims: List[int],
        num_bases: int,
    ):
        super().__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.num_relations = num_relations
        self.hidden_dims = hidden_dims
        self.num_bases = num_bases

        # cool trick to iterate through a list pairwise
        # https://stackoverflow.com/questions/5434891/iterate-a-list-as-pair-current-next-in-python
        a, b = itertools.tee([self.entity_input_dim] + self.hidden_dims)
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
        node features: (batch, num_entity, input_dim)
        relation features: (batch, num_relations, input_dim)
        adjacency matrix: (batch, num_relations, num_entity, num_entity)
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
        output = self.self_attn_layer_norm(output)
        output, _ = self.self_attn(
            output, output, output, key_padding_mask=key_padding_mask
        )
        output += residual

        # linear layer
        residual = output
        output = self.linear_layer_norm(output)
        output = self.linear_layers(output)
        output += residual

        return output
