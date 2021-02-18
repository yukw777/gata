import torch
import torch.nn as nn
import itertools

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
