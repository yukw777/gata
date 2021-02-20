import pytest
import torch

from graph_updater import (
    RelationalGraphConvolution,
    RGCNHighwayConnections,
    GraphEncoder,
    DepthwiseSeparableConv1d,
)


@pytest.mark.parametrize(
    "entity_input_dim,relation_input_dim,num_relations,out_dim,"
    "num_bases,num_entity,batch_size,output_size",
    [
        (10, 20, 5, 25, 3, 7, 5, (5, 7, 25)),
        (20, 20, 10, 20, 5, 10, 3, (3, 10, 20)),
    ],
)
def test_r_gcn(
    entity_input_dim,
    relation_input_dim,
    num_relations,
    out_dim,
    num_bases,
    num_entity,
    batch_size,
    output_size,
):
    rgcn = RelationalGraphConvolution(
        entity_input_dim, relation_input_dim, num_relations, out_dim, num_bases
    )
    assert (
        rgcn(
            torch.rand(batch_size, num_entity, entity_input_dim),
            torch.rand(batch_size, num_relations, relation_input_dim),
            torch.rand(batch_size, num_relations, num_entity, num_entity),
        ).size()
        == output_size
    )


@pytest.mark.parametrize(
    "entity_input_dim,relation_input_dim,num_relations,out_dim,"
    "num_bases,num_entity,batch_size,output_size",
    [
        (10, 20, 5, 25, 3, 7, 5, (5, 7, 25)),
        (20, 20, 10, 20, 5, 10, 3, (3, 10, 20)),
    ],
)
def test_r_gcn_highway_connections(
    entity_input_dim,
    relation_input_dim,
    num_relations,
    out_dim,
    num_bases,
    num_entity,
    batch_size,
    output_size,
):
    rgcn = RGCNHighwayConnections(
        entity_input_dim, relation_input_dim, num_relations, out_dim, num_bases
    )
    assert (
        rgcn(
            torch.rand(batch_size, num_entity, entity_input_dim),
            torch.rand(batch_size, num_relations, relation_input_dim),
            torch.rand(batch_size, num_relations, num_entity, num_entity),
        ).size()
        == output_size
    )


@pytest.mark.parametrize(
    "entity_input_dim,relation_input_dim,num_relations,hidden_dims,"
    "num_bases,num_entity,batch_size,output_size",
    [
        (10, 20, 5, [10, 20, 30], 3, 7, 5, (5, 7, 30)),
        (10, 20, 5, [30, 30, 30], 3, 7, 5, (5, 7, 30)),
        (20, 20, 10, [30, 30, 30], 5, 10, 3, (3, 10, 30)),
        (20, 20, 10, [30, 20, 10], 5, 10, 3, (3, 10, 10)),
    ],
)
def test_graph_encoder(
    entity_input_dim,
    relation_input_dim,
    num_relations,
    hidden_dims,
    num_bases,
    num_entity,
    batch_size,
    output_size,
):
    graph_encoder = GraphEncoder(
        entity_input_dim, relation_input_dim, num_relations, hidden_dims, num_bases
    )
    assert (
        graph_encoder(
            torch.rand(batch_size, num_entity, entity_input_dim),
            torch.rand(batch_size, num_relations, relation_input_dim),
            torch.rand(batch_size, num_relations, num_entity, num_entity),
        ).size()
        == output_size
    )


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,batch_size,seq_len_in,seq_len_out",
    [
        (10, 5, 3, 2, 5, 5),
        (15, 4, 2, 3, 10, 11),
    ],
)
def test_depthwise_separable_conv_1d(
    in_channels, out_channels, kernel_size, batch_size, seq_len_in, seq_len_out
):
    ds_conv = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size)
    assert ds_conv(torch.rand(batch_size, in_channels, seq_len_in)).size() == (
        batch_size,
        out_channels,
        seq_len_out,
    )
