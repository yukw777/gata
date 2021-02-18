import pytest
import torch

from graph_updater import (
    RelationalGraphConvolution,
    RGCNHighwayConnections,
    GraphEncoder,
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
