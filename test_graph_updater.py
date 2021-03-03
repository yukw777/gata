import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_updater import (
    RelationalGraphConvolution,
    RGCNHighwayConnections,
    GraphEncoder,
    DepthwiseSeparableConv1d,
    PositionalEncoder,
    PositionalEncoderTensor2Tensor,
    TextEncoderConvBlock,
    TextEncoderBlock,
    TextEncoder,
    masked_softmax,
    ContextQueryAttention,
    ReprAggregator,
    GraphUpdater,
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
    "num_bases,num_entity,batch_size",
    [
        (10, 20, 5, [10, 20, 30], 3, 7, 5),
        (10, 20, 5, [30, 30, 30], 3, 7, 5),
        (20, 20, 10, [30, 30, 30], 5, 10, 3),
        (20, 20, 10, [30, 20, 10], 5, 10, 3),
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
        == (batch_size, num_entity, hidden_dims[-1])
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


@pytest.mark.parametrize(
    "channels,max_len,batch_size,seq_len",
    [
        (4, 10, 5, 6),
        (10, 12, 3, 10),
    ],
)
def test_pos_encoder_tensor2tensor(channels, max_len, batch_size, seq_len):
    pe = PositionalEncoderTensor2Tensor(channels, max_len)
    encoded = pe(
        torch.zeros(
            batch_size,
            seq_len,
            channels,
        )
    )
    assert encoded.size() == (batch_size, seq_len, channels)
    # sanity check, make sure the values of the first dimension of both halves
    # of the channels is sin(0, 1, 2, ...) and cos(0, 1, 2, ...)
    for i in range(batch_size):
        assert encoded[i, :, 0].equal(torch.sin(torch.arange(seq_len).float()))
        assert encoded[i, :, channels // 2].equal(
            torch.cos(torch.arange(seq_len).float())
        )


@pytest.mark.parametrize(
    "d_model,max_len,batch_size,seq_len",
    [
        (4, 10, 5, 6),
        (10, 12, 3, 10),
    ],
)
def test_pos_encoder(d_model, max_len, batch_size, seq_len):
    pe = PositionalEncoder(d_model, max_len)
    encoded = pe(
        torch.zeros(
            batch_size,
            seq_len,
            d_model,
        )
    )
    assert encoded.size() == (batch_size, seq_len, d_model)
    # sanity check, make sure the values of the first dimension is sin(0, 1, 2, ...)
    # and the second dimension is cos(0, 1, 2, ...)
    for i in range(batch_size):
        assert encoded[i, :, 0].equal(torch.sin(torch.arange(seq_len).float()))
        assert encoded[i, :, 1].equal(torch.cos(torch.arange(seq_len).float()))


@pytest.mark.parametrize(
    "channels,kernel_size,batch_size,seq_len",
    [
        (10, 3, 2, 5),
        (15, 5, 3, 10),
        (15, 11, 5, 20),
    ],
)
def test_text_enc_conv_block(channels, kernel_size, batch_size, seq_len):
    conv = TextEncoderConvBlock(channels, kernel_size)
    assert conv(torch.rand(batch_size, seq_len, channels)).size() == (
        batch_size,
        seq_len,
        channels,
    )


@pytest.mark.parametrize(
    "num_conv_layers,kernel_size,hidden_dim,num_heads,batch_size,seq_len",
    [
        (1, 3, 10, 1, 3, 5),
        (3, 5, 12, 3, 3, 10),
    ],
)
def test_text_enc_block(
    num_conv_layers, kernel_size, hidden_dim, num_heads, batch_size, seq_len
):
    text_enc_block = TextEncoderBlock(
        num_conv_layers, kernel_size, hidden_dim, num_heads
    )
    # random tensors and increasing masks
    assert text_enc_block(
        torch.rand(batch_size, seq_len, hidden_dim),
        torch.tensor(
            [[1.0] * (i + 1) + [0.0] * (seq_len - i - 1) for i in range(batch_size)]
        ),
    ).size() == (
        batch_size,
        seq_len,
        hidden_dim,
    )


@pytest.mark.parametrize(
    "word_emb_dim,num_enc_blocks,enc_block_num_conv_layers,enc_block_kernel_size,"
    "enc_block_hidden_dim,enc_block_num_heads,batch_size,seq_len",
    [
        (10, 1, 1, 3, 8, 1, 1, 1),
        (10, 1, 1, 3, 8, 1, 2, 5),
        (20, 3, 5, 5, 10, 5, 3, 7),
    ],
)
def test_text_encoder(
    word_emb_dim,
    num_enc_blocks,
    enc_block_num_conv_layers,
    enc_block_kernel_size,
    enc_block_hidden_dim,
    enc_block_num_heads,
    batch_size,
    seq_len,
):
    text_encoder = TextEncoder(
        word_emb_dim,
        num_enc_blocks,
        enc_block_num_conv_layers,
        enc_block_kernel_size,
        enc_block_hidden_dim,
        enc_block_num_heads,
    )
    # random word ids and increasing masks
    assert text_encoder(
        torch.rand(batch_size, seq_len, word_emb_dim),
        torch.tensor(
            [[1.0] * (i + 1) + [0.0] * (seq_len - i - 1) for i in range(batch_size)]
        ),
    ).size() == (
        batch_size,
        seq_len,
        enc_block_hidden_dim,
    )


def test_masked_softmax():
    batched_input = torch.tensor([[1, 2, 3], [1, 1, 2], [3, 2, 1]]).float()
    batched_mask = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]]).float()
    batched_output = masked_softmax(batched_input, batched_mask, dim=1)

    # compare the result from masked_softmax with regular softmax with filtered values
    for input, mask, output in zip(batched_input, batched_mask, batched_output):
        assert output[output != 0].equal(F.softmax(input[mask == 1], dim=0))


@pytest.mark.parametrize(
    "hidden_dim,batch_size,ctx_seq_len,query_seq_len",
    [
        (10, 1, 2, 4),
        (10, 3, 5, 10),
    ],
)
def test_cqattn_trilinear(hidden_dim, batch_size, ctx_seq_len, query_seq_len):
    ra = ContextQueryAttention(hidden_dim)
    batched_ctx = torch.rand(batch_size, ctx_seq_len, hidden_dim)
    batched_query = torch.rand(batch_size, query_seq_len, hidden_dim)
    batched_similarity = ra.trilinear_for_attention(batched_ctx, batched_query)

    # compare the result from the optimized version to the one from the naive version
    combined_w = torch.cat([ra.w_C, ra.w_Q, ra.w_CQ]).squeeze()
    for similarity, ctx, query in zip(batched_similarity, batched_ctx, batched_query):
        for i in range(ctx_seq_len):
            for j in range(query_seq_len):
                naive_s_ij = torch.matmul(
                    combined_w, torch.cat([ctx[i], query[j], ctx[i] * query[j]])
                )
                assert similarity[i, j].isclose(naive_s_ij, atol=1e-6)


@pytest.mark.parametrize(
    "hidden_dim,batch_size,ctx_seq_len,query_seq_len",
    [
        (10, 1, 3, 5),
        (10, 3, 5, 7),
    ],
)
def test_cqattn(hidden_dim, batch_size, ctx_seq_len, query_seq_len):
    # test against non masked version as masked softmax is the weak point
    ra = ContextQueryAttention(hidden_dim)
    ctx = torch.rand(batch_size, ctx_seq_len, hidden_dim)
    query = torch.rand(batch_size, query_seq_len, hidden_dim)
    output = ra(
        ctx,
        query,
        torch.ones(batch_size, ctx_seq_len),
        torch.ones(batch_size, query_seq_len),
    )
    assert output.size() == (batch_size, ctx_seq_len, 4 * hidden_dim)

    # (batch, ctx_seq_len, query_seq_len)
    similarity = ra.trilinear_for_attention(ctx, query)
    # (batch, ctx_seq_len, query_seq_len)
    s_ctx = F.softmax(similarity, dim=1)
    # (batch, ctx_seq_len, query_seq_len)
    s_query = F.softmax(similarity, dim=2)
    # (batch, ctx_seq_len, hidden_dim)
    P = torch.bmm(s_query, query)
    # (batch, ctx_seq_len, hidden_dim)
    Q = torch.bmm(torch.bmm(s_query, s_ctx.transpose(1, 2)), ctx)

    # (batch, ctx_seq_len, 4 * hidden_dim)
    no_mask_output = torch.cat([ctx, P, ctx * P, ctx * Q], dim=2)

    assert output.equal(no_mask_output)


@pytest.mark.parametrize(
    "hidden_dim,batch_size,repr1_seq_len,repr2_seq_len",
    [
        (10, 1, 3, 5),
        (10, 3, 5, 7),
    ],
)
def test_repr_aggr(hidden_dim, batch_size, repr1_seq_len, repr2_seq_len):
    ra = ReprAggregator(hidden_dim)
    repr1 = torch.rand(batch_size, repr1_seq_len, hidden_dim)
    repr2 = torch.rand(batch_size, repr2_seq_len, hidden_dim)
    repr12, repr21 = ra(
        repr1,
        repr2,
        torch.ones(batch_size, repr1_seq_len),
        torch.ones(batch_size, repr2_seq_len),
    )
    assert repr12.size() == (batch_size, repr1_seq_len, hidden_dim)
    assert repr21.size() == (batch_size, repr2_seq_len, hidden_dim)


@pytest.mark.parametrize(
    "hidden_dim,word_emb_dim,num_nodes,node_emb_dim,num_relations,relation_emb_dim,"
    "text_encoder_num_blocks,text_encoder_num_conv_layers,text_encoder_kernel_size,"
    "text_encoder_num_heads,graph_encoder_num_conv_layers,graph_encoder_num_bases,"
    "batch,obs_len,prev_action_len,rnn_prev_hidden",
    [
        (10, 20, 3, 16, 6, 12, 1, 1, 3, 1, 1, 1, 1, 5, 7, None),
        (10, 20, 3, 16, 6, 12, 1, 1, 3, 1, 1, 1, 1, 5, 7, torch.rand(1, 10)),
        (12, 24, 5, 32, 8, 16, 3, 6, 5, 4, 4, 3, 3, 7, 3, None),
        (12, 24, 5, 32, 8, 16, 3, 6, 5, 4, 4, 3, 3, 7, 3, torch.rand(3, 12)),
    ],
)
def test_graph_updater_forward(
    hidden_dim,
    word_emb_dim,
    num_nodes,
    node_emb_dim,
    num_relations,
    relation_emb_dim,
    text_encoder_num_blocks,
    text_encoder_num_conv_layers,
    text_encoder_kernel_size,
    text_encoder_num_heads,
    graph_encoder_num_conv_layers,
    graph_encoder_num_bases,
    batch,
    obs_len,
    prev_action_len,
    rnn_prev_hidden,
):
    word_embeddings = nn.Embedding(100, word_emb_dim)
    word_embeddings.weight.requires_grad = False
    gu = GraphUpdater(
        hidden_dim,
        word_emb_dim,
        num_nodes,
        node_emb_dim,
        num_relations,
        relation_emb_dim,
        text_encoder_num_blocks,
        text_encoder_num_conv_layers,
        text_encoder_kernel_size,
        text_encoder_num_heads,
        graph_encoder_num_conv_layers,
        graph_encoder_num_bases,
        word_embeddings,
        torch.rand(num_nodes, word_emb_dim),
        torch.rand(num_relations, word_emb_dim),
    )
    results = gu(
        torch.randint(100, (batch, obs_len)),
        torch.randint(100, (batch, prev_action_len)),
        torch.randint(2, (batch, obs_len)).float(),
        torch.randint(2, (batch, prev_action_len)).float(),
        rnn_prev_hidden,
    )
    assert results["h_t"].size() == (batch, hidden_dim)
    assert results["g_t"].size() == (batch, num_relations, num_nodes, num_nodes)

    # pretraining
    gu.pretraining = True
    results = gu(
        torch.randint(100, (batch, obs_len)),
        torch.randint(100, (batch, prev_action_len)),
        torch.randint(2, (batch, obs_len)).float(),
        torch.randint(2, (batch, prev_action_len)).float(),
        rnn_prev_hidden,
    )
    assert results["h_t"].size() == (batch, hidden_dim)
    assert results["g_t"].size() == (batch, num_relations, num_nodes, num_nodes)
    assert results["h_ag"].size() == (batch, prev_action_len, hidden_dim)
    assert results["h_ga"].size() == (batch, num_nodes, hidden_dim)
