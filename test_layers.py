import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import (
    RelationalGraphConvolution,
    RGCNHighwayConnections,
    GraphEncoder,
    DepthwiseSeparableConv1d,
    PositionalEncoder,
    PositionalEncoderTensor2Tensor,
    TextEncoderConvBlock,
    TextEncoderBlock,
    TextEncoder,
    ContextQueryAttention,
    ReprAggregator,
    EncoderMixin,
    WordNodeRelInitMixin,
)
from utils import increasing_mask
from preprocessor import PAD, UNK, BOS, EOS


@pytest.mark.parametrize(
    "node_input_dim,relation_input_dim,num_relations,out_dim,"
    "num_bases,num_nodes,batch_size",
    [
        (10, 20, 5, 25, 3, 7, 5),
        (20, 20, 10, 20, 5, 10, 3),
    ],
)
def test_r_gcn(
    node_input_dim,
    relation_input_dim,
    num_relations,
    out_dim,
    num_bases,
    num_nodes,
    batch_size,
):
    rgcn = RelationalGraphConvolution(
        node_input_dim, relation_input_dim, num_relations, out_dim, num_bases
    )
    assert (
        rgcn(
            torch.rand(batch_size, num_nodes, node_input_dim),
            torch.rand(batch_size, num_relations, relation_input_dim),
            torch.rand(batch_size, num_relations, num_nodes, num_nodes),
        ).size()
        == (batch_size, num_nodes, out_dim)
    )


@pytest.mark.parametrize(
    "node_input_dim,relation_input_dim,num_relations,out_dim,"
    "num_bases,num_nodes,batch_size",
    [
        (10, 20, 5, 25, 3, 7, 5),
        (20, 20, 10, 20, 5, 10, 3),
    ],
)
def test_r_gcn_get_supports(
    node_input_dim,
    relation_input_dim,
    num_relations,
    out_dim,
    num_bases,
    num_nodes,
    batch_size,
):
    rgcn = RelationalGraphConvolution(
        node_input_dim, relation_input_dim, num_relations, out_dim, num_bases
    )
    node_features = torch.rand(batch_size, num_nodes, node_input_dim)
    relation_features = torch.rand(batch_size, num_relations, relation_input_dim)
    adj = torch.rand(batch_size, num_relations, num_nodes, num_nodes)
    assert rgcn.get_supports(node_features, relation_features, adj).equal(
        rgcn.optimized_get_supports(node_features, relation_features, adj)
    )


@pytest.mark.parametrize(
    "node_input_dim,relation_input_dim,num_relations,out_dim,"
    "num_bases,num_nodes,batch_size,output_size",
    [
        (10, 20, 5, 25, 3, 7, 5, (5, 7, 25)),
        (20, 20, 10, 20, 5, 10, 3, (3, 10, 20)),
    ],
)
def test_r_gcn_highway_connections(
    node_input_dim,
    relation_input_dim,
    num_relations,
    out_dim,
    num_bases,
    num_nodes,
    batch_size,
    output_size,
):
    rgcn = RGCNHighwayConnections(
        node_input_dim, relation_input_dim, num_relations, out_dim, num_bases
    )
    assert (
        rgcn(
            torch.rand(batch_size, num_nodes, node_input_dim),
            torch.rand(batch_size, num_relations, relation_input_dim),
            torch.rand(batch_size, num_relations, num_nodes, num_nodes),
        ).size()
        == output_size
    )


@pytest.mark.parametrize(
    "node_input_dim,relation_input_dim,num_relations,hidden_dims,"
    "num_bases,num_nodes,batch_size",
    [
        (10, 20, 5, [10, 20, 30], 3, 7, 5),
        (10, 20, 5, [30, 30, 30], 3, 7, 5),
        (20, 20, 10, [30, 30, 30], 5, 10, 3),
        (20, 20, 10, [30, 20, 10], 5, 10, 3),
    ],
)
def test_graph_encoder(
    node_input_dim,
    relation_input_dim,
    num_relations,
    hidden_dims,
    num_bases,
    num_nodes,
    batch_size,
):
    graph_encoder = GraphEncoder(
        node_input_dim, relation_input_dim, num_relations, hidden_dims, num_bases
    )
    assert (
        graph_encoder(
            torch.rand(batch_size, num_nodes, node_input_dim),
            torch.rand(batch_size, num_relations, relation_input_dim),
            torch.rand(batch_size, num_relations, num_nodes, num_nodes),
        ).size()
        == (batch_size, num_nodes, hidden_dims[-1])
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
        increasing_mask(batch_size, seq_len),
    ).size() == (
        batch_size,
        seq_len,
        hidden_dim,
    )


@pytest.mark.parametrize(
    "num_enc_blocks,enc_block_num_conv_layers,enc_block_kernel_size,"
    "enc_block_hidden_dim,enc_block_num_heads,batch_size,seq_len",
    [
        (1, 1, 3, 8, 1, 1, 1),
        (1, 1, 3, 8, 1, 2, 5),
        (3, 5, 5, 10, 5, 3, 7),
    ],
)
def test_text_encoder(
    num_enc_blocks,
    enc_block_num_conv_layers,
    enc_block_kernel_size,
    enc_block_hidden_dim,
    enc_block_num_heads,
    batch_size,
    seq_len,
):
    text_encoder = TextEncoder(
        num_enc_blocks,
        enc_block_num_conv_layers,
        enc_block_kernel_size,
        enc_block_hidden_dim,
        enc_block_num_heads,
    )
    # random word ids and increasing masks
    assert text_encoder(
        torch.rand(batch_size, seq_len, enc_block_hidden_dim),
        torch.tensor(
            [[1.0] * (i + 1) + [0.0] * (seq_len - i - 1) for i in range(batch_size)]
        ),
    ).size() == (
        batch_size,
        seq_len,
        enc_block_hidden_dim,
    )


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
    "num_words,hidden_dim,node_emb_dim,rel_emb_dim,num_node,num_relations,"
    "batch_size,seq_len",
    [
        (100, 12, 24, 36, 10, 12, 5, 6),
    ],
)
def test_encoder_mixin(
    num_words,
    hidden_dim,
    node_emb_dim,
    rel_emb_dim,
    num_node,
    num_relations,
    batch_size,
    seq_len,
):
    class TestEncoder(EncoderMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(num_words, hidden_dim)
            self.text_encoder = TextEncoder(1, 1, 1, hidden_dim, 1)
            self.graph_encoder = GraphEncoder(
                hidden_dim + node_emb_dim,
                hidden_dim + rel_emb_dim,
                num_relations,
                [hidden_dim],
                1,
            )
            self.node_embeddings = nn.Embedding(num_node, node_emb_dim)
            self.relation_embeddings = nn.Embedding(num_relations, rel_emb_dim)

            self.node_name_word_ids = torch.randint(num_words, (num_node, 3))
            self.node_name_mask = increasing_mask(num_node, 3)
            self.rel_name_word_ids = torch.randint(num_words, (num_relations, 2))
            self.rel_name_mask = increasing_mask(num_relations, 2)

    te = TestEncoder()
    assert (
        te.encode_text(
            torch.randint(num_words, (batch_size, seq_len)),
            increasing_mask(batch_size, seq_len),
        ).size()
        == (batch_size, seq_len, hidden_dim)
    )
    assert te.get_node_features().size() == (num_node, hidden_dim + node_emb_dim)
    assert te.get_relation_features().size() == (
        num_relations,
        hidden_dim + rel_emb_dim,
    )
    assert te.encode_graph(
        torch.rand(batch_size, num_relations, num_node, num_node)
    ).size() == (batch_size, num_node, hidden_dim)


def test_word_node_rel_init_mixin():
    class TestWordNodeRelInitMixin(WordNodeRelInitMixin):
        pass

    test_init_mixin = TestWordNodeRelInitMixin()

    # default values
    (
        node_name_word_ids,
        node_name_mask,
        rel_name_word_ids,
        rel_name_mask,
    ) = test_init_mixin.init_word_node_rel()
    assert test_init_mixin.preprocessor.word_vocab == [PAD, UNK, BOS, EOS]
    assert test_init_mixin.num_words == 4
    assert test_init_mixin.num_nodes == 1
    assert test_init_mixin.num_relations == 2
    assert node_name_word_ids.size() == (1, 1)
    assert node_name_mask.size() == (1, 1)
    assert rel_name_word_ids.size() == (2, 2)
    assert rel_name_mask.size() == (2, 2)

    # provide vocab files
    (
        node_name_word_ids,
        node_name_mask,
        rel_name_word_ids,
        rel_name_mask,
    ) = test_init_mixin.init_word_node_rel(
        word_vocab_path="vocabs/word_vocab.txt",
        node_vocab_path="vocabs/node_vocab.txt",
        relation_vocab_path="vocabs/relation_vocab.txt",
    )
    assert len(test_init_mixin.preprocessor.word_vocab) == 772
    assert test_init_mixin.num_words == 772
    assert test_init_mixin.num_nodes == 99
    assert test_init_mixin.num_relations == 20
    assert node_name_word_ids.size() == (99, 4)
    assert node_name_mask.size() == (99, 4)
    assert rel_name_word_ids.size() == (20, 3)
    assert rel_name_mask.size() == (20, 3)
