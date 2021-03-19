import pytest
import torch
import torch.nn as nn

from graph_updater import GraphUpdater


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
    num_words = 100
    word_embeddings = nn.Embedding(num_words, word_emb_dim)
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
        torch.randint(num_words, (num_nodes, 5)),
        torch.randint(2, (num_nodes, 5)).float(),
        torch.randint(num_words, (num_relations, 3)),
        torch.randint(2, (num_relations, 3)).float(),
    )
    results = gu(
        torch.randint(num_words, (batch, obs_len)),
        torch.randint(num_words, (batch, prev_action_len)),
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
