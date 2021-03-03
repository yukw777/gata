import pytest
import torch

from train_graph_updater import TextDecoderBlock, TextDecoder


@pytest.mark.parametrize(
    "hidden_dim,num_heads,batch_size,input_seq_len,num_node,prev_action_len",
    [
        (10, 1, 1, 3, 5, 4),
        (20, 2, 3, 5, 10, 8),
    ],
)
def test_text_decoder_block(
    hidden_dim, num_heads, batch_size, input_seq_len, num_node, prev_action_len
):
    decoder_block = TextDecoderBlock(hidden_dim, num_heads)
    assert (
        decoder_block(
            torch.rand(batch_size, input_seq_len, hidden_dim),
            torch.tensor(
                [
                    [1.0] * (i + 1) + [0.0] * (input_seq_len - i - 1)
                    for i in range(batch_size)
                ]
            ),
            torch.rand(batch_size, num_node, hidden_dim),
            torch.rand(batch_size, prev_action_len, hidden_dim),
            torch.tensor(
                [
                    [1.0] * (i + 1) + [0.0] * (prev_action_len - i - 1)
                    for i in range(batch_size)
                ]
            ),
        ).size()
        == (batch_size, input_seq_len, hidden_dim)
    )


@pytest.mark.parametrize(
    "num_dec_blocks,dec_block_hidden_dim,dec_block_num_heads,"
    "batch_size,input_seq_len,num_node,prev_action_len",
    [
        (1, 10, 1, 1, 3, 5, 4),
        (1, 20, 2, 3, 5, 10, 8),
        (3, 10, 1, 1, 3, 5, 4),
        (3, 20, 2, 3, 5, 10, 8),
    ],
)
def test_text_decoder(
    num_dec_blocks,
    dec_block_hidden_dim,
    dec_block_num_heads,
    batch_size,
    input_seq_len,
    num_node,
    prev_action_len,
):
    decoder = TextDecoder(num_dec_blocks, dec_block_hidden_dim, dec_block_num_heads)
    assert (
        decoder(
            torch.rand(batch_size, input_seq_len, dec_block_hidden_dim),
            torch.tensor(
                [
                    [1.0] * (i + 1) + [0.0] * (input_seq_len - i - 1)
                    for i in range(batch_size)
                ]
            ),
            torch.rand(batch_size, num_node, dec_block_hidden_dim),
            torch.rand(batch_size, prev_action_len, dec_block_hidden_dim),
            torch.tensor(
                [
                    [1.0] * (i + 1) + [0.0] * (prev_action_len - i - 1)
                    for i in range(batch_size)
                ]
            ),
        ).size()
        == (batch_size, input_seq_len, dec_block_hidden_dim)
    )