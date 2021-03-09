import pytest
import torch

from hydra.experimental import initialize, compose

from train_graph_updater import TextDecoderBlock, TextDecoder, main


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


def test_main():
    with initialize(config_path="train_graph_updater_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data.train_path=test-data/test-data.json",
                "data.train_batch_size=2",
                "data.train_num_workers=0",
                "data.val_path=test-data/test-data.json",
                "data.val_batch_size=2",
                "data.val_num_workers=0",
                "data.test_path=test-data/test-data.json",
                "data.test_batch_size=2",
                "data.test_num_workers=0",
                "+pl_trainer.fast_dev_run=true",
            ],
        )
        main(cfg)


def test_main_test():
    with initialize(config_path="train_graph_updater_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data.test_path=test-data/test-data.json",
                "data.test_batch_size=2",
                "data.test_num_workers=0",
                "eval.run_test=true",
                "eval.checkpoint_path=test-data/test.ckpt",
                "+pl_trainer.limit_test_batches=1",
            ],
        )
        main(cfg)
