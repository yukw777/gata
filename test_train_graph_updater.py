import pytest
import torch

from hydra.experimental import initialize, compose

from train_graph_updater import TextDecoderBlock, TextDecoder, main, GraphUpdaterObsGen
from preprocessor import PAD, UNK, BOS, EOS


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


def test_graph_updater_obs_gen_default_init():
    g = GraphUpdaterObsGen()
    default_word_vocab = [PAD, UNK, BOS, EOS]
    assert g.preprocessor.word_vocab == default_word_vocab
    assert g.graph_updater.word_embeddings[0].weight.size() == (
        len(default_word_vocab),
        g.hparams.word_emb_dim,
    )

    # default node_vocab = ['node']
    assert g.graph_updater.node_name_word_ids.size() == (len(g.node_vocab), 1)
    assert g.graph_updater.node_name_mask.size() == (len(g.node_vocab), 1)

    # default relation_vocab = ['relation', 'relation reverse']
    assert g.graph_updater.rel_name_word_ids.size() == (len(g.relation_vocab), 2)
    assert g.graph_updater.rel_name_mask.size() == (len(g.relation_vocab), 2)


def test_main(tmp_path):
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
                "model_size=tiny",
                f"+pl_trainer.default_root_dir={tmp_path}",
                "pl_trainer.max_epochs=1",
            ],
        )
        main(cfg)


def test_main_test_only(tmp_path):
    with initialize(config_path="train_graph_updater_conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data.test_path=test-data/test-data.json",
                "data.test_batch_size=2",
                "data.test_num_workers=0",
                "eval.test_only=true",
                "eval.checkpoint_path=test-data/test.ckpt",
                "model_size=tiny",
                f"+pl_trainer.default_root_dir={tmp_path}",
                "+pl_trainer.limit_test_batches=1",
            ],
        )
        main(cfg)


@pytest.mark.parametrize("step,multiplier", [(0, 0), (3, 0.5), (15, 1), (10000, 1)])
def test_learning_rate_warmup(step, multiplier):
    g = GraphUpdaterObsGen(steps_for_lr_warmup=16)
    assert g.learning_rate_warmup(step) == multiplier
