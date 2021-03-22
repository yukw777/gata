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


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("rnn_prev_hidden", [True, False])
@pytest.mark.parametrize(
    "batch_size,obs_len,prev_action_len",
    [
        (1, 10, 4),
        (4, 12, 8),
    ],
)
def test_graph_updater_obs_gen_forward(
    batch_size, obs_len, prev_action_len, rnn_prev_hidden, training
):
    g = GraphUpdaterObsGen()
    g.train(training)
    episode_data = {
        "obs_word_ids": torch.randint(g.num_words, (batch_size, obs_len)),
        "obs_mask": torch.randint(2, (batch_size, obs_len)).float(),
        "prev_action_word_ids": torch.randint(
            g.num_words, (batch_size, prev_action_len)
        ),
        "prev_action_mask": torch.randint(2, (batch_size, prev_action_len)).float(),
        "groundtruth_obs_word_ids": torch.randint(g.num_words, (batch_size, obs_len)),
    }
    results = g(
        episode_data,
        rnn_prev_hidden=torch.rand(batch_size, g.hparams.hidden_dim)
        if rnn_prev_hidden
        else None,
    )
    assert results["h_t"].size() == (batch_size, g.hparams.hidden_dim)
    assert results["batch_loss"].size() == (batch_size,)
    if not training:
        assert results["pred_obs_word_ids"].size() == (batch_size, obs_len)
        # decoded_obs_word_ids has variable lengths
        assert results["decoded_obs_word_ids"].size(0) == batch_size
        assert results["decoded_obs_word_ids"].ndim == 2


@pytest.mark.parametrize("batch_size,num_node,prev_action_len", [(1, 3, 5), (3, 10, 7)])
def test_graph_updater_obs_gen_greedy_decode(batch_size, num_node, prev_action_len):
    g = GraphUpdaterObsGen()
    decoded = g.greedy_decode(
        torch.rand(batch_size, num_node, g.hparams.hidden_dim),
        torch.rand(batch_size, prev_action_len, g.hparams.hidden_dim),
        torch.randint(2, (batch_size, prev_action_len)).float(),
    )
    assert decoded.ndim == 2
    assert decoded.size(0) == batch_size
    # [BOS] + max_decode_len
    assert decoded.size(1) <= g.hparams.max_decode_len + 1
    # Always start with BOS
    assert decoded[:, 0].equal(
        torch.tensor([g.preprocessor.word_to_id(BOS)] * batch_size)
    )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("hidden", [True, False])
@pytest.mark.parametrize(
    "batch_size,obs_len,prev_action_len,max_episode_len",
    [
        (1, 5, 3, 6),
        (5, 7, 4, 12),
    ],
)
def test_graph_updater_obs_gen_process_batch(
    batch_size, obs_len, prev_action_len, max_episode_len, hidden, training
):
    g = GraphUpdaterObsGen()
    g.train(training)
    batch = [
        {
            "obs_word_ids": torch.randint(g.num_words, (batch_size, obs_len)),
            "obs_mask": torch.randint(2, (batch_size, obs_len)).float(),
            "prev_action_word_ids": torch.randint(
                g.num_words, (batch_size, prev_action_len)
            ),
            "prev_action_mask": torch.randint(2, (batch_size, prev_action_len)).float(),
            "groundtruth_obs_word_ids": torch.randint(
                g.num_words, (batch_size, obs_len)
            ),
            "step_mask": torch.randint(2, (batch_size,)).float(),
        }
        for _ in range(max_episode_len)
    ]
    h_t = torch.rand(batch_size, g.hparams.hidden_dim) if hidden else None
    results = g.process_batch(batch, h_t=h_t)
    assert len(results["losses"]) == max_episode_len
    assert all(loss.ndim == 0 for loss in results["losses"])
    assert len(results["hiddens"]) == max_episode_len
    assert all(
        hidden.size() == (batch_size, g.hparams.hidden_dim)
        for hidden in results["hiddens"]
    )
    if not training:
        assert len(results["preds"]) == max_episode_len
        assert all(pred.size() == (batch_size, obs_len) for pred in results["preds"])
        assert len(results["decoded"]) == max_episode_len
        assert all(dec.ndim == 2 for dec in results["decoded"])
        assert all(dec.size(0) == batch_size for dec in results["decoded"])
        assert len(results["f1s"]) <= max_episode_len * batch_size
        assert all(f1.ndim == 0 for f1 in results["f1s"])


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
