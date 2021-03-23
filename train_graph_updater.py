import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import random
import wandb
import math

from urllib.parse import urlparse
from typing import List, Dict, Tuple, Optional, Any
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import LambdaLR

from utils import (
    load_fasttext,
    generate_square_subsequent_mask,
    calculate_seq_f1,
    batchify,
)
from preprocessor import BOS, EOS
from graph_updater import GraphUpdater
from layers import PositionalEncoderTensor2Tensor, WordNodeRelInitMixin
from optimizers import RAdam
from graph_updater_data import GraphUpdaterObsGenDataModule


class TextDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim has to be even for positional encoding"
        self.num_heads = num_heads

        self.pos_encoder = PositionalEncoderTensor2Tensor(hidden_dim, 512)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.node_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.prev_action_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.combine_node_prev_action = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU()
        )
        self.linear_layer_norm = nn.LayerNorm(hidden_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        input: torch.Tensor,
        input_mask: torch.Tensor,
        node_hidden: torch.Tensor,
        prev_action_hidden: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input: (batch, input_seq_len, hidden_dim)
        input_mask: (batch, input_seq_len)
        node_hidden: (batch, num_node, hidden_dim)
        prev_action_hidden: (batch, prev_action_len, hidden_dim)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, input_seq_len, hidden_dim)
        """
        # calculate attention mask for decoding
        # this is the mask that prevents MultiheadAttention
        # from attending to future values
        input_seq_len = input.size(1)
        attn_mask = generate_square_subsequent_mask(input_seq_len).to(input.device)
        # (input_seq_len, input_seq_len)

        # add the positional encodings
        pos_encoded_input = self.pos_encoder(input)

        # self attention layer
        input_residual = pos_encoded_input
        # MultiheadAttention expects batch dim to be 1 for q, k, v
        # but 0 for key_padding_mask, so we need to transpose
        transposed_pos_encoded_input = pos_encoded_input.transpose(0, 1)
        input_attn, _ = self.self_attn(
            transposed_pos_encoded_input,
            transposed_pos_encoded_input,
            transposed_pos_encoded_input,
            key_padding_mask=input_mask == 0,
            attn_mask=attn_mask,
        )
        input_attn = input_attn.transpose(0, 1)
        input_attn *= input_mask.unsqueeze(-1)
        input_attn += input_residual
        # (batch, input_seq_len, hidden_dim)

        # calculate self attention for the nodes and previous action
        # strictly speaking, we should calculate attention masks for these
        # based on input_mask, but due to this bug:
        # https://github.com/pytorch/pytorch/issues/41508
        # it returns nan's if we apply attention masks. So let's just skip it.
        # It's OK, b/c we apply input_mask when we combine these.
        # apply layer norm to the input self attention output to calculate the query
        query = self.self_attn_layer_norm(input_attn).transpose(0, 1)
        # (input_seq_len, batch, hidden_dim)

        # self attention for the nodes
        # no key_padding_mask, since we use all the nodes
        # (batch * num_heads, input_seq_len, num_node)
        transposed_node_hidden = node_hidden.transpose(0, 1)
        node_attn, _ = self.node_attn(
            query, transposed_node_hidden, transposed_node_hidden
        )
        node_attn = node_attn.transpose(0, 1)
        # (batch, input_seq_len, hidden_dim)

        # self attention for the previous action
        # key_padding_mask is from prev_action_mask
        # (batch * num_heads, input_seq_len, prev_action_len)
        transposed_prev_action_hidden = prev_action_hidden.transpose(0, 1)
        prev_action_attn, _ = self.prev_action_attn(
            query,
            transposed_prev_action_hidden,
            transposed_prev_action_hidden,
            key_padding_mask=prev_action_mask == 0,
        )
        prev_action_attn = prev_action_attn.transpose(0, 1)
        # (batch, input_seq_len, hidden_dim)

        # combine self attention for the previous action and nodes with
        # input self attention
        combined_self_attn = self.combine_node_prev_action(
            torch.cat([prev_action_attn, node_attn], dim=-1)
        )
        combined_self_attn *= input_mask.unsqueeze(-1)
        combined_self_attn += input_attn
        # (batch, input_seq_len, hidden_dim)

        # linear layer
        output = self.linear_layer_norm(combined_self_attn)
        output = self.linear_layers(output)
        output += combined_self_attn
        # (batch, input_seq_len, hidden_dim)

        return output


class TextDecoder(nn.Module):
    def __init__(
        self, num_dec_blocks: int, dec_block_hidden_dim: int, dec_block_num_heads: int
    ) -> None:
        super().__init__()
        self.dec_blocks = nn.ModuleList(
            TextDecoderBlock(dec_block_hidden_dim, dec_block_num_heads)
            for _ in range(num_dec_blocks)
        )

    def forward(
        self,
        input: torch.Tensor,
        input_mask: torch.Tensor,
        node_hidden: torch.Tensor,
        prev_action_hidden: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input: (batch, input_seq_len, hidden_dim)
        input_mask: (batch, input_seq_len)
        node_hidden: (batch, num_node, hidden_dim)
        prev_action_hidden: (batch, prev_action_len, hidden_dim)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, input_seq_len, hidden_dim)
        """
        # (batch_size, input_seq_len, hidden_dim)
        output = input
        for dec_block in self.dec_blocks:
            output = dec_block(
                output, input_mask, node_hidden, prev_action_hidden, prev_action_mask
            )
        # (batch_size, input_seq_len, hidden_dim)

        return output


class GraphUpdaterObsGen(WordNodeRelInitMixin, pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 8,
        word_emb_dim: int = 300,
        node_emb_dim: int = 12,
        relation_emb_dim: int = 10,
        text_encoder_num_blocks: int = 1,
        text_encoder_num_conv_layers: int = 3,
        text_encoder_kernel_size: int = 5,
        text_encoder_num_heads: int = 1,
        graph_encoder_num_cov_layers: int = 4,
        graph_encoder_num_bases: int = 3,
        text_decoder_num_blocks: int = 1,
        text_decoder_num_heads: int = 1,
        learning_rate: float = 5e-4,
        sample_k_gen_obs: int = 5,
        max_decode_len: int = 200,
        steps_for_lr_warmup: int = 10000,
        pretrained_word_embedding_path: Optional[str] = None,
        word_vocab_path: Optional[str] = None,
        node_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "hidden_dim",
            "word_emb_dim",
            "node_emb_dim",
            "relation_emb_dim",
            "text_encoder_num_blocks",
            "text_encoder_num_conv_layers",
            "text_encoder_kernel_size",
            "text_encoder_num_heads",
            "graph_encoder_num_cov_layers",
            "graph_encoder_num_bases",
            "text_decoder_num_blocks",
            "text_decoder_num_heads",
            "learning_rate",
            "sample_k_gen_obs",
            "max_decode_len",
            "steps_for_lr_warmup",
        )

        # initialize word (preprocessor), node and relation stuff
        (
            node_name_word_ids,
            node_name_mask,
            rel_name_word_ids,
            rel_name_mask,
        ) = self.init_word_node_rel(
            word_vocab_path=to_absolute_path(word_vocab_path)
            if word_vocab_path is not None
            else None,
            node_vocab_path=to_absolute_path(node_vocab_path)
            if node_vocab_path is not None
            else None,
            relation_vocab_path=to_absolute_path(relation_vocab_path)
            if relation_vocab_path is not None
            else None,
        )

        # load pretrained word embedding and freeze it
        if pretrained_word_embedding_path is not None:
            pretrained_word_embedding = load_fasttext(
                to_absolute_path(pretrained_word_embedding_path), self.preprocessor
            )
        else:
            pretrained_word_embedding = nn.Embedding(self.num_words, word_emb_dim)
        pretrained_word_embedding.weight.requires_grad = False

        # graph updater
        self.graph_updater = GraphUpdater(
            self.hparams.hidden_dim,  # type: ignore
            self.hparams.word_emb_dim,  # type: ignore
            len(self.node_vocab),
            self.hparams.node_emb_dim,  # type: ignore
            len(self.relation_vocab),
            self.hparams.relation_emb_dim,  # type: ignore
            self.hparams.text_encoder_num_blocks,  # type: ignore
            self.hparams.text_encoder_num_conv_layers,  # type: ignore
            self.hparams.text_encoder_kernel_size,  # type: ignore
            self.hparams.text_encoder_num_heads,  # type: ignore
            self.hparams.graph_encoder_num_cov_layers,  # type: ignore
            self.hparams.graph_encoder_num_bases,  # type: ignore
            pretrained_word_embedding,
            node_name_word_ids,
            node_name_mask,
            rel_name_word_ids,
            rel_name_mask,
        )
        self.graph_updater.pretraining = True

        # text decoder
        self.text_decoder = TextDecoder(
            text_decoder_num_blocks, hidden_dim, text_decoder_num_heads
        )
        self.target_word_prj = nn.Linear(hidden_dim, self.num_words, bias=False)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=self.preprocessor.pad_id, reduction="none"
        )

    def forward(  # type: ignore
        self,
        episode_data: Dict[str, torch.Tensor],
        rnn_prev_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        episode_data:
        {
            'obs_word_ids': tensor of shape (batch, obs_len),
            'obs_mask': tensor of shape (batch, obs_len),
            'prev_action_word_ids': tensor of shape (batch, prev_action_len),
            'prev_action_mask': tensor of shape (batch, prev_action_len),
            'groundtruth_obs_word_ids': tensor of shape (batch, obs_len),
        }
        rnn_prev_hidden: (batch, hidden_dim)

        output:
        {
            'h_t': hidden state of the rnn cell at time t; (batch, hidden_dim),
            'batch_loss': batch loss for this episode data. (batch),
            'pred_obs_word_ids': predicted observation word IDs. Only for eval.
                (batch, obs_len),
            'decoded_obs_word_ids': decoded observation word IDs. Only for eval.
                (batch, obs_len),
        }
        """
        # graph updater
        graph_updater_results = self.graph_updater(
            episode_data["obs_word_ids"],
            episode_data["prev_action_word_ids"],
            episode_data["obs_mask"],
            episode_data["prev_action_mask"],
            rnn_prev_hidden=rnn_prev_hidden,
        )

        # decode
        decoder_output = self.text_decoder(
            graph_updater_results["prj_obs"],
            episode_data["obs_mask"],
            graph_updater_results["h_ga"],
            graph_updater_results["h_ag"],
            episode_data["prev_action_mask"],
        )
        # (batch, obs_len, hidden_dim)
        decoder_output = self.target_word_prj(decoder_output)
        # (batch, obs_len, num_words)

        batch_size = decoder_output.size(0)
        batch_loss = (
            self.ce_loss(
                decoder_output.view(-1, decoder_output.size(-1)),
                episode_data["groundtruth_obs_word_ids"].flatten(),
            )
            .view(batch_size, -1)
            .sum(dim=1)
        )
        # (batch)

        results = {
            "h_t": graph_updater_results["h_t"].detach(),
            "batch_loss": batch_loss,
        }

        if self.training:
            return results

        results["pred_obs_word_ids"] = (
            (
                decoder_output
                * (episode_data["groundtruth_obs_word_ids"] != 0).float().unsqueeze(-1)
            )
            .argmax(dim=-1)
            .detach()
        )
        # (batch, obs_len)
        results["decoded_obs_word_ids"] = self.greedy_decode(
            graph_updater_results["h_ga"],
            graph_updater_results["h_ag"],
            episode_data["prev_action_mask"],
        )
        # (batch, decoded_len)

        return results

    def greedy_decode(
        self,
        node_hidden: torch.Tensor,
        prev_action_hidden: torch.Tensor,
        prev_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Start with "bos" and greedy decode.
        node_hidden: (batch, num_node, hidden_dim)
        prev_action_hidden: (batch, prev_action_len, hidden_dim)
        prev_action_mask: (batch, prev_action_len)

        output: (batch, decoded_len)
        """
        # start with bos tokens
        batch_size = node_hidden.size(0)
        bos_id = self.preprocessor.word_to_id(BOS)
        decoded_word_ids = torch.tensor(
            [[bos_id] for _ in range(batch_size)], device=self.device
        )
        # (batch, 1)
        eos_id = self.preprocessor.word_to_id(EOS)
        eos_mask = torch.tensor([False] * batch_size, device=self.device)
        # (batch)
        for _ in range(self.hparams.max_decode_len):  # type: ignore
            input = self.graph_updater.word_embeddings(decoded_word_ids)
            # (batch, curr_decode_len, hidden_dim)
            input_mask = decoded_word_ids.ne(self.preprocessor.pad_id).float()
            # (batch, curr_decode_len)
            decoder_output = self.target_word_prj(
                self.text_decoder(
                    input, input_mask, node_hidden, prev_action_hidden, prev_action_mask
                )
            )
            # (batch, curr_decode_len, num_words)
            preds = decoder_output[:, -1].argmax(dim=-1)
            # (batch)

            # add new decoded words to decoded_word_ids
            # if we've hit eos, we add padding
            decoded_word_ids = torch.cat(
                [
                    decoded_word_ids,
                    preds.masked_fill(eos_mask, self.preprocessor.pad_id).unsqueeze(-1),
                ],
                dim=-1,
            )
            # (batch, curr_decode_len + 1)

            # update the eos_mask. once it's True b/c we hit eos, it never goes back.
            eos_mask = eos_mask.logical_or(preds == eos_id)
            # (batch)

            # if all the sequences have reached eos, break
            if eos_mask.all():
                break
        return decoded_word_ids

    def process_batch(
        self,
        batch: List[Dict[str, torch.Tensor]],
        h_t: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        batch: [
            {
                'obs_word_ids': tensor of shape (batch, obs_len),
                'obs_mask': tensor of shape (batch, obs_len),
                'prev_action_word_ids': tensor of shape (batch, prev_action_len),
                'prev_action_mask': tensor of shape (batch, prev_action_len),
                'groundtruth_obs_word_ids': tensor of shape (batch, obs_len),
                'step_mask': tensor of shape (batch),
            },
            ...
        ]
        h_t: (batch, hidden_dim)

        output: {
            'losses': [scalar masked mean batch loss, ...], length == max_episode_len
            'hiddens': [rnn hidden of shape (batch, hidden_dim), ...],
                length == max_episode_len
            'preds': [predicted word ids of shape (batch, obs_len), ...],
                length == max_episode_len, eval only
            'decoded': [decoded word ids of shape (batch, decoded_len), ...],
                length == max_episode_len, eval only
            'f1s': [scalar f1 scores, ...], length <= max_episode_len * batch, eval only
        }
        """
        losses: List[torch.Tensor] = []
        f1s: List[torch.Tensor] = []
        preds: List[torch.Tensor] = []
        decoded: List[torch.Tensor] = []
        hiddens: List[torch.Tensor] = []
        eos_id = self.preprocessor.word_to_id(EOS)
        for i, episode_data in enumerate(batch):
            results = self(episode_data, rnn_prev_hidden=h_t)
            h_t = results["h_t"]
            assert h_t is not None
            hiddens.append(h_t)
            losses.append(
                torch.sum(results["batch_loss"] * episode_data["step_mask"])
                / episode_data["step_mask"].sum()
            )
            if not self.training:
                preds.append(results["pred_obs_word_ids"])
                decoded.append(results["decoded_obs_word_ids"])
                for j, (
                    padded_decoded_word_ids,
                    padded_groundtruth_word_ids,
                ) in enumerate(
                    zip(
                        results["decoded_obs_word_ids"].tolist(),
                        episode_data["groundtruth_obs_word_ids"].tolist(),
                    ),
                ):
                    if episode_data["step_mask"][j] == 0:
                        # if step is masked, skip
                        continue

                    # cut at eos
                    try:
                        decoded_eos_id = padded_decoded_word_ids.index(eos_id)
                    except ValueError:
                        decoded_eos_id = -1
                    try:
                        groundtruth_eos_id = padded_groundtruth_word_ids.index(eos_id)
                    except ValueError:
                        groundtruth_eos_id = -1

                    # calculate f1
                    f1s.append(
                        torch.tensor(
                            calculate_seq_f1(
                                padded_decoded_word_ids[:decoded_eos_id],
                                padded_groundtruth_word_ids[:groundtruth_eos_id],
                            )
                        )
                    )

        results = {"losses": losses, "hiddens": hiddens}
        if self.training:
            return results

        results["preds"] = preds
        results["decoded"] = decoded
        results["f1s"] = f1s
        return results

    def training_step(  # type: ignore
        self,
        batch: List[Dict[str, torch.Tensor]],
        batch_idx: int,
        hiddens: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        results = self.process_batch(batch, h_t=hiddens)
        loss = torch.stack(results["losses"]).mean()
        self.log("train_loss", loss, prog_bar=True)
        return {
            "loss": loss,
            "hiddens": results["hiddens"][-1],
        }

    def tbptt_split_batch(self, batch, split_size: int):
        return list(batchify(batch, split_size))

    def eval_step(
        self,
        batch: List[Dict[str, torch.Tensor]],
        log_key_prefix: str,
    ) -> List[Tuple[str, str, str]]:
        results = self.process_batch(batch)
        self.log(
            log_key_prefix + "loss",
            torch.stack(results["losses"]).mean(),
            sync_dist=True,
        )
        self.log(log_key_prefix + "f1", torch.stack(results["f1s"]).mean())
        return self.gen_decoded_groundtruth_pred_table(
            batch, results["preds"], results["decoded"]
        )

    def validation_step(  # type: ignore
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    ) -> List[Tuple[str, str, str]]:
        return self.eval_step(batch, "val_")

    def gen_decoded_groundtruth_pred_table(
        self,
        episode_seq: List[Dict[str, torch.Tensor]],
        preds: List[torch.Tensor],
        decoded: List[torch.Tensor],
    ) -> List[Tuple[str, str, str]]:

        groundtruth_obs_word_ids = [
            word_ids
            for episode in episode_seq
            for word_ids in episode["groundtruth_obs_word_ids"].tolist()
        ]
        # decode all the predicted observations
        pred_obs_word_ids = [
            word_ids
            for pred_obs_word_ids in preds
            for word_ids in pred_obs_word_ids.tolist()
        ]
        # decode all the decoded observations
        decoded_obs_word_ids = [
            word_ids
            for decoded_obs_word_ids in decoded
            for word_ids in decoded_obs_word_ids.tolist()
        ]
        return list(
            zip(
                [
                    obs
                    for obs in self.preprocessor.decode(groundtruth_obs_word_ids)
                    if len(obs.split()) > 1
                ],
                [
                    obs
                    for obs in self.preprocessor.decode(pred_obs_word_ids)
                    if len(obs.split()) > 1
                ],
                [
                    obs
                    for obs in self.preprocessor.decode(decoded_obs_word_ids)
                    if len(obs.split()) > 1
                ],
            )
        )

    def wandb_log_gen_obs(
        self, outputs: List[List[List[str]]], table_title: str
    ) -> None:
        flat_outputs = [item for sublist in outputs for item in sublist]
        data = (
            random.sample(flat_outputs, self.hparams.sample_k_gen_obs)  # type: ignore
            if len(flat_outputs) >= self.hparams.sample_k_gen_obs  # type: ignore
            else flat_outputs
        )
        self.logger.experiment.log(
            {
                table_title: wandb.Table(
                    data=data, columns=["Groundtruth", "Predicted", "Decoded"]
                )
            }
        )

    def validation_epoch_end(self, outputs: List[List[List[str]]]) -> None:
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(
                outputs, f"Generated Observations Val Epoch {self.current_epoch}"
            )

    def test_step(  # type: ignore
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    ) -> List[Tuple[str, str, str]]:
        return self.eval_step(batch, "test_")

    def test_epoch_end(self, outputs: List[List[List[str]]]) -> None:
        if isinstance(self.logger, WandbLogger):
            self.wandb_log_gen_obs(
                outputs, f"Generated Observations Test Epoch {self.current_epoch}"
            )

    def learning_rate_warmup(self, step: int) -> float:
        if step < self.hparams.steps_for_lr_warmup:  # type: ignore
            return math.log2(step + 1) / math.log2(
                self.hparams.steps_for_lr_warmup  # type: ignore
            )
        return 1.0

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = LambdaLR(optimizer, self.learning_rate_warmup)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class WandbSaveCallback(Callback):
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> dict:
        if isinstance(trainer.logger, WandbLogger):
            wandb.save(f"gata/{trainer.logger.version}/checkpoints/*.ckpt")
        return {}


@hydra.main(config_path="train_graph_updater_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    # seed
    pl.seed_everything(42)

    # trainer
    trainer_config = OmegaConf.to_container(cfg.pl_trainer, resolve=True)
    assert isinstance(trainer_config, dict)
    trainer_config["logger"] = instantiate(cfg.logger) if "logger" in cfg else True
    if isinstance(trainer_config["logger"], WandbLogger):
        trainer_config["callbacks"] = [WandbSaveCallback()]
    trainer = pl.Trainer(
        **trainer_config,
        checkpoint_callback=ModelCheckpoint(monitor="val_loss", mode="min"),
    )

    # set up data module
    dm = GraphUpdaterObsGenDataModule(**cfg.data)

    # test
    if not cfg.eval.test_only:
        # instantiate the lightning module
        lm = GraphUpdaterObsGen(
            **cfg.model, **cfg.train, max_decode_len=cfg.eval.max_decode_len
        )

        # fit
        trainer.fit(lm, datamodule=dm)

        # test
        trainer.test(datamodule=dm)
    else:
        assert (
            cfg.eval.checkpoint_path is not None
        ), "missing checkpoint path for testing"
        parsed = urlparse(cfg.eval.checkpoint_path)
        if parsed.scheme == "":
            # local path
            ckpt_path = to_absolute_path(cfg.eval.checkpoint_path)
        else:
            # remote path
            ckpt_path = cfg.eval.checkpoint_path
        model = GraphUpdaterObsGen.load_from_checkpoint(
            ckpt_path, **cfg.model, max_decode_len=cfg.eval.max_decode_len
        )
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
