import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import List, Dict, Tuple
from utils import load_fasttext, masked_mean, generate_square_subsequent_mask

from preprocessor import SpacyPreprocessor
from graph_updater import GraphUpdater, PositionalEncoderTensor2Tensor
from optimizers import RAdam


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
        pos_encoded_input = pos_encoded_input.transpose(0, 1)
        input_attn, _ = self.self_attn(
            pos_encoded_input,
            pos_encoded_input,
            pos_encoded_input,
            key_padding_mask=input_mask == 0,
            attn_mask=attn_mask,
        )
        input_attn = input_attn.transpose(0, 1)
        input_attn *= input_mask.unsqueeze(-1)
        input_attn += input_residual
        # (batch, input_seq_len, hidden_dim)

        # calculate self attention for the nodes and previous action
        # apply layer norm to the input self attention output to calculate the query
        query = self.self_attn_layer_norm(input_attn).transpose(0, 1)
        # (input_seq_len, batch, hidden_dim)

        # self attention for the nodes
        # no key_padding_mask, since we use all the nodes
        # attn_mask is calculated from input_mask
        num_node = node_hidden.size(1)
        node_attn_mask = (
            input_mask.unsqueeze(-1)
            .expand(-1, -1, num_node)
            .repeat(self.num_heads, 1, 1)
            == 0
        )
        # (batch * num_heads, input_seq_len, num_node)
        node_hidden = node_hidden.transpose(0, 1)
        node_attn, _ = self.node_attn(
            query, node_hidden, node_hidden, attn_mask=node_attn_mask
        )
        node_attn = node_attn.transpose(0, 1)
        # (batch, input_seq_len, hidden_dim)

        # self attention for the previous action
        # key_padding_mask is from prev_action_mask
        # attn_mask is calculated from input_mask
        prev_action_len = prev_action_hidden.size(1)
        prev_action_attn_mask = (
            input_mask.unsqueeze(-1)
            .expand(-1, -1, prev_action_len)
            .repeat(self.num_heads, 1, 1)
        )
        # (batch * num_heads, input_seq_len, prev_action_len)
        prev_action_hidden = prev_action_hidden.transpose(0, 1)
        prev_action_attn, _ = self.prev_action_attn(
            query,
            prev_action_hidden,
            prev_action_hidden,
            key_padding_mask=prev_action_mask == 0,
            attn_mask=prev_action_attn_mask,
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


class GraphUpdaterObsGen(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        word_vocab_path: str,
        pretrained_word_embedding_path: str,
        word_emb_dim: int,
        node_vocab_path: str,
        node_emb_dim: int,
        relation_vocab_path: str,
        relation_emb_dim: int,
        text_encoder_num_blocks: int,
        text_encoder_num_conv_layers: int,
        text_encoder_kernel_size: int,
        text_encoder_num_heads: int,
        graph_encoder_num_cov_layers: int,
        graph_encoder_num_bases: int,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # load preprocessor
        self.preprocessor = SpacyPreprocessor.load_from_file(word_vocab_path)

        # load pretrained word embedding and freeze it
        pretrained_word_embedding = load_fasttext(
            pretrained_word_embedding_path, self.preprocessor
        )
        pretrained_word_embedding.weight.requires_grad = False

        # load node vocab
        with open(node_vocab_path, "r") as f:
            self.node_vocab = [node_name.strip() for node_name in f]

        # load relation vocab
        with open(relation_vocab_path, "r") as f:
            self.relation_vocab = [relation_name.strip() for relation_name in f]
        # add reverse relations
        self.relation_vocab += [rel + " reverse" for rel in self.relation_vocab]

        # calculate mean masked node name embeddings
        node_name_word_ids, node_name_mask = self.preprocessor.preprocess(
            self.node_vocab
        )
        node_name_embeddings = masked_mean(
            pretrained_word_embedding(node_name_word_ids), node_name_mask
        )
        rel_name_word_ids, rel_name_mask = self.preprocessor.preprocess(
            self.relation_vocab
        )
        rel_name_embeddings = masked_mean(
            pretrained_word_embedding(rel_name_word_ids), rel_name_mask
        )

        # graph updater
        self.graph_updater = GraphUpdater(
            hidden_dim,
            word_emb_dim,
            len(self.node_vocab),
            node_emb_dim,
            len(self.relation_vocab),
            relation_emb_dim,
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            text_encoder_num_heads,
            graph_encoder_num_cov_layers,
            graph_encoder_num_bases,
            pretrained_word_embedding,
            node_name_embeddings,
            rel_name_embeddings,
        )
        self.graph_updater.pretraining = True

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(
        self, batch: Tuple[List[Dict[str, torch.Tensor]], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        episode_batch, episode_mask = batch
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.learning_rate)
