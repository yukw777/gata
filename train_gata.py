import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import gym
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from textworld import EnvInfos

from utils import load_textworld_games
from layers import WordNodeRelInitMixin
from action_selector import ActionSelector
from graph_updater import GraphUpdater


def request_infos_for_train() -> EnvInfos:
    request_infos = EnvInfos()
    request_infos.admissible_commands = True
    request_infos.description = False
    request_infos.location = False
    request_infos.facts = False
    request_infos.last_action = False
    request_infos.game = True

    return request_infos


def request_infos_for_eval() -> EnvInfos:
    request_infos = EnvInfos()
    request_infos.admissible_commands = True
    request_infos.description = True
    request_infos.location = True
    request_infos.facts = True
    request_infos.last_action = True
    request_infos.game = True
    return request_infos


def get_game_dir(
    base_dir_path: str,
    dataset: str,
    difficulty_level: int,
    training_size: Optional[int] = None,
) -> str:
    return os.path.join(
        base_dir_path,
        dataset + ("" if training_size is None else f"_{training_size}"),
        f"difficulty_level_{difficulty_level}",
    )


@dataclass
class Transition:
    """
    Represents a transition in one single game.
    """

    # game observation
    ob: str
    # action candidates
    action_cands: List[str]
    # current graph
    current_graph: torch.Tensor
    # chosen action ID
    action_id: int
    # received reward
    reward: float
    # next observation
    next_ob: str
    # next action candidates
    next_action_cands: List[str]
    # next graph
    next_graph: torch.Tensor
    # done
    done: bool


class TransitionCache:
    def __init__(self, batch_size: int) -> None:
        # cache[i][j] = j'th transition of i'th game
        self.cache: List[List[Transition]] = [[] for _ in range(batch_size)]

    def batch_add(
        self,
        obs: List[str],
        batch_action_cands: List[List[str]],
        current_graphs: torch.Tensor,
        actions_idx: List[int],
        rewards: List[float],
        dones: List[bool],
        next_obs: List[str],
        batch_next_action_cands: List[List[str]],
        next_graphs: torch.Tensor,
    ) -> None:
        for i, (
            ob,
            action_cands,
            current_graph,
            action_id,
            reward,
            done,
            next_ob,
            next_action_cands,
            next_graph,
        ) in enumerate(
            zip(
                obs,
                batch_action_cands,
                current_graphs,
                actions_idx,
                rewards,
                dones,
                next_obs,
                batch_next_action_cands,
                next_graphs,
            )
        ):
            if len(self.cache[i]) > 0 and done and self.cache[i][-1].done:
                # this game is already done, don't add this transition
                continue
            self.cache[i].append(
                Transition(
                    ob=ob,
                    action_cands=action_cands,
                    current_graph=current_graph,
                    action_id=action_id,
                    reward=reward,
                    next_ob=next_ob,
                    next_action_cands=next_action_cands,
                    next_graph=next_graph,
                    done=done,
                )
            )

    def get_avg_rewards(self) -> List[float]:
        return [
            np.mean([transition.reward for transition in game])  # type: ignore
            for game in self.cache
        ]


class GATADoubleDQN(WordNodeRelInitMixin, pl.LightningModule):
    def __init__(
        self,
        difficulty_level: int = 1,
        training_size: int = 1,
        max_episode_steps: int = 100,
        game_batch_size: int = 25,
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
        action_scorer_num_heads: int = 1,
        epsilon_anneal_from: float = 1.0,
        epsilon_anneal_to: float = 0.1,
        epsilon_anneal_episodes: int = 20000,
        game_reward_discount: float = 0.9,
        word_vocab_path: Optional[str] = None,
        node_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
        pretrained_graph_updater: Optional[GraphUpdater] = None,
        train_env: Optional[gym.Env] = None,
        val_env: Optional[gym.Env] = None,
        test_env: Optional[gym.Env] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "difficulty_level",
            "training_size",
            "max_episode_steps",
            "game_batch_size",
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
            "action_scorer_num_heads",
            "epsilon_anneal_from",
            "epsilon_anneal_to",
            "epsilon_anneal_episodes",
            "game_reward_discount",
        )

        # load envs
        if train_env is None:
            # load the test rl data
            self.train_env = load_textworld_games(
                "test-data/rl_games",
                "train",
                request_infos_for_train(),
                max_episode_steps,
                game_batch_size,
            )
        else:
            self.train_env = train_env
        if val_env is None:
            # load the test rl data
            self.val_env = load_textworld_games(
                "test-data/rl_games",
                "val",
                request_infos_for_eval(),
                max_episode_steps,
                game_batch_size,
            )
        else:
            self.val_env = val_env
        if test_env is None:
            # load the test rl data
            self.test_env = load_textworld_games(
                "test-data/rl_games",
                "test",
                request_infos_for_eval(),
                max_episode_steps,
                game_batch_size,
            )
        else:
            self.test_env = test_env

        # initialize word (preprocessor), node and relation stuff
        (
            node_name_word_ids,
            node_name_mask,
            rel_name_word_ids,
            rel_name_mask,
        ) = self.init_word_node_rel(
            word_vocab_path=word_vocab_path,
            node_vocab_path=node_vocab_path,
            relation_vocab_path=relation_vocab_path,
        )

        # main action selector
        self.action_selector = ActionSelector(
            hidden_dim,
            self.num_words,
            word_emb_dim,
            self.num_nodes,
            node_emb_dim,
            self.num_relations,
            relation_emb_dim,
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            text_encoder_num_heads,
            graph_encoder_num_cov_layers,
            graph_encoder_num_bases,
            action_scorer_num_heads,
            node_name_word_ids,
            node_name_mask,
            rel_name_word_ids,
            rel_name_mask,
        )

        # target action selector
        self.target_action_selector = ActionSelector(
            hidden_dim,
            self.num_words,
            word_emb_dim,
            self.num_nodes,
            node_emb_dim,
            self.num_relations,
            relation_emb_dim,
            text_encoder_num_blocks,
            text_encoder_num_conv_layers,
            text_encoder_kernel_size,
            text_encoder_num_heads,
            graph_encoder_num_cov_layers,
            graph_encoder_num_bases,
            action_scorer_num_heads,
            node_name_word_ids,
            node_name_mask,
            rel_name_word_ids,
            rel_name_mask,
        )
        # we don't train the target action selector
        for param in self.target_action_selector.parameters():
            param.requires_grad = False
        # update the target action selector weights to those of the main action selector
        self.update_target_action_selector()

        # graph updater
        if pretrained_graph_updater is None:
            self.graph_updater = GraphUpdater(
                hidden_dim,
                word_emb_dim,
                self.num_nodes,
                node_emb_dim,
                self.num_relations,
                relation_emb_dim,
                text_encoder_num_blocks,
                text_encoder_num_conv_layers,
                text_encoder_kernel_size,
                text_encoder_num_heads,
                graph_encoder_num_cov_layers,
                graph_encoder_num_bases,
                nn.Embedding(self.num_words, word_emb_dim),
                node_name_word_ids,
                node_name_mask,
                rel_name_word_ids,
                rel_name_mask,
            )
        else:
            self.graph_updater = pretrained_graph_updater
        # we use graph updater only to get the current graph representations
        self.graph_updater.eval()
        # we don't want to train the graph updater
        for param in self.graph_updater.parameters():
            param.requires_grad = False

        # loss
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def update_target_action_selector(self) -> None:
        self.target_action_selector.load_state_dict(self.action_selector.state_dict())

    def forward(  # type: ignore
        self,
        obs_word_ids: torch.Tensor,
        obs_mask: torch.Tensor,
        current_graph: torch.Tensor,
        action_cand_word_ids: torch.Tensor,
        action_cand_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use the main action selector to get the action scores based on the game state.

        obs_word_ids: (batch, obs_len)
        obs_mask: (batch, obs_len)
        current_graph: (batch, num_relation, num_node, num_node)
        action_cand_word_ids: (batch, num_action_cands, action_cand_len)
        action_cand_mask: (batch, num_action_cands, action_cand_len)

        output:
            action scores of shape (batch, num_action_cands)
            action mask of shape (batch, num_action_cands)
        """
        return self.action_selector(
            obs_word_ids,
            obs_mask,
            current_graph,
            action_cand_word_ids,
            action_cand_mask,
        )

    @staticmethod
    def get_q_values(
        action_scores: torch.Tensor,
        action_mask: torch.Tensor,
        actions_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the state action values of the given actions.

        action_scores: (batch, num_action_cands)
        action_mask: (batch, num_action_cands)
        actions_idx: (batch)

        output: state action values (batch)
        """
        actions_idx = actions_idx.unsqueeze(1)
        return torch.squeeze(
            action_scores.gather(1, actions_idx) * action_mask.gather(1, actions_idx),
            dim=1,
        )

    def training_step(  # type: ignore
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # double deep q learning

        # calculate the current q values
        action_scores, action_mask = self(
            batch["obs_word_ids"],
            batch["obs_mask"],
            batch["current_graph"],
            batch["action_cand_word_ids"],
            batch["action_cand_mask"],
        )
        q_values = self.get_q_values(action_scores, action_mask, batch["actions_idx"])

        with torch.no_grad():
            # select the next actions with the best q values
            next_action_scores, next_action_mask = self(
                batch["next_obs_word_ids"],
                batch["next_obs_mask"],
                batch["next_graph"],
                batch["next_action_cand_word_ids"],
                batch["next_action_cand_mask"],
            )
            next_actions_idx = self.action_selector.select_max_q(
                next_action_scores, next_action_mask
            )

            # calculate the next q values using the target action selector
            next_tgt_action_scores, next_tgt_action_mask = self.target_action_selector(
                batch["next_obs_word_ids"],
                batch["next_obs_mask"],
                batch["next_graph"],
                batch["next_action_cand_word_ids"],
                batch["next_action_cand_mask"],
            )
            next_q_values = self.get_q_values(
                next_tgt_action_scores, next_tgt_action_mask, next_actions_idx
            )
        # TODO: loss calculation and updates for prioritized experience replay
        # Note: no need to mask the next Q values as "done" states are not even added
        # to the replay buffer
        return self.smooth_l1_loss(
            q_values,
            batch["rewards"]
            + next_q_values * self.hparams.game_reward_discount,  # type: ignore
        )
