import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import hydra
import itertools
import gym
import random
import glob

from urllib.parse import urlparse
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning.loggers import WandbLogger
from dataclasses import dataclass, field
from typing import (
    Optional,
    Dict,
    List,
    Iterator,
    Callable,
    Iterable,
    Any,
    Tuple,
)
from textworld import EnvInfos
from torch.utils.data import IterableDataset, DataLoader, Dataset

from utils import load_textworld_games
from layers import WordNodeRelInitMixin
from action_selector import ActionSelector
from graph_updater import GraphUpdater
from agent import EpsilonGreedyAgent
from optimizers import RAdam
from train_graph_updater import GraphUpdaterObsGen
from callbacks import (
    EqualModelCheckpoint,
    EqualNonZeroModelCheckpoint,
    RLEarlyStopping,
    WandbSaveCallback,
)


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


def get_game_files(
    base_dir_path: str,
    dataset: str,
    difficulty_levels: List[int],
    training_size: Optional[int] = None,
) -> List[str]:
    # construct the game directories
    game_dirs = [
        os.path.join(
            base_dir_path,
            dataset + ("" if training_size is None else f"_{training_size}"),
            f"difficulty_level_{difficulty_level}",
        )
        for difficulty_level in difficulty_levels
    ]
    game_files: List[str] = []
    if training_size is not None:
        # for training games, sample equally among difficulty levels
        num_game_files_per_level = training_size // len(difficulty_levels)
        for game_dir in game_dirs:
            game_files.extend(
                random.sample(
                    glob.glob(os.path.join(game_dir, "*.z8")), num_game_files_per_level
                )
            )
    else:
        # for val and test games, collect all
        for game_dir in game_dirs:
            game_files.extend(glob.glob(os.path.join(game_dir, "*.z8")))
    return game_files


@dataclass
class Transition:
    """
    Represents a transition in one single episode.
    """

    # episode observation
    ob: str = ""
    # previous action
    prev_action: str = ""
    # RNN prev hidden states
    rnn_prev_hidden: torch.Tensor = field(
        default_factory=lambda: torch.empty(0), compare=False, hash=True
    )
    # action candidates
    action_cands: List[str] = field(default_factory=list)
    # chosen action ID
    action_id: int = 0
    # cumulative reward after the action
    cum_reward: int = 0
    # step reward after the action
    # this is a float b/c of multi-step learning (discounted rewards)
    step_reward: float = 0
    # next observation
    next_ob: str = ""
    # next action candidates
    next_action_cands: List[str] = field(default_factory=list)
    # RNN current hidden states, or the hidden states right before the next state
    # this is primarily used for multi-step learning
    rnn_curr_hidden: torch.Tensor = field(
        default_factory=lambda: torch.empty(0), compare=False, hash=True
    )
    # done
    done: bool = False

    def __eq__(self, other):
        return (
            self.ob == other.ob
            and self.action_cands == other.action_cands
            and self.rnn_prev_hidden.equal(other.rnn_prev_hidden)
            and self.action_id == other.action_id
            and self.cum_reward == other.cum_reward
            and self.step_reward == other.step_reward
            and self.next_ob == other.next_ob
            and self.next_action_cands == other.next_action_cands
            and self.rnn_curr_hidden.equal(other.rnn_curr_hidden)
            and self.done == other.done
        )


class TransitionCache:
    def __init__(self, batch_size: int) -> None:
        # cache[i][j] = j'th transition of i'th episode
        self.cache: List[List[Transition]] = [[] for _ in range(batch_size)]

    def batch_add(
        self,
        obs: List[str],
        prev_actions: List[str],
        rnn_prev_hiddens: torch.Tensor,
        batch_action_cands: List[List[str]],
        actions_idx: List[int],
        cum_rewards: List[float],
        step_rewards: List[float],
        next_obs: List[str],
        batch_next_action_cands: List[List[str]],
        rnn_curr_hiddens: torch.Tensor,
        dones: List[bool],
    ) -> None:
        for i, (
            ob,
            prev_action,
            rnn_prev_hidden,
            action_cands,
            action_id,
            cum_reward,
            step_reward,
            next_ob,
            next_action_cands,
            rnn_curr_hidden,
            done,
        ) in enumerate(
            zip(
                obs,
                prev_actions,
                rnn_prev_hiddens,
                batch_action_cands,
                actions_idx,
                cum_rewards,
                step_rewards,
                next_obs,
                batch_next_action_cands,
                rnn_curr_hiddens,
                dones,
            )
        ):
            if len(self.cache[i]) > 0 and done and self.cache[i][-1].done:
                # this episode is already done, don't add this transition
                continue
            self.cache[i].append(
                Transition(
                    ob=ob,
                    prev_action=prev_action,
                    rnn_prev_hidden=rnn_prev_hidden,
                    action_cands=action_cands,
                    action_id=action_id,
                    cum_reward=cum_reward,
                    step_reward=step_reward,
                    next_ob=next_ob,
                    next_action_cands=next_action_cands,
                    rnn_curr_hidden=rnn_curr_hidden,
                    done=done,
                )
            )

    def get_game_rewards(self) -> List[int]:
        return [episode[-1].cum_reward for episode in self.cache]

    def get_game_steps(self) -> List[int]:
        return [len(episode) for episode in self.cache]

    def get_avg_rewards(self) -> List[float]:
        """
        TextWorld returns cumulative rewards at each step, so in order to calculate
        the average rewards, we just need to divide the final rewards by the number
        of transitions
        """
        return [episode[-1].cum_reward / len(episode) for episode in self.cache]


class ReplayBufferDataset(IterableDataset):
    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        return self.generate_batch()


class ReplayBuffer:
    """
    Prioritized Experience Replay Buffer is based on
    PERBuffer from PyTorch Lightning Bolts.
    """

    def __init__(
        self,
        capacity: int,
        reward_threshold: float,
        sample_batch_size: int,
        eps: float,
        alpha: float,
        beta_from: float,
        beta_frames: int,
        multi_step: int,
        reward_discount: float,
    ):
        self.capacity = capacity
        self.reward_threshold = reward_threshold
        self.sample_batch_size = sample_batch_size
        self.eps = eps
        self.alpha = alpha
        self.beta_from = beta_from
        self.beta_frames = beta_frames
        self.beta = beta_from
        self.multi_step = multi_step
        self.reward_discount = reward_discount

        self.buffer: List[Transition] = []
        self.buffer_next_id = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def update_beta(self, step: int) -> None:
        slope = (1.0 - self.beta_from) / self.beta_frames
        self.beta = min(1.0, self.beta_from + step * slope)

    def push(self, t_cache: TransitionCache) -> None:
        buffer_avg_reward = 0.0
        if len(self.buffer) > 0:
            buffer_avg_reward = np.mean(  # type: ignore
                [transition.step_reward for transition in self.buffer]
            )
        for avg_reward, transitions in zip(t_cache.get_avg_rewards(), t_cache.cache):
            if avg_reward >= buffer_avg_reward * self.reward_threshold:
                self._extend_limited_list(transitions)

    def _extend_limited_list(self, transitions: List[Transition]) -> None:
        for t in transitions:
            # get the max priority before adding
            max_prio = self.priorities.max() if self.buffer else 1.0

            if len(self.buffer) < self.capacity:
                self.buffer.append(t)
            else:
                self.buffer[self.buffer_next_id] = t

            # set the priority to the max so that it will resampled soon
            self.priorities[self.buffer_next_id] = max_prio

            self.buffer_next_id += 1
            self.buffer_next_id %= self.capacity

    def sample(self) -> Optional[Dict[str, Any]]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.buffer_next_id]

        # alpha determines the level of prioritization
        # 0 = no prioritization
        probs = prios ** self.alpha
        probs /= probs.sum()

        # sample indices based on probs
        sampled_indices = np.random.choice(
            len(self.buffer), self.sample_batch_size, p=probs, replace=False
        )
        sampled_steps = np.random.randint(
            1, self.multi_step + 1, size=len(sampled_indices)
        )
        samples, indices, steps = self.sample_multi_step(sampled_indices, sampled_steps)
        if len(samples) == 0:
            return None

        # weight of each sample to compensate for the bias added in
        # with prioritising samples
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # return the samples, the indices chosen and the weight of each sample
        return {
            "samples": samples,
            "steps": steps,
            "indices": indices,
            "weights": weights.tolist(),
        }

    def sample_multi_step(
        self, sampled_indices: List[int], sampled_steps: List[int]
    ) -> Tuple[List[Transition], List[int], List[int]]:
        multi_step_samples: List[Transition] = []
        indices: List[int] = []
        steps: List[int] = []
        for idx, step in zip(sampled_indices, sampled_steps):
            # make sure the game is not done before "step" steps.
            # we loop around since the buffer is circular
            if any(
                self.buffer[i % len(self.buffer)].done for i in range(idx, idx + step)
            ):
                continue
            head_t = self.buffer[idx]
            # loop around since it might be in the front
            tail_t = self.buffer[(idx + step) % len(self.buffer)]
            step_reward = sum(
                self.reward_discount ** i
                * self.buffer[(idx + i) % len(self.buffer)].step_reward
                for i in range(step + 1)
            )
            multi_step_samples.append(
                Transition(
                    ob=head_t.ob,
                    prev_action=head_t.prev_action,
                    rnn_prev_hidden=head_t.rnn_prev_hidden,
                    action_cands=head_t.action_cands,
                    action_id=head_t.action_id,
                    cum_reward=tail_t.cum_reward,
                    step_reward=step_reward,
                    next_ob=tail_t.next_ob,
                    next_action_cands=tail_t.next_action_cands,
                    rnn_curr_hidden=tail_t.rnn_curr_hidden,
                    done=tail_t.done,
                )
            )
            indices.append(idx)
            steps.append(step)

        return multi_step_samples, indices, steps

    def update_priorities(
        self, batch_idx: List[int], batch_priorities: List[float]
    ) -> None:
        for i, prio in zip(batch_idx, batch_priorities):
            self.priorities[i] = prio + self.eps


class RLEvalDataset(Dataset):
    """
    A dummy dataset for reinforcement learning evaluation.
    We need this to "kick off" the validation/test loop in PL.
    Without this, PL wouldn't run them.
    """

    def __init__(self, num_gamefiles: int) -> None:
        self.num_gamefiles = num_gamefiles

    def __getitem__(self, idx: int) -> int:
        return idx

    def __len__(self) -> int:
        return self.num_gamefiles


class GATADoubleDQN(WordNodeRelInitMixin, pl.LightningModule):
    DIFFICULTY_LEVEL_MAP = {1: [3], 2: [7], 3: [5], 4: [9], 5: [3, 7, 5, 9]}

    def __init__(
        self,
        base_data_dir: Optional[str] = None,
        difficulty_level: int = 1,
        train_data_size: int = 1,
        train_max_episode_steps: int = 50,
        train_game_batch_size: int = 25,
        training_step_freq: int = 50,
        replay_buffer_capacity: int = 500000,
        replay_buffer_populate_episodes: int = 100,
        replay_buffer_reward_threshold: float = 0.1,
        replay_buffer_eps: float = 1e-6,
        replay_buffer_alpha: float = 0.6,
        replay_buffer_beta_from: float = 0.4,
        replay_buffer_beta_frames: int = 100000,
        replay_buffer_multi_step: int = 3,
        train_sample_batch_size: int = 64,
        learning_rate: float = 1e-3,
        target_net_update_frequency: int = 500,
        eval_max_episode_steps: int = 100,
        eval_game_batch_size: int = 20,
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
        reward_discount: float = 0.9,
        ckpt_patience: int = 3,
        word_vocab_path: Optional[str] = None,
        node_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
        pretrained_graph_updater: Optional[GraphUpdater] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "difficulty_level",
            "train_data_size",
            "train_max_episode_steps",
            "train_game_batch_size",
            "training_step_freq",
            "replay_buffer_capacity",
            "replay_buffer_populate_episodes",
            "replay_buffer_reward_threshold",
            "replay_buffer_eps",
            "replay_buffer_alpha",
            "replay_buffer_beta_from",
            "replay_buffer_beta_frames",
            "replay_buffer_multi_step",
            "train_sample_batch_size",
            "learning_rate",
            "target_net_update_frequency",
            "eval_max_episode_steps",
            "eval_game_batch_size",
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
            "reward_discount",
            "ckpt_patience",
        )

        # load the test rl data
        if base_data_dir is None:
            abs_base_data_dir = to_absolute_path("test-data/rl_games")
            train_game_files = glob.glob(os.path.join(abs_base_data_dir, "*.z8"))
            val_game_files = glob.glob(os.path.join(abs_base_data_dir, "*.z8"))
            test_game_files = glob.glob(os.path.join(abs_base_data_dir, "*.z8"))
        else:
            abs_base_data_dir = to_absolute_path(base_data_dir)
            train_game_files = get_game_files(
                abs_base_data_dir,
                "train",
                self.DIFFICULTY_LEVEL_MAP[difficulty_level],
                training_size=train_data_size,
            )
            val_game_files = get_game_files(
                abs_base_data_dir,
                "valid",
                self.DIFFICULTY_LEVEL_MAP[difficulty_level],
            )
            test_game_files = get_game_files(
                abs_base_data_dir,
                "test",
                self.DIFFICULTY_LEVEL_MAP[difficulty_level],
            )

        self.train_env = load_textworld_games(
            train_game_files,
            "train",
            request_infos_for_train(),
            train_max_episode_steps,
            train_game_batch_size,
        )
        # load the val rl data
        self.val_env = load_textworld_games(
            val_game_files,
            "val",
            request_infos_for_eval(),
            eval_max_episode_steps,
            eval_game_batch_size,
        )
        # load the test rl data
        self.test_env = load_textworld_games(
            test_game_files,
            "test",
            request_infos_for_eval(),
            eval_max_episode_steps,
            eval_game_batch_size,
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

        # online action selector
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
        if pretrained_graph_updater is not None:
            # load the pretrained graph encoder weights
            action_selector_state_dict = self.action_selector.state_dict()

            def is_graph_encoder_key(key: str) -> bool:
                return any(
                    key.startswith(graph_encoder_key)
                    for graph_encoder_key in [
                        "graph_encoder",
                        "word_embeddings",
                        "node_embeddings",
                        "relation_embeddings",
                    ]
                )

            pretrained_graph_encoder_state_dict = {
                k: v
                for k, v in pretrained_graph_updater.state_dict().items()
                if is_graph_encoder_key(k)
            }
            action_selector_state_dict.update(pretrained_graph_encoder_state_dict)
            self.action_selector.load_state_dict(action_selector_state_dict)

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
        # update the target action selector weights to those of
        # the online action selector
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
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")

        # agent
        self.agent = EpsilonGreedyAgent(
            self.graph_updater,
            self.action_selector,
            self.preprocessor,
            self.hparams.epsilon_anneal_from,  # type: ignore
            self.hparams.epsilon_anneal_to,  # type: ignore
            self.hparams.epsilon_anneal_episodes,  # type: ignore
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            replay_buffer_capacity,
            replay_buffer_reward_threshold,
            train_sample_batch_size,
            replay_buffer_eps,
            replay_buffer_alpha,
            replay_buffer_beta_from,
            replay_buffer_beta_frames,
            replay_buffer_multi_step,
            reward_discount,
        )

        # bookkeeping
        self.total_episode_steps = 0
        self.ckpt_patience_count = 0

        # metrics
        self.game_rewards: List[int] = []
        self.game_normalized_rewards: List[float] = []
        self.game_steps: List[int] = []

    def seed_envs(self, seed: int) -> None:
        self.train_env.seed(seed)
        self.val_env.seed(seed)
        self.test_env.seed(seed)

    def update_target_action_selector(self) -> None:
        self.target_action_selector.load_state_dict(self.action_selector.state_dict())

    def forward(  # type: ignore
        self,
        obs_word_ids: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_action_word_ids: torch.Tensor,
        prev_action_mask: torch.Tensor,
        rnn_prev_hidden: torch.Tensor,
        action_cand_word_ids: torch.Tensor,
        action_cand_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Use the online action selector to get the action scores based on the game state.

        obs_word_ids: (batch, obs_len)
        obs_mask: (batch, obs_len)
        prev_action_word_ids: (batch, prev_action_len)
        prev_action_mask: (batch, prev_action_len)
        rnn_prev_hidden: (batch, hidden_dim)
        action_cand_word_ids: (batch, num_action_cands, action_cand_len)
        action_cand_mask: (batch, num_action_cands, action_cand_len)
        action_mask: (batch, num_action_cands)

        output: {
            'action_scores: (batch, num_action_cands),
            'rnn_curr_hidden': (batch, hidden_dim),
            'current_graph': (batch, num_relation, num_node, num_node)
        }
        """
        results = self.graph_updater(
            obs_word_ids,
            prev_action_word_ids,
            obs_mask,
            prev_action_mask,
            rnn_prev_hidden=rnn_prev_hidden,
        )
        return {
            "action_scores": self.action_selector(
                obs_word_ids,
                obs_mask,
                results["g_t"],
                action_cand_word_ids,
                action_cand_mask,
                action_mask,
            ),
            "rnn_curr_hidden": results["h_t"],
            "current_graph": results["g_t"],
        }

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
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        # double deep q learning

        # calculate the current q values
        results = self(
            batch["obs_word_ids"],
            batch["obs_mask"],
            batch["prev_action_word_ids"],
            batch["prev_action_mask"],
            batch["rnn_prev_hidden"],
            batch["action_cand_word_ids"],
            batch["action_cand_mask"],
            batch["action_mask"],
        )
        q_values = self.get_q_values(
            results["action_scores"], batch["action_mask"], batch["actions_idx"]
        )

        with torch.no_grad():
            # select the next actions with the best q values
            next_results = self(
                batch["next_obs_word_ids"],
                batch["next_obs_mask"],
                batch["curr_action_word_ids"],
                batch["curr_action_mask"],
                batch["rnn_curr_hidden"],
                batch["next_action_cand_word_ids"],
                batch["next_action_cand_mask"],
                batch["next_action_mask"],
            )
            next_actions_idx = self.action_selector.select_max_q(
                next_results["action_scores"], batch["next_action_mask"]
            )

            # calculate the next q values using the target action selector
            next_tgt_action_scores = self.target_action_selector(
                batch["next_obs_word_ids"],
                batch["next_obs_mask"],
                next_results["current_graph"],
                batch["next_action_cand_word_ids"],
                batch["next_action_cand_mask"],
                batch["next_action_mask"],
            )
            # Note: no need to mask the next Q values as
            # "done" states are not even added to the replay buffer
            next_q_values = self.get_q_values(
                next_tgt_action_scores, batch["next_action_mask"], next_actions_idx
            )

        target_q_values = batch["rewards"] + next_q_values * (
            self.hparams.reward_discount ** batch["steps"]  # type: ignore
        )

        # update priorities for the replay buffer
        abs_td_error = torch.abs(q_values - target_q_values)
        self.replay_buffer.update_priorities(
            batch["indices"].tolist(), abs_td_error.tolist()
        )

        # calculate loss
        loss = torch.mean(
            batch["weights"] * self.smooth_l1_loss(q_values, target_q_values)
        )
        self.log_dict(
            {
                "train_loss": loss,
                "epsilon": self.agent.epsilon,
            },
            on_step=False,
            on_epoch=True,
        )
        return loss

    def training_epoch_end(self, _) -> None:
        self.log_dict(
            {
                "train_avg_game_rewards": torch.tensor(
                    self.game_rewards, dtype=torch.float
                ).mean(),
                "train_avg_game_normalized_rewards": torch.tensor(
                    self.game_normalized_rewards
                ).mean(),
                "train_avg_game_steps": torch.tensor(
                    self.game_steps, dtype=torch.float
                ).mean(),
            }
        )

    def eval_step(self, env: gym.Env) -> Dict[str, torch.Tensor]:  # type: ignore
        prev_actions: Optional[List[str]] = None
        rnn_prev_hidden: Optional[torch.Tensor] = None
        obs, infos = env.reset()
        steps: List[int] = [0] * len(obs)
        for step in itertools.count():
            actions, rnn_prev_hidden = self.agent.act(
                obs,
                infos["admissible_commands"],
                prev_actions=prev_actions,
                rnn_prev_hidden=rnn_prev_hidden,
            )
            obs, rewards, dones, infos = env.step(actions)
            for i, done in enumerate(dones):
                if steps[i] == 0 and done:
                    steps[i] = step
            if all(dones):
                break

        return {
            "game_rewards": torch.tensor(rewards, dtype=torch.float),
            "game_normalized_rewards": torch.tensor(
                [
                    reward / game.metadata["max_score"]
                    for reward, game in zip(rewards, infos["game"])
                ]
            ),
            "game_steps": torch.tensor(steps, dtype=torch.float),
        }

    def eval_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], prefix: str
    ) -> None:
        self.log_dict(
            {
                prefix
                + "_avg_game_rewards": torch.cat(
                    [output["game_rewards"] for output in outputs]
                ).mean(),
                prefix
                + "_avg_game_normalized_rewards": torch.cat(
                    [output["game_normalized_rewards"] for output in outputs]
                ).mean(),
                prefix
                + "_avg_game_steps": torch.cat(
                    [output["game_steps"] for output in outputs]
                ).mean(),
            }
        )

    def validation_step(  # type: ignore
        self, _batch: torch.Tensor, _batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.eval_step(self.val_env)

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self.eval_epoch_end(outputs, "val")

    def on_validation_end(self) -> None:
        train_model_ckpt, val_model_ckpt = list(
            filter(
                lambda c: isinstance(c, EqualModelCheckpoint), self.trainer.callbacks
            )
        )
        if val_model_ckpt.best_model_score is not None:
            # val_avg_game_normalized_rewards > 0 since something has been saved.
            val_current_score = self.trainer.logger_connector.callback_metrics[
                "val_avg_game_normalized_rewards"
            ]
            if val_current_score < val_model_ckpt.best_model_score:
                self.ckpt_patience_count += 1
            else:
                self.ckpt_patience_count = 0
        elif train_model_ckpt.best_model_score is not None:
            train_current_score = self.trainer.logger_connector.callback_metrics[
                "train_avg_game_normalized_rewards"
            ]
            if train_current_score < train_model_ckpt.best_model_score:
                self.ckpt_patience_count += 1
            else:
                self.ckpt_patience_count = 0
        if self.ckpt_patience_count >= self.hparams.ckpt_patience:  # type: ignore
            # reload the best checkpoint
            if val_model_ckpt.best_model_score is not None:
                self.print("patience ran out. loading from best validation checkpoint")
                state_dict = torch.load(val_model_ckpt.best_model_path)["state_dict"]
            else:
                # we can assume that train_model_ckpt.best_model_score is not None here
                self.print("patience ran out. loading from best training checkpoint")
                state_dict = torch.load(train_model_ckpt.best_model_path)["state_dict"]
            self.load_state_dict(state_dict)
            self.update_target_action_selector()
            self.ckpt_patience_count = 0

    def test_step(  # type: ignore
        self, _batch: torch.Tensor, _batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.eval_step(self.test_env)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self.eval_epoch_end(outputs, "test")

    def train_dataloader(self) -> DataLoader:
        self.populate_replay_buffer()
        return DataLoader(  # type: ignore
            ReplayBufferDataset(self.gen_train_batch),
            # disable automatic batching as it's done in the replay buffer
            batch_size=None,
            collate_fn=self.prepare_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            RLEvalDataset(len(self.val_env.gamefiles)),
            batch_size=self.hparams.eval_game_batch_size,  # type: ignore
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            RLEvalDataset(len(self.test_env.gamefiles)),
            batch_size=self.hparams.eval_game_batch_size,  # type: ignore
        )

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.learning_rate)

    def populate_replay_buffer(self) -> None:
        episodes_played = 0

        # we don't want to sample, so just return None
        def sample() -> Optional[Dict[str, Any]]:
            return None

        def act_random(_: torch.Tensor, action_mask: torch.Tensor) -> List[int]:
            return self.agent.select_random(action_mask).tolist()

        # noop
        def episode_end() -> None:
            pass

        while (
            episodes_played
            < self.hparams.replay_buffer_populate_episodes  # type: ignore
        ):
            for _ in self.play_episodes(sample, act_random, episode_end):
                pass
            episodes_played += self.train_env.batch_size

    def gen_train_batch(self) -> Iterator[Dict[str, Any]]:
        """
        Generate train batches by playing multiple episodes in parallel.
        Generation stops when all the parallel episodes are done.
        The number of parallel episodes is self.train_env.batch_size.
        This means that one epoch = self.train_env.batch_size episodes
        """

        def sample() -> Optional[Dict[str, Any]]:
            if (
                self.total_episode_steps
                % self.hparams.training_step_freq  # type: ignore
                == 0
            ):
                # return a sample if we're at the training step frequency
                return self.replay_buffer.sample()
            return None

        def act_epsilon_greedy(
            action_scores: torch.Tensor, action_mask: torch.Tensor
        ) -> List[int]:
            return self.agent.select_epsilon_greedy(
                self.agent.action_selector.select_max_q(action_scores, action_mask),
                self.agent.select_random(action_mask),
            ).tolist()

        def episode_end() -> None:
            self.total_episode_steps += 1

        # As the agent gets better, it'll take fewer steps, and eventually
        # it may not even take "training_step_freq" steps per batch of episodes,
        # which means there is no batch for that epoch. PL doesn't like that
        # (https://github.com/PyTorchLightning/pytorch-lightning/issues/6810)
        # so keep playing episods until we get at least one batch.
        sampled = False
        while not sampled:
            for batch in self.play_episodes(sample, act_epsilon_greedy, episode_end):
                sampled = True
                yield batch

        # set up for the next batch of episodes
        if (
            self.current_epoch
            % self.hparams.target_net_update_frequency  # type: ignore
        ):
            self.update_target_action_selector()
        episodes_played = self.current_epoch * self.train_env.batch_size
        self.agent.update_epsilon(episodes_played)

    def play_episodes(
        self,
        sample: Callable[[], Optional[Dict[str, Any]]],
        action_select_fn: Callable[[torch.Tensor, torch.Tensor], List[int]],
        episode_end_fn: Callable[[], None],
    ) -> Iterator[Dict[str, Any]]:
        transition_cache = TransitionCache(self.train_env.batch_size)
        # in order to avoid having to deal with None, just start with zeros
        # this is OK, b/c Graph Updater already initializes rnn_prev_hidden
        # to zeros if it's None.
        rnn_prev_hidden: torch.Tensor = torch.zeros(
            self.train_env.batch_size,
            self.hparams.hidden_dim,  # type: ignore
            device=self.device,
        )
        prev_actions: List[str] = ["restart"] * self.train_env.batch_size
        prev_cum_rewards: List[int] = [0] * self.train_env.batch_size
        dones: List[bool] = [False] * self.train_env.batch_size

        raw_obs, infos = self.train_env.reset()
        # clean observations
        obs = self.agent.preprocessor.batch_clean(raw_obs)
        # filter action cands
        action_cands = self.agent.filter_action_cands(infos["admissible_commands"])
        while True:
            sampled = sample()
            if sampled is not None:
                yield sampled

            if all(dones):
                # if all the previous episodes are done, we can stop
                break

            results = self.agent.calculate_action_scores(
                obs,
                action_cands,
                prev_actions=prev_actions,
                rnn_prev_hidden=rnn_prev_hidden,
            )

            # select actions randomly
            actions_idx = action_select_fn(
                results["action_scores"], results["action_mask"]
            )

            # take a step
            actions = self.agent.decode_actions(action_cands, actions_idx)
            next_raw_obs, cum_rewards, dones, next_infos = self.train_env.step(actions)

            # clean next observations
            next_obs = self.agent.preprocessor.batch_clean(next_raw_obs)

            # filter next action cands
            next_action_cands = self.agent.filter_action_cands(
                next_infos["admissible_commands"]
            )

            # calculate step rewards
            step_rewards = [
                curr - prev for prev, curr in zip(prev_cum_rewards, cum_rewards)
            ]

            # add the transition to the cache
            rnn_curr_hidden = results["rnn_curr_hidden"]
            transition_cache.batch_add(
                obs,
                prev_actions,
                rnn_prev_hidden,
                action_cands,
                actions_idx,
                cum_rewards,
                step_rewards,
                next_obs,
                next_action_cands,
                rnn_curr_hidden,
                dones,
            )

            # set up the next step
            obs = next_obs
            action_cands = next_action_cands
            prev_actions = actions
            prev_cum_rewards = cum_rewards
            rnn_prev_hidden = rnn_curr_hidden
            episode_end_fn()

        # push transitions into the buffer
        self.replay_buffer.push(transition_cache)

        # collect metrics
        self.game_rewards = transition_cache.get_game_rewards()
        self.game_normalized_rewards = [
            reward / game.metadata["max_score"]
            for reward, game in zip(self.game_rewards, infos["game"])
        ]
        self.game_steps = transition_cache.get_game_steps()

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs: List[str] = []
        prev_actions: List[str] = []
        rnn_prev_hiddens: List[torch.Tensor] = []
        action_cands: List[List[str]] = []
        actions_idx: List[int] = []
        curr_actions: List[str] = []
        rewards: List[float] = []
        next_obs: List[str] = []
        next_action_cands: List[List[str]] = []
        rnn_curr_hiddens: List[torch.Tensor] = []
        for transition in batch["samples"]:
            obs.append(transition.ob)
            prev_actions.append(transition.prev_action)
            rnn_prev_hiddens.append(transition.rnn_prev_hidden)
            action_cands.append(transition.action_cands)
            actions_idx.append(transition.action_id)
            curr_actions.append(transition.action_cands[transition.action_id])
            rewards.append(transition.step_reward)
            next_obs.append(transition.next_ob)
            next_action_cands.append(transition.next_action_cands)
            rnn_curr_hiddens.append(transition.rnn_curr_hidden)

        # preprocess
        obs_word_ids, obs_mask = self.agent.preprocessor.preprocess(obs)
        (
            action_cand_word_ids,
            action_cand_mask,
            action_mask,
        ) = self.agent.preprocess_action_cands(action_cands)
        prev_action_word_ids, prev_action_mask = self.agent.preprocessor.preprocess(
            prev_actions
        )
        curr_action_word_ids, curr_action_mask = self.agent.preprocessor.preprocess(
            curr_actions
        )
        next_obs_word_ids, next_obs_mask = self.agent.preprocessor.preprocess(next_obs)
        (
            next_action_cand_word_ids,
            next_action_cand_mask,
            next_action_mask,
        ) = self.agent.preprocess_action_cands(next_action_cands)

        return {
            "obs_word_ids": obs_word_ids,
            "obs_mask": obs_mask,
            "prev_action_word_ids": prev_action_word_ids,
            "prev_action_mask": prev_action_mask,
            "rnn_prev_hidden": torch.stack(rnn_prev_hiddens),
            "action_cand_word_ids": action_cand_word_ids,
            "action_cand_mask": action_cand_mask,
            "action_mask": action_mask,
            "actions_idx": torch.tensor(actions_idx),
            "rewards": torch.tensor(rewards),
            "curr_action_word_ids": curr_action_word_ids,
            "curr_action_mask": curr_action_mask,
            "next_obs_word_ids": next_obs_word_ids,
            "next_obs_mask": next_obs_mask,
            "next_action_cand_word_ids": next_action_cand_word_ids,
            "next_action_cand_mask": next_action_cand_mask,
            "next_action_mask": next_action_mask,
            "rnn_curr_hidden": torch.stack(rnn_curr_hiddens),
            "steps": torch.tensor(batch["steps"]),
            "weights": torch.tensor(batch["weights"]),
            "indices": torch.tensor(batch["indices"]),
        }


@hydra.main(config_path="train_gata_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    # seed
    pl.seed_everything(42)

    # trainer
    trainer_config = OmegaConf.to_container(cfg.pl_trainer, resolve=True)
    assert isinstance(trainer_config, dict)
    trainer_config["logger"] = instantiate(cfg.logger) if "logger" in cfg else True
    val_monitor = "val_avg_game_normalized_rewards"
    train_monitor = "train_avg_game_normalized_rewards"
    trainer_config["callbacks"] = [
        RLEarlyStopping(
            val_monitor,
            train_monitor,
            cfg.train.early_stop_threshold,
            patience=cfg.train.early_stop_patience,
        ),
        EqualModelCheckpoint(
            monitor=train_monitor,
            mode="max",
            filename="{epoch}-{step}-{train_avg_game_normalized_rewards:.2f}",
        ),
        # because of a bug:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/6467
        # we need to pass in the validation model checkpoint last
        # the train model checkpoint won't really be used
        # after the validation metrics goes over 0.
        EqualNonZeroModelCheckpoint(
            monitor=val_monitor,
            mode="max",
            filename="{epoch}-{step}-{val_avg_game_normalized_rewards:.2f}",
        ),
    ]
    if isinstance(trainer_config["logger"], WandbLogger):
        trainer_config["callbacks"].append(WandbSaveCallback())
    trainer = pl.Trainer(**trainer_config)

    # instantiate the lightning module
    if not cfg.eval.test_only:
        lm_model_config = OmegaConf.to_container(cfg.model, resolve=True)
        assert isinstance(lm_model_config, dict)
        if cfg.model.pretrained_graph_updater is not None:
            graph_updater_obs_gen = GraphUpdaterObsGen.load_from_checkpoint(
                to_absolute_path(cfg.model.pretrained_graph_updater.ckpt_path),
                word_vocab_path=cfg.model.pretrained_graph_updater.word_vocab_path,
                node_vocab_path=cfg.model.pretrained_graph_updater.node_vocab_path,
                relation_vocab_path=(
                    cfg.model.pretrained_graph_updater.relation_vocab_path
                ),
            )
            lm_model_config[
                "pretrained_graph_updater"
            ] = graph_updater_obs_gen.graph_updater
        lm = GATADoubleDQN(**lm_model_config, **cfg.train, **cfg.data)
        lm.seed_envs(42)

        # fit
        trainer.fit(lm)

        # test
        trainer.test()
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

        model = GATADoubleDQN.load_from_checkpoint(
            ckpt_path,
            word_vocab_path=cfg.model.word_vocab_path,
            node_vocab_path=cfg.model.node_vocab_path,
            relation_vocab_path=cfg.model.relation_vocab_path,
        )
        trainer.test(model=model)


if __name__ == "__main__":
    main()
