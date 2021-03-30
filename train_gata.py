import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import hydra
import itertools
import gym

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning.loggers import WandbLogger
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Iterator, Deque, Callable, Iterable, Generator
from textworld import EnvInfos
from torch.utils.data import IterableDataset, DataLoader, Dataset
from collections import deque

from utils import load_textworld_games, WandbSaveCallback
from layers import WordNodeRelInitMixin
from action_selector import ActionSelector
from graph_updater import GraphUpdater
from agent import EpsilonGreedyAgent
from optimizers import RAdam
from train_graph_updater import GraphUpdaterObsGen


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
    Represents a transition in one single episode.
    """

    # episode observation
    ob: str = ""
    # action candidates
    action_cands: List[str] = field(default_factory=list)
    # current graph
    current_graph: torch.Tensor = field(
        default_factory=lambda: torch.empty(0), compare=False, hash=True
    )
    # chosen action ID
    action_id: int = 0
    # cumulative reward after the action
    cum_reward: int = 0
    # step reward after the action
    step_reward: int = 0
    # next observation
    next_ob: str = ""
    # next action candidates
    next_action_cands: List[str] = field(default_factory=list)
    # next graph
    next_graph: torch.Tensor = field(
        default_factory=lambda: torch.empty(0), compare=False, hash=True
    )
    # done
    done: bool = False

    def __eq__(self, other):
        return (
            self.ob == other.ob
            and self.action_cands == other.action_cands
            and self.current_graph.equal(other.current_graph)
            and self.action_id == other.action_id
            and self.cum_reward == other.cum_reward
            and self.step_reward == other.step_reward
            and self.next_ob == other.next_ob
            and self.next_action_cands == other.next_action_cands
            and self.next_graph.equal(other.next_graph)
            and self.done == other.done
        )


class TransitionCache:
    def __init__(self, batch_size: int) -> None:
        # cache[i][j] = j'th transition of i'th episode
        self.cache: List[List[Transition]] = [[] for _ in range(batch_size)]

    def batch_add(
        self,
        obs: List[str],
        batch_action_cands: List[List[str]],
        current_graphs: torch.Tensor,
        actions_idx: List[int],
        cum_rewards: List[float],
        step_rewards: List[float],
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
            cum_reward,
            step_reward,
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
                cum_rewards,
                step_rewards,
                dones,
                next_obs,
                batch_next_action_cands,
                next_graphs,
            )
        ):
            if len(self.cache[i]) > 0 and done and self.cache[i][-1].done:
                # this episode is already done, don't add this transition
                continue
            self.cache[i].append(
                Transition(
                    ob=ob,
                    action_cands=action_cands,
                    current_graph=current_graph,
                    action_id=action_id,
                    cum_reward=cum_reward,
                    step_reward=step_reward,
                    next_ob=next_ob,
                    next_action_cands=next_action_cands,
                    next_graph=next_graph,
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
        )

        # load the test rl data
        self.train_env = load_textworld_games(
            to_absolute_path(
                get_game_dir(
                    base_data_dir,
                    "train",
                    difficulty_level,
                    training_size=train_data_size,
                )
            )
            if base_data_dir is not None
            else "test-data/rl_games",
            "train",
            request_infos_for_train(),
            train_max_episode_steps,
            train_game_batch_size,
        )
        # load the val rl data
        self.val_env = load_textworld_games(
            to_absolute_path(get_game_dir(base_data_dir, "valid", difficulty_level))
            if base_data_dir is not None
            else "test-data/rl_games",
            "val",
            request_infos_for_eval(),
            eval_max_episode_steps,
            eval_game_batch_size,
        )
        # load the test rl data
        self.test_env = load_textworld_games(
            to_absolute_path(get_game_dir(base_data_dir, "test", difficulty_level))
            if base_data_dir is not None
            else "test-data/rl_games",
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
        self.smooth_l1_loss = nn.SmoothL1Loss()

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
        self.buffer: Deque[Transition] = deque(maxlen=replay_buffer_capacity)

        # bookkeeping
        self.total_episode_steps = 0

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
        current_graph: torch.Tensor,
        action_cand_word_ids: torch.Tensor,
        action_cand_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use the online action selector to get the action scores based on the game state.

        obs_word_ids: (batch, obs_len)
        obs_mask: (batch, obs_len)
        current_graph: (batch, num_relation, num_node, num_node)
        action_cand_word_ids: (batch, num_action_cands, action_cand_len)
        action_cand_mask: (batch, num_action_cands, action_cand_len)
        action_mask: (batch, num_action_cands)

        output: action scores of shape (batch, num_action_cands)
        """
        return self.action_selector(
            obs_word_ids,
            obs_mask,
            current_graph,
            action_cand_word_ids,
            action_cand_mask,
            action_mask,
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
        action_scores = self(
            batch["obs_word_ids"],
            batch["obs_mask"],
            batch["current_graph"],
            batch["action_cand_word_ids"],
            batch["action_cand_mask"],
            batch["action_mask"],
        )
        q_values = self.get_q_values(
            action_scores, batch["action_mask"], batch["actions_idx"]
        )

        with torch.no_grad():
            # select the next actions with the best q values
            next_action_scores = self(
                batch["next_obs_word_ids"],
                batch["next_obs_mask"],
                batch["next_graph"],
                batch["next_action_cand_word_ids"],
                batch["next_action_cand_mask"],
                batch["next_action_mask"],
            )
            next_actions_idx = self.action_selector.select_max_q(
                next_action_scores, batch["next_action_mask"]
            )

            # calculate the next q values using the target action selector
            next_tgt_action_scores = self.target_action_selector(
                batch["next_obs_word_ids"],
                batch["next_obs_mask"],
                batch["next_graph"],
                batch["next_action_cand_word_ids"],
                batch["next_action_cand_mask"],
                batch["next_action_mask"],
            )
            next_q_values = self.get_q_values(
                next_tgt_action_scores, batch["next_action_mask"], next_actions_idx
            )
        # TODO: loss calculation and updates for prioritized experience replay
        # Note: no need to mask the next Q values as "done" states are not even added
        # to the replay buffer
        loss = self.smooth_l1_loss(
            q_values,
            batch["rewards"]
            + next_q_values * self.hparams.reward_discount,  # type: ignore
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

    def test_step(  # type: ignore
        self, _batch: torch.Tensor, _batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.eval_step(self.test_env)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self.eval_epoch_end(outputs, "test")

    def train_dataloader(self) -> DataLoader:
        self.populate_replay_buffer()
        return DataLoader(
            ReplayBufferDataset(self.gen_train_batch),
            batch_size=self.hparams.train_sample_batch_size,  # type: ignore
            collate_fn=self.prepare_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            RLEvalDataset(len(self.val_env.gamefiles)),
            batch_size=self.hparams.eval_game_batch_size,  # type: ignore
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            RLEvalDataset(len(self.val_env.gamefiles)),
            batch_size=self.hparams.eval_game_batch_size,  # type: ignore
        )

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.learning_rate)

    def populate_replay_buffer(self) -> None:
        episodes_played = 0

        # we don't want to sample, so just yield None
        def sample() -> Generator[Optional[Transition], None, None]:
            yield None

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

    def gen_train_batch(self) -> Iterator[Transition]:
        """
        Generate train batches by playing multiple episodes in parallel.
        Generation stops when all the parallel episodes are done.
        The number of parallel episodes is self.train_env.batch_size.
        This means that one epoch = self.train_env.batch_size episodes
        """

        def sample() -> Generator[Optional[Transition], None, None]:
            if (
                self.total_episode_steps
                % self.hparams.training_step_freq  # type: ignore
                == 0
            ):
                # if we've played enough episodes to learn and we're at the
                # yield frequency, yield a batch
                yield from self.sample()

        def act_epsilon_greedy(
            action_scores: torch.Tensor, action_mask: torch.Tensor
        ) -> List[int]:
            return self.agent.select_epsilon_greedy(
                self.agent.action_selector.select_max_q(action_scores, action_mask),
                self.agent.select_random(action_mask),
            ).tolist()

        def episode_end() -> None:
            self.total_episode_steps += 1

        yield from self.play_episodes(sample, act_epsilon_greedy, episode_end)

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
        sample: Callable[[], Generator[Optional[Transition], None, None]],
        action_select_fn: Callable[[torch.Tensor, torch.Tensor], List[int]],
        episode_end_fn: Callable[[], None],
    ) -> Iterator[Transition]:
        transition_cache = TransitionCache(self.train_env.batch_size)
        rnn_prev_hidden: Optional[torch.Tensor] = None
        prev_actions: Optional[List[str]] = None
        prev_obs: List[str] = []
        prev_action_cands: List[List[str]] = []
        prev_graph: torch.Tensor = torch.empty(0)
        prev_actions_idx: List[int] = []
        prev_cum_rewards: List[float] = [0.0] * self.train_env.batch_size
        prev_step_rewards: List[float] = []
        prev_dones: List[bool] = [False] * self.train_env.batch_size

        raw_obs, infos = self.train_env.reset()
        while True:
            for sampled in sample():
                if sampled is None:
                    continue
                yield sampled

            if all(prev_dones):
                # if all the previous episodes are done, we can stop
                break

            # clean observations
            obs = self.agent.preprocessor.batch_clean(raw_obs)

            # filter action cands
            action_cands = self.agent.filter_action_cands(infos["admissible_commands"])
            results = self.agent.calculate_action_scores(
                obs,
                action_cands,
                prev_actions=prev_actions,
                rnn_prev_hidden=rnn_prev_hidden,
            )
            curr_graph = results["curr_graph"].cpu()

            # select actions randomly
            actions_idx = action_select_fn(
                results["action_scores"], results["action_mask"]
            )

            # take a step
            prev_actions = self.agent.decode_actions(action_cands, actions_idx)
            raw_obs, cum_rewards, dones, infos = self.train_env.step(prev_actions)

            # if we took a step before, add a transition
            if prev_actions is not None:
                # add the previous transition to the cache
                transition_cache.batch_add(
                    prev_obs,
                    prev_action_cands,
                    prev_graph,
                    prev_actions_idx,
                    prev_cum_rewards,
                    prev_step_rewards,
                    prev_dones,
                    obs,
                    action_cands,
                    curr_graph,
                )

            # set up the next step
            # first save the previous state
            prev_obs = obs
            prev_action_cands = action_cands
            prev_graph = curr_graph
            prev_actions_idx = actions_idx
            prev_step_rewards = [
                curr - prev for prev, curr in zip(prev_cum_rewards, cum_rewards)
            ]
            prev_cum_rewards = cum_rewards
            prev_dones = dones
            rnn_prev_hidden = results["rnn_curr_hidden"]
            episode_end_fn()

        # push transitions into the buffer
        self.push_to_buffer(transition_cache)

        # collect metrics
        self.game_rewards = transition_cache.get_game_rewards()
        self.game_normalized_rewards = [
            reward / game.metadata["max_score"]
            for reward, game in zip(self.game_rewards, infos["game"])
        ]
        self.game_steps = transition_cache.get_game_steps()

    def push_to_buffer(self, t_cache: TransitionCache) -> None:
        buffer_avg_reward = 0.0
        if len(self.buffer) > 0:
            buffer_avg_reward = np.mean(  # type: ignore
                [transition.step_reward for transition in self.buffer]
            )
        for avg_reward, transitions in zip(t_cache.get_avg_rewards(), t_cache.cache):
            if (
                avg_reward
                >= buffer_avg_reward
                * self.hparams.replay_buffer_reward_threshold  # type: ignore
            ):
                self.buffer.extend(transitions)

    def prepare_batch(self, transitions: List[Transition]) -> Dict[str, torch.Tensor]:
        obs: List[str] = []
        action_cands: List[List[str]] = []
        curr_graphs: List[torch.Tensor] = []
        actions_idx: List[int] = []
        rewards: List[float] = []
        next_obs: List[str] = []
        next_action_cands: List[List[str]] = []
        next_graphs: List[torch.Tensor] = []
        for transition in transitions:
            obs.append(transition.ob)
            action_cands.append(transition.action_cands)
            curr_graphs.append(transition.current_graph)
            actions_idx.append(transition.action_id)
            rewards.append(transition.step_reward)
            next_obs.append(transition.next_ob)
            next_action_cands.append(transition.next_action_cands)
            next_graphs.append(transition.next_graph)

        # preprocess
        obs_word_ids, obs_mask = self.agent.preprocessor.preprocess(obs)
        (
            action_cand_word_ids,
            action_cand_mask,
            action_mask,
        ) = self.agent.preprocess_action_cands(action_cands)
        next_obs_word_ids, next_obs_mask = self.agent.preprocessor.preprocess(next_obs)
        (
            next_action_cand_word_ids,
            next_action_cand_mask,
            next_action_mask,
        ) = self.agent.preprocess_action_cands(next_action_cands)

        return {
            "obs_word_ids": obs_word_ids,
            "obs_mask": obs_mask,
            "current_graph": torch.stack(curr_graphs),
            "action_cand_word_ids": action_cand_word_ids,
            "action_cand_mask": action_cand_mask,
            "action_mask": action_mask,
            "actions_idx": torch.tensor(actions_idx),
            "rewards": torch.tensor(rewards),
            "next_obs_word_ids": next_obs_word_ids,
            "next_obs_mask": next_obs_mask,
            "next_graph": torch.stack(next_graphs),
            "next_action_cand_word_ids": next_action_cand_word_ids,
            "next_action_cand_mask": next_action_cand_mask,
            "next_action_mask": next_action_mask,
        }

    def sample(self) -> Iterator[Transition]:
        for idx in np.random.choice(
            len(self.buffer),
            size=self.hparams.train_sample_batch_size,  # type: ignore
            replace=False,
        ):
            yield self.buffer[idx]


@hydra.main(config_path="train_gata_conf", config_name="config")
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
    trainer = pl.Trainer(**trainer_config)

    # instantiate the lightning module
    lm_model_config = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(lm_model_config, dict)
    if cfg.model.pretrained_graph_updater is not None:
        graph_updater_obs_gen = GraphUpdaterObsGen.load_from_checkpoint(
            to_absolute_path(cfg.model.pretrained_graph_updater.ckpt_path),
            word_vocab_path=cfg.model.pretrained_graph_updater.word_vocab_path,
            node_vocab_path=cfg.model.pretrained_graph_updater.node_vocab_path,
            relation_vocab_path=cfg.model.pretrained_graph_updater.relation_vocab_path,
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


if __name__ == "__main__":
    main()
