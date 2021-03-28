import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import itertools
import hydra

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning.loggers import WandbLogger
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Iterator, Deque, Callable, Iterable
from textworld import EnvInfos
from torch.utils.data import IterableDataset, DataLoader
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
    # received reward
    reward: float = 0.0
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
        if super().__eq__(other):
            return self.current_graph.equal(
                other.current_graph
            ) and self.next_graph.equal(other.next_graph)
        return False


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
                # this episode is already done, don't add this transition
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
            np.mean([transition.reward for transition in episode])  # type: ignore
            for episode in self.cache
        ]


class ReplayBufferDataset(IterableDataset):
    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        return self.generate_batch()


class GATADoubleDQN(WordNodeRelInitMixin, pl.LightningModule):
    def __init__(
        self,
        base_data_dir: Optional[str] = None,
        difficulty_level: int = 1,
        train_data_size: int = 1,
        max_episodes: int = 100000,
        train_max_episode_steps: int = 50,
        train_game_batch_size: int = 25,
        episodes_before_learning: int = 100,
        training_step_freq: int = 50,
        replay_buffer_capacity: int = 500000,
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
            "max_episodes",
            "train_max_episode_steps",
            "train_game_batch_size",
            "episodes_before_learning",
            "training_step_freq",
            "replay_buffer_capacity",
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use the online action selector to get the action scores based on the game state.

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
            + next_q_values * self.hparams.reward_discount,  # type: ignore
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            ReplayBufferDataset(self.gen_train_batch),
            batch_size=self.hparams.train_sample_batch_size,  # type: ignore
            collate_fn=self.prepare_batch,
        )

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.learning_rate)

    def gen_train_batch(self) -> Iterator[Transition]:
        episodes_played = 0
        total_episode_steps = 0
        while episodes_played < self.hparams.max_episodes:  # type: ignore
            transition_cache = TransitionCache(self.train_env.batch_size)
            rnn_prev_hidden: Optional[torch.Tensor] = None
            prev_actions: Optional[List[str]] = None
            prev_obs: List[str] = []
            prev_action_cands: List[List[str]] = []
            prev_graph: torch.Tensor = torch.empty(0)
            prev_actions_idx: List[int] = []
            prev_rewards: List[float] = []
            prev_dones: List[bool] = [False] * self.train_env.batch_size

            # Play a batch of episodes.
            # In order to ensure that all of the transitions have the next state
            # so we collect transitions one step shifted.
            raw_obs, infos = self.train_env.reset()
            for step in itertools.count():
                # Check if we should yield a sample
                if (
                    episodes_played
                    >= self.hparams.episodes_before_learning  # type: ignore
                    and total_episode_steps
                    % self.hparams.training_step_freq  # type: ignore
                    == 0
                ):
                    # if we've played enough episodes to learn and we're at the
                    # yield frequency, yield a batch
                    yield from self.sample()

                if all(prev_dones):
                    # if all the previous episodes are done, we can stop
                    break

                # now take a step
                # clean observations
                obs = self.agent.preprocessor.batch_clean(raw_obs)

                # filter action cands
                action_cands = self.agent.filter_action_cands(
                    infos["admissible_commands"]
                )
                results = self.agent.calculate_action_scores(
                    obs,
                    action_cands,
                    prev_actions=prev_actions,
                    rnn_prev_hidden=rnn_prev_hidden,
                )
                curr_graph = results["curr_graph"].cpu()
                # if we took a step before, add a transition
                if prev_actions is not None:
                    # add the previous transition to the cache
                    transition_cache.batch_add(
                        prev_obs,
                        prev_action_cands,
                        prev_graph,
                        prev_actions_idx,
                        prev_rewards,
                        prev_dones,
                        obs,
                        action_cands,
                        curr_graph,
                    )

                # now pick an action
                if (
                    episodes_played
                    < self.hparams.episodes_before_learning  # type: ignore
                ):
                    # select actions randomly
                    actions_idx = self.agent.select_random(
                        results["action_mask"]
                    ).tolist()
                else:
                    # select actions based on the epsilon greedy strategy
                    actions_idx = self.agent.select_epsilon_greedy(
                        self.agent.action_selector.select_max_q(
                            results["action_scores"], results["action_mask"]
                        ),
                        self.agent.select_random(results["action_mask"]),
                    ).tolist()
                    # (batch)

                # take a step
                prev_actions = self.agent.decode_actions(action_cands, actions_idx)
                raw_obs, rewards, dones, infos = self.train_env.step(prev_actions)

                # set up the next step
                # first save the previous state
                prev_obs = obs
                prev_action_cands = action_cands
                prev_graph = curr_graph
                prev_actions_idx = actions_idx
                prev_rewards = rewards
                prev_dones = dones
                rnn_prev_hidden = results["rnn_curr_hidden"]
                total_episode_steps += 1
            # push transitions into the buffer
            self.push_to_buffer(transition_cache)

            # set up for the next batch of episodes
            # episodes_played increments by self.train_env.batch_size
            # so we have to compare these to mods
            if (
                (episodes_played + self.train_env.batch_size)
                % self.hparams.target_net_update_frequency  # type: ignore
                <= episodes_played
                % self.hparams.target_net_update_frequency  # type: ignore
            ):
                self.update_target_action_selector()
            if episodes_played >= self.hparams.episodes_before_learning:  # type: ignore
                self.agent.update_epsilon(
                    episodes_played
                    - self.hparams.episodes_before_learning  # type: ignore
                )
            episodes_played += self.train_env.batch_size

    def push_to_buffer(self, t_cache: TransitionCache) -> None:
        buffer_avg_reward = 0.0
        if len(self.buffer) > 0:
            buffer_avg_reward = np.mean(  # type: ignore
                [transition.reward for transition in self.buffer]
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
            rewards.append(transition.reward)
            next_obs.append(transition.next_ob)
            next_action_cands.append(transition.next_action_cands)
            next_graphs.append(transition.next_graph)

        # preprocess
        obs_word_ids, obs_mask = self.agent.preprocessor.preprocess(obs)
        (
            action_cand_word_ids,
            action_cand_mask,
        ) = self.agent.preprocess_action_cands(action_cands)
        next_obs_word_ids, next_obs_mask = self.agent.preprocessor.preprocess(next_obs)
        (
            next_action_cand_word_ids,
            next_action_cand_mask,
        ) = self.agent.preprocess_action_cands(next_action_cands)

        return {
            "obs_word_ids": obs_word_ids,
            "obs_mask": obs_mask,
            "current_graph": torch.stack(curr_graphs),
            "action_cand_word_ids": action_cand_word_ids,
            "action_cand_mask": action_cand_mask,
            "actions_idx": torch.tensor(actions_idx),
            "rewards": torch.tensor(rewards),
            "next_obs_word_ids": next_obs_word_ids,
            "next_obs_mask": next_obs_mask,
            "next_graph": torch.stack(next_graphs),
            "next_action_cand_word_ids": next_action_cand_word_ids,
            "next_action_cand_mask": next_action_cand_mask,
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


if __name__ == "__main__":
    main()
