import os
import pytorch_lightning as pl

from typing import Optional
from textworld import EnvInfos
from hydra.utils import to_absolute_path

from utils import load_textworld_games


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


class GATADoubleDQN(pl.LightningModule):
    def __init__(
        self,
        game_dir: str = "data/rl.0.2",
        difficulty_level: int = 1,
        training_size: int = 1,
        max_episode_steps: int = 100,
        game_batch_size: int = 5,
    ) -> None:
        super().__init__()
        game_dir = to_absolute_path(game_dir)

        # load envs
        self.train_env = load_textworld_games(
            get_game_dir(
                game_dir, "train", difficulty_level, training_size=training_size
            ),
            "train",
            request_infos_for_train(),
            max_episode_steps,
            game_batch_size,
        )
        self.val_env = load_textworld_games(
            get_game_dir(game_dir, "valid", difficulty_level),
            "val",
            request_infos_for_train(),
            max_episode_steps,
            game_batch_size,
        )
        self.test_env = load_textworld_games(
            get_game_dir(game_dir, "test", difficulty_level),
            "test",
            request_infos_for_train(),
            max_episode_steps,
            game_batch_size,
        )
