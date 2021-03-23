import pytest

from train_gata import request_infos_for_train, request_infos_for_eval, get_game_dir


def test_request_infos_for_train():
    infos = request_infos_for_train()
    assert infos.admissible_commands is True
    assert infos.description is False
    assert infos.location is False
    assert infos.facts is False
    assert infos.last_action is False
    assert infos.game is True


def test_request_infos_for_eval():
    infos = request_infos_for_eval()
    assert infos.admissible_commands is True
    assert infos.description is True
    assert infos.location is True
    assert infos.facts is True
    assert infos.last_action is True
    assert infos.game is True


@pytest.mark.parametrize(
    "base_dir_path,dataset,difficulty_level,training_size,expected_game_dir",
    [
        ("base_dir", "train", 1, 1, "base_dir/train_1/difficulty_level_1"),
        ("base_dir", "train", 2, 10, "base_dir/train_10/difficulty_level_2"),
        ("base_dir", "valid", 10, None, "base_dir/valid/difficulty_level_10"),
        ("base_dir", "test", 20, None, "base_dir/test/difficulty_level_20"),
    ],
)
def test_get_game_dir(
    base_dir_path, dataset, difficulty_level, training_size, expected_game_dir
):
    assert (
        get_game_dir(
            base_dir_path, dataset, difficulty_level, training_size=training_size
        )
        == expected_game_dir
    )
