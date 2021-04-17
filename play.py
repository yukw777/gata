import itertools
import textworld
import gym

from train_gata import request_infos_for_eval, GATADoubleDQN
from agent import Agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("game_file")
    parser.add_argument("gata_double_dqn_ckpt")
    args = parser.parse_args()

    gata_double_dqn = GATADoubleDQN.load_from_checkpoint(
        args.gata_double_dqn_ckpt,
        word_vocab_path="vocabs/word_vocab.txt",
        node_vocab_path="vocabs/node_vocab.txt",
        relation_vocab_path="vocabs/relation_vocab.txt",
    )
    agent = Agent(
        gata_double_dqn.graph_updater,
        gata_double_dqn.action_selector,
        gata_double_dqn.preprocessor,
    )

    env_id = textworld.gym.register_game(
        args.game_file, request_infos=request_infos_for_eval()
    )
    env = gym.make(env_id)

    prev_actions = None
    rnn_prev_hidden = None
    ob, info = env.reset()
    print(ob)
    for step in itertools.count():
        input("Press Enter to make a move ")
        actions, rnn_prev_hidden = agent.act(
            [ob],
            [info["admissible_commands"]],
            prev_actions=prev_actions,
            rnn_prev_hidden=rnn_prev_hidden,
        )
        action = actions[0]
        print(f"\n>> {action}\n")
        ob, reward, done, info = env.step(action)
        print(ob)
        if done:
            break
        prev_actions = actions
    env.close()
    print(f"normalized reward: {reward/step}")
