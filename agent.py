import torch

from typing import List, Optional, Tuple, Dict
from itertools import chain

from graph_updater import GraphUpdater
from action_selector import ActionSelector
from preprocessor import SpacyPreprocessor


class Agent:
    def __init__(
        self,
        graph_updater: GraphUpdater,
        action_selector: ActionSelector,
        preprocessor: SpacyPreprocessor,
    ) -> None:
        assert (
            graph_updater.node_embeddings.weight.device
            == action_selector.node_embeddings.weight.device
        )
        self.graph_updater = graph_updater
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = graph_updater.node_embeddings.weight.device

    @torch.no_grad()
    def calculate_action_scores(
        self,
        obs: List[str],
        action_cands: List[List[str]],
        prev_actions: Optional[List[str]] = None,
        rnn_prev_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        If prev_actions is None, use ['restart', ...]

        output: {
            'action_scores': (batch, num_action_cands)
            'action_mask': (batch, num_action_cands)
            'rnn_curr_hidden': (batch, hidden_dim)
            'curr_graph': (batch, num_relation, num_node, num_node)
        }
        """
        # preprocess observations
        obs_word_ids, obs_mask = self.preprocessor.preprocess(obs, device=self.device)

        # set initial prev_actions if necessary
        if prev_actions is None:
            prev_actions = ["restart"] * len(obs)
        # preprocess previous actions
        prev_action_word_ids, prev_action_mask = self.preprocessor.preprocess(
            prev_actions, device=self.device
        )

        # preprocess action candidates
        (
            action_cand_word_ids,
            action_cand_mask,
        ) = self.preprocess_action_cands(action_cands)

        # calculate the current graph
        graph_updater_results = self.graph_updater(
            obs_word_ids,
            prev_action_word_ids,
            obs_mask,
            prev_action_mask,
            rnn_prev_hidden=rnn_prev_hidden,
        )

        # based on the current graph, calculate the q values
        action_scores, action_mask = self.action_selector(
            obs_word_ids,
            obs_mask,
            graph_updater_results["g_t"],
            action_cand_word_ids,
            action_cand_mask,
        )

        return {
            "action_scores": action_scores,
            "action_mask": action_mask,
            "rnn_curr_hidden": graph_updater_results["h_t"],
            "curr_graph": graph_updater_results["g_t"],
        }

    @torch.no_grad()
    def act(
        self,
        raw_obs: List[str],
        raw_action_cands: List[List[str]],
        prev_actions: Optional[List[str]] = None,
        rnn_prev_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Take a batch of raw observations, action candidates, previous actions
        and previous rnn hidden states and return a matching batch of actions that
        maximizes the q value, as well as the new hidden state for the RNN cell.
        """
        # clean observations
        obs = self.preprocessor.batch_clean(raw_obs)

        # filter action cands
        action_cands = self.filter_action_cands(raw_action_cands)

        results = self.calculate_action_scores(
            obs,
            action_cands,
            prev_actions,
            rnn_prev_hidden=rnn_prev_hidden,
        )
        actions_idx = self.action_selector.select_max_q(
            results["action_scores"], results["action_mask"]
        )

        # decode the action strings
        return (
            self.decode_actions(action_cands, actions_idx.tolist()),
            results["rnn_curr_hidden"],
        )

    @staticmethod
    def decode_actions(
        action_cands: List[List[str]], actions_idx: List[int]
    ) -> List[str]:
        return [cands[i] for cands, i in zip(action_cands, actions_idx)]

    @staticmethod
    def filter_action_cands(batch_action_cands: List[List[str]]) -> List[List[str]]:
        """
        batch_action_cands: a batch of "admissible commands" from the game

        returns: (
            filtered batch of action candidates: no look or examine actions
                (except for examine cookbook),
        """
        return [
            list(
                filter(
                    lambda cmd: cmd == "examine cookbook"
                    or cmd.split()[0] not in {"examine", "look"},
                    action_cands,
                )
            )
            for action_cands in batch_action_cands
        ]

    def preprocess_action_cands(
        self, action_cands: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        action_cands: filtered action candidates

        returns: (
            action_cand_word_ids of shape (batch, num_action_cands, action_cand_len)
            action_cand_mask of shape (batch, num_action_cands, action_cand_len)
        )
        """
        # preprocess by flattening out action candidates
        (
            flat_action_cand_word_ids,
            flat_action_cand_mask,
        ) = self.preprocessor.preprocess(
            list(chain.from_iterable(action_cands)), device=self.device
        )

        max_num_action_cands = max(map(len, action_cands))
        max_action_cand_len = flat_action_cand_word_ids.size(1)

        # now pad by max_num_action_cands
        action_cand_word_ids_list: List[torch.Tensor] = []
        action_cand_mask_list: List[torch.Tensor] = []
        i = 0
        for cands in action_cands:
            unpadded_action_cand_word_ids = flat_action_cand_word_ids[
                i : i + len(cands)
            ]
            unpadded_action_cand_mask = flat_action_cand_mask[i : i + len(cands)]
            pad_len = max_num_action_cands - len(cands)
            if pad_len > 0:
                padded_action_cand_word_ids = torch.cat(
                    [
                        unpadded_action_cand_word_ids,
                        torch.zeros(pad_len, max_action_cand_len, dtype=torch.long),
                    ]
                )
                padded_action_cand_mask = torch.cat(
                    [
                        unpadded_action_cand_mask,
                        torch.zeros(pad_len, max_action_cand_len),
                    ]
                )
            else:
                padded_action_cand_word_ids = unpadded_action_cand_word_ids
                padded_action_cand_mask = unpadded_action_cand_mask
            action_cand_word_ids_list.append(padded_action_cand_word_ids)
            action_cand_mask_list.append(padded_action_cand_mask)
            i += len(cands)
        return torch.stack(action_cand_word_ids_list), torch.stack(
            action_cand_mask_list
        )


class EpsilonGreedyAgent(Agent):
    def __init__(
        self,
        graph_updater: GraphUpdater,
        action_selector: ActionSelector,
        preprocessor: SpacyPreprocessor,
        epsilon_anneal_from: float,
        epsilon_anneal_to: float,
        epsilon_anneal_episodes: int,
    ):
        super().__init__(graph_updater, action_selector, preprocessor)
        self.epsilon_anneal_from = epsilon_anneal_from
        self.epsilon_anneal_to = epsilon_anneal_to
        self.epsilon_anneal_episodes = epsilon_anneal_episodes
        self.epsilon = self.epsilon_anneal_from

    @staticmethod
    @torch.no_grad()
    def select_random(action_mask: torch.Tensor) -> torch.Tensor:
        """
        Randomly draw an action.
        We use the action_mask here b/c we want all the unmasked actions
        to have the same probability to be selected, and the masked actions
        zero probabilities.

        action_mask: (batch, num_action_cands)
        output: (batch)
        """
        return torch.multinomial(action_mask, 1).squeeze(dim=1)

    @torch.no_grad()
    def select_epsilon_greedy(
        self, max_q_actions_idx: torch.Tensor, random_actions_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        max_q_actions_idx: (batch)
        random_actions_idx: (batch)

        output: selected actions based on the epsilon greedy strategy (batch)
        """
        # epsilon greedy: epsilon is the probability for using random actions
        batch_size = max_q_actions_idx.size(0)
        choose_random = torch.bernoulli(
            torch.tensor([self.epsilon] * batch_size, device=self.device)
        )
        # (batch)
        return (
            choose_random * random_actions_idx + (1 - choose_random) * max_q_actions_idx
        ).long()

    def update_epsilon(self, step: int) -> None:
        """
        Update the epsilon value linearly based on the current step
        """
        slope = (
            self.epsilon_anneal_to - self.epsilon_anneal_from
        ) / self.epsilon_anneal_episodes
        self.epsilon = min(
            self.epsilon_anneal_to, step * slope + self.epsilon_anneal_from
        )
