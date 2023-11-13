""":mod:`pokerzoo.env_v0` implements classes and utilities related to
poker environments.
"""

from gymnasium.spaces import Discrete, MultiBinary
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pokerkit import clean_values, State
import numpy as np


def env(
        deck,
        hand_types,
        streets,
        betting_structure,
        ante_trimming_status,
        raw_antes,
        raw_blinds_or_straddles,
        bring_in,
        raw_starting_stacks,
        player_count,
        render_mode='human',
        illegal_reward=-1,
):
    """Create a poker environment with recommended wrappers.

    Usually, if you want uniform antes, set ``ante_trimming_status`` to
    ``True``.  If you want non-uniform antes like big blind antes, set
    it to ``False``.

    :param deck: The deck.
    :param hand_types: The hand types.
    :param streets: The streets.
    :param betting_structure: The betting structure.
    :param ante_trimming_status: The ante trimming stat
    :param raw_antes: The raw antes.
    :param raw_blinds_or_straddles: The raw blinds or straddles.
    :param bring_in: The bring-in.
    :param raw_starting_stacks: The raw starting stacks.
    :param player_count: The number of players.
    :param render_mode: The optional render mode, defaults to
                        ``'human'``.
    :return: The environment with recommended wrappers.
    """
    env = raw_env(
        deck,
        hand_types,
        streets,
        betting_structure,
        ante_trimming_status,
        raw_antes,
        raw_blinds_or_straddles,
        bring_in,
        raw_starting_stacks,
        player_count,
        render_mode,
        illegal_reward,
    )
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

    return env


class PokerEnv(AECEnv):
    """The class for raw poker environments."""

    metadata = {'render_modes': ['human'], 'name': 'poker_v0'}

    def __init__(
        self,
        deck,
        hand_types,
        streets,
        betting_structure,
        ante_trimming_status,
        raw_antes,
        raw_blinds_or_straddles,
        bring_in,
        raw_starting_stacks,
        player_count,
        render_mode=None,
        illegal_reward=-1,
    ):
        super().__init__()

        self.deck = deck
        self.hand_types = hand_types
        self.streets = streets
        self.betting_structure = betting_structure
        self.ante_trimming_status = ante_trimming_status
        self.antes = clean_values(raw_antes, player_count)
        self.blinds_or_straddles = clean_values(
            raw_blinds_or_straddles,
            player_count,
        )
        self.bring_in = bring_in
        self.starting_stacks = clean_values(raw_starting_stacks, player_count)
        self.player_count = player_count
        self.render_mode = render_mode
        self.illegal_reward = illegal_reward
        self.raw_state = None
        self.possible_agents = list(range(player_count))
        self.agents = []
        self.observation_spaces = {
            agent: Dict(
                {
                },
            ) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: Dict(
                {
                    'dealing': MultiBinary(52),
                    'betting': Discrete(
                        self.starting_stacks[agent] + 2,
                        start=-1,
                    ),
                },
            ) for agent in self.possible_agents
        }
        self.terminations = {}
        self.truncations = {}
        self.rewards = {}
        self._cumulative_rewards = {}
        self.infos = {}
        self.agent_selection = None

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)

            return

        assert self.raw_state is not None
        assert self.agent_selection is not None

        action_status = True

        if action == -1:
            self.raw_state.fold()
        elif action == self.raw_state.checking_or_calling_amount:
            self.raw_state.check_or_call()
        elif (
                self.raw_state.min_completion_betting_or_raising_to_amount
                <= action
                <= self.raw_state.max_completion_betting_or_raising_to_amount
        ):
            self.raw_state.complete_bet_or_raise_to(action)
        else:
            action_status = False
        
        if not action_status:
            self.rewards[self.agent_selection] = self.illegal_reward

        self._update()

        if self.render_mode is not None:
            self.render()

    def reset(self, seed=None, options=None):
        self.raw_state = State(
            tuple(Automation),
            self.deck,
            self.hand_types,
            self.streets,
            self.betting_structure,
            self.ante_trimming_status,
            self.antes,
            self.blinds_or_straddles,
            self.bring_in,
            self.starting_stacks,
            self.player_count,
        )

        self.agents.clear()
        self.agents.extend(self.possible_agents)

        for agent in self.agents:
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.rewards[agent] = 0
            self._cumulative_rewards[agent] = 0
            self.infos[agent] = {}

        self.agent_selection = None

        self._update()

    def _update(self):
        assert self.raw_state is not None

        if self.raw_state.status is False:
            for agent in self.agents:
                self.rewards[agent] = (
                    self.raw_state.stacks[agent]
                    - self.raw_state.starting_stacks[agent]
                )

        if self.raw_state.stander_pat_or_discarder_index is not None:
            self.agent_selection = (
                self.raw_state.stander_pat_or_discarder_index
            )
        elif self.actor_index is not None:
            self.agent_selection = self.raw_state.actor_index
        else:
            self.agent_selection = None

        self._accumulate_rewards()

    def observe(self, agent):
        pass

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = 'Game over'

        print(string)

    def state(self):
        pass
