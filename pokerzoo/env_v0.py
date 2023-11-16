""":mod:`pokerzoo.env_v0` implements classes and utilities related to
poker environments.
"""

from gymnasium.spaces import Discrete, MultiBinary
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pokerkit import clean_values, State
import numpy as np

CARD_COUNT = 54
STANDING_PAT_OR_DISCARDING = 0
FOLDING = 1
CHECKING_OR_CALLING = 2
COMPLETION_BETTING_OR_RAISING_TO = 3


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
        antes,
        blinds_or_straddles,
        bring_in,
        starting_stacks,
        player_count,
        chip_sizes,
        completion_betting_or_raising_to_sizes,
        render_mode=None,
        illegal_reward=-1,
    ):
        super().__init__()

        self.deck = deck
        """The deck."""
        self.hand_types = hand_types
        """The hand types."""
        self.streets = streets
        """The streets."""
        self.betting_structure = betting_structure
        """The betting structure."""
        self.ante_trimming_status = ante_trimming_status
        """The ante trimming stat"""
        self.antes = antes
        """The antes."""
        self.blinds_or_straddles = blinds_or_straddles,
        """The blinds or straddles."""
        self.bring_in = bring_in
        """The bring-in."""
        self.starting_stacks = starting_stacks
        """The starting stacks."""
        self.player_count = player_count
        """The number of players."""
        self.chip_sizes = chip_sizes
        """The chip sizes."""
        self.completion_betting_or_raising_to_sizes = completion_betting_or_raising_to_sizes
        """The completion, bet, or raise to sizes."""
        self.render_mode = render_mode
        """The optional render mode, defaults to ``'human'``."""
        self.illegal_reward = illegal_reward
        """The illegal reward."""
        self.raw_state = None
        """The raw state."""
        self.state_space = Sequence(
            Dict(
                {
                    'down_cards': Tuple(
                        MultiBinary(CARD_COUNT) for _ in self.possible_agents
                    ),
                    'down_card_counts': Tuple(
                        MultiBinary(CARD_COUNT) for _ in self.possible_agents
                    ),
                    'up_cards': Tuple(
                        MultiBinary(CARD_COUNT) for _ in self.possible_agents
                    ),
                    'board_cards': MultiBinary(CARD_COUNT),
                    'statuses': MultiBinary(player_count),
                    'bets': Tuple(
                        MultiBinary(len(chip_sizes)) for _ in self.possible_agents
                    ),
                    'stacks': Tuple(
                        MultiBinary(len(chip_sizes)) for _ in self.possible_agents
                    ),
                    'pot_contributions': Tuple(
                        MultiBinary(len(chip_sizes)) for _ in self.possible_agents
                    ),
                    'stander_pat_or_discarder': MultiBinary(player_count),
                    'actor': MultiBinary(player_count),
                },
            ),
        )
        """The state space."""
        self._state = []
        self.observations = {}
        """The observations."""
        self.possible_agents = list(range(player_count))
        self.agents = []
        self.observation_spaces = {
            agent: Sequence(
                Dict(
                    {
                        'down_cards': MultiBinary(CARD_COUNT),
                        'down_card_counts': Tuple(
                            MultiBinary(CARD_COUNT) for _ in self.possible_agents
                        ),
                        'up_cards': Tuple(
                            MultiBinary(CARD_COUNT) for _ in self.possible_agents
                        ),
                        'board_cards': MultiBinary(CARD_COUNT),
                        'statuses': MultiBinary(player_count),
                        'bets': Tuple(
                            MultiBinary(len(chip_sizes)) for _ in self.possible_agents
                        ),
                        'stacks': Tuple(
                            MultiBinary(len(chip_sizes)) for _ in self.possible_agents
                        ),
                        'pot_contributions': Tuple(
                            MultiBinary(len(chip_sizes)) for _ in self.possible_agents
                        ),
                        'stander_pat_or_discarder': MultiBinary(player_count),
                        'actor': MultiBinary(player_count),
                    },
                ),
            ) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: Dict(
                {
                    'discarded_cards': MultiBinary(CARD_COUNT),
                    'index': Discrete(
                        len(completion_betting_or_raising_to_sizes)
                        + COMPLETION_BETTING_OR_RAISING_TO,
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

        if (
                (
                    action['index'] != STANDING_PAT_OR_DISCARDING
                    and action['discarded_cards'].any()
                )
                or not (
                    0
                    <= action['index']
                    < (
                        COMPLETION_BETTING_OR_RAISING_TO
                        + len(self.completion_betting_or_raising_to_sizes)
                    )
                )
        ):
            self.rewards[self.agent_selection] = self.illegal_reward
            status = False
        else:
            try:
                match action['index']:
                    case STANDING_PAT_OR_DISCARDING:
                        self.raw_state.stand_pat_or_discard()  # TODO
                    case FOLDING:
                        self.raw_state.fold()
                    case CHECKING_OR_CALLING:
                        self.raw_state.check_or_call()
                    case index:
                        self.raw_state.complete_bet_or_raise_to()  # TODO
            except ValueError:
                status = False
            else:
                status = True

        if status:
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

        sub_state = {
            'down_cards': Tuple(
                MultiBinary(CARD_COUNT) for _ in self.possible_agents
            ),
            'down_card_counts': Tuple(
                MultiBinary(CARD_COUNT) for _ in self.possible_agents
            ),
            'up_cards': Tuple(
                MultiBinary(CARD_COUNT) for _ in self.possible_agents
            ),
            'board_cards': MultiBinary(CARD_COUNT),
            'statuses': MultiBinary(player_count),
            'bets': Tuple(
                MultiBinary(len(chip_sizes)) for _ in self.possible_agents
            ),
            'stacks': Tuple(
                MultiBinary(len(chip_sizes)) for _ in self.possible_agents
            ),
            'pot_contributions': Tuple(
                MultiBinary(len(chip_sizes)) for _ in self.possible_agents
            ),
            'stander_pat_or_discarder': MultiBinary(player_count),
            'actor': MultiBinary(player_count),
        }

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
        return self.observations[agent]

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
        return self.states


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
        chip_sizes,
        completion_betting_or_raising_to_sizes,
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
    :param chip_sizes: The chip sizes.
    :param completion_betting_or_raising_to_sizes: The completion,
                                                   betting, or raising
                                                   to sizes.
    :param render_mode: The optional render mode, defaults to
                        ``'human'``.
    :param illegal_reward: The illegal reward.
    :return: The environment with recommended wrappers.
    """
    env = PokerEnv(
        deck,
        hand_types,
        streets,
        betting_structure,
        ante_trimming_status,
        clean_values(raw_antes),
        clean_values(raw_blinds_or_straddles),
        bring_in,
        clean_values(raw_starting_stacks),
        player_count,
        chip_sizes,
        completion_betting_or_raising_to_sizes,
        render_mode,
        illegal_reward,
    )
    env = wrappers.OrderEnforcingWrapper(env)

    return env


def env_like(
        state,
        chip_sizes,
        completion_betting_or_raising_to_sizes,
        render_mode='human',
        illegal_reward=-1,
):
    """Create a poker environment with recommended wrappers with a
    template state.

    Usually, if you want uniform antes, set ``ante_trimming_status`` to
    ``True``.  If you want non-uniform antes like big blind antes, set
    it to ``False``.

    :param state: The template state.
    :param chip_sizes: The chip sizes.
    :param completion_betting_or_raising_to_sizes: The completion,
                                                   betting, or raising
                                                   to sizes.
    :param render_mode: The optional render mode, defaults to
                        ``'human'``.
    :param illegal_reward: The illegal reward.
    :return: The environment with recommended wrappers.
    """
    return env(
        state.deck,
        state.hand_types,
        state.streets,
        state.betting_structure,
        state.ante_trimming_status,
        state.antes,
        state.blinds_or_straddles,
        state.bring_in,
        state.starting_stacks,
        state.player_count,
        chip_sizes,
        completion_betting_or_raising_to_sizes,
        render_mode,
        illegal_reward,
    )
