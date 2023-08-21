# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
import copy
import enum
from typing import List, Any

import numpy as np
import pyspiel
from numpy import ndarray

_NUM_PLAYERS = 2
_NUM_ROWS = 6
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_NUM_CHESS_TYPES = 6
_DICT_CHESS_CELL = {9: 6, 8: 5, 7: 4, 6: 3, 2: 2, 1: 1, 0: 0}
_DICT_CHESS_NAME = {9: "雷", 8: "师", 7: "旅", 6: "团", 2: "炸", 1: "旗", 0: "空"}

_GAME_TYPE = pyspiel.GameType(
    short_name="junqi1",
    long_name="Python Simplized Junqi1",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)  #

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS * _NUM_CELLS + 1,
    max_chance_outcomes=0,  #
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=200  # _NUM_CELLS * _NUM_CELLS
)


class GamePhase(enum.IntEnum):
    """Enum game phrase."""
    DEPLOYING: int = 0
    SELECTING: int = 1
    MOVING: int = 2


class ChessType(enum.IntEnum):
    """Enum chess type to make the code easy-reading."""
    MINE: int = 9
    GENERAL: int = 8
    COLONEL: int = 7
    CAPTAIN: int = 6
    BOMB: int = 2
    FLAG: int = 1
    NONE: int = 0


class JunQiGame(pyspiel.Game):
    """A Python version of the JunQi game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return JunQiState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return JunQiObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


class JunQiState(pyspiel.State):
    """A python version of the JunQi state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)

        self._cur_player: int = 0
        self._player0_score: float = 0.0
        self._is_terminal: bool = False

        self.game_phase: GamePhase = GamePhase.DEPLOYING
        self.game_real_length: int = 0
        self.game_length: int = 0

        self.selected_pos: list[[int, int], [int, int]] = [[0, 0], [_NUM_ROWS - 1, _NUM_COLS - 1]]
        self.decode_action: list[[int, int]] = [0, 0] * (_NUM_COLS * _NUM_ROWS)
        self.board: list[list[Chess]] = [[Chess(0, -1)] * _NUM_COLS for _ in range(_NUM_ROWS)]
        self.chess_list: list[list[int]] = [[9, 8, 8, 7, 7, 6, 6, 2, 1],
                                            [9, 8, 8, 7, 7, 6, 6, 2, 1]]
        self.obs_mov: list[list[list[int]]] = [[[0] * _NUM_COLS for _ in range(_NUM_ROWS)],
                                               [[0] * _NUM_COLS for _ in range(_NUM_ROWS)]]
        self.obs_attack: bool = False

        for i in range(_NUM_COLS * _NUM_ROWS):
            self.decode_action[i] = [i // _NUM_COLS, i % _NUM_COLS]

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def current_player(self) -> int or any:
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player: int) -> list[Any]:
        """Returns a list of legal actions."""
        actions: list[bool] = [False] * (_NUM_COLS * _NUM_ROWS)
        actions_idx_list: list[int] = []
        if self.game_phase == GamePhase.DEPLOYING:
            for i in range(_NUM_COLS * _NUM_ROWS // 2):
                actions[i] = True if self.chess_list[player][i] != 0 else False
        elif self.game_phase == GamePhase.SELECTING:
            for i in range(_NUM_ROWS):
                for j in range(_NUM_COLS):
                    if (self.board[i][j].country == player
                            and self.board[i][j].type != ChessType.MINE
                            and self.board[i][j].type != ChessType.FLAG
                            and self._is_legal_start([i, j], player)):
                        actions[i * _NUM_COLS + j] = True
        elif self.game_phase == GamePhase.MOVING:
            from_pos: list[int, int] = self.selected_pos[player]
            for to_pos in [[from_pos[0] + 1, from_pos[1]],
                           [from_pos[0] - 1, from_pos[1]],
                           [from_pos[0], from_pos[1] + 1],
                           [from_pos[0], from_pos[1] - 1]]:
                if self._is_legal_destination(to_pos, player):
                    actions[to_pos[0] * _NUM_COLS + to_pos[1]] = True
        for i in range(len(actions)):
            if actions[i]:
                actions_idx_list.append(i)
        if len(actions_idx_list) == 0:
            actions_idx_list.append(_NUM_COLS * _NUM_ROWS)
        return actions_idx_list

    def _is_legal_destination(self, to_pos: list[int, int], player: int) -> bool:
        """Check whether the destination of a move is legal."""
        r, c = to_pos[0], to_pos[1]
        return True if (0 <= r < _NUM_ROWS and 0 <= c < _NUM_COLS
                        and self.board[r][c].country != player) else False

    def _is_legal_start(self, from_pos: list[int, int], player: int) -> bool:
        """Check whether the start point of a move is legal."""
        for to_pos in [[from_pos[0] + 1, from_pos[1]],
                       [from_pos[0] - 1, from_pos[1]],
                       [from_pos[0], from_pos[1] + 1],
                       [from_pos[0], from_pos[1] - 1]]:
            if self._is_legal_destination(to_pos, player):
                return True
        return False

    def _apply_action(self, action: int) -> None:
        """Applies the specified action to the state."""
        # TODO: Remove copy module if we can, and refactor the code to easier ones.
        # TODO: Try to make the code easier
        print(self.serialize(), end="\n\n") if self.game_phase != GamePhase.SELECTING else print("", end="")
        player = self._cur_player

        if action == _NUM_COLS * _NUM_ROWS:
            # End game by no legal move.
            self._is_terminal = True
            self._player0_score = -1.0 if player == 0 else 1.0
            return

        if self.game_phase == GamePhase.DEPLOYING:
            r, c = self.selected_pos[player][0], self.selected_pos[player][1]
            self.board[r][c] = Chess(self.chess_list[player][action], player)
            self.chess_list[player][action] = 0
            if player == 0:
                r += (c + 1) // _NUM_COLS
                c = (c + 1) % _NUM_COLS
            else:
                r += (c - 1) // _NUM_COLS
                c = (c - 1) % _NUM_COLS
                if r == (_NUM_ROWS // 2) - 1 and c == _NUM_COLS - 1:
                    # Deployment phase ended, start selecting-moving phase
                    self.game_phase = GamePhase.SELECTING
            # TODO: Maybe this line is not necessary.
            self.selected_pos[player][0], self.selected_pos[player][1] = r, c
            self._cur_player = 1 - self._cur_player

        elif self.game_phase == GamePhase.SELECTING:
            self.selected_pos[player] = copy.deepcopy(self.decode_action[action])
            self.game_phase = GamePhase.MOVING

        elif self.game_phase == GamePhase.MOVING:
            self.obs_mov: list[list[list[int]]] = [[[0] * _NUM_COLS for _ in range(_NUM_ROWS)],
                                                   [[0] * _NUM_COLS for _ in range(_NUM_ROWS)]]

            attacker: Chess = copy.deepcopy(self.board[self.selected_pos[player][0]][self.selected_pos[player][1]])
            defender: Chess = copy.deepcopy(self.board[self.decode_action[action][0]][self.decode_action[action][1]])

            if defender.type == ChessType.NONE:
                self.board[self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(attacker)

                self.obs_mov[player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -1
                self.obs_mov[1 - player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -1

            else:
                self.obs_attack = True
                self.obs_mov[player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -2
                self.obs_mov[1 - player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -2
                if defender.type == ChessType.FLAG:
                    # End game by captured flag.
                    self._is_terminal = True
                    self._player0_score = 1.0 if self._cur_player == 0 else -1.0
                elif (attacker.type == ChessType.BOMB
                      or defender.type == ChessType.BOMB
                      or attacker.type == defender.type):
                    self.board[self.decode_action[action][0]][self.decode_action[action][1]] = Chess(0, -1)
                elif attacker.type > defender.type:
                    self.board[self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(attacker)
                elif attacker.type < defender.type:
                    pass

            self.obs_mov[player][self.decode_action[action][0]][self.decode_action[action][1]] = 1
            self.obs_mov[1 - player][self.decode_action[action][0]][self.decode_action[action][1]] = 1

            self.board[self.selected_pos[player][0]][self.selected_pos[player][1]] = Chess(0, -1)

            self._cur_player = 1 - self._cur_player
            self.game_real_length += 1
            self.game_phase = GamePhase.SELECTING

        self.game_length += 1

    def _action_to_string(self, player, action):
        """Action -> string."""
        to_pos = self.decode_action[action]
        return "({},{})".format("0" if player == 0 else "1", to_pos)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]

    def serialize(self):
        return _board_to_string(self.board)

    def serialize_action(self, action):
        return f"Action:{self.decode_action[action]} AKA {action}"

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return _board_to_string(self.board)


class JunQiObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        # TODO: Add interface for the number of moves of history

        self.num_history_move: int = 20
        self.num_steps_to_attack: int = 20
        scalar_shape: int = 1
        prev_select_shape: int = 1

        # Observation components. See paper page 37.
        # Note that we deleted "lakes on the map".
        #
        #                          Private Info.      Public Info. -i    Public Info. i     Move m i
        self.shape = (  # pointers      |                     |                 |               |
            _NUM_ROWS, _NUM_COLS, (_NUM_CHESS_TYPES + _NUM_CHESS_TYPES + _NUM_CHESS_TYPES + self.num_history_move +
                                   scalar_shape + scalar_shape + scalar_shape + scalar_shape + prev_select_shape))
        #                          Remain Len.    Remain Mov.    Game Phase     Pha.Sele.Mov.  Prev. Selectoin

        self.tensor = np.zeros(np.prod(self.shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, self.shape)}

        self.mov_idx: int = 0
        idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 1
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                self.dict["observation"][row][col][idx] = self.num_steps_to_attack

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        prev_obs = copy.deepcopy(obs)
        obs.fill(0)

        _idx = 0
        self.mov_idx = self.mov_idx + (1 if (self.mov_idx < self.num_history_move
                                             and state.game_phase == GamePhase.SELECTING) else 0)

        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                chess = state.board[row][col]

                # The player’s own private information.
                # Shape: _NUM_ROWS * _NUM_COLS * _NUM_CHESS_TYPES tensor.
                _idx = 0
                for t in range(1, _NUM_CHESS_TYPES + 1):
                    if _DICT_CHESS_CELL[chess.type] == t:
                        obs[row][col][t] = 1

                # The opponent’s public information.
                # Contains all 0’s during the deployment phase.
                # Shape: _NUM_ROWS * _NUM_COLS * _NUM_CHESS_TYPES tensor.
                # TODO: #unrevealed(t) in paper page 36.
                _idx = _NUM_CHESS_TYPES
                for i in range(1, _NUM_CHESS_TYPES + 1):
                    if chess.country == 1 - player:
                        obs[row][col][i] = 2
                for t in range(self.num_history_move):
                    if prev_obs[row][col][t] == 1 and chess.country == 1 - player:
                        for i in range(1, _NUM_CHESS_TYPES + 1):
                            obs[row][col][i] = 3
                            break
                # ...
                # TODO: Add the situation "if the piece at (r, c) is known to have type t".
                #       For example 40 died and then the position of the flag
                #       should be a public information.
                # ...
                # obs[:][:][_NUM_CHESS_TYPES:2 * _NUM_CHESS_TYPES - 1] = pub_oppo
                # To be continued.

                # The player’s own public information.
                # Contains all 0’s during the deployment phase.
                # Shape: _NUM_ROWS * _NUM_COLS * _NUM_CHESS_TYPES tensor.
                _idx = 2 * _NUM_CHESS_TYPES
                for i in range(1, _NUM_CHESS_TYPES + 1):
                    if chess.country == player:
                        obs[row][col][i] = 2
                for t in range(self.num_history_move):
                    if prev_obs[row][col][t] == 1 and chess.country == player:
                        for i in range(1, _NUM_CHESS_TYPES + 1):
                            obs[row][col][i] = 3
                            break
                # ...
                # TODO: Add the situation "if the piece at (r, c) is known to have type t".
                #       For example 40 died and then the position of the flag
                #       should be a public information.
                # ...
                # obs[:][:][_NUM_CHESS_TYPES:2 * _NUM_CHESS_TYPES - 1] = pub_oppo
                # To be continued.

                # An encoding of the last 40(or other number) moves.
                # Here we used a scrolling index.
                # Shape: _NUM_ROWS * _NUM_COLS * self.num_history_move tensor.
                # TODO: Need changes in state
                _idx = 3 * _NUM_CHESS_TYPES
                obs[row][col][self.mov_idx + _idx] = copy.deepcopy(state.obs_mov[player][row][col])

                # The ratio of the game length to the maximum length
                # before the game is considered a draw.
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move
                obs[row][col][_idx] = _GAME_INFO.max_game_length - state.game_length

                # The ratio of the number of moves since the last attack
                # to the maximum number of moves without attack before
                # the game is considered a draw.
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 1
                obs[row][col][_idx] = prev_obs[row][col][0] - 1 if not state.obs_attack else self.num_steps_to_attack

                # The phase of the game.
                # Either deployment (1) or play (0).
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 2
                obs[row][col][_idx] = 1 if state.game_phase == GamePhase.DEPLOYING else 0
                _idx += 1

                # An indication of whether the agent needs to select a
                # piece (0) or target square (1) for an already selected piece.
                # 0 during deployment phase.
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 3
                obs[row][col][_idx] = 1 if state.game_phase == GamePhase.MOVING else 0

                # The piece selected in the previous step (1 for the selected
                # piece, 0 elsewhere), if applicable, otherwise all 0’s.
                # Shape: _NUM_COLS * _NUM_ROWS tensor -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 4
                obs[row][col][_idx] = 1 if (state.game_phase == GamePhase.MOVING
                                            and state.selected_pos[player] == [row, col]) else 0

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        # TODO: Add string formed observation for debugging and logging.
        pass

    @staticmethod
    def _fill_obs_by_d3(obs: np.ndarray, idx: int, value: int or bool or float) -> None:
        for r in range(_NUM_ROWS):
            for c in range(_NUM_COLS):
                obs[r][c][idx] = value


# Helper functions for game details.


def _board_to_string(board):
    """Returns a string representation of the board."""
    return "\n".join("".join([str(chess) for chess in row]) for row in board)


class Chess:
    def __init__(self, num=-10, country=-1):
        self.num = num
        self.type = ChessType(num)
        self.name = _DICT_CHESS_NAME[num]
        self.country = country if num != 0 else -1

    def __str__(self):
        if self.type == ChessType.NONE:
            return f"\033[;;m{self.name}\033[0m"
        elif self.country == 0:
            return f"\033[;30;43m{self.name}\033[0m"
        elif self.country == 1:
            return f"\033[;30;42m{self.name}\033[0m"

    def __repr__(self):
        return repr(self.num)

    def __eq__(self, other):
        return self.num == other

    def __lt__(self, other):
        return self.num < other

    def __gt__(self, other):
        return self.num > other


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, JunQiGame)
