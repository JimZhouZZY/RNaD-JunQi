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

import numpy as np
import pyspiel

"""
0## 01
1## 23
2#o 45
3## 67
4## 89
5## 10 11
6## 12 13
7#o 14 15
8## 16 17
9## 18 19
"""

_NUM_PLAYERS = 2
_NUM_MAX_PEACE_STEP = 10
_NUM_ROWS = 4
_NUM_COLS = 2
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_NUM_CHESS_TYPES = 4
_NUM_CHESS_QUANTITY = 4

_ALPHA_EAG = 2.0

_NUM_CHESS_QUANTITY_BY_TYPE = {
    1: 1, 2: 1, 3: 1, 4: 1,
}

_CHESS_WEIGHT_BY_TYPE = {
    12: 8, 11: 6, 10: 10, 9: 9,
    8: 8, 7: 7, 6: 6, 5: 5,
    4: 4, 3: 3, 2: 7, 1: 147,
    0: 0, -10: 0,
}

_NUM_ALL_WEIGHT = 294

_MAX_DFS_DEPTH = 100

_RAILWAY_ROW_IDX = []
_RAILWAY_COL_IDX = []
_CAMP_POSITIONS = []

_BLOCK_DOWN_POSITIONS = []
_BLOCK_UP_POSITIONS = []

_BLOCK_MINE_ACTIONS = []
_BLOCK_BOMB_ACTIONS = []

_FLAG_POSITIONS = [[0, 0], [0, 1], [3, 0], [3, 1]]
_FLAG_ACTIONS = [0, 1, 6, 7]

_DICT_CHESS_NAME = {
    12: "|炸弹|", 11: "|地雷|", 10: "|司令|", 9: "|炸弹|",
    8: "|地雷|", 7: "|司令|", 6: "|军长|", 5: "|师长|",
    4: "|旅长|", 3: "|团长|", 2: "|营长|", 1: "|军旗|",
    0: "|空空|", -10: "|口口|"
}

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
    provides_factored_observation_string=True
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=40,
)


class GamePhase(enum.IntEnum):
    """Enum game phrase."""
    DEPLOYING: int = 0
    SELECTING: int = 1
    MOVING: int = 2


class MapType(enum.IntEnum):
    """Enum game phrase."""
    NORMAL: int = 0
    RAILWAY: int = 1
    CAMP: int = 2
    BLOCK_UP: int = 3
    BLOCK_DOWN: int = 4


class ChessType(enum.IntEnum):
    """Enum chess type to make the code easy-reading."""
    BOMB: int = 9
    MINE: int = 8
    GENERAL: int = 7
    LIEUTENANT_GENERAL: int = 6
    MAJOR_GENERAL: int = 5
    BRIGADIER_GENERAL: int = 4
    COLONEL: int = 3
    LIEUTENANT_COLONEL: int = 2
    FLAG: int = 1
    NONE: int = 0
    UNKOWN: int = -10
    ENGINEER: int = -1000


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
        self._is_fake_terminal: bool = [False, False]
        self._end_in_peace: bool = False

        self.game_phase: GamePhase = GamePhase.DEPLOYING
        self.game_length_real: int = 0
        self.game_length_peace: int = 0
        self.game_length: int = 0

        self.general_dead = [False, False]
        self.selected_pos = [[0, 0], [_NUM_ROWS - 1, _NUM_COLS - 1]]
        self.flags_pos = [[0,0], [0,0]]
        self.chess_deploy_idx = [0, 0]
        self.decode_action = [0, 0] * (_NUM_COLS * _NUM_ROWS)
        self.board = [[Chess(0, -1)] * _NUM_COLS for _ in range(_NUM_ROWS)]
        self.chess_list = [
            [
                1, 4, 3, 2,
            ],
            [
                1, 4, 3, 2,
            ],
        ]
        self.map = [[MapType.NORMAL] * _NUM_COLS for _ in range(_NUM_ROWS)]
        for pos in _CAMP_POSITIONS:
            self.map[pos[0]][pos[1]] = MapType.CAMP
        for pos in _BLOCK_UP_POSITIONS:
            self.map[pos[0]][pos[1]] = MapType.BLOCK_UP
        for pos in _BLOCK_DOWN_POSITIONS:
            self.map[pos[0]][pos[1]] = MapType.BLOCK_DOWN

        self.obs_mov = [[[0] * _NUM_COLS for _ in range(_NUM_ROWS)],
                        [[0] * _NUM_COLS for _ in range(_NUM_ROWS)]]
        self.obs_attack: bool = False

        self.obs_pub = np.ones((2, _NUM_ROWS, _NUM_COLS, _NUM_CHESS_TYPES)) + 0
        self.obs_oppo_pub = np.ones((2, _NUM_ROWS, _NUM_COLS, _NUM_CHESS_TYPES)) + 0
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                if row <= 1:
                    self.obs_pub[0][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0
                    self.obs_oppo_pub[1][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0
                if row >= 2:
                    self.obs_pub[1][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0
                    self.obs_oppo_pub[0][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0
                if [row, col] not in _FLAG_POSITIONS:
                    self.obs_pub[0][row][col][0] = 0
                    self.obs_pub[1][row][col][0] = 0
                    self.obs_oppo_pub[0][row][col][0] = 0
                    self.obs_oppo_pub[1][row][col][0] = 0
                if [row, col] in _CAMP_POSITIONS:
                    self.obs_pub[0][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0
                    self.obs_pub[1][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0
                    self.obs_oppo_pub[0][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0
                    self.obs_oppo_pub[1][row][col] = np.zeros(_NUM_CHESS_TYPES) + 0

        for i in range(_NUM_COLS * _NUM_ROWS):
            self.decode_action[i] = [i // _NUM_COLS, i % _NUM_COLS]
        pass

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def current_player(self) -> int or any:
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player: int):
        """Returns a list of legal actions."""
        actions: list[bool] = [False] * (_NUM_COLS * _NUM_ROWS)
        actions_idx_list: list[int] = []
        if self.game_phase == GamePhase.DEPLOYING:
            selected_chess = self.chess_list[player][self.chess_deploy_idx[player]]
            stt = 0 if player == 0 else _NUM_ROWS // 2
            end = _NUM_ROWS // 2 if player == 0 else _NUM_ROWS
            for row in range(stt, end):
                for col in range(_NUM_COLS):
                    i = row * _NUM_COLS + col
                    if selected_chess == ChessType.FLAG and i not in _FLAG_ACTIONS:
                        continue
                    if selected_chess == ChessType.MINE and i in _BLOCK_MINE_ACTIONS:
                        continue
                    if selected_chess == ChessType.BOMB and i in _BLOCK_BOMB_ACTIONS:
                        continue
                    if (self.board[row][col].type == ChessType.NONE
                            and self.map[row][col] != MapType.CAMP):
                        actions[i] = True

        elif self.game_phase == GamePhase.SELECTING:
            for i in range(_NUM_ROWS):
                for j in range(_NUM_COLS):
                    #and [i, j] not in _FLAG_POSITIONS
                    if (self.board[i][j].country == player
                            and self.board[i][j].type != ChessType.MINE
                            and self.board[i][j].type != ChessType.FLAG
                            and self._get_legal_onestep_action(
                                [i, j], player, booleantype=True)):
                        actions[i * _NUM_COLS + j] = True
        elif self.game_phase == GamePhase.MOVING:
            from_pos = self.selected_pos[player]
            for to_pos in self._get_legal_destination(from_pos, player):
                actions[to_pos[0] * _NUM_COLS + to_pos[1]] = True
        for i in range(len(actions)):
            if actions[i]:
                i_obs = i if player == 0 else (_NUM_CELLS-1-i)
                if player == 1:
                    i_obs = (i_obs - 1) if i_obs % 2 == 1 else (i_obs + 1)
                actions_idx_list.append(i_obs)
        if (not self._is_terminal) and len(actions_idx_list) == 0:
            actions_idx_list.append(_NUM_CELLS - 1)
            self._is_terminal = True
            if len(self._legal_actions(1-self._cur_player)) == 0:
                self._player0_score = 0.0
            self._player0_score = -1.0 if self._cur_player == 0 else 1.0
            
        #print(actions_idx_list)
        return actions_idx_list

    def _get_legal_destination(self, from_pos, player: int):
        """Check whether the destination of a move is legal."""
        legal_destination = []
        legal_destination += self._get_legal_onestep_action(from_pos, player, booleantype=False)

        if self.board[from_pos[0]][from_pos[1]].type == ChessType.ENGINEER:
            legal_destination += self._search_engineer_path(from_pos)

        if from_pos[0] in _RAILWAY_ROW_IDX:
            x = from_pos[1] + 1
            while 0 <= x < _NUM_COLS:
                if self.board[from_pos[0]][x].type == ChessType.NONE:
                    legal_destination.append([from_pos[0], x])
                else:
                    if self.board[from_pos[0]][x].country == 1 - player:
                        legal_destination.append([from_pos[0], x])
                    break
                x += 1
            x = from_pos[1] - 1
            while 0 <= x < _NUM_COLS:
                if self.board[from_pos[0]][x].type == ChessType.NONE:
                    legal_destination.append([from_pos[0], x])
                else:
                    if self.board[from_pos[0]][x].country == 1 - player:
                        legal_destination.append([from_pos[0], x])
                    break
                x -= 1
        if from_pos[1] in _RAILWAY_COL_IDX:
            y = from_pos[0] + 1
            while 0 <= y < _NUM_ROWS:
                if self.board[y][from_pos[1]].type == ChessType.NONE:
                    legal_destination.append([y, from_pos[1]])
                else:
                    if self.board[y][from_pos[1]].country == 1 - player:
                        legal_destination.append([y, from_pos[1]])
                    break
                y += 1
            y = from_pos[0] - 1
            while 0 <= y < _NUM_ROWS:
                if self.board[y][from_pos[1]].type == ChessType.NONE:
                    legal_destination.append([y, from_pos[1]])
                else:
                    if self.board[y][from_pos[1]].country == 1 - player:
                        legal_destination.append([y, from_pos[1]])
                    break
                y -= 1
        return legal_destination

    def _get_legal_onestep_action(self, from_pos, player: int, booleantype=False):
        """Check whether the start point of a move is legal."""
        map_from_pos = self.map[from_pos[0]][from_pos[1]]
        available_pos = [
            [from_pos[0], from_pos[1] + 1],
            [from_pos[0], from_pos[1] - 1],
        ]
        lean_pos = [
            [from_pos[0] + 1, from_pos[1] + 1],
            [from_pos[0] - 1, from_pos[1] + 1],
            [from_pos[0] - 1, from_pos[1] - 1],
            [from_pos[0] + 1, from_pos[1] - 1],
        ]
        available_pos += lean_pos
        if map_from_pos != MapType.BLOCK_DOWN:
            available_pos.append([from_pos[0] + 1, from_pos[1]])
        if map_from_pos != MapType.BLOCK_UP:
            available_pos.append([from_pos[0] - 1, from_pos[1]])

        ans_pos = []

        for to_pos in available_pos:
            r, c = to_pos[0], to_pos[1]
            if 0 <= r < _NUM_ROWS and 0 <= c < _NUM_COLS:
                destination_type = self.map[r][c]
                if map_from_pos == MapType.CAMP:
                    if (destination_type == MapType.CAMP
                            and self.board[r][c] == ChessType.NONE):
                        ans_pos.append(to_pos)
                    elif (destination_type != MapType.CAMP
                          and self.board[r][c].country != player):
                        ans_pos.append(to_pos)
                else:
                    if (destination_type == MapType.CAMP
                            and self.board[r][c] == ChessType.NONE):
                        ans_pos.append(to_pos)
                    elif (destination_type != MapType.CAMP
                          and to_pos not in lean_pos
                          and self.board[r][c].country != player):
                        ans_pos.append(to_pos)
        if booleantype:
            return False if ans_pos == [] else True
        else:
            return ans_pos

    def _search_engineer_path(self, from_pos):
        visited = set()
        reachable_points = []

        def dfs(current_pos):
            visited.add(tuple(current_pos))
            reachable_points.append(list(current_pos))
            
            if list(current_pos) in _BLOCK_UP_POSITIONS:
                directions = [(0, 1), (0, -1), (1, 0)]
            elif list(current_pos) in _BLOCK_DOWN_POSITIONS:
                directions = [(0, 1), (0, -1), (-1, 0)]
            else:
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

            for dx, dy in directions:
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if (
                        0 <= new_pos[0] < _NUM_ROWS
                        and 0 <= new_pos[1] < _NUM_COLS
                        and tuple(new_pos) not in visited
                ):
                    if self.board[new_pos[0]][new_pos[1]].country == 1 - self.current_player():
                        visited.add(tuple(new_pos))
                        reachable_points.append(list(new_pos))
                    elif self.board[new_pos[0]][new_pos[1]].country == self.current_player():
                        continue
                    elif [new_pos[0], new_pos[1]] in _CAMP_POSITIONS:
                        visited.add(tuple(new_pos))
                        reachable_points.append(list(new_pos))
                    elif ((new_pos[0] not in _RAILWAY_ROW_IDX) or (new_pos[1] not in _RAILWAY_COL_IDX) or new_pos[0] in [0, 11] 
                        and [dx, dy] not in [[1, 1], [1, -1], [-1, 1], [-1, -1]]):
                        visited.add(tuple(new_pos))
                        reachable_points.append(list(new_pos))
                    else:
                        reachable_points.append(list(new_pos))
                        dfs(new_pos)

        dfs(from_pos)
        reachable_points.pop(0)
        return reachable_points

    def _apply_action(self, action: int) -> None:
        """Applies the specified action to the state."""
        # TODO: Remove copy module if we can, and refactor the code to easier ones.
        # TODO: Try to make the code easier
        # print(self.serialize(), end="\n\n") if self.game_phase != GamePhase.SELECTING else print("", end="")
        player = self._cur_player
        #print(player)
        #print(self.decode_action[action],self.selected_pos)
        if player == 1:
            action = (action - 1) if action % 2 == 1 else (action + 1)
        
        action = action if self._cur_player == 0 else (7 - action)


        if self._is_terminal:
            self._cur_player = 1 - self._cur_player
            return
        
        if self._is_fake_terminal[player]:
            self._cur_player = 1 - self._cur_player
            return

        if self.game_phase == GamePhase.DEPLOYING:
            self.selected_pos[player] = copy.deepcopy(self.decode_action[action])
            # print(self.chess_deploy_idx)
            r, c = self.selected_pos[player][0], self.selected_pos[player][1]
            self.board[r][c] = Chess(self.chess_list[player][self.chess_deploy_idx[player]], player)
            self.chess_deploy_idx[player] += (1 if self.chess_deploy_idx[player] < _NUM_CHESS_QUANTITY else 0)
            if self.chess_deploy_idx[player] == 0:
                self.flags_pos[player] = [r, c]
            if self.chess_deploy_idx[0] == self.chess_deploy_idx[1] == _NUM_CHESS_QUANTITY:
                self.game_phase = GamePhase.SELECTING
            self._cur_player = 1 - self._cur_player

        elif self.game_phase == GamePhase.SELECTING:
            self.selected_pos[player] = copy.deepcopy(self.decode_action[action])
            self.game_phase = GamePhase.MOVING

        elif self.game_phase == GamePhase.MOVING:
            self.obs_attack = False
            self.obs_mov: list[list[list[int]]] = [[[0] * _NUM_COLS for _ in range(_NUM_ROWS)],
                                                   [[0] * _NUM_COLS for _ in range(_NUM_ROWS)]]

            attacker: Chess = copy.deepcopy(self.board[self.selected_pos[player][0]][self.selected_pos[player][1]])
            defender: Chess = copy.deepcopy(self.board[self.decode_action[action][0]][self.decode_action[action][1]])
            
            obs_defender = self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]]
            defender_min = 0
            for i in range(1, _NUM_CHESS_TYPES):
                obs_defender[i] == 1
                defender_min = i

            if defender.type == ChessType.NONE:
                self.board[self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(attacker)
                
                self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(self.obs_pub[player][self.selected_pos[player][0]][self.selected_pos[player][1]]) # May be useless.
                self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(self.obs_pub[1-player][self.selected_pos[player][0]][self.selected_pos[player][1]])
                self.obs_oppo_pub[player][self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(self.obs_oppo_pub[player][self.selected_pos[player][0]][self.selected_pos[player][1]])
                self.obs_oppo_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(self.obs_oppo_pub[1-player][self.selected_pos[player][0]][self.selected_pos[player][1]]) # May be useless.

                #self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.MINE-1] = 0 # May be useless.
                #self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.MINE-1] = 0
                #self.obs_oppo_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.MINE-1] = 0
                #self.obs_oppo_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.MINE-1] = 0 # May be useless.

                self.obs_mov[player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -1
                self.obs_mov[1 - player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -1

            else:
                self.obs_attack = True
                
                self.obs_mov[player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -0.5
                self.obs_mov[1 - player][self.selected_pos[player][0]][self.selected_pos[player][1]] = -0.5
                
                if defender.type == ChessType.FLAG:
                    # End game by captured flag.
                    self._is_terminal = True
                    self._player0_score = 1.0 if self._cur_player == 0 else -1.0
                    self.board[self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(attacker)
                elif (attacker.type == ChessType.BOMB
                      or defender.type == ChessType.BOMB
                      or attacker.type == defender.type):
                    if attacker.type == ChessType.GENERAL:
                        self.board[self.flags_pos[player][0]][self.flags_pos[player][1]].revealed = True
                    if defender.type == ChessType.GENERAL:
                        self.board[self.flags_pos[1 - player][0]][self.flags_pos[1 - player][1]].revealed = True
                    self.board[self.decode_action[action][0]][self.decode_action[action][1]] = Chess(0, -1)
                    self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]] = np.zeros(_NUM_CHESS_TYPES) + 0
                elif ((attacker.type > defender.type)
                      or
                      (
                        attacker.type == ChessType.ENGINEER
                        and defender.type == ChessType.MINE
                      )):
                    self.board[self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(attacker)
                    self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(self.obs_pub[1-player][self.selected_pos[player][0]][self.selected_pos[player][1]])
                    self.obs_oppo_pub[player][self.decode_action[action][0]][self.decode_action[action][1]] = copy.deepcopy(self.obs_oppo_pub[player][self.selected_pos[player][0]][self.selected_pos[player][1]])

                    if attacker.type == ChessType.ENGINEER and defender.type == ChessType.MINE:
                        self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]] = np.zeros(_NUM_CHESS_TYPES) + 0
                        self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]][2-1] = 1
                        self.obs_oppo_pub[player][self.decode_action[action][0]][self.decode_action[action][1]] = np.zeros(_NUM_CHESS_TYPES) + 0
                        self.obs_oppo_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][2-1] = 1
                    else:
                        for i in range(1, _NUM_CHESS_TYPES + 1, 1):
                            if i <= defender.type:
                                self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]][i-1] = 0
                        #self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.MINE-1] = 0
                        #self.obs_pub[1 - player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.BOMB-1] = 0
                        
                        self.obs_oppo_pub[player][self.decode_action[action][0]][self.decode_action[action][1]] =  np.zeros(_NUM_CHESS_TYPES) + 0
                        for i in range(defender_min+1, 10, 1):
                            if i >= _NUM_CHESS_TYPES:
                                continue
                            self.obs_oppo_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][i] = 1
                elif attacker.type < defender.type:
                    if attacker.type == ChessType.GENERAL:
                        self.board[self.flags_pos[player][0]][self.flags_pos[player][1]].revealed = True
                        self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]] = np.zeros(_NUM_CHESS_TYPES) + 0
                        #self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.MINE-1] = 1
                    if attacker.type == ChessType.ENGINEER:
                        pass
                        #self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.MINE-1] = 0
                    else:
                        for i in range(1, _NUM_CHESS_TYPES + 1, 1):
                            if i <= attacker.type:
                                self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][i-1] = 0
                        #self.obs_pub[player][self.decode_action[action][0]][self.decode_action[action][1]][ChessType.BOMB-1] = 0
                        

            self.game_length_peace += 1
            if self.obs_attack:
                self.game_length_peace = 0

            self.obs_mov[player][self.decode_action[action][0]][self.decode_action[action][1]] = 1
            self.obs_mov[1 - player][self.decode_action[action][0]][self.decode_action[action][1]] = 1

            self.board[self.selected_pos[player][0]][self.selected_pos[player][1]] = Chess(0, -1)

            self.obs_pub[player][self.selected_pos[player][0]][self.selected_pos[player][1]] = np.zeros(_NUM_CHESS_TYPES)
            self.obs_pub[1 - player][self.selected_pos[player][0]][self.selected_pos[player][1]] = np.zeros(_NUM_CHESS_TYPES)
            self.obs_oppo_pub[player][self.selected_pos[player][0]][self.selected_pos[player][1]] = np.zeros(_NUM_CHESS_TYPES)
            self.obs_oppo_pub[1 - player][self.selected_pos[player][0]][self.selected_pos[player][1]] = np.zeros(_NUM_CHESS_TYPES)

            self._cur_player = 1 - self._cur_player
            self.game_length_real += 1
            self.game_phase = GamePhase.SELECTING

        self.game_length += 1
        if self.game_length >= _GAME_INFO.max_game_length:
            self._is_terminal = True
            self._end_in_peace = True
            return

        if self.game_length_peace > _NUM_MAX_PEACE_STEP:
            # End game by maximum game lenth.
            self._is_terminal = True
            self._end_in_peace = True
            return
            

    def _action_to_string(self, player, action):
        """Action -> string."""
        to_pos = self.decode_action[action]
        return "({},{})".format("0" if player == 0 else "1", to_pos)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if self._end_in_peace:
            self._player0_score = 0
            
        return [self._player0_score, -self._player0_score]

    def serialize(self):
        return _board_to_string(self.board)

    def serialize_pov(self, player) -> str:
        pov_board: list[list[Chess]] = [[Chess(0, -1)] * _NUM_COLS for _ in range(_NUM_ROWS)]
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                if self.board[row][col].country == player:
                    pov_board[row][col] = self.board[row][col]
                elif self.board[row][col].country == 1 - player:
                    pov_board[row][col] = Chess(-10, 1 - player)
        return _board_to_string_pov(pov_board, self.board)

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

        self.num_history_move: int = 10
        scalar_shape: int = 1
        prev_select_shape: int = 1

        # Observation components. See paper page 37.
        # Note that we deleted "lakes on the map".
        #
        #                         Map    Private Info.      Public Info. -i    Public Info. i     Move m i
        self.shape = (  # pointers |          |                     |                 |               |
            _NUM_ROWS, _NUM_COLS, (1   + _NUM_CHESS_TYPES + _NUM_CHESS_TYPES + _NUM_CHESS_TYPES + self.num_history_move +
                                   scalar_shape + scalar_shape + scalar_shape + scalar_shape + prev_select_shape))
        #                          Remain Len.    Remain Mov.    Game Phase     Pha.Sele.Mov.  Prev. Selectoin

        self.tensor = np.zeros(np.prod(self.shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, self.shape)}

        self.mov_idx: int = 0

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        prev_obs = copy.deepcopy(obs)
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                for i in range(self.shape[2]):
                    if not 3 * _NUM_CHESS_TYPES + 1 <= i < 3 * _NUM_CHESS_TYPES + self.num_history_move + 1:
                        obs[row][col][i] = 0

        _idx = 0
        obs_row = 0
        obs_col = 0
        self.mov_idx = (self.mov_idx + 1) if (self.mov_idx <= self.num_history_move
                                              and state.game_phase == GamePhase.SELECTING) else 0

        for row in range(_NUM_ROWS):
            if player == 1:
                obs_row = _NUM_ROWS - 1 - row
            else:
                obs_row = row
            for col in range(_NUM_COLS):
                if player == 1:
                    #obs_col = _NUM_COLS - 1 - col
                    obs_col = col
                else:
                    obs_col = col

                chess = state.board[row][col]

                # Map information.
                _idx = 0
                obs[obs_row][obs_col][_idx] = -0.5
                if (row in _RAILWAY_ROW_IDX or col in _RAILWAY_COL_IDX) and row != 0 and row != _NUM_ROWS:
                    obs[obs_row][obs_col][_idx] = 0.25
                if [row, col] in _CAMP_POSITIONS:
                    obs[obs_row][obs_col][_idx] = 0.5
                if [row, col] in _FLAG_POSITIONS:
                    obs[obs_row][obs_col][_idx] = 1
                if [row, col] in _BLOCK_UP_POSITIONS or [row, col] in _BLOCK_DOWN_POSITIONS:
                    obs[obs_row][obs_col][_idx] = 0
                

                # The player’s own private information.
                # Shape: _NUM_ROWS * _NUM_COLS * _NUM_CHESS_TYPES tensor.
                _idx = 1
                if chess.country == player:
                    for t in range(1, _NUM_CHESS_TYPES + 1):
                        if chess.type == t:
                            obs[obs_row][obs_col][_idx + t - 1] = 1

                # The opponent’s public information.
                # Contains all 0’s during the deployment phase.
                # Shape: _NUM_ROWS * _NUM_COLS * _NUM_CHESS_TYPES tensor.
                # TODO: #unrevealed(t) in paper page 36.
                _idx = _NUM_CHESS_TYPES + 1

                if state.game_phase == GamePhase.DEPLOYING:
                    for i in range(_NUM_CHESS_TYPES):
                        obs[obs_row][obs_col][_idx + i] = 0
                elif chess.revealed and chess.country == 1 - player:
                    obs[obs_row][obs_col][_idx + 0] = 1.0
                elif chess.country == 1 - player:
                    a = 0
                    for i in range(_NUM_CHESS_TYPES):
                        a += state.obs_pub[player][row][col][i]
                    if a != 0:
                        for i in range(_NUM_CHESS_TYPES):
                            obs[obs_row][obs_col][_idx + i] = state.obs_pub[player][row][col][i] / (a)
                    else:
                        obs[obs_row][obs_col][_idx + i] = 0
                # ...
                # TODO: Add the situation "if the piece at (r, c) is known to have type t".
                #       For example 40 died and then the position of the flag
                #       should be a public information.
                # ...
                # To be continued.

                # The player’s own public information.
                # Contains all 0’s during the deployment phase.
                # Shape: _NUM_ROWS * _NUM_COLS * _NUM_CHESS_TYPES tensor.
                _idx = 2 * _NUM_CHESS_TYPES + 1

                if state.game_phase == GamePhase.DEPLOYING:
                    for i in range(_NUM_CHESS_TYPES):
                        obs[obs_row][obs_col][_idx + i] = 0
                elif chess.revealed and chess.country == player:
                    obs[obs_row][obs_col][_idx + 0] = 1.0
                elif chess.country == player:
                    a = 0
                    for i in range(_NUM_CHESS_TYPES):
                        a += state.obs_oppo_pub[player][row][col][i]
                    if a != 0:
                        for i in range(_NUM_CHESS_TYPES):
                            obs[obs_row][obs_col][_idx + i] = state.obs_oppo_pub[player][row][col][i] / (a)
                    else:
                        obs[obs_row][obs_col][_idx + i] = 1
                # ...
                # TODO: Add the situation "if the piece at (r, c) is known to have type t".
                #       For example 40 died and then the position of the flag
                #       should be a public information.
                # ...
                # To be continued.


                
                # An encoding of the last 40(or other number) moves.
                # Here we used a scrolling index.
                # Shape: _NUM_ROWS * _NUM_COLS * self.num_history_move tensor.
                # TODO: Need changes in state
                _idx = 3 * _NUM_CHESS_TYPES + 1
                if state.game_phase == GamePhase.SELECTING:
                    for i in range(self.num_history_move - 1):
                        obs[obs_row][obs_col][_idx + i] = obs[obs_row][obs_col][_idx + i + 1]
                    obs[obs_row][obs_col][_idx + self.num_history_move - 1] = state.obs_mov[player][row][col]
                # print(obs[obs_row][obs_col][self.mov_idx + _idx])
                

                # The ratio of the game length to the maximum length
                # before the game is considered a draw.
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 1
                r = state.game_length / _GAME_INFO.max_game_length
                obs[obs_row][obs_col][_idx] = 1 - ((1 - r)**_ALPHA_EAG)

                # The ratio of the number of moves since the last attack
                # to the maximum number of moves without attack before
                # the game is considered a draw.
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 1 + 1
                r = state.game_length_peace / _NUM_MAX_PEACE_STEP
                obs[obs_row][obs_col][_idx] = 1 - ((1 - r)**_ALPHA_EAG)

                # The phase of the game.
                # Either deployment (1) or play (0).
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 2 + 1
                obs[obs_row][obs_col][_idx] = 1 if state.game_phase == GamePhase.DEPLOYING else 0
                _idx += 1

                # An indication of whether the agent needs to select a
                # piece (0) or target square (1) for an already selected piece.
                # 0 during deployment phase.
                # Shape: Scalar -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 3 + 1
                obs[obs_row][obs_col][_idx] = 1 if state.game_phase == GamePhase.MOVING else 0

                # The piece selected in the previous step (1 for the selected
                # piece, 0 elsewhere), if applicable, otherwise all 0’s.
                # Shape: _NUM_COLS * _NUM_ROWS tensor -> _NUM_COLS * _NUM_ROWS * 1 tensor.
                _idx = 3 * _NUM_CHESS_TYPES + self.num_history_move + 4 + 1
                obs[obs_row][obs_col][_idx] = 1 if (state.game_phase == GamePhase.MOVING
                                            and state.selected_pos[player] == [row, col]) else 0

        # self.string_from(state, player) # Uncomment this if you are debugging on observations

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        # TODO: Add string formed observation for debugging and logging.
        obs = self.dict["observation"]
        obs_title = ["Map", "Prv.", "Pub. -i", "Pub. i", "History", "LenMax",
                     "LenAtk", "Game phase", "Select", "Prev.Selection"]
        str_obs = [""] * 10
        str_pub = [""] * _NUM_CHESS_TYPES
        str_pub_oppo = [""] * _NUM_CHESS_TYPES
        str_his = [""] * self.num_history_move

        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                str_obs[0] += ",".join(map(str, [obs[row][col][0]])) + ", "
                str_obs[1] += ",".join(map(str, np.where(obs[row][col][1:_NUM_CHESS_TYPES + 1] == 1)[0] + 1)) + ", "
                str_obs[2] += ",".join(map(str, [obs[row][col][_NUM_CHESS_TYPES + 1]])) + ", "
                str_obs[3] += ",".join(map(str, [obs[row][col][2 * _NUM_CHESS_TYPES + 1]])) + ", "
                for i in range(_NUM_CHESS_TYPES):
                    str_pub[i] += ",".join(map(str, [obs[row][col][1 * _NUM_CHESS_TYPES + 1 + i]])) + ", "
                    str_pub_oppo[i] += ",".join(map(str, [obs[row][col][2 * _NUM_CHESS_TYPES + 1 + i]])) + ", "
                for i in range(self.num_history_move):
                    str_his[i] += ",".join(map(str, [obs[row][col][3 * _NUM_CHESS_TYPES + 1 + i]])) + ", "
                #str_obs[4] += ",".join(map(str, [obs[row][col][3 * _NUM_CHESS_TYPES + self.num_history_move -1]])) + ", "
                str_obs[5] += ",".join(map(str,
                                           [obs[row][col][3 * _NUM_CHESS_TYPES + self.num_history_move + 1]])) + ", "
                str_obs[6] += ",".join(map(str,
                                           [obs[row][col][3 * _NUM_CHESS_TYPES + self.num_history_move + 1 + 1]])) + ", "
                str_obs[7] += ",".join(map(str,
                                           [obs[row][col][3 * _NUM_CHESS_TYPES + self.num_history_move + 2 + 1]])) + ", "
                str_obs[8] += ",".join(map(str,
                                           [obs[row][col][3 * _NUM_CHESS_TYPES + self.num_history_move + 3 + 1]])) + ", "
                str_obs[9] += ",".join(map(str,
                                           [obs[row][col][3 * _NUM_CHESS_TYPES + self.num_history_move + 4 + 1]])) + ", "

            for i in range(len(str_obs)):
                str_obs[i] += "\n"
            for i in range(_NUM_CHESS_TYPES):
                str_pub[i]  += "\n"
                str_pub_oppo[i]  += "\n"
            for i in range(self.num_history_move):
                    str_his[i] += "\n"

        print("Current Player: " + str(player), str(state.current_player()))
        [print(obs_title[i] + "\n" + str_obs[i]) for i in range(len(obs_title))]
        [print("Pub" + str(i+1) + "\n" + str_pub[i]) for i in range(_NUM_CHESS_TYPES)]
        [print("Pub oppo " + str(i+1) + "\n" + str_pub_oppo[i]) for i in range(_NUM_CHESS_TYPES)]
        [print("History " + str(i+1) + "\n" + str_his[i]) for i in range(self.num_history_move)]


# Helper functions for game details.


def _board_to_string(board):
    """Returns a string representation of the board."""
    return "\n".join("".join([str(chess) for chess in row]) for row in board)


def _board_to_string_pov(pov_board, board):
    """Returns a string representation of the board."""
    str_board: str = ""
    for row in range(_NUM_ROWS):
        for col in range(_NUM_COLS):
            str_board += str(pov_board[row][col])
        str_board += "      |      "
        for col in range(_NUM_COLS):
            str_board += str(board[row][col])
        str_board += "\n"
    return str_board


class Chess:
    def __init__(self, num=-10, country=-1):
        self.num = num
        self.type = ChessType(num)
        self.name = _DICT_CHESS_NAME[num]
        self.country = country if num != 0 else -1
        self.revealed: bool = False

    def __str__(self):
        if self.type == ChessType.NONE:
            return f"\033[;30;44m{self.name}\033[0m"
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
