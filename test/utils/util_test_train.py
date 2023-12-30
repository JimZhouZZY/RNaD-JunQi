import numpy as np
_NUM_PLAYERS = 2
_NUM_MAX_PEACE_STEP = 50
_NUM_ROWS = 10
_NUM_COLS = 2
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_NUM_CHESS_TYPES = 9
_NUM_CHESS_QUANTITY = 9

_ALPHA_EAG = 2.0

_NUM_CHESS_QUANTITY_BY_TYPE = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, # 11 = Mine, 12 = Bomb
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
_RAILWAY_COL_IDX = [0]
_CAMP_POSITIONS = [[2, 1],[7, 1]]

_BLOCK_DOWN_POSITIONS = []
_BLOCK_UP_POSITIONS = []

_BLOCK_MINE_ACTIONS = range(4, 16)
_BLOCK_BOMB_ACTIONS = range(8, 12)

_FLAG_POSITIONS = [[0, 0], [0, 1], [9, 0], [9, 1],]
_FLAG_ACTIONS = [0, 1, 18, 19]
num_history_move = 40

def string_from(self, obs, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    # TODO: Add string formed observation for debugging and logging.
    #obs = self.dict["observation"]
    obs = np.reshape(obs, (10,2,73))
    obs_title = ["Map", "Prv.", "Pub. -i", "Pub. i", "History", "LenMax",
                 "LenAtk", "Game phase", "Select", "Prev.Selection"]
    str_obs = [""] * 10
    str_pub = [""] * _NUM_CHESS_TYPES
    str_pub_oppo = [""] * _NUM_CHESS_TYPES
    str_his = [""] * num_history_move

    for row in range(_NUM_ROWS):
        for col in range(_NUM_COLS):
            str_obs[0] += ",".join(map(str, [obs[row][col][0]])) + ", "
            str_obs[1] += ",".join(map(str, np.where(obs[row][col][1:_NUM_CHESS_TYPES + 1] == 1)[0] + 1)) + ", "
            str_obs[2] += ",".join(map(str, [obs[row][col][_NUM_CHESS_TYPES + 1]])) + ", "
            str_obs[3] += ",".join(map(str, [obs[row][col][2 * _NUM_CHESS_TYPES + 1]])) + ", "
            for i in range(_NUM_CHESS_TYPES):
                str_pub[i] += ",".join(map(str, [obs[row][col][1 * _NUM_CHESS_TYPES + 1 + i]])) + ", "
                str_pub_oppo[i] += ",".join(map(str, [obs[row][col][2 * _NUM_CHESS_TYPES + 1 + i]])) + ", "

            for i in range(num_history_move):
                str_his[i] += ",".join(map(str, [obs[row][col][3 * _NUM_CHESS_TYPES + 1 + i]])) + ", "

            #str_obs[4] += ",".join(map(str, [obs[row][col][3 * _NUM_CHESS_TYPES + num_history_move -1]])) + ", "
            str_obs[5] += ",".join(map(str,[obs[row][col][3 * _NUM_CHESS_TYPES + num_history_move + 1]])) + ", "
            str_obs[6] += ",".join(map(str,[obs[row][col][3 * _NUM_CHESS_TYPES + num_history_move + 1 + 1]])) + ", "
            str_obs[7] += ",".join(map(str,[obs[row][col][3 * _NUM_CHESS_TYPES + num_history_move + 2 + 1]])) + ", "
            str_obs[8] += ",".join(map(str,[obs[row][col][3 * _NUM_CHESS_TYPES + num_history_move + 3 + 1]])) + ", "
            str_obs[9] += ",".join(map(str,[obs[row][col][3 * _NUM_CHESS_TYPES + num_history_move + 4 + 1]])) + ", "

        for i in range(len(str_obs)):
            str_obs[i] += "\n"

        for i in range(_NUM_CHESS_TYPES):
            str_pub[i]  += "\n"
            str_pub_oppo[i]  += "\n"

        for i in range(num_history_move):
                str_his[i] += "\n"


    #print("Current Player: " + str(player), str(state.current_player()))
    [print(obs_title[i] + "\n" + str_obs[i]) for i in range(len(obs_title))]
    [print("Pub" + str(i+1) + "\n" + str_pub[i]) for i in range(_NUM_CHESS_TYPES)]
    [print("Pub oppo " + str(i+1) + "\n" + str_pub_oppo[i]) for i in range(_NUM_CHESS_TYPES)]
    [print("History " + str(i+1) + "\n" + str_his[i]) for i in range(num_history_move)]
