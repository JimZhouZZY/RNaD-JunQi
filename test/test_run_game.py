import numpy as np
from open_spiel.python import policy as policy_lib
import pyspiel

game = pyspiel.load_game("junqi1")
state = game.new_initial_state()

while not state.is_terminal():
    state.apply_action(np.random.choice(state.legal_actions()))
    print(str(state) + '\n')
    print(state._is_terminal, state._is_fake_terminal)
    #input()

print(state.returns(), state.game_length)
