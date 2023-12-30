import numpy as np
from open_spiel.python import policy as policy_lib
import pyspiel

game = pyspiel.load_game("junqi1")
state = game.new_initial_state()

for _ in range(53):
  state.observation_tensor()
  state.apply_action(np.random.choice(state.legal_actions()))
  '''
  for row in range(12):
    for col in range(5):
      print(state.obs_mov[0][row][col], end =" ")
    print("\n", end="")
  print("\n", end="")
  '''
  #input()

print(str(state) + '\n')

print("obs_pub:")
for i in range(2):
  print(f"Agent: {i}")
  for t in range(6):
    print(f"Chess Type: {t}")
    for row in range(8):
      for col in range(2):
        print(state.obs_pub[i][row][col][t], end=" ")
      print("\n", end="")
    print("\n", end="")
  print("\n", end="")

print(str(state) + '\n')
print("obs_oppo_pub:")
for i in range(2):
  print(f"Agent: {i}")
  for t in range(6):
    print(f"Chess Type: {t}")
    for row in range(8):
      for col in range(2):
        print(state.obs_oppo_pub[i][row][col][t], end=" ")
      print("\n", end="")
    print("\n", end="")
  print("\n", end="")

print(str(state) + '\n')
print("History:")
for i in range(2):
  print(f"Agent: {i}")
  for row in range(8):
      for col in range(2):
        print(state.obs_mov[i][row][col], end=" ")
      print("\n", end="")
  print("\n", end="")
#print(str(state) + '\n')

obs_tensor = state.observation_tensor()
#print(np.array(obs_tensor, dtype=np.float64))
print(str(state) + '\n')
state.observation_string()
