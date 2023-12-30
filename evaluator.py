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

"""Python RNAD example."""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy
from absl import app
from absl import flags
from absl import logging
import matplotlib
import matplotlib.pyplot as plt
import time, sys, os
import numpy as np
import pickle
import time
import threading
import warnings
from numpy import random

import learner as rnad

_exit = 0
_TIME_GAP = 60

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 150, "Number of traversals/games")
flags.DEFINE_string("game_name", "junqi1", "Name of the game")

loss_values = []
epochs = []


class JunQiSolver(rnad.RNaDLearner):
    pass

config = rnad.RNaDConfig(
    game_name='junqi1',
    trajectory_max=200,
    state_representation=rnad.StateRepresentation.INFO_SET,
    policy_network_layers=(256, 256),
    batch_size=1,
    learning_rate=0.00005,
    adam=rnad.AdamConfig(),
    clip_gradient=10_000,
    target_network_avg=0.001,
    entropy_schedule_repeats=(1,),
    entropy_schedule_size=(20_000,),
    eta_reward_transform=0.2,
    nerd=rnad.NerdConfig(),
    c_vtrace=1.0,
    finetune=rnad.FineTuning(),
    seed=24, )


def main(unused_argv):
    # logging.info("Loading %s", FLAGS.game_name)
    # game = pyspiel.load_game(FLAGS.game_name)
    rnad_solver = JunQiSolver(config)

    while True:
        with open('data/model.pkl', 'rb') as f:
            rnad_solver = pickle.load(f)
        
        with open('data/model1.pkl', 'rb') as f:
            rnad_solver1 = pickle.load(f)

        rnad_solver.pit_random()
        #rnad_solver.pit()

        rnad_solver.config = rnad.RNaDConfig(
            game_name='junqi1',
            batch_size=1,
        )
        rnad_solver1.config = rnad.RNaDConfig(
            game_name='junqi1',
            batch_size=1,
        )
        random_player = 0
        reward = [0, 0]
        num = 0
        wins = 0
        loses = 0
        peaces = 0
        for _ in range(1000):
            start_time = time.perf_counter()
            states = [
                rnad_solver._play_chance(rnad_solver._game.new_initial_state())
                for _ in range(1)
            ]
            env_step = rnad_solver._batch_of_states_as_env_step(states)
            state = states[0]
            while not state.is_terminal():
                a = None
                if state.current_player() == random_player:
                    #actions = state.legal_actions()
                    #idx = random.randint(0, len(actions) - 1)
                    #a = [actions[idx]]
                    a, actor_step = rnad_solver1.actor_step(env_step)
                else:
                    a, actor_step = rnad_solver.actor_step(env_step)
            states = rnad_solver._batch_of_states_apply_action(states, a)
            env_step = rnad_solver._batch_of_states_as_env_step(states)
            reward[0] += state.returns()[1 - random_player]
            reward[1] += state.returns()[random_player]
            if state.returns()[1 - random_player] == 1:
                wins += 1
            elif state.returns()[random_player] == 1:
                loses += 1
            else:
                peaces += 1
            num += 1
            end_time = time.perf_counter()
            t = end_time - start_time
            #print(random_player, state.returns())
            #print("=========================")
            print(f"Number: {num}, Reward: {[state.returns()[1 - random_player], state.returns()[random_player]]}, in {t} seconds.")
            #print(state.game_length, state.game_length_real, 1-random_player)
            #print(str(state) + '\n')
            print(f"Score: {reward}, WinRate: {wins / num}, LoseRate: {loses / num}, PeaceRate: {peaces / num}")
            print("=========================")
            random_player = 1 - random_player
        print("=========================")
        print("=========================")
        print("=========================")
        input()


if __name__ == "__main__":
    app.run(main)
    _exit = 1
    sys.exit()
