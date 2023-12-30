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

import learner as rnad

_exit = 0
_TIME_GAP = 60

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 150, "Number of traversals/games")
flags.DEFINE_string("game_name", "junqi1", "Name of the game")

loss_values = []
epochs = []

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


class JunQiSolver(rnad.RNaDLearner):
    pass

config = rnad.RNaDConfig(
    game_name='junqi1',
    trajectory_max=300,
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
    seed=42, )


def main(unused_argv):
    # logging.info("Loading %s", FLAGS.game_name)
    # game = pyspiel.load_game(FLAGS.game_name)
    rnad_solver = JunQiSolver(config)


    with open('data/model.pkl', 'rb') as f:
        rnad_solver = pickle.load(f)

    #rnad_solver.pit_random()
    rnad_solver.pit()


if __name__ == "__main__":
    app.run(main)
    _exit = 1
    sys.exit()
