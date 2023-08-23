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

from open_spiel.python.algorithms.rnad import rnad_for_junqi as rnad

_exit = 0
_TIME_GAP = 300

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 150, "Number of traversals/games")
flags.DEFINE_string("game_name", "junqi1", "Name of the game")

loss_values = []
iterations_list = []


def command_line_action(time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        print("Choose an action from {}:".format(legal_actions))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action


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


class JunQiSolver(rnad.RNaDSolver):
    pass


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def plot():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.ion()  # Turn on interactive mode for dynamic updating
        fig, ax = plt.subplots()
        # matplotlib.use('TkAgg')
        ax.set_title('Training Loss Curve')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        while not _exit:
            plt.pause(_TIME_GAP)
            ax.clear()
            ax.plot(iterations_list, loss_values)  # , marker='o', linestyle='-')
            ax.set_title('Training Loss Curve')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            plt.pause(0.01)
            #plt.draw()
            plt.savefig(len(iterations_list))
            plt.savefig(str(len(iterations_list)))
            plt.pause(0.01)


def print_loss(any, i):
    loss_values.append(any['loss'].tolist())
    iterations_list.append(any['learner_steps'])
    if i % 1 == 0:
        print(
            f"[DATA] Loss: {round(any['loss'].tolist(), 4)}, "
              f"Actor Steps: {any['actor_steps']}, "
              f"Learner Steps: {any['learner_steps']}"
        )


config = rnad.RNaDConfig(
    game_name='junqi1_py38',
    trajectory_max=200,
    state_representation=rnad.StateRepresentation.OBSERVATION,
    policy_network_layers=(256, 256),
    batch_size=512,
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
    seed=42,
)


def main(unused_argv):
    # logging.info("Loading %s", FLAGS.game_name)
    # game = pyspiel.load_game(FLAGS.game_name)
    with open('model.pkl', 'wb') as f:
        pass
    with open('model.pkl', 'rb') as f:
        if f.readline() != b'':
            print("[INFO] Model data found, continuing training...")
            rnad_solver = pickle.load(f)
        else:
            rnad_solver = JunQiSolver(config)
    i = 0
    iterations = 1e2
    t_list = []
    t_std = time.perf_counter()
    threading.Thread(target=plot).start()
    try:
        while (i <= iterations):
            t_start = time.perf_counter()
            print_loss(rnad_solver.step(), i)
            t_end = time.perf_counter()
            t_list.append(t_end - t_start)
            print(
                f"[INFO] Training in progress: {100 * i / iterations}% [{i} of {iterations} iterations] " + "Time used: " + time.strftime(
                    "%H:%M:%S", time.gmtime(t_end - t_std)) + " ETA: " + time.strftime(
                    "%H:%M:%S", time.gmtime(numpy.average(t_list) * (iterations - i))))
            i += 1
    except KeyboardInterrupt:
        with open('model.pkl', 'wb') as f:
            pickle.dump(rnad_solver, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(rnad_solver, f)

    rnad_solver.pit()


if __name__ == "__main__":
    app.run(main)
    _exit = 1
    sys.exit()
