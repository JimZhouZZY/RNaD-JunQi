import os, sys
import time
import traceback
import pickle
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
#os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import actor
import learner
from learner import ActorStep, TimeStep, EnvStep
from config import generate_actor_config, generate_learner_config
from config import _NUM_ACTORS, _BATCH_SIZE, _SAVE_GAP, _FILE_PATH
print(f"Training started with {_NUM_ACTORS} agents and batch size={_BATCH_SIZE}")

from test.utils.timeit import timeit
from utils import print_loss
from test.utils.obs_string import string_from

import ray

ray.init()


@timeit
def update_actor_net(gradients):
    for actor in actor_agents:
        # Network params.
        actor.params = gradients[0]
        actor.params_target = gradients[1]
        actor.params_prev = gradients[2]
        actor.params_prev_ = gradients[3]
        # Optimizer state.
        #actor.optimizer.state = gradients[4]
        #actor.optimizer_target.state = gradients[5]

@timeit
def collect_timestep(ts):
    time_steps = []
    for t in range(40):
        env_steps = []
        actor_steps = []
        prev_env_steps = []
        for d in range(_NUM_ACTORS):
            for prev_env_step in ts[d][t][0]:
                prev_env_steps.append(
                    EnvStep(
                        valid = prev_env_step.valid,
                        obs = prev_env_step.obs,
                        legal = prev_env_step.legal,
                        player_id = prev_env_step.player_id,
                        rewards = prev_env_step.rewards,
                        )
                    )
            
            #prev_env_steps.extend(ts[d][t][0])
            actor_steps.append(ts[d][t][1])
            #env_steps.extend(ts[d][t][2])
            for env_step in ts[d][t][2]:
                env_steps.append(
                    EnvStep(
                        valid = env_step.valid,
                        obs = env_step.obs,
                        legal = env_step.legal,
                        player_id = env_step.player_id,
                        rewards = env_step.rewards,
                        )
                    )
            
        prev_env_step = jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *prev_env_steps)
        env_step = jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *env_steps)
        actor_steps = np.concatenate(actor_steps, axis=1)
        actor_step = ActorStep(policy=actor_steps[0], action_oh=actor_steps[1], rewards=())
        time_steps.append(
            TimeStep(
                env=prev_env_step,
                actor=ActorStep(
                    action_oh=actor_step.action_oh,
                    policy=actor_step.policy,
                    rewards=env_step.rewards
                    ),
            )
        )
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *time_steps)

@timeit
def learner_step(timestep):
    return learner_agent.learn(timestep)

@timeit
def actor_step():
    return ray.get([actor_agents[i].act.remote() for i in range(_NUM_ACTORS)])

def train():
    i = 0
    iterations = 1e6
    t_list = []
    t_std = time.perf_counter()
    while(i <= iterations):
        try:
            t_start = time.perf_counter()
            timestep = collect_timestep(actor_step())
            gradients, logs = learner_step(timestep)
            update_actor_net(gradients)
            t_end = time.perf_counter()
            t_list.append(t_end - t_start)
            print(f"[INFO] Training in progress: {100 * i / iterations}% [{i} of {iterations} iterations] " + "Time used: " + time.strftime(
                    "%H:%M:%S", time.gmtime(t_end - t_std)) + " ETA: " + time.strftime(
                    "%H:%M:%S", time.gmtime(np.average(t_list) * (iterations - i))))
            i += 1
            loss = np.average(logs['loss'])
            learner_steps = int(logs['learner_steps'])
            loss_values.append(loss)
            iterations_list.append(learner_steps)
            try:
                #print(logs)
                print_loss(loss, learner_steps, i)
                if i % _SAVE_GAP == 0:
                    plt.plot(iterations_list, loss_values)
                    plt.savefig(_FILE_PATH+str(len(iterations_list)))
                    with open(_FILE_PATH+'model_auto_saved.pkl', 'wb') as f:
                        pickle.dump(learner_agent, f)
                    with open(_FILE_PATH+'loss.txt', 'w') as file:
                        for value in loss_values:
                            file.write(str(value) + '\n')
            except Exception as e:
                traceback.print_exc()
                sys.exit(0)
            except KeyboardInterrupt:
                sys.exit(0)
        except Exception as e:
            #plt.plot(iterations_list, loss_values)
            #plt.savefig(_FILE_PATH+str(len(iterations_list)))
            with open(_FILE_PATH+'model_auto_saved.pkl', 'wb') as f:
                pickle.dump(learner_agent, f)
            with open(_FILE_PATH+'loss.txt', 'w') as file:
                for value in loss_values:
                    file.write(str(value) + '\n')
            traceback.print_exc()
            #os.system("shutdown now -h")
            sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)


loss_values = []
iterations_list = []
actor_agents = []
gradients = None
if os.path.exists("data/loss.txt"):
    with open(_FILE_PATH+'loss.txt', 'r') as file:
        loss_values = file.read().splitlines()
    iterations_list = list(range(1, len(loss_values)+1))

actor_configs = [generate_actor_config(_BATCH_SIZE, _NUM_ACTORS) for _ in range(_NUM_ACTORS)]
for i in range(_NUM_ACTORS):
    actor_agents.append(actor.RNaDActor.remote(actor_configs[i]))

if os.path.exists("data/model.pkl"):
    print("[INFO] Model data found, continuing training...")
    with open('data/model.pkl', 'rb') as f:
        learner_agent = pickle.load(f)
    gradients =  [learner_agent.params, learner_agent.params_target, learner_agent.params_prev, learner_agent.params_prev_,
         learner_agent.optimizer, learner_agent.optimizer_target]
    update_actor_net(gradients)
else:
    learner_config = generate_learner_config(_BATCH_SIZE)
    learner_agent = learner.RNaDLearner(learner_config)

if __name__ == "__main__":
    train()

