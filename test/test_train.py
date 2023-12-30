import numpy as np
import jax

import rnad_actor
import rnad_learner
from open_spiel.python.algorithms.rnad import rnad_for_junqi as rnad
#from rnad_actor import ActorStep, TimeStep

_NUM_ACTORS = 1
_BATCH_SIZE = 1

actor_config = rnad_actor.RNaDConfig(
    game_name='junqi1',
    trajectory_max=300,
    state_representation=rnad_actor.StateRepresentation.OBSERVATION,
    policy_network_layers=(16, 16, 16, 16),
    batch_size=_BATCH_SIZE//_NUM_ACTORS,
    learning_rate=0.00005,
    adam=rnad_actor.AdamConfig(),
    clip_gradient=10_000,
    target_network_avg=0.001,
    entropy_schedule_repeats=(1,),
    entropy_schedule_size=(20_000,),
    eta_reward_transform=0.2,
    nerd=rnad_actor.NerdConfig(),
    c_vtrace=1.0,
    finetune=rnad_actor.FineTuning(),
    seed=42,
)

learner_config = rnad_learner.RNaDConfig(
    game_name='junqi1',
    trajectory_max=300,
    state_representation=rnad_learner.StateRepresentation.OBSERVATION,
    policy_network_layers=(16, 16, 16, 16),
    batch_size=_BATCH_SIZE,
    learning_rate=0.00005,
    adam=rnad_learner.AdamConfig(),
    clip_gradient=10_000,
    target_network_avg=0.001,
    entropy_schedule_repeats=(1,),
    entropy_schedule_size=(20_000,),
    eta_reward_transform=0.2,
    nerd=rnad_learner.NerdConfig(),
    c_vtrace=1.0,
    finetune=rnad_learner.FineTuning(),
    seed=42,
)

actor = rnad_actor.RNaDSolver(actor_config)
rnader = rnad.RNaDSolver(actor_config)
rnader2 = rnad.RNaDSolver(actor_config)
learner = rnad_learner.RNaDSolver(learner_config)

def update_actor_net(gradients):
    # Network params.
    actor.params = gradients[0]
    actor.params_target = gradients[1]
    actor.params_prev = gradients[2]
    actor.params_prev_ = gradients[3]
    # Optimizer state.
    #actor.optimizer.state = gradients[4]
    #actor.optimizer_target.state = gradients[5]


def collect_timestep(ts):
    time_steps = []
    for t in range(300):
        env_steps = []
        actor_steps = []
        prev_env_steps = []
        for d in range(_NUM_ACTORS):
            prev_env_steps.extend(ts[d][t][0])
            actor_steps.append(ts[d][t][1])
            env_steps.extend(ts[d][t][2])
        prev_env_step = jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *prev_env_steps)
        env_step = jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *env_steps)
        actor_steps = np.concatenate(actor_steps, axis=1)
        actor_step = rnad_learner.ActorStep(policy=actor_steps[0], action_oh=actor_steps[1], rewards=())
        time_steps.append(
            rnad_learner.TimeStep(
                env=prev_env_step,
                actor=rnad_learner.ActorStep(
                    action_oh=actor_step.action_oh,
                    policy=actor_step.policy,
                    rewards=env_step.rewards),
            )
        )
    #print(len(time_steps))
    #print(time_steps[0])
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *time_steps)


def train():
    while(True):
        timestep = collect_timestep(ray.get([actor.act.remote() for _ in range(_NUM_ACTORS)]))
        ts_test = rnader.act()
        print(ts_test)
        input()
        #print(len(timesteps), len(timesteps[0]), type(timesteps[0]))
        #timestep = np.concatenate((timesteps[0],timesteps[1],timesteps[2],timesteps[3]), axis=1)
        gradients, logs = ray.get(learner.learn.remote(timestep))
        update_actor_net(gradients)
        print(logs)


ts_opt = rnader.act()
ts_opt = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *ts_opt)
#ts_opt2 = rnader2.act()
#ts_opt2 = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *ts_opt2)
ts = collect_timestep([actor.act()])

np.set_printoptions(threshold=np.inf)
#print(ts.actor)
#print(ts_opt.actor)

np.savetxt("ts_env_valid.txt", ts.env.valid,fmt="%s")
np.savetxt("ts_env_obs_0.txt", ts.env.obs[0],fmt="%s")
np.savetxt("ts_env_obs_51.txt", ts.env.obs[51],fmt="%s")
#np.savetxt("ts_env_legal.txt", ts.env.legal,fmt="%s")
np.savetxt("ts_env_player_id.txt", ts.env.player_id,fmt="%s")
#np.savetxt("ts_env_rewards.txt", ts.env.rewards,fmt="%s")
#np.savetxt("ts_actor_action_oh.txt", ts.actor.action_oh,fmt="%s")
#np.savetxt("ts_actor_policy.txt", ts.actor.policy,fmt="%s")
#np.savetxt("ts_actor_rewards.txt", ts.actor.rewards,fmt="%s")

np.savetxt("ts_opt_env_valid.txt", ts_opt.env.valid,fmt="%s")
np.savetxt("ts_opt_env_obs_0.txt", ts_opt.env.obs[0],fmt="%s")
np.savetxt("ts_opt_env_obs_51.txt", ts_opt.env.obs[51],fmt="%s")
#np.savetxt("ts_env_legal.txt", ts.env.legal,fmt="%s")
np.savetxt("ts_opt_env_player_id.txt", ts_opt.env.player_id,fmt="%s")

from utils.util_test_train import string_from

#string_from(0,ts.env.obs[0],1)

_, log1 = learner.learn(ts_opt)
_, log2 = learner.learn(ts)
#_, log3 = learner.learn(ts_opt2)

log1
log2
#log3

np.array_equal(ts_opt.env.valid,ts.env.valid)
np.array_equal(ts_opt.env.obs,ts.env.obs)
np.array_equal(ts_opt.env.legal,ts.env.legal)
np.array_equal(ts_opt.env.player_id,ts.env.player_id)
np.array_equal(ts_opt.env.rewards,ts.env.rewards)
np.array_equal(ts_opt.actor.action_oh,ts.actor.action_oh)
np.array_equal(ts_opt.actor.policy,ts.actor.policy)
np.array_equal(ts_opt.actor.rewards,ts.actor.rewards)
