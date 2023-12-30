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
"""Python implementation of R-NaD (https://arxiv.org/pdf/2206.15378.pdf)."""
import random
import sys
import enum
import functools
import time
from typing import Any, Callable, Sequence, Tuple, Union, Mapping, Optional
import chex
import haiku as hk
import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tree
import ray
import tensorflow as tf

import numpy as np
import optax

from open_spiel.python import policy as policy_lib
import pyspiel
from network.cnn_paper_32 import PyramidModule, ValueHeadModule, PolicyHeadModule
#from rnad.data_class import *
#from rnad.entropy_schedule import EntropySchedule
#from rnad.func import *
from rnad.rnad import *

# Some handy aliases.
# Since most of these are just aliases for a "bag of tensors", the goal
# is to improve the documentation, and not to actually enforce correctness
# through pytype.
Params = chex.ArrayTree

def _legal_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
    """A soft-max policy that respects legal_actions."""
    try:
        chex.assert_equal_shape((logits, legal_actions))
    except AssertionError:
        logits = jnp.reshape(logits, (8))
        chex.assert_equal_shape((logits, legal_actions))
    # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
    l_min = logits.min(axis=-1, keepdims=True)
    logits = jnp.where(legal_actions, logits, l_min)
    logits -= logits.max(axis=-1, keepdims=True)
    logits *= legal_actions
    exp_logits = jnp.where(legal_actions, jnp.exp(logits),
                           0)  # Illegal actions become 0.
    exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / exp_logits_sum


def _player_others(player_ids: chex.Array, valid: chex.Array,
                   player: int) -> chex.Array:
    """A vector of 1 for the current player and -1 for others.

  Args:
    player_ids: Tensor [...] containing player ids (0 <= player_id < N).
    valid: Tensor [...] containing whether these states are valid.
    player: The player id as int.

  Returns:
    player_other: is 1 for the current player and -1 for others [..., 1].
  """
    chex.assert_equal_shape((player_ids, valid))
    current_player_tensor = (player_ids == player).astype(jnp.int32)  # pytype: disable=attribute-error  # numpy-scalars

    res = 2 * current_player_tensor - 1
    res = res * valid
    return jnp.expand_dims(res, axis=-1)


def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array,
                  valid: chex.Array) -> chex.Array:
    """Returns a ratio of policy pi/mu when selecting action a.

  By convention, this ratio is 1 on non valid states
  Args:
    pi: the policy of shape [..., A].
    mu: the sampling policy of shape [..., A].
    actions_oh: a one-hot encoding of the current actions of shape [..., A].
    valid: 0 if the state is not valid and else 1 of shape [...].

  Returns:
    pi/mu on valid states and 1 otherwise. The shape is the same
    as pi, mu or actions_oh but without the last dimension A.
  """
    chex.assert_equal_shape((pi, mu, actions_oh))
    chex.assert_shape((valid,), actions_oh.shape[:-1])

    def _select_action_prob(pi):
        return (jnp.sum(actions_oh * pi, axis=-1, keepdims=False) * valid +
                (1 - valid))

    pi_actions_prob = _select_action_prob(pi)
    mu_actions_prob = _select_action_prob(mu)
    return pi_actions_prob / mu_actions_prob


def _where(pred: chex.Array, true_data: chex.ArrayTree,
           false_data: chex.ArrayTree) -> chex.ArrayTree:
    """Similar to jax.where but treats `pred` as a broadcastable prefix."""

    def _where_one(t, f):
        chex.assert_equal_rank((t, f))
        # Expand the dimensions of pred if true_data and false_data are higher rank.
        p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
        return jnp.where(p, t, f)

    return tree.tree_map(_where_one, true_data, false_data)


def _has_played(valid: chex.Array, player_id: chex.Array,
                player: int) -> chex.Array:
    """Compute a mask of states which have a next state in the sequence."""
    chex.assert_equal_shape((valid, player_id))

    def _loop_has_played(carry, x):
        valid, player_id = x
        chex.assert_equal_shape((valid, player_id))

        our_res = jnp.ones_like(player_id)
        opp_res = carry
        reset_res = jnp.zeros_like(carry)

        our_carry = carry
        opp_carry = carry
        reset_carry = jnp.zeros_like(player_id)

        # pyformat: disable
        return _where(valid, _where((player_id == player),
                                    (our_carry, our_res),
                                    (opp_carry, opp_res)),
                      (reset_carry, reset_res))
        # pyformat: enable

    _, result = lax.scan(
        f=_loop_has_played,
        init=jnp.zeros_like(player_id[-1]),
        xs=(valid, player_id),
        reverse=True)
    return result


@ray.remote(num_cpus=1,num_gpus=0)
class RNaDActor(policy_lib.Policy):
    """Implements a solver for the R-NaD Algorithm.

  See https://arxiv.org/abs/2206.15378.

  Define all networks. Derive losses & learning steps. Initialize the game
  state and algorithmic variables.
  """

    def __init__(self, config: RNaDConfig, pit=False):
        self.config = config
        self._pit = pit

        # Learner and actor step counters.
        self.learner_steps = 0
        self.actor_steps = 0

        self.init()

    def init(self):
        """Initialize the network and losses."""
        # The random facilities for jax and numpy.
        self._rngkey = jax.random.PRNGKey(self.config.seed)
        self._np_rng = np.random.RandomState(self.config.seed)
        # TODO(author16): serialize both above to get the fully deterministic behaviour.

        # Create a game and an example of a state.
        self._game = pyspiel.load_game(self.config.game_name)
        self._ex_state = self._play_chance(self._game.new_initial_state())
        
        # The network.
        def network(env_step) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            conv_torso = PyramidModule(N=2, M=2)  # env_step.obs

            dm0 = self.config.batch_size if env_step.obs.shape != (1*8*28,) else 1
            shape = (dm0, 4, 2, 28)
            reshaped_obs = jnp.reshape(env_step.obs, shape)

            torso = conv_torso(reshaped_obs)

            pm_value_head = ValueHeadModule()
            pm_policy_head = PolicyHeadModule()

            logit = pm_policy_head(torso)
            logit = jnp.reshape(logit, (dm0, 8))
            v = pm_value_head(torso)
            v = jnp.reshape(v, (dm0, 8))

            pi = _legal_policy(logit, env_step.legal)
            log_pi = legal_log_policy(logit, env_step.legal)
            return pi, v, log_pi, logit
        
        self.network = hk.without_apply_rng(hk.transform(network))

        # The machinery related to updating parameters/learner.
        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats)
        self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=False)

        # Create initial parameters.
        env_step = self._state_as_env_step(self._ex_state)
        key = self._next_rng_key()  # Make sure to use the same key for all.
        self.params = self.network.init(key, env_step)
        self.params_target = self.network.init(key, env_step)
        self.params_prev = self.network.init(key, env_step)
        self.params_prev_ = self.network.init(key, env_step)

        # Parameter optimizers.
        self.optimizer = optax_optimizer(
            self.params,
            optax.chain(
                optax.scale_by_adam(
                    eps_root=0.0,
                    **self.config.adam,
                ), optax.scale(-self.config.learning_rate),
                optax.clip(self.config.clip_gradient)))
        self.optimizer_target = optax_optimizer(
            self.params_target, optax.sgd(self.config.target_network_avg))

    def loss(self, params: Params, params_target: Params, params_prev: Params,
             params_prev_: Params, ts: TimeStep, alpha: float,
             learner_steps: int) -> float:
        rollout = jax.vmap(self.network.apply, (None, 0), 0)
        pi, v, log_pi, logit = rollout(params, ts.env)

        policy_pprocessed = self.config.finetune(pi, ts.env.legal, learner_steps)

        _, v_target, _, _ = rollout(params_target, ts.env)
        _, _, log_pi_prev, _ = rollout(params_prev, ts.env)
        _, _, log_pi_prev_, _ = rollout(params_prev_, ts.env)
        # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
        # For the stability reasons, reward changes smoothly between iterations.
        # The mixing between old and new reward transform is a convex combination
        # parametrised by alpha.
        log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        for player in range(self._game.num_players()):
            reward = ts.actor.rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = v_trace(
                v_target,
                ts.env.valid,
                ts.env.player_id,
                ts.actor.policy,
                policy_pprocessed,
                log_policy_reg,
                _player_others(ts.env.player_id, ts.env.valid, player),
                ts.actor.action_oh,
                reward,
                player,
                lambda_=1.0,
                c=self.config.c_vtrace,
                rho=np.inf,
                eta=self.config.eta_reward_transform)
            v_target_list.append(v_target_)
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(policy_target_)
        loss_v = get_loss_v([v] * self._game.num_players(), v_target_list,
                            has_played_list)

        is_vector = jnp.expand_dims(jnp.ones_like(ts.env.valid), axis=-1)
        importance_sampling_correction = [is_vector] * self._game.num_players()
        # Uses v-trace to define q-values for Nerd
        loss_nerd = get_loss_nerd(
            [logit] * self._game.num_players(), [pi] * self._game.num_players(),
            v_trace_policy_target_list,
            ts.env.valid,
            ts.env.player_id,
            ts.env.legal,
            importance_sampling_correction,
            clip=self.config.nerd.clip,
            threshold=self.config.nerd.beta)
        return loss_v + loss_nerd  # pytype: disable=bad-return-type  # numpy-scalars

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_parameters(
          self,
          params: Params,
          params_target: Params,
          params_prev: Params,
          params_prev_: Params,
          optimizer: Optimizer,
          optimizer_target: Optimizer,
          timestep: TimeStep,
          alpha: float,
          learner_steps: int,
          update_target_net: bool):
        """A jitted pure-functional part of the `step`."""
        loss_val, grad = self._loss_and_grad(params, params_target, params_prev,
                                             params_prev_, timestep, alpha,
                                             learner_steps)
        # Update `params`` using the computed gradient.
        params = optimizer(params, grad)
        # Update `params_target` towards `params`.
        params_target = optimizer_target(
            params_target, tree.tree_map(lambda a, b: a - b, params_target, params))

        # Rolls forward the prev and prev_ params if update_target_net is 1.
        # pyformat: disable
        params_prev, params_prev_ = jax.lax.cond(
            update_target_net,
            lambda: (params_target, params_prev),
            lambda: (params_prev, params_prev_))
        # pyformat: enable

        logs = {
            "loss": loss_val,
        }
        return (params, params_target, params_prev, params_prev_, optimizer,
                optimizer_target), logs

    def __getstate__(self):
        """To serialize the agent."""
        return dict(
            # RNaD config.
            config=self.config,

            # Learner and actor step counters.
            learner_steps=self.learner_steps,
            actor_steps=self.actor_steps,

            # The randomness keys.
            np_rng=self._np_rng.get_state(),
            rngkey=self._rngkey,

            # Network params.
            params=self.params,
            params_target=self.params_target,
            params_prev=self.params_prev,
            params_prev_=self.params_prev_,
            # Optimizer state.
            optimizer=self.optimizer.state,  # pytype: disable=attribute-error  # always-use-return-annotations
            optimizer_target=self.optimizer_target.state,
            # pytype: disable=attribute-error  # always-use-return-annotations
        )

    def __setstate__(self, state):
        """To deserialize the agent."""
        # RNaD config.
        self.config = state["config"]

        self.init()

        # Learner and actor step counters.
        self.learner_steps = state["learner_steps"]
        self.actor_steps = state["actor_steps"]

        # The randomness keys.
        self._np_rng.set_state(state["np_rng"])
        self._rngkey = state["rngkey"]

        # Network params.
        self.params = state["params"]
        self.params_target = state["params_target"]
        self.params_prev = state["params_prev"]
        self.params_prev_ = state["params_prev_"]
        # Optimizer state.
        self.optimizer.state = state["optimizer"]
        self.optimizer_target.state = state["optimizer_target"]

    def split(self, arr):
        """Splits the first axis of `arr` evenly across the number of devices."""
        return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

    def step(self):
        """One step of the algorithm, that plays the game and improves params."""
        timestep = self.collect_batch_trajectory()
        alpha, update_target_net = self._entropy_schedule(self.learner_steps)
        (self.params, self.params_target, self.params_prev, self.params_prev_,
         self.optimizer, self.optimizer_target), logs = self.update_parameters(
             self.params, self.params_target, self.params_prev, self.params_prev_,
             self.optimizer, self.optimizer_target, timestep, alpha,
             self.learner_steps, update_target_net)
        self.learner_steps += 1
        logs.update({
            "actor_steps": self.actor_steps,
            "learner_steps": self.learner_steps,
        })
        return logs

    def act(self):
        return self.collect_batch_trajectory()

    def _next_rng_key(self) -> chex.PRNGKey:
        """Get the next rng subkey from class rngkey.

        Must *not* be called from under a jitted function!

        Returns:
          A fresh rng_key.
        """
        self._rngkey, subkey = jax.random.split(self._rngkey)
        return subkey

    def _state_as_env_step(self, state: pyspiel.State) -> EnvStep:
        # A terminal state must be communicated to players, however since
        # it's a terminal state things like the state_representation or
        # the set of legal actions are meaningless and only needed
        # for the sake of creating well a defined trajectory tensor.
        # Therefore the code below:
        # - extracts the rewards
        # - if the state is terminal, uses a dummy other state for other fields.
        rewards = np.array(state.returns(), dtype=np.float64)

        valid = not state.is_terminal()
        if not valid:
            state = self._ex_state

        if self.config.state_representation == StateRepresentation.OBSERVATION:
            obs = state.observation_tensor()
        elif self.config.state_representation == StateRepresentation.INFO_SET:
            obs = state.information_state_tensor()
        else:
            raise ValueError(
                f"Invalid StateRepresentation: {self.config.state_representation}.")

        # TODO(author16): clarify the story around rewards and valid.
        return EnvStep(
            obs=np.array(obs, dtype=np.float64),
            legal=np.array(state.legal_actions_mask(), dtype=np.int8),
            player_id=np.array(state.current_player(), dtype=np.float64),
            valid=np.array(valid, dtype=np.float64),
            rewards=rewards)

    def action_probabilities(self,
                             state: pyspiel.State,
                             player_id: Any = None):
        """Returns action probabilities dict for a single batch."""
        env_step = self._batch_of_states_as_env_step([state])
        probs = self._network_jit_apply_and_post_process(
            self.params_target, env_step)
        probs = jax.device_get(probs[0])  # Squeeze out the 1-element batch.
        return {
            action: probs[action]
            for action, valid in enumerate(jax.device_get(env_step.legal[0]))
            if valid
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply_and_post_process(
            self, params: Params, env_step: EnvStep) -> chex.Array:
        pi, _, _, _ = self.network.apply(params, env_step)
        pi = self.config.finetune.post_process_policy(pi, env_step.legal)
        return pi

    @functools.partial(jax.jit, static_argnums=(0,))
    def _network_jit_apply(
            self, params: Params, env_step: EnvStep) -> chex.Array:
        pi, _, _, _ = self.network.apply(params, env_step)
        return pi

    def actor_step(self, env_step: EnvStep):
        pi = self._network_jit_apply(self.params, env_step)
        pi = np.asarray(pi).astype("float64")
        # TODO(author18): is this policy normalization really needed?
        pi = pi / np.sum(pi, axis=-1, keepdims=True)

        action = np.apply_along_axis(
            lambda x: self._np_rng.choice(range(pi.shape[1]), p=x), axis=-1, arr=pi)
        # TODO(author16): reapply the legal actions mask to bullet-proof sampling.
        action_oh = np.zeros(pi.shape, dtype="float64")
        action_oh[range(pi.shape[0]), action] = 1.0

        actor_step_list = [pi, action_oh] # [B, A] # [[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]]

        return action, actor_step_list

    def pit(self):
        self.config = RNaDConfig(
            game_name='junqi1',
            batch_size=1,
        )
        human_player = 0
        while True:
            states = [
                self._play_chance(self._game.new_initial_state())
                for _ in range(1)
            ]
            env_step = self._batch_of_states_as_env_step(states)
            state = states[0]
            # env_step.legal = np.squeeze(env_step.legal)
            print(env_step)
            for _ in range(self.config.trajectory_max):
                a, actor_step = self.actor_step(env_step)
                if human_player == state.current_player():
                    print("=========================")
                    print(self.get_stringed_board(state, human_player))
                    print("=========================")
                    print(f"Result from RNaD: {a}")
                    a = [command_line_action(state, a) for _ in range(1)]
                states = self._batch_of_states_apply_action(states, a)
                env_step = self._batch_of_states_as_env_step(states)
            print(state.game_length)
            print(state.game_length_real)
            human_player = 1 - human_player

    def pit_random(self):
        self.config = RNaDConfig(
            game_name='junqi1',
            batch_size=1,
        )
        random_player = 0
        reward = [0, 0]
        num = 0
        wins = 0
        loses = 0
        peaces = 0
        while True:
            start_time = time.perf_counter()
            states = [
                self._play_chance(self._game.new_initial_state())
                for _ in range(1)
            ]
            env_step = self._batch_of_states_as_env_step(states)
            state = states[0]
            while not state.is_terminal():
                if state.current_player() == random_player:
                    actions = state.legal_actions()
                    idx = random.randint(0, len(actions) - 1)
                    a = [actions[idx]]
                else:
                    a, actor_step = self.actor_step(env_step)
                states = self._batch_of_states_apply_action(states, a)
                env_step = self._batch_of_states_as_env_step(states)
            reward[0] += state.returns()[1 - random_player]
            reward[1] += state.returns()[random_player]
            if state.returns()[1 - random_player] == 1:
                wins += 1
            elif state.returns()[random_player] == 1:
                loses += 1
            else:
                peaces += 1
            random_player = 1 - random_player
            num += 1
            end_time = time.perf_counter()
            t = end_time - start_time
            print(f"Number: {num}, Reward: {[state.returns()[1 - random_player], state.returns()[random_player]]}, in {t} seconds.")
            print(f"Score: {reward}, WinRate: {wins / num}, LoseRate: {loses / num}, PeaceRate: {peaces / num}")

    def collect_batch_trajectory(self):
        states = [
            self._play_chance(self._game.new_initial_state())
            for _ in range(self.config.batch_size)
        ]
        timesteps = []

        env_step, env_step_list = self._batch_of_states_as_env_step(states)
        for _ in range(self.config.trajectory_max):
            prev_env_step_list = env_step_list
            a, actor_step_list = self.actor_step(env_step)

            states = self._batch_of_states_apply_action(states, a)
            env_step, env_step_list = self._batch_of_states_as_env_step(states)
            timesteps.append([prev_env_step_list, actor_step_list, env_step_list])
        # Concatenate all the timesteps together to form a single rollout [T, B, ..]
        # return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *timesteps)
        return timesteps

    def _batch_of_states_as_env_step(self,
                                     states: Sequence[pyspiel.State]) -> EnvStep:
        envs = [self._state_as_env_step(state) for state in states]
        return jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *envs), envs

    def _batch_of_states_apply_action(
            self, states: Sequence[pyspiel.State],
            actions: chex.Array) -> Sequence[pyspiel.State]:
        """Apply a batch of `actions` to a parallel list of `states`."""
        for state, action in zip(states, list(actions)):
            if not state.is_terminal():
                self.actor_steps += 1
                state.apply_action(action)
                self._play_chance(state)
        return states

    def _play_chance(self, state: pyspiel.State) -> pyspiel.State:
        """Plays the chance nodes until we end up at another type of node.

    Args:
      state: to be updated until it does not correspond to a chance node.
    Returns:
      The same input state object, but updated. The state is returned
      only for convenience, to allow chaining function calls.
    """
        while state.is_chance_node():
            chance_outcome, chance_proba = zip(*state.chance_outcomes())
            action = self._np_rng.choice(chance_outcome, p=chance_proba)
            state.apply_action(action)
        return state

    @staticmethod
    def get_stringed_board(state, human_player=0):
        return state.serialize_pov(human_player)

    def get_stringed_action(self):
        actions = self._ex_state.legal_actions()
        return "\n".join(self._ex_state.serialize_action(action) for action in actions)

