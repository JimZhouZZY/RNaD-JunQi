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

@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""
    b1: float = 0.0
    b2: float = 0.999
    eps: float = 10e-8


@chex.dataclass(frozen=True)
class NerdConfig:
    """Nerd related params."""
    beta: float = 2.0
    clip: float = 10_000


class StateRepresentation(str, enum.Enum):
    INFO_SET = "info_set"
    OBSERVATION = "observation"


@chex.dataclass(frozen=True)
class EnvStep:
    """Holds the tensor data representing the current game state."""
    # Indicates whether the state is a valid one or just a padding. Shape: [...]
    # The terminal state being the first one to be marked !valid.
    # All other tensors in EnvStep contain data, but only for valid timesteps.
    # Once !valid the data needs to be ignored, since it's a duplicate of
    # some other previous state.
    # The rewards is the only exception that contains reward values
    # in the terminal state, which is marked !valid.
    # TODO(author16): This is a confusion point and would need to be clarified.
    valid: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    # The single tensor representing the state observation. Shape: [..., ??]
    obs: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    # The legal actions mask for the current player. Shape: [..., A]
    legal: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    # The current player id as an int. Shape: [...]
    player_id: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    # The rewards of all the players. Shape: [..., P]
    rewards: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@chex.dataclass(frozen=True)
class ActorStep:
    """The actor step tensor summary."""
    # The action (as one-hot) of the current player. Shape: [..., A]
    action_oh: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    # The policy of the current player. Shape: [..., A]
    policy: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    # The rewards of all the players. Shape: [..., P]
    # Note - these are rewards obtained *after* the actor step, and thus
    # these are the same as EnvStep.rewards visible before the *next* step.
    rewards: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@chex.dataclass(frozen=True)
class TimeStep:
    """The tensor data for one game transition (env_step, actor_step)."""
    env: EnvStep = EnvStep()
    actor: ActorStep = ActorStep()


@chex.dataclass(frozen=True)
class FineTuning:
    """Fine tuning options, aka policy post-processing.

  Even when fully trained, the resulting softmax-based policy may put
  a small probability mass on bad actions. This results in an agent
  waiting for the opponent (itself in self-play) to commit an error.

  To address that the policy is post-processed using:
  - thresholding: any action with probability smaller than self.threshold
    is simply removed from the policy.
  - discretization: the probability values are rounded to the closest
    multiple of 1/self.discretization.

  The post-processing is used on the learner, and thus must be jit-friendly.
  """
    # The learner step after which the policy post processing (aka finetuning)
    # will be enabled when learning. A strictly negative value is equivalent
    # to infinity, ie disables finetuning completely.
    from_learner_steps: int = -1
    # All policy probabilities below `threshold` are zeroed out. Thresholding
    # is disabled if this value is non-positive.
    policy_threshold: float = 0.03
    # Rounds the policy probabilities to the "closest"
    # multiple of 1/`self.discretization`.
    # Discretization is disabled for non-positive values.
    policy_discretization: int = 32

    def __call__(self, policy: chex.Array, mask: chex.Array,
                 learner_steps: int) -> chex.Array:
        """A configurable fine tuning of a policy."""
        chex.assert_equal_shape((policy, mask))
        do_finetune = jnp.logical_and(self.from_learner_steps >= 0,
                                      learner_steps > self.from_learner_steps)

        return jnp.where(do_finetune, self.post_process_policy(policy, mask),
                         policy)

    def post_process_policy(
            self,
            policy: chex.Array,
            mask: chex.Array,
    ) -> chex.Array:
        """Unconditionally post process a given masked policy."""
        chex.assert_equal_shape((policy, mask))
        policy = self._threshold(policy, mask)
        policy = self._discretize(policy)
        return policy

    def _threshold(self, policy: chex.Array, mask: chex.Array) -> chex.Array:
        """Remove from the support the actions 'a' where policy(a) < threshold."""
        chex.assert_equal_shape((policy, mask))
        if self.policy_threshold <= 0:
            return policy

        mask = mask * (
            # Values over the threshold.
                (policy >= self.policy_threshold) +
                # Degenerate case is when policy is less than threshold *everywhere*.
                # In that case we just keep the policy as-is.
                (jnp.max(policy, axis=-1, keepdims=True) < self.policy_threshold))
        return mask * policy / jnp.sum(mask * policy, axis=-1, keepdims=True)

    def _discretize(self, policy: chex.Array) -> chex.Array:
        """Round all action probabilities to a multiple of 1/self.discretize."""
        if self.policy_discretization <= 0:
            return policy

        # The unbatched/single policy case:
        if len(policy.shape) == 1:
            return self._discretize_single(policy)

        # policy may be [B, A] or [T, B, A], etc. Thus add hk.BatchApply.
        dims = len(policy.shape) - 1

        # TODO(author18): avoid mixing vmap and BatchApply since the two could
        # be folded into either a single BatchApply or a sequence of vmaps, but
        # not the mix.
        vmapped = jax.vmap(self._discretize_single)
        policy = hk.BatchApply(vmapped, num_dims=dims)(policy)

        return policy

    def _discretize_single(self, mu: chex.Array) -> chex.Array:
        """A version of self._discretize but for the unbatched data."""
        # TODO(author18): try to merge _discretize and _discretize_single
        # into one function that handles both batched and unbatched cases.
        if len(mu.shape) == 2:
            mu_ = jnp.squeeze(mu, axis=0)
        else:
            mu_ = mu
        n_actions = mu_.shape[-1]
        roundup = jnp.ceil(mu_ * self.policy_discretization).astype(jnp.int32)
        result = jnp.zeros_like(mu_)
        order = jnp.argsort(-mu_)  # Indices of descending order.
        weight_left = self.policy_discretization

        def f_disc(i, order, roundup, weight_left, result):
            x = jnp.minimum(roundup[order[i]], weight_left)
            result = jax.numpy.where(weight_left >= 0, result.at[order[i]].add(x),
                                     result)
            weight_left -= x
            return i + 1, order, roundup, weight_left, result

        def f_scan_scan(carry, x):
            i, order, roundup, weight_left, result = carry
            i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
                i, order, roundup, weight_left, result)
            carry_next = (i_next, order_next, roundup_next, weight_left_next,
                          result_next)
            return carry_next, x

        (_, _, _, weight_left_next, result_next), _ = jax.lax.scan(
            f_scan_scan,
            init=(jnp.asarray(0), order, roundup, weight_left, result),
            xs=None,
            length=n_actions)

        result_next = jnp.where(weight_left_next > 0,
                                result_next.at[order[0]].add(weight_left_next),
                                result_next)
        if len(mu.shape) == 2:
            result_next = jnp.expand_dims(result_next, axis=0)
        return result_next / self.policy_discretization
    

@chex.dataclass(frozen=True)
class RNaDConfig:
    """Configuration parameters for the RNaDSolver."""
    # The game parameter string including its name and parameters.
    game_name: str
    # The games longer than this value are truncated. Must be strictly positive.
    trajectory_max: int = 500

    # The content of the EnvStep.obs tensor.
    state_representation: StateRepresentation = StateRepresentation.INFO_SET

    # Network configuration.
    policy_network_layers: Sequence[int] = (256, 256)

    # The batch size to use when learning/improving parameters.
    batch_size: int = 256  # 256
    # The learning rate for `params`.
    learning_rate: float = 0.00005
    # The config related to the ADAM optimizer used for updating `params`.
    adam: AdamConfig = AdamConfig()
    # All gradients values are clipped to [-clip_gradient, clip_gradient].
    clip_gradient: float = 10_000
    # The "speed" at which `params_target` is following `params`.
    target_network_avg: float = 0.001

    # RNaD algorithm configuration.
    # Entropy schedule configuration. See EntropySchedule class documentation.
    entropy_schedule_repeats: Sequence[int] = (1,)
    entropy_schedule_size: Sequence[int] = (20_000,)
    # The weight of the reward regularisation term in RNaD.
    eta_reward_transform: float = 0.8  # 0.2
    nerd: NerdConfig = NerdConfig()
    c_vtrace: float = 1.0

    # Options related to fine tuning of the agent.
    finetune: FineTuning = FineTuning()

    # The seed that fully controls the randomness.
    seed: int = 42
