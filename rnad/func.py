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

# V-Trace
#
# Custom implementation of VTrace to handle trajectories having a mix of
# different player steps. The standard rlax.vtrace can't be applied here
# out of the box because a trajectory could look like '121211221122'.


def v_trace(
        v: chex.Array,
        valid: chex.Array,
        player_id: chex.Array,
        acting_policy: chex.Array,
        merged_policy: chex.Array,
        merged_log_policy: chex.Array,
        player_others: chex.Array,
        actions_oh: chex.Array,
        reward: chex.Array,
        player: int,
        # Scalars below.
        eta: float,
        lambda_: float,
        c: float,
        rho: float,
) -> Tuple[Any, Any, Any]:
    """Custom VTrace for trajectories with a mix of different player steps."""
    gamma = 1.0

    has_played = _has_played(valid, player_id, player)

    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
    inv_mu = _policy_ratio(
        jnp.ones_like(merged_policy), acting_policy, actions_oh, valid)

    eta_reg_entropy = (-eta *
                       jnp.sum(merged_policy * merged_log_policy, axis=-1) *
                       jnp.squeeze(player_others, axis=-1))
    eta_log_policy = -eta * merged_log_policy * player_others

    @chex.dataclass(frozen=True)
    class LoopVTraceCarry:
        """The carry of the v-trace scan loop."""
        reward: chex.Array
        # The cumulated reward until the end of the episode. Uncorrected (v-trace).
        # Gamma discounted and includes eta_reg_entropy.
        reward_uncorrected: chex.Array
        next_value: chex.Array
        next_v_target: chex.Array
        importance_sampling: chex.Array

    init_state_v_trace = LoopVTraceCarry(
        reward=jnp.zeros_like(reward[-1]),
        reward_uncorrected=jnp.zeros_like(reward[-1]),
        next_value=jnp.zeros_like(v[-1]),
        next_v_target=jnp.zeros_like(v[-1]),
        importance_sampling=jnp.ones_like(policy_ratio[-1]))

    def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
        (cs, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions_oh,
         eta_log_policy) = x

        reward_uncorrected = (
                reward + gamma * carry.reward_uncorrected + eta_reg_entropy)
        discounted_reward = reward + gamma * carry.reward

        # V-target:
        our_v_target = (
                v + jnp.expand_dims(
            jnp.minimum(rho, cs * carry.importance_sampling), axis=-1) *
                (jnp.expand_dims(reward_uncorrected, axis=-1) +
                 gamma * carry.next_value - v) + lambda_ * jnp.expand_dims(
            jnp.minimum(c, cs * carry.importance_sampling), axis=-1) * gamma *
                (carry.next_v_target - carry.next_value))

        opp_v_target = jnp.zeros_like(our_v_target)
        reset_v_target = jnp.zeros_like(our_v_target)

        # Learning output:
        our_learning_output = (
                v +  # value
                eta_log_policy +  # regularisation
                actions_oh * jnp.expand_dims(inv_mu, axis=-1) *
                (jnp.expand_dims(discounted_reward, axis=-1) + gamma * jnp.expand_dims(
                    carry.importance_sampling, axis=-1) * carry.next_v_target - v))

        opp_learning_output = jnp.zeros_like(our_learning_output)
        reset_learning_output = jnp.zeros_like(our_learning_output)

        # State carry:
        our_carry = LoopVTraceCarry(
            reward=jnp.zeros_like(carry.reward),
            next_value=v,
            next_v_target=our_v_target,
            reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
            importance_sampling=jnp.ones_like(carry.importance_sampling))
        opp_carry = LoopVTraceCarry(
            reward=eta_reg_entropy + cs * discounted_reward,
            reward_uncorrected=reward_uncorrected,
            next_value=gamma * carry.next_value,
            next_v_target=gamma * carry.next_v_target,
            importance_sampling=cs * carry.importance_sampling)
        reset_carry = init_state_v_trace

        # Invalid turn: init_state_v_trace and (zero target, learning_output)
        # pyformat: disable
        return _where(valid,  # pytype: disable=bad-return-type  # numpy-scalars
                      _where((player_id == player),
                             (our_carry, (our_v_target, our_learning_output)),
                             (opp_carry, (opp_v_target, opp_learning_output))),
                      (reset_carry, (reset_v_target, reset_learning_output)))
        # pyformat: enable

    _, (v_target, learning_output) = lax.scan(
        f=_loop_v_trace,
        init=init_state_v_trace,
        xs=(policy_ratio, player_id, v, reward, eta_reg_entropy, valid, inv_mu,
            actions_oh, eta_log_policy),
        reverse=True)

    return v_target, has_played, learning_output


def get_loss_v(v_list: Sequence[chex.Array],
               v_target_list: Sequence[chex.Array],
               mask_list: Sequence[chex.Array]) -> chex.Array:
    """Define the loss function for the critic."""
    chex.assert_trees_all_equal_shapes(v_list, v_target_list)
    # v_list and v_target_list come with a degenerate trailing dimension,
    # which mask_list tensors do not have.
    chex.assert_shape(mask_list, v_list[0].shape[:-1])
    loss_v_list = []
    for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
        assert v_n.shape[0] == v_target.shape[0]

        loss_v = jnp.expand_dims(
            mask, axis=-1) * (v_n - lax.stop_gradient(v_target)) ** 2
        normalization = jnp.sum(mask)
        loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0))

        loss_v_list.append(loss_v)
    return sum(loss_v_list)


def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array,
                               threshold: float,
                               threshold_center: chex.Array) -> chex.Array:
    """Apply the force with below a given threshold."""
    chex.assert_equal_shape((decision_outputs, force, threshold_center))
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = jnp.minimum(force, 0.0)
    force_positive = jnp.maximum(force, 0.0)
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * lax.stop_gradient(clipped_force)


def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
    """The `normalization` is the number of steps over which loss is computed."""
    chex.assert_equal_shape((loss, mask))
    loss = jnp.sum(loss * mask)
    normalization = jnp.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def get_loss_nerd(logit_list: Sequence[chex.Array],
                  policy_list: Sequence[chex.Array],
                  q_vr_list: Sequence[chex.Array],
                  valid: chex.Array,
                  player_ids: Sequence[chex.Array],
                  legal_actions: chex.Array,
                  importance_sampling_correction: Sequence[chex.Array],
                  clip: float = 100,
                  threshold: float = 2) -> chex.Array:
    """Define the nerd loss."""
    assert isinstance(importance_sampling_correction, list)
    loss_pi_list = []
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
            zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)):
        assert logit_pi.shape[0] == q_vr.shape[0]
        # loss policy
        adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True)
        adv_pi = is_c * adv_pi  # importance sampling correction
        adv_pi = jnp.clip(adv_pi, a_min=-clip, a_max=clip)
        adv_pi = lax.stop_gradient(adv_pi)

        logits = logit_pi - jnp.mean(
            logit_pi * legal_actions, axis=-1, keepdims=True)

        threshold_center = jnp.zeros_like(logits)

        nerd_loss = jnp.sum(
            legal_actions *
            apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
            axis=-1)
        nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
        loss_pi_list.append(nerd_loss)
    return sum(loss_pi_list)

