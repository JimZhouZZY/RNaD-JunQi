import chex
import haiku as hk
import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tree
from typing import Any, Callable, Sequence, Tuple, Union, Mapping, Optional

class PyramidModule(hk.Module):

    def __init__(self, M, N):
        super().__init__()

        self.N = N
        self.M = M

        self.conv = hk.Conv2D(
            output_channels=16,
            kernel_shape=2,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv")


    def __call__(self, input):
        out = self.conv(input)
        return out


class ValueHeadModule(hk.Module):

    def __init__(self):
        super().__init__()

        self.conv = hk.Conv2D(
            output_channels=1,
            kernel_shape=2,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv")

    def __call__(self, input):
        out = self.conv(input)

        #out = jnp.reshape(jnp.append(jnp.reshape(jnp.delete(out, 5, axis=1), (1,60)), 0), (1, 61))


        #print(jnp.shape(out))
        # out = self.linear(out)

        #print(jnp.shape(out), 1111111)

        return out


class PolicyHeadModule(hk.Module):

    def __init__(self):
        super().__init__()

        self.conv = hk.Conv2D(
            output_channels=1,
            kernel_shape=2,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv")

    def __call__(self, input):
        #out = self.pm_policy_head(input)
        out = self.conv(input)
        # out = jax.nn.softmax(out)
        #out = jnp.reshape(jnp.append(jnp.reshape(jnp.delete(out, 5, axis=1), (1,60)), 0), (1, 61))

        return out
