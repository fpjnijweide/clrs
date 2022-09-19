from __future__ import division
from __future__ import print_function

import collections

# import sonnet as hk
# import tensorflow as tf
from typing import NamedTuple

import haiku as hk
import jax.numpy as jnp
from jax import nn

from clrs._src.memory_models.dnc import addressing
from clrs._src.memory_models.dnc import util

AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))

class AccessState(NamedTuple):
    memory: jnp.array
    read_weights: jnp.array
    write_weights: jnp.array
    linkage: jnp.array
    usage: jnp.array


def _erase_and_write(memory, address, reset_weights, values):

    return memory


class DNC(hk.RNNCore):
class DNCMemoryAccess(hk.RNNCore):
    """Access module of the Differentiable Neural Computer.

    This memory module supports multiple read and write heads. It makes use of:

          num_writes: The number of write heads (fixed at 1 in the paper).
          name: The name of the module.
        """
        super(DNC, self).__init__(name=name)
        super(DNCMemoryAccess, self).__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads

    def initial_state(self, node_fts):
        batch_size, n, h = node_fts.shape

        read_vector_list = []
        for i in range(self.read_head_num):
            new_read_vector = jnp.tanh(hk.get_parameter(f"NTM_read_vector_{i}", shape=[self.memory_vector_dim, ],
                                                        init=hk.initializers.VarianceScaling(1.0, "fan_avg",
                                                                                             "uniform")))
            read_vector_list.append(new_read_vector)

        w_var_list = []
        for i in range(self.read_head_num + self.write_head_num):
            new_w_var = nn.softmax(hk.get_parameter(f"NTM_w_{i}", shape=[self.memory_size, ],
                                                    init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")))
            w_var_list.append(new_w_var)
        memory: jnp.array
        read_weights: jnp.array
        write_weights: jnp.array
        linkage: jnp.array
        usage: jnp.array

        if self.init_mode == 'learned':
            M = jnp.tanh(hk.get_parameter("NTM_M", shape=[self.memory_size, self.memory_vector_dim],
                                          init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")))
        elif self.init_mode == 'random':
            M = jnp.tanh(
                hk.initializers.RandomNormal(stddev=0.5)([self.memory_size, self.memory_vector_dim], dtype=jnp.float32))
        elif self.init_mode == 'constant':
            M = jnp.full([self.memory_size, self.memory_vector_dim], 1e-6)
        else:
            raise RuntimeError("invalid init mode")

        state = NTMState(M, w_var_list, read_vector_list)
        if batch_size is not None:
            state = add_batch(state, batch_size)
        state = AccessState(memory, read_weights,write_weights,linkage,usage)
        return state

    def __call__(self, inputs, prev_state: AccessState):


        return read_weights

    # @property
    # def state_size(self):
    #     """Returns a tuple of the shape of the state tensors."""
    #     return AccessState(
    #         memory=jnp.TensorShape([self._memory_size, self._word_size]),
    #         read_weights=jnp.TensorShape([self._num_reads, self._memory_size]),
    #         write_weights=jnp.TensorShape([self._num_writes, self._memory_size]),
    #         linkage=self._linkage.state_size,
    #         usage=self._freeness.state_size)
    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        return AccessState(
            memory=(self._memory_size, self._word_size),
            read_weights=(self._num_reads, self._memory_size),
            write_weights=(self._num_writes, self._memory_size),
            linkage=self._linkage.state_size,
            usage=self._freeness.state_size)

    # @property
    # def output_size(self):
    #     """Returns the output shape."""
    #     return jnp.TensorShape([self._num_reads, self._word_size])
    @property
    def output_size(self):
        """Returns the output shape."""
        return (self._num_reads, self._word_size)
