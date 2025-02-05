# Copyright 2017 Google Inc.
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
# ==============================================================================
"""DNC access modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# import sonnet as hk
# import tensorflow as tf
import haiku as hk
import jax.numpy as jnp
from jax import nn

from clrs._src.memory_models.dnc import addressing
from clrs._src.memory_models.dnc import util

AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))


def _erase_and_write(memory, address, reset_weights, values):
    """Module to erase and write in the external memory.

    Erase operation:
      M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

    Add operation:
      M_t(i) = M_t'(i) + w_t(i) * a_t

    where e are the reset_weights, w the write weights and a the values.

    Args:
      memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
      address: 3-D tensor `[batch_size, num_writes, memory_size]`.
      reset_weights: 3-D tensor `[batch_size, num_writes, word_size]`.
      values: 3-D tensor `[batch_size, num_writes, word_size]`.

    Returns:
      3-D tensor of shape `[batch_size, num_writes, word_size]`.
    """
    expand_address = jnp.expand_dims(address, 3)
    reset_weights = jnp.expand_dims(reset_weights, 2)
    weighted_resets = expand_address * reset_weights
    reset_gate = jnp.prod(1 - weighted_resets, 1)
    memory *= reset_gate

    new_addr = jnp.transpose(jnp.conj(address),axes=[0,2,1])
    add_matrix = jnp.matmul(new_addr, values)
    memory += add_matrix

    return memory


class DNCAccessModule(hk.RNNCore):
    """Access module of the Differentiable Neural Computer.

    This memory module supports multiple read and write heads. It makes use of:

    *   `addressing.TemporalLinkage` to track the temporal ordering of writes in
        memory for each write head.
    *   `addressing.FreenessAllocator` for keeping track of memory usage, where
        usage increase when a memory location is written to, and decreases when
        memory is read from that the controller says can be freed.

    Write-address selection is done by an interpolation between content-based
    lookup and using unused memory.

    Read-address selection is done by an interpolation of content-based lookup
    and following the link graph in the forward or backwards read direction.
    """

    def __init__(self,
                 memory_size,
                 word_size,
                 num_reads=1,
                 num_writes=1,
                 name='memory_access'):
        """Creates a MemoryAccess module.

        Args:
          memory_size: The number of memory slots (N in the DNC paper).
          word_size: The width of each memory slot (W in the DNC paper)
          num_reads: The number of read heads (R in the DNC paper).
          num_writes: The number of write heads (fixed at 1 in the paper).
          name: The name of the module.
        """
        super(DNCAccessModule, self).__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes

        self._write_content_weights_mod = addressing.CosineWeights(
            num_writes, word_size, name='write_content_weights')
        self._read_content_weights_mod = addressing.CosineWeights(
            num_reads, word_size, name='read_content_weights')

        self._linkage = addressing.TemporalLinkage(memory_size, num_writes)
        self._freeness = addressing.Freeness(memory_size)

        self.read_nodes_amount = self._num_reads
        self.write_nodes_amount = 4*self._num_writes + 2*self._num_reads

        self.write_ints_dense_layers = [hk.Linear(output_size=3)] * self._num_writes
        self.read_ints_dense_layers = [hk.Linear(output_size=2 + (1 + 2 * self._num_writes))] * self._num_reads



    def initial_state(self, batch_size):
        return AccessState(memory=jnp.zeros([batch_size,*self.state_size.memory]),
                           read_weights=jnp.ones([batch_size,*self.state_size.read_weights])*1e-6,
                           write_weights=jnp.ones([batch_size,*self.state_size.write_weights])*1e-6,
                           linkage=self._linkage.initial_state(batch_size),
                           usage=jnp.ones([batch_size,self.state_size.usage])*1e-6)

    def __call__(self, inputs, prev_state: AccessState):
        """Connects the MemoryAccess module into the graph.

        Args:
          inputs: tensor of shape `[batch_size, input_size]`. This is used to
              control this access module.
          prev_state: Instance of `AccessState` containing the previous state.

        Returns:
          A tuple `(output, next_state)`, where `output` is a tensor of shape
          `[batch_size, num_reads, word_size]`, and `next_state` is the new
          `AccessState` named tuple at the current time t.
        """
        # Update usage using inputs['free_gate'] and previous read & write weights.
        usage = self._freeness(
            write_weights=prev_state.write_weights,
            free_gate=inputs['free_gate'],
            read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage)

        # Write to memory.
        write_weights = self._write_weights(inputs, prev_state.memory, usage)
        memory = _erase_and_write(
            prev_state.memory,
            address=write_weights,
            reset_weights=inputs['erase_vectors'],
            values=inputs['write_vectors'])

        _, linkage_state = self._linkage(write_weights, prev_state.linkage)

        # Read from memory.
        read_weights = self._read_weights(
            inputs,
            memory=memory,
            prev_read_weights=prev_state.read_weights,
            link=linkage_state.link)
        read_words = jnp.matmul(read_weights, memory)

        return (read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage))

    def prepare_memory_input(self, concatenated_ret, n):
        """Applies transformations to `inputs` to get control for this module."""
        #
        # def _linear(first_dim, second_dim, name, activation=None):
        #     """Returns a linear transformation of `inputs`, followed by a reshape."""
        #     linear = hk.Linear(first_dim * second_dim, name=name)(inputs)
        #     if activation is not None:
        #         linear = activation(linear, name=name + '_activation')
        #     return jnp.reshape(linear, [-1, first_dim, second_dim])

        # TODO maybe squeeze. check dimensions


        # 1 read node, for each read node

        # write nodes
        #
        # read_keys 1 read
        # free_gate int read
        # read_strengths int read
        # read_mode 1 CUSTOM read
        # 2 write nodes per read head

        # allocation_gate int write
        # write_gate int write
        # write_strengths int write
        # write_keys 1 write
        # write_vectors, erase_vectors = 2* write
        # 4 write nodes per write head

        real_ret, _, write_ret = jnp.split(concatenated_ret, [n, n + self.read_nodes_amount], axis=1)
        total_nodes_for_read_heads,total_nodes_for_write_heads = jnp.split(write_ret, [self._num_reads*2], axis=1)

        list_of_nodes_for_each_read_head = jnp.split(total_nodes_for_read_heads, self._num_reads, axis=1)
        list_of_nodes_for_each_write_head = jnp.split(total_nodes_for_write_heads, self._num_writes, axis=1)

        erase_vectors = []
        write_vectors = []
        write_keys=[]
        allocation_gates = []
        write_gates = []
        write_strengths = []
        for write_head_index, write_nodes_for_head in enumerate(list_of_nodes_for_each_write_head):
            write_ints_pre_dense,write_key,write_vector,erase_vector = jnp.split(write_nodes_for_head, 4, axis=1)
            erase_vector= nn.sigmoid(erase_vector)
            post_dense_layer = self.write_ints_dense_layers[write_head_index](write_ints_pre_dense)
            allocation_gate,write_gate,write_strength = jnp.split(post_dense_layer, 3, axis=2)
            allocation_gate = nn.sigmoid(allocation_gate)
            write_gate = nn.sigmoid(write_gate)

            erase_vectors.append(erase_vector.squeeze())
            write_vectors.append(write_vector.squeeze())
            write_keys.append(write_key.squeeze())
            allocation_gates.append(allocation_gate.squeeze())
            write_gates.append(write_gate.squeeze())
            write_strengths.append(write_strength.squeeze())

        erase_vectors = jnp.stack(erase_vectors,axis=1)
        write_vectors = jnp.stack(write_vectors,axis=1)
        write_keys = jnp.stack(write_keys,axis=1)
        allocation_gates = jnp.stack(allocation_gates,axis=1)
        write_gates = jnp.stack(write_gates,axis=1)
        write_strengths = jnp.stack(write_strengths,axis=1)




        read_keys = []
        free_gates = []
        read_strengths = []
        read_modes = []
        for read_head_index, read_nodes_for_head in enumerate(list_of_nodes_for_each_read_head):
            read_key, read_ints_pre_dense = jnp.split(read_nodes_for_head, 2, axis=1)
            read_ints_post_dense = self.read_ints_dense_layers[read_head_index](read_ints_pre_dense)
            free_gate,read_strength,read_mode = jnp.split(read_ints_post_dense, [1,2], axis=2)
            # read_mode = self.read_mode_dense_layers[read_head_index](read_mode_pre_dense)
            free_gate = nn.sigmoid(free_gate)
            read_mode = nn.softmax(read_mode)

            read_keys.append(read_key.squeeze())
            free_gates.append(free_gate.squeeze())
            read_strengths.append(read_strength.squeeze())
            read_modes.append(read_mode.squeeze())

        read_keys = jnp.stack(read_keys,axis=1)
        free_gates = jnp.stack(free_gates,axis=1)
        read_strengths = jnp.stack(read_strengths,axis=1)
        read_modes = jnp.stack(read_modes,axis=1)
        # # v_t^i - The vectors to write to memory, for each write head `i`.
        # write_vectors = _linear(self._num_writes, self._word_size, 'write_vectors')
        #
        # # e_t^i - Amount to erase the memory by before writing, for each write head.
        # erase_vectors = _linear(self._num_writes, self._word_size, 'erase_vectors',
        #                         nn.sigmoid)
        #
        # # f_t^j - Amount that the memory at the locations read from at the previous
        # # time step can be declared unused, for each read head `j`.
        # free_gate = nn.sigmoid(
        #     hk.Linear(self._num_reads, name='free_gate')(inputs))
        #
        # # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # # identify this gate with allocation vs writing (as defined below).
        # allocation_gate = nn.sigmoid(
        #     hk.Linear(self._num_writes, name='allocation_gate')(inputs))
        #
        # # g_t^{w, i} - Overall gating of write amount for each write head.
        # write_gate = nn.sigmoid(
        #     hk.Linear(self._num_writes, name='write_gate')(inputs))
        #
        # # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # # each write head), and content-based lookup, for each read head.
        # num_read_modes = 1 + 2 * self._num_writes
        # read_mode = nn.softmax(
        #     _linear(self._num_reads, num_read_modes, name='read_mode'))
        #
        # # Parameters for the (read / write) "weights by content matching" modules.
        # write_keys = _linear(self._num_writes, self._word_size, 'write_keys')
        # write_strengths = hk.Linear(self._num_writes, name='write_strengths')(
        #     inputs)
        #
        # read_keys = _linear(self._num_reads, self._word_size, 'read_keys')
        # read_strengths = hk.Linear(self._num_reads, name='read_strengths')(inputs)

        result = {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gate': free_gates,
            'allocation_gate': allocation_gates,
            'write_gate': write_gates,
            'read_mode': read_modes,
        }
        return result,real_ret

    def _write_weights(self, inputs, memory, usage):
        """Calculates the memory locations to write to.

        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.

        Args:
          inputs: Collection of inputs to the access module, including controls for
              how to chose memory writing, such as the content to look-up and the
              weighting between content-based and allocation-based addressing.
          memory: A tensor of shape  `[batch_size, memory_size, word_size]`
              containing the current memory contents.
          usage: Current memory usage, which is a tensor of shape `[batch_size,
              memory_size]`, used for allocation-based addressing.

        Returns:
          tensor of shape `[batch_size, num_writes, memory_size]` indicating where
              to write to (if anywhere) for each write head.
        """
        # c_t^{w, i} - The content-based weights for each write head.
        write_content_weights = self._write_content_weights_mod(
            memory, inputs['write_content_keys'],
            inputs['write_content_strengths'])

        # a_t^i - The allocation weights for each write head.
        write_allocation_weights = self._freeness.write_allocation_weights(
            usage=usage,
            write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
            num_writes=self._num_writes)

        # Expands gates over memory locations.
        allocation_gate = jnp.expand_dims(inputs['allocation_gate'], -1)
        write_gate = jnp.expand_dims(inputs['write_gate'], -1)

        # w_t^{w, i} - The write weightings for each write head.
        return write_gate * (allocation_gate * write_allocation_weights +
                             (1 - allocation_gate) * write_content_weights)

    def _read_weights(self, inputs, memory, prev_read_weights, link):
        """Calculates read weights for each read head.

        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.

        Args:
          inputs: Controls for this access module. This contains the content-based
              keys to lookup, and the weightings for the different read modes.
          memory: A tensor of shape `[batch_size, memory_size, word_size]`
              containing the current memory contents to do content-based lookup.
          prev_read_weights: A tensor of shape `[batch_size, num_reads,
              memory_size]` containing the previous read locations.
          link: A tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` containing the temporal write transition graphs.

        Returns:
          A tensor of shape `[batch_size, num_reads, memory_size]` containing the
          read weights for each read head.
        """
        # c_t^{r, i} - The content weightings for each read head.
        content_weights = self._read_content_weights_mod(
            memory, inputs['read_content_keys'], inputs['read_content_strengths'])

        # Calculates f_t^i and b_t^i.
        forward_weights = self._linkage.directional_read_weights(
            link, prev_read_weights, forward=True)
        backward_weights = self._linkage.directional_read_weights(
            link, prev_read_weights, forward=False)

        backward_mode = inputs['read_mode'][:, :, :self._num_writes]
        forward_mode = (
            inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes])
        content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]

        read_weights = (
                jnp.expand_dims(content_mode,2) * content_weights * content_weights + jnp.sum(
            jnp.expand_dims(forward_mode, 3) * forward_weights, 2) +
                jnp.sum(jnp.expand_dims(backward_mode, 3) * backward_weights, 2))

        return read_weights

    # def prepare_memory_input(self, concatenated_ret, n):
    #     # TODO reduce for loops?
    #
    #     real_ret, _, write_ret = jnp.split(concatenated_ret, [n, n + self.read_head_num], axis=1)
    #     w_nodes, a_e_nodes = jnp.split(write_ret, [self.w_nodes_amount], axis=1)
    #     list_of_params_for_each_w = jnp.split(w_nodes, self.write_head_num+self.read_head_num, axis=1)
    #     list_of_params_for_each_write = jnp.split(a_e_nodes, self.write_head_num, axis=1)
    #     w_params_for_batch = []
    #     for i, params in enumerate(list_of_params_for_each_w):
    #         s_t_pre_dense, beta_g_y_t, k_t = jnp.split(params, 3, axis=1)
    #         s_t = self.s_t_dense_layer(s_t_pre_dense.squeeze())
    #         beta_g_y_t_post_dense = self.beta_g_y_t_dense_layer(beta_g_y_t.squeeze())
    #
    #         beta_t, g_t, gamma_t = jnp.split(beta_g_y_t_post_dense, 3, axis=1)
    #         k = jnp.tanh(k_t.squeeze())
    #         beta = nn.softplus(beta_t)
    #         g = nn.sigmoid(g_t)
    #         s = nn.softmax(
    #             s_t
    #         )
    #         gamma = nn.softplus(gamma_t) + 1
    #         head_parameters = (k, beta, g, s, gamma)
    #         w_params_for_batch.append(head_parameters)
    #     e_a_params_for_batch = []
    #     for i, params in enumerate(list_of_params_for_each_write):
    #         e_t, a_t = jnp.split(params, 2, axis=1)
    #         e_t = e_t.squeeze()
    #         a_t = a_t.squeeze()
    #         e_a_final = (e_t, a_t)
    #         e_a_params_for_batch.append(e_a_final)
    #     final_tuple = (w_params_for_batch, e_a_params_for_batch)
    #     return final_tuple, real_ret

    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        return AccessState(
            memory=(self._memory_size, self._word_size),
            read_weights=(self._num_reads, self._memory_size),
            write_weights=(self._num_writes, self._memory_size),
            linkage=self._linkage.state_size,
            usage=self._freeness.state_size)

    @property
    def output_size(self):
        """Returns the output shape."""
        return (self._num_reads, self._word_size)
