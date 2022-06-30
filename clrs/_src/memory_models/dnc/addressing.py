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
"""DNC addressing modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Ensure values are greater than epsilon to avoid numerical instability.
import jax
from clrs._src.memory_models.dnc import util

# import sonnet as hk
# import tensorflow as tf

import haiku as hk
import jax.numpy as jnp
from jax import nn

_EPSILON = 1e-6

TemporalLinkageState = collections.namedtuple('TemporalLinkageState',
                                              ('link', 'precedence_weights'))

# def fill_diagonal_own_implementation(matrix,arr):
#     arr_as_matrix = jnp.diag(arr)
#     matrix_diag_as_arr = jnp.diagonal(matrix)
#     matrix_only_diag = jnp.diag(matrix_diag_as_arr)
#     matrix_diag_0 = matrix - (matrix_only_diag)
#     final_mat = matrix_diag_0 + arr_as_matrix
#     return final_mat

# def fill_diagonal(a: jnp.ndarray, val: jnp.ndarray, wrap=False):
#     # Direct copy from the numpy source at https://github.com/numpy/numpy/blob/v1.23.0/numpy/lib/index_tricks.py#L779-L910
#     # I take no credit for this code, except for adding "jnp." in a few locations
#     if a.ndim < 2:
#         raise ValueError("array must be at least 2-d")
#     end = None
#     if a.ndim == 2:
#         # Explicit, fast formula for the common case.  For 2-d arrays, we
#         # accept rectangular ones.
#         step = a.shape[1] + 1
#         # This is needed to don't have tall matrix have the diagonal wrap.
#         if not wrap:
#             end = a.shape[1] * a.shape[1]
#     else:
#         # For more than d=2, the strided formula is only valid for arrays with
#         # all dimensions equal, so we check first.
#         if not jnp.alltrue(jnp.diff(a.shape) == 0):
#             raise ValueError("All dimensions of input must be of equal length")
#         step = 1 + (jnp.cumprod(a.shape[:-1])).sum()
#
#     # Write the value out into the diagonal.
#     a.flat[:end:step] = val

def _vector_norms(m):
    squared_norms = jnp.sum(m * m, axis=2, keepdims=True)
    return jnp.sqrt(squared_norms + _EPSILON)


def weighted_softmax(activations, strengths, strengths_op):
    """Returns softmax over activations multiplied by positive strengths.

    Args:
      activations: A tensor of shape `[batch_size, num_heads, memory_size]`, of
        activations to be transformed. Softmax is taken over the last dimension.
      strengths: A tensor of shape `[batch_size, num_heads]` containing strengths to
        multiply by the activations prior to the softmax.
      strengths_op: An operation to transform strengths before softmax.

    Returns:
      A tensor of same shape as `activations` with weighted softmax applied.
    """
    transformed_strengths = jnp.expand_dims(strengths_op(strengths), -1)
    sharp_activations = activations * transformed_strengths
    softmax = hk.BatchApply(f=nn.softmax)
    return softmax(sharp_activations)


class CosineWeights(hk.Module):
    """Cosine-weighted attention.

    Calculates the cosine similarity between a query and each word in memory, then
    applies a weighted softmax to return a sharp distribution.
    """

    def __init__(self,
                 num_heads,
                 word_size,
                 strength_op=nn.softplus,
                 name='cosine_weights'):
        """Initializes the CosineWeights module.

        Args:
          num_heads: number of memory heads.
          word_size: memory word size.
          strength_op: operation to apply to strengths (default is nn.softplus).
          name: module name (default 'cosine_weights')
        """
        super(CosineWeights, self).__init__(name=name)
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = strength_op

    def __call__(self, memory, keys, strengths):
        """Connects the CosineWeights module into the graph.

        Args:
          memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
          keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
          strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

        Returns:
          Weights tensor of shape `[batch_size, num_heads, memory_size]`.
        """
        # Calculates the inner product between the query vector and words in memory.
        dot = jnp.matmul(keys, memory, adjoint_b=True)

        # Outer product to compute denominator (euclidean norm of query and memory).
        memory_norms = _vector_norms(memory)
        key_norms = _vector_norms(keys)
        norm = jnp.matmul(key_norms, memory_norms, adjoint_b=True)

        # Calculates cosine similarity between the query vector and words in memory.
        similarity = dot / (norm + _EPSILON)

        return weighted_softmax(similarity, strengths, self._strength_op)



class TemporalLinkage(hk.RNNCore):
    """Keeps track of write order for forward and backward addressing.

    This is a pseudo-RNNCore module, whose state is a pair `(link,
    precedence_weights)`, where `link` is a (collection of) graphs for (possibly
    multiple) write heads (represented by a tensor with values in the range
    [0, 1]), and `precedence_weights` records the "previous write locations" used
    to build the link graphs.

    The function `directional_read_weights` computes addresses following the
    forward and backward directions in the link graphs.
    """

    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        """Construct a TemporalLinkage module.

        Args:
          memory_size: The number of memory slots.
          num_writes: The number of write heads.
          name: Name of the module.
        """
        super(TemporalLinkage, self).__init__(name=name)
        self._memory_size = memory_size
        self._num_writes = num_writes

    def __call__(self, write_weights, prev_state) -> (None, TemporalLinkageState):
        """Calculate the updated linkage state given the write weights.

        Args:
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the memory addresses of the different write heads.
          prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
              shape `[batch_size, num_writes, memory_size, memory_size]`, and a
              tensor `precedence_weights` of shape `[batch_size, num_writes,
              memory_size]` containing the aggregated history of recent writes.

        Returns:
          A `TemporalLinkageState` tuple `next_state`, which contains the updated
          link and precedence weights.
        """
        link = self._link(prev_state.link, prev_state.precedence_weights,
                          write_weights)
        precedence_weights = self._precedence_weights(prev_state.precedence_weights,
                                                      write_weights)
        return None, TemporalLinkageState(
            link=link, precedence_weights=precedence_weights)

    def directional_read_weights(self, link, prev_read_weights, forward):
        """Calculates the forward or the backward read weights.

        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.

        Args:
          link: tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the link graphs L_t.
          prev_read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` containing the previous read weights w_{t-1}^r.
          forward: Boolean indicating whether to follow the "future" direction in
              the link graph (True) or the "past" direction (False).

        Returns:
          tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
        """
        # We calculate the forward and backward directions for each pair of
        # read and write heads; hence we need to tile the read weights and do a
        # sort of "outer product" to get this.
        expanded_read_weights = jnp.stack([prev_read_weights] * self._num_writes,
                                          1)
        result = jnp.matmul(expanded_read_weights, link, adjoint_b=forward)
        # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
        return jnp.transpose(result, perm=[0, 2, 1, 3])



    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs.

        For each write head, the link is a directed graph (represented by a matrix
        with entries in range [0, 1]) whose vertices are the memory locations, and
        an edge indicates temporal ordering of writes.

        Args:
          prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the previous link graphs for each write
              head.
          prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` which is the previous "aggregated" write weights for
              each write head.
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new locations in memory written to.

        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
          containing the new link graphs for each write head.
        """
        batch_size = jnp.shape(prev_link)[0]
        write_weights_i = jnp.expand_dims(write_weights, 3)
        write_weights_j = jnp.expand_dims(write_weights, 2)
        prev_precedence_weights_j = jnp.expand_dims(prev_precedence_weights, 2)
        prev_link_scale = 1 - write_weights_i - write_weights_j
        new_link = write_weights_i * prev_precedence_weights_j
        link = prev_link_scale * prev_link + new_link
        link = link*1
        # Return the link with the diagonal set to zero, to remove self-looping
        # edges.
        link[jnp.diag_indices_from[link]] = jnp.zeros([batch_size, self._num_writes, self._memory_size],dtype=link.dtype)
        # return fill_diagonal(
        #     link,
        #     jnp.zeros(
        #         [batch_size, self._num_writes, self._memory_size],
        #         dtype=link.dtype))
        return link

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights given the current write weights.

        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.

        Args:
          prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` containing the previous precedence weights.
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new write weights.

        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size]` containing the
          new precedence weights.
        """
        write_sum = jnp.sum(write_weights, 2, keepdims=True)
        return (1 - write_sum) * prev_precedence_weights + write_weights

    # @property
    # def state_size(self):
    #     """Returns a `TemporalLinkageState` tuple of the state tensors' shapes."""
    #     return TemporalLinkageState(
    #         link=jnp.TensorShape(
    #             [self._num_writes, self._memory_size, self._memory_size]),
    #         precedence_weights=jnp.TensorShape([self._num_writes,
    #                                             self._memory_size]), )


class Freeness(hk.Module):
    """Memory usage that is increased by writing and decreased by reading.

    This module is a pseudo-RNNCore whose state is a tensor with values in
    the range [0, 1] indicating the usage of each of `memory_size` memory slots.

    The usage is:

    *   Increased by writing, where usage is increased towards 1 at the write
        addresses.
    *   Decreased by reading, where usage is decreased after reading from a
        location when free_gate is close to 1.

    The function `write_allocation_weights` can be invoked to get free locations
    to write to for a number of write heads.
    """

    def __init__(self, memory_size, name='freeness'):
        """Creates a Freeness module.

        Args:
          memory_size: Number of memory slots.
          name: Name of the module.
        """
        super(Freeness, self).__init__(name=name)
        self._memory_size = memory_size

    def __call__(self, write_weights, free_gate, read_weights, prev_usage):
        """Calculates the new memory usage u_t.

        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.

        Args:
          write_weights: tensor of shape `[batch_size, num_writes,
              memory_size]` giving write weights at previous time step.
          free_gate: tensor of shape `[batch_size, num_reads]` which indicates
              which read heads read memory that can now be freed.
          read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` giving read weights at previous time step.
          prev_usage: tensor of shape `[batch_size, memory_size]` giving
              usage u_{t - 1} at the previous time step, with entries in range
              [0, 1].

        Returns:
          tensor of shape `[batch_size, memory_size]` representing updated memory
          usage.
        """
        # Calculation of usage is not differentiable with respect to write weights.
        write_weights = jax.lax.stop_gradient(write_weights)
        usage = self._usage_after_write(prev_usage, write_weights)
        usage = self._usage_after_read(usage, free_gate, read_weights)
        return usage

    def write_allocation_weights(self, usage, write_gates, num_writes):
        """Calculates freeness-based locations for writing to.

        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)

        Args:
          usage: A tensor of shape `[batch_size, memory_size]` representing
              current memory usage.
          write_gates: A tensor of shape `[batch_size, num_writes]` with values in
              the range [0, 1] indicating how much each write head does writing
              based on the address returned here (and hence how much usage
              increases).
          num_writes: The number of write heads to calculate write weights for.

        Returns:
          tensor of shape `[batch_size, num_writes, memory_size]` containing the
              freeness-based write locations. Note that this isn't scaled by
              `write_gate`; this scaling must be applied externally.
        """
        # expand gatings over memory locations
        write_gates = jnp.expand_dims(write_gates, -1)

        allocation_weights = []
        for i in range(num_writes):
            allocation_weights.append(self._allocation(usage))
            # update usage to take into account writing to this new allocation
            usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

        # Pack the allocation weights for the write heads into one tensor.
        return jnp.stack(allocation_weights, axis=1)

    def _usage_after_write(self, prev_usage, write_weights):
        """Calcualtes the new usage after writing to memory.

        Args:
          prev_usage: tensor of shape `[batch_size, memory_size]`.
          write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.

        Returns:
          New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        # Calculate the aggregated effect of all write heads
        write_weights = 1 - util.reduce_prod(1 - write_weights, 1)
        return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        """Calcualtes the new usage after reading and freeing from memory.

        Args:
          prev_usage: tensor of shape `[batch_size, memory_size]`.
          free_gate: tensor of shape `[batch_size, num_reads]` with entries in the
              range [0, 1] indicating the amount that locations read from can be
              freed.
          read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.

        Returns:
          New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        free_gate = jnp.expand_dims(free_gate, -1)
        free_read_weights = free_gate * read_weights
        phi = util.reduce_prod(1 - free_read_weights, 1, name='phi')
        return prev_usage * phi

    def _allocation(self, usage):
        r"""Computes allocation by sorting `usage`.

        This corresponds to the value a = a_t[\phi_t[j]] in the paper.

        Args:
          usage: tensor of shape `[batch_size, memory_size]` indicating current
              memory usage. This is equal to u_t in the paper when we only have one
              write head, but for multiple write heads, one should update the usage
              while iterating through the write heads to take into account the
              allocation returned by this function.

        Returns:
          Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        # Ensure values are not too small prior to cumprod.
        usage = _EPSILON + (1 - _EPSILON) * usage

        nonusage = 1 - usage
        sorted_nonusage, indices = jax.lax.top_k(
            nonusage, k=self._memory_size)
        sorted_usage = 1 - sorted_nonusage
        prod_sorted_usage = jnp.cumprod(sorted_usage, axis=1, exclusive=True)
        sorted_allocation = sorted_nonusage * prod_sorted_usage
        inverse_indices = util.batch_invert_permutation(indices)

        # This final line "unsorts" sorted_allocation, so that the indexing
        # corresponds to the original indexing of `usage`.
        return util.batch_gather(sorted_allocation, inverse_indices)

    # @property
    # def state_size(self):
    #     """Returns the shape of the state tensor."""
    #     return jnp.TensorShape([self._memory_size])
