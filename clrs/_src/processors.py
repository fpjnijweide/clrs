# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of baseline processor networks."""

import abc
from typing import Any, Callable, List, Optional, Type

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
# from clrs._src.memory import NTM
from clrs._src.memory_models import extend_features
from clrs._src.memory_models.ntm.ntm_memory import NTM

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class Processor(hk.Module):
  """Processor abstract base class."""

  @abc.abstractmethod
  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **kwargs,
  ) -> _Array:
    """Processor inference step.

    Args:
      node_fts: Node features.
      edge_fts: Edge features.
      graph_fts: Graph features.
      adj_mat: Graph adjacency matrix.
      hidden: Hidden features.
      **kwargs: Extra kwargs.

    Returns:
      Output of processor inference step.
    """
    pass

  @property
  def inf_bias(self):
    return False

  @property
  def inf_bias_edge(self):
    return False


class GAT(Processor):
  """Graph Attention Network (Velickovic et al., ICLR 2018)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GAT inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    a_1 = hk.Linear(self.nb_heads)
    a_2 = hk.Linear(self.nb_heads)
    a_e = hk.Linear(self.nb_heads)
    a_g = hk.Linear(self.nb_heads)

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    att_1 = jnp.expand_dims(a_1(z), axis=-1)
    att_2 = jnp.expand_dims(a_2(z), axis=-1)
    att_e = a_e(edge_fts)
    att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

    logits = (
        jnp.transpose(att_1, (0, 2, 1, 3)) +  # + [B, H, N, 1]
        jnp.transpose(att_2, (0, 2, 3, 1)) +  # + [B, H, 1, N]
        jnp.transpose(att_e, (0, 3, 1, 2)) +  # + [B, H, N, N]
        jnp.expand_dims(att_g, axis=-1)       # + [B, H, 1, 1]
    )                                         # = [B, H, N, N]
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class GATFull(GAT):
  """Graph Attention Network with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class GATv2(Processor):
  """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      residual: bool = True,
      use_ln: bool = False,
      name: str = 'gatv2_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.nb_heads = nb_heads
    if out_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the width!')
    self.head_size = out_size // nb_heads
    if self.mid_size % nb_heads != 0:
      raise ValueError('The number of attention heads must divide the message!')
    self.mid_head_size = self.mid_size // nb_heads
    self.activation = activation
    self.residual = residual
    self.use_ln = use_ln

  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """GATv2 inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj_mat - 1.0) * 1e9
    bias_mat = jnp.tile(bias_mat[..., None],
                        (1, 1, 1, self.nb_heads))     # [B, N, N, H]
    bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

    w_1 = hk.Linear(self.mid_size)
    w_2 = hk.Linear(self.mid_size)
    w_e = hk.Linear(self.mid_size)
    w_g = hk.Linear(self.mid_size)

    a_heads = []
    for _ in range(self.nb_heads):
      a_heads.append(hk.Linear(1))

    values = m(z)                                      # [B, N, H*F]
    values = jnp.reshape(
        values,
        values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, H, F]
    values = jnp.transpose(values, (0, 2, 1, 3))              # [B, H, N, F]

    pre_att_1 = w_1(z)
    pre_att_2 = w_2(z)
    pre_att_e = w_e(edge_fts)
    pre_att_g = w_g(graph_fts)

    pre_att = (
        jnp.expand_dims(pre_att_1, axis=1) +     # + [B, 1, N, H*F]
        jnp.expand_dims(pre_att_2, axis=2) +     # + [B, N, 1, H*F]
        pre_att_e +                              # + [B, N, N, H*F]
        jnp.expand_dims(pre_att_g, axis=(1, 2))  # + [B, 1, 1, H*F]
    )                                            # = [B, N, N, H*F]

    pre_att = jnp.reshape(
        pre_att,
        pre_att.shape[:-1] + (self.nb_heads, self.mid_head_size)
    )  # [B, N, N, H, F]

    pre_att = jnp.transpose(pre_att, (0, 3, 1, 2, 4))  # [B, H, N, N, F]

    # This part is not very efficient, but we agree to keep it this way to
    # enhance readability, assuming `nb_heads` will not be large.
    logit_heads = []
    for head in range(self.nb_heads):
      logit_heads.append(
          jnp.squeeze(
              a_heads[head](jax.nn.leaky_relu(pre_att[:, head])),
              axis=-1)
      )  # [B, N, N]

    logits = jnp.stack(logit_heads, axis=1)  # [B, H, N, N]

    coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)  # [B, H, N, F]
    ret = jnp.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]
    ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, H*F]

    if self.residual:
      ret += skip(z)

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class GATv2Full(GATv2):
  """Graph Attention Network v2 with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGN(Processor):
  """Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = jax.nn.relu,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      use_ln: bool = False,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes
    self.use_ln = use_ln

  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MPNN inference step."""

    b, n, _ = node_fts.shape
    assert edge_fts.shape[:-1] == (b, n, n)
    assert graph_fts.shape[:-1] == (b,)
    assert adj_mat.shape == (b, n, n)

    z = jnp.concatenate([node_fts, hidden], axis=-1)
    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(z)
    msg_2 = m_2(z)
    msg_e = m_e(edge_fts)
    msg_g = m_g(graph_fts)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))
    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
      msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
    elif self.reduction == jnp.max:
      maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
                         msgs,
                         -BIG_NUMBER)
      msgs = jnp.max(maxarg, axis=1)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

    h_1 = o1(z)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    if self.use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class DeepSets(PGN):
  """Deep Sets (Zaheer et al., NeurIPS 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, nb_nodes: int,
               batch_size: int) -> _Array:
    adj_mat = jnp.repeat(
        jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class MPNN(PGN):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


class PGNMask(PGN):
  """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  @property
  def inf_bias(self):
    return True

  @property
  def inf_bias_edge(self):
    return True


class MemNetMasked(Processor):
  """Implementation of End-to-End Memory Networks.

  Inspired by the description in https://arxiv.org/abs/1503.08895.
  """

  def __init__(
      self,
      vocab_size: int,
      sentence_size: int,
      linear_output_size: int,
      embedding_size: int = 16,
      memory_size: Optional[int] = 128,
      num_hops: int = 1,
      nonlin: Callable[[Any], Any] = jax.nn.relu,
      apply_embeddings: bool = True,
      init_func: hk.initializers.Initializer = jnp.zeros,
      use_ln: bool = False,
      name: str = 'memnet') -> None:
    """Constructor.

    Args:
      vocab_size: the number of words in the dictionary (each story, query and
        answer come contain symbols coming from this dictionary).
      sentence_size: the dimensionality of each memory.
      linear_output_size: the dimensionality of the output of the last layer
        of the model.
      embedding_size: the dimensionality of the latent space to where all
        memories are projected.
      memory_size: the number of memories provided.
      num_hops: the number of layers in the model.
      nonlin: non-linear transformation applied at the end of each layer.
      apply_embeddings: flag whether to aply embeddings.
      init_func: initialization function for the biases.
      use_ln: whether to use layer normalisation in the model.
      name: the name of the model.
    """
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._sentence_size = sentence_size
    self._memory_size = memory_size
    self._linear_output_size = linear_output_size
    self._num_hops = num_hops
    self._nonlin = nonlin
    self._apply_embeddings = apply_embeddings
    self._init_func = init_func
    self._use_ln = use_ln
    # Encoding part: i.e. "I" of the paper.
    self._encodings = _position_encoding(sentence_size, embedding_size)

  def __call__(
      self,
      node_fts: _Array,
      edge_fts: _Array,
      graph_fts: _Array,
      adj_mat: _Array,
      hidden: _Array,
      **unused_kwargs,
  ) -> _Array:
    """MemNet inference step."""

    del hidden
    node_and_graph_fts = jnp.concatenate([node_fts, graph_fts[:, None]],
                                         axis=1)
    edge_fts_padded = jnp.pad(edge_fts * adj_mat[..., None],
                              ((0, 0), (0, 1), (0, 1), (0, 0)))
    nxt_hidden = jax.vmap(self._apply, (1), 1)(node_and_graph_fts,
                                               edge_fts_padded)

    # Broadcast hidden state corresponding to graph features across the nodes.
    nxt_hidden = nxt_hidden[:, :-1] + nxt_hidden[:, -1:]
    return nxt_hidden

  def _apply(self, queries: _Array, stories: _Array) -> _Array:
    """Apply Memory Network to the queries and stories.

    Args:
      queries: Tensor of shape [batch_size, sentence_size].
      stories: Tensor of shape [batch_size, memory_size, sentence_size].

    Returns:
      Tensor of shape [batch_size, vocab_size].
    """
    if self._apply_embeddings:
      query_biases = hk.get_parameter(
          'query_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      stories_biases = hk.get_parameter(
          'stories_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      memory_biases = hk.get_parameter(
          'memory_contents',
          shape=[self._memory_size, self._embedding_size],
          init=self._init_func)
      output_biases = hk.get_parameter(
          'output_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)

      nil_word_slot = jnp.zeros([1, self._embedding_size])

    # This is "A" in the paper.
    if self._apply_embeddings:
      stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
      memory_embeddings = jnp.take(
          stories_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      memory_embeddings = jnp.pad(
          memory_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(memory_embeddings)[1]),
           (0, 0), (0, 0)))
      memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
    else:
      memory = stories

    # This is "B" in the paper. Also, when there are no queries (only
    # sentences), then there these lines are substituted by
    # query_embeddings = 0.1.
    if self._apply_embeddings:
      query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
      query_embeddings = jnp.take(
          query_biases, queries.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(queries.shape) + [self._embedding_size])
      # This is "u" in the paper.
      query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
    else:
      query_input_embedding = queries

    # This is "C" in the paper.
    if self._apply_embeddings:
      output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
      output_embeddings = jnp.take(
          output_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      output_embeddings = jnp.pad(
          output_embeddings,
          ((0, 0), (0, self._memory_size - jnp.shape(output_embeddings)[1]),
           (0, 0), (0, 0)))
      output = jnp.sum(output_embeddings * self._encodings, 2)
    else:
      output = stories

    intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

    # Output_linear is "H".
    output_linear = hk.Linear(self._linear_output_size, with_bias=False)

    for hop_number in range(self._num_hops):
      query_input_embedding_transposed = jnp.transpose(
          jnp.expand_dims(query_input_embedding, -1), [0, 2, 1])

      # Calculate probabilities.
      probs = jax.nn.softmax(
          jnp.sum(memory * query_input_embedding_transposed, 2))

      # Calculate output of the layer by multiplying by C.
      transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
      transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

      # This is "o" in the paper.
      layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

      # Finally the answer
      if hop_number == self._num_hops - 1:
        # Please note that in the TF version we apply the final linear layer
        # in all hops and this results in shape mismatches.
        output_layer = output_linear(query_input_embedding + layer_output)
      else:
        output_layer = intermediate_linear(query_input_embedding + layer_output)

      query_input_embedding = output_layer
      if self._nonlin:
        output_layer = self._nonlin(output_layer)

    # This linear here is "W".
    ret = hk.Linear(self._vocab_size, with_bias=False)(output_layer)

    if self._use_ln:
      ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ret = ln(ret)

    return ret


class MemNetFull(MemNetMasked):
  """Memory Networks with full adjacency matrix."""

  def __call__(self, node_fts: _Array, edge_fts: _Array, graph_fts: _Array,
               adj_mat: _Array, hidden: _Array, **unused_kwargs) -> _Array:
    adj_mat = jnp.ones_like(adj_mat)
    return super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden)


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
  """Position Encoding described in section 4.1 [1]."""
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)


class MemoryAugmentedProcessor(Processor):
    """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

    def __init__(
            self,
            processor_type: Type[Processor],
            memory_type: Type[hk.RNNCore],
            name: str = 'ntm_network',
            *args, **kwargs
    ):

        super().__init__(name=name)
        self.processor = processor_type(*args, **kwargs)
        self.memory = memory_type()

        self.memory_state = None

        self.read_node_fts = None

        self.write_node_fts_layer = hk.initializers.RandomNormal(stddev=0.5)
        self.read_node_fts_layer = hk.initializers.RandomNormal(stddev=0.5)
        self.write_edge_fts_layer = hk.initializers.RandomNormal()
        self.read_edge_fts_layer = hk.initializers.RandomNormal()



    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        """inference step."""
        b, n, _ = node_fts.shape

        if not self.memory_state:
            self.memory_state = self.memory.initial_state(node_fts)

        adj_mat_new, hidden_new, new_edge_fts, node_fts_new = extend_features(adj_mat, edge_fts, hidden, node_fts,self.memory.read_nodes_amount,self.memory.write_nodes_amount,self.write_node_fts_layer,self.write_edge_fts_layer,self.read_node_fts,self.read_edge_fts_layer,self.read_node_fts_layer)

        # Network is called here
        ret = self.processor(node_fts_new, new_edge_fts, graph_fts, adj_mat_new, hidden_new, **unused_kwargs)

        memory_input, ret = self.memory.prepare_memory_input(ret, n)

        self.read_node_fts, self.memory_state = self.memory(memory_input,self.memory_state)


        # TODO do for PGN?
        # TODO do for deque
        # TODO do for own architecture
        # TODO do for DNC



        # TODO experiment with memory size and such

        return ret


    # def extend_features(self, adj_mat, edge_fts, hidden, node_fts):
    #     b, n, _ = node_fts.shape
    #     write_node_fts_shape = jnp.array(node_fts.shape)
    #     write_node_fts_shape = write_node_fts_shape.at[0].set(1)
    #     write_node_fts_shape = write_node_fts_shape.at[1].set(self.write_nodes_amount)
    #     self.write_node_fts = self.write_node_fts_layer(shape=write_node_fts_shape, dtype=jnp.float32)
    #     self.write_node_fts = jnp.repeat(self.write_node_fts, b, axis=0)
    #     if self.read_node_fts is None:
    #         read_node_fts_shape = jnp.array(node_fts.shape)
    #         read_node_fts_shape = read_node_fts_shape.at[0].set(1)
    #         read_node_fts_shape = read_node_fts_shape.at[1].set(self.read_nodes_amount)
    #
    #         self.read_node_fts = self.write_node_fts_layer(shape=read_node_fts_shape, dtype=jnp.float32)
    #         self.read_node_fts = jnp.repeat(self.read_node_fts, b, axis=0)
    #     # Setting the new node fts
    #     node_fts_new = jnp.concatenate([node_fts, self.read_node_fts, self.write_node_fts], axis=1)
    #     new_edge_feat_shape = jnp.array(edge_fts.shape)
    #     edge_fts_embedding_size = new_edge_feat_shape[-1]
    #     edge_fts_embedding_size_for_write = (self.write_nodes_amount, edge_fts_embedding_size)
    #     new_edge_feat_shape = new_edge_feat_shape.at[1].set(n + self.write_nodes_amount + self.read_nodes_amount)
    #     new_edge_feat_shape = new_edge_feat_shape.at[2].set(n + self.write_nodes_amount + self.read_nodes_amount)
    #     new_edge_fts = jnp.zeros(new_edge_feat_shape)
    #     new_edge_fts = new_edge_fts.at[:, :n, :n, :].set(edge_fts)
    #     # The edge features to/from write/read nodes will be generated via a dense layer
    #     write_edge_fts = self.write_edge_fts_layer(shape=edge_fts_embedding_size_for_write, dtype=jnp.float32)
    #     read_edge_fts = self.read_edge_fts_layer(shape=(1, edge_fts_embedding_size), dtype=jnp.float32)
    #     write_edge_fts_tiled = jnp.tile(write_edge_fts, [b, n, 1, 1])
    #     read_edge_fts_tiled = jnp.tile(read_edge_fts, [b, self.read_nodes_amount, n, 1])
    #     new_edge_fts = new_edge_fts.at[:, :n,
    #                    n + self.read_nodes_amount:n + self.read_nodes_amount + self.write_nodes_amount, :].set(
    #         write_edge_fts_tiled)  # From normal nodes to write nodes: edges
    #     new_edge_fts = new_edge_fts.at[:, n:n + self.read_nodes_amount, :n, :].set(
    #         read_edge_fts_tiled)  # From read nodes to normal nodes: edges
    #     # Setting the ajdacency matrix
    #     adj_mat_new = jnp.zeros([b, n + self.write_nodes_amount + self.read_nodes_amount,
    #                              n + self.write_nodes_amount + self.read_nodes_amount])
    #     adj_mat_new = adj_mat_new.at[:, :n, :n].set(adj_mat)
    #     adj_mat_new = adj_mat_new.at[:, :n,
    #                   n + self.read_nodes_amount:n + self.read_nodes_amount + self.write_nodes_amount].set(
    #         1)  # From normal nodes to write nodes: edges
    #     adj_mat_new = adj_mat_new.at[:, n:n + self.read_nodes_amount, :n].set(
    #         1)  # From read nodes to normal nodes: edges
    #     hidden_new_shape = jnp.array(hidden.shape)
    #     hidden_new_shape = hidden_new_shape.at[1].set(
    #         hidden_new_shape[1] + self.read_nodes_amount + self.write_nodes_amount)
    #     hidden_new = jnp.zeros(hidden_new_shape, dtype=jnp.float32)
    #     hidden_new = hidden_new.at[:, :n, :].set(hidden)  # TODO what to do about hidden
    #     return adj_mat_new, hidden_new, new_edge_fts, node_fts_new


ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_heads: Optional[int] = None) -> ProcessorFactory:
  """Returns a processor factory.

  Args:
    kind: One of the available types of processor.
    use_ln: Whether the processor passes the output through a layernorm layer.
    nb_heads: Number of attention heads for GAT processors.
  Returns:
    A callable that takes an `out_size` parameter (equal to the hidden
    dimension of the network) and returns a processor instance.
  """
  def _factory(out_size: int):
    if kind == 'deepsets':
      processor = DeepSets(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    elif kind == 'gat':
      processor = GAT(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gat_full':
      processor = GATFull(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2':
      processor = GATv2(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_full':
      processor = GATv2Full(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_ntm':
      processor = MemoryAugmentedProcessor(
          processor_type=GATv2,
          memory_type=NTM,
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'memnet_full':
      processor = MemNetFull(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'memnet_masked':
      processor = MemNetMasked(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    elif kind == 'pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    elif kind == 'pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    else:
      raise ValueError('Unexpected processor kind ' + kind)

    return processor

  return _factory
