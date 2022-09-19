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
"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import NamedTuple

import jax.numpy as jnp
import haiku as hk

from clrs._src.memory_models.dnc.access import DNCAccessModule

class DNCState(NamedTuple):
    access_output: jnp.array
    access_state: jnp.array



class DNCNetwork(hk.RNNCore):
  """DNC core module.

  Contains controller and memory access module.
  """

  def __init__(self,
               access_config,
               controller_config,
               output_size,
               clip_value=None,
               name='dnc'):
    """Initializes the DNC core.

    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').

    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(DNCNetwork, self).__init__(name=name)

    with self._enter_variable_scope():
      # self._controller = snt.LSTM(**controller_config)
      self._access = DNCAccessModule(**access_config)

    self._access_output_size = jnp.prod(self._access.output_size.as_list())
    self._output_size = output_size
    self._clip_value = clip_value or 0

    self._output_size = output_size
    self._state_size = DNCState(
        access_output=self._access_output_size,
        access_state=self._access.state_size,
    )

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return jnp.clip(x, -self._clip_value, self._clip_value)
    else:
      return x

  def __call__(self, inputs, prev_state):
    """Connects the DNC core into the graph.

    Args:
      inputs: Tensor input.
      prev_state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_state` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.

    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `DNCState` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """

    # prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    # prev_controller_state = prev_state.controller_state

    # controller_input = jnp.concatenate(
    #     [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    # controller_output, controller_state = self._controller(
    #     controller_input, prev_controller_state)
    #
    # controller_output = self._clip_if_enabled(controller_output)
    # controller_state = tf.contrib.framework.nest.map_structure(self._clip_if_enabled, controller_state)

    access_output, access_state = self._access(inputs,
                                               prev_access_state)


    return inputs, DNCState(
        access_output=access_output,
        access_state=access_state,
    )

  def initial_state(self, batch_size, dtype=jnp.float32):
    return DNCState(
        access_state=self._access.initial_state(batch_size),
        access_output=jnp.zeros(
            [batch_size] + self._access.output_size.as_list()))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
