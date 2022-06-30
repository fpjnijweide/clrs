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
"""DNC util ops and modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as jnp
# import tensorflow as jnp
import jax.numpy as jnp


def batch_invert_permutation(permutations):
    """Returns batched `jnp.invert_permutation` for every row in `permutations`."""
    perm = jnp.cast(permutations, jnp.float32)
    dim = int(perm.get_shape()[-1])
    size = jnp.cast(jnp.shape(perm)[0], jnp.float32)
    delta = jnp.cast(jnp.shape(perm)[-1], jnp.float32)
    rg = jnp.range(0, size * delta, delta, dtype=jnp.float32)
    rg = jnp.expand_dims(rg, 1)
    rg = jnp.tile(rg, [1, dim])
    perm = jnp.add(perm, rg)
    flat = jnp.reshape(perm, [-1])
    perm = jnp.invert_permutation(jnp.cast(flat, jnp.int32))
    perm = jnp.reshape(perm, [-1, dim])
    return jnp.subtract(perm, jnp.cast(rg, jnp.int32))


def batch_gather(values, indices):
    """Returns batched `jnp.gather` for every row in the ijnput."""
    idx = jnp.expand_dims(indices, -1)
    size = jnp.shape(indices)[0]
    rg = jnp.range(size, dtype=jnp.int32)
    rg = jnp.expand_dims(rg, -1)
    rg = jnp.tile(rg, [1, int(indices.get_shape()[-1])])
    rg = jnp.expand_dims(rg, -1)
    gidx = jnp.concat([rg, idx], -1)
    return jnp.gather_nd(values, gidx)


def one_hot(length, index):
    """Return an nd array of given `length` filled with 0s and a 1 at `index`."""
    result = jnp.zeros(length)
    result[index] = 1
    return result


def reduce_prod(x, axis, name=None):
    """Efficient reduce product over axis.

    Uses jnp.cumprod and jnp.gather_nd as a workaround to the poor performance of calculating jnp.reduce_prod's gradient on CPU.
    """
    cp = jnp.cumprod(x, axis, reverse=True)
    size = jnp.shape(cp)[0]
    idx1 = jnp.range(jnp.cast(size, jnp.float32), dtype=jnp.float32)
    idx2 = jnp.zeros([size], jnp.float32)
    indices = jnp.stack([idx1, idx2], 1)
    return jnp.gather_nd(cp, jnp.cast(indices, jnp.int32))
