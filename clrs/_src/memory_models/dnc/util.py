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

import jax.lax
# import numpy as jnp
# import tensorflow as jnp
import jax.numpy as jnp


def invert_permutation(p):
    # Full credits for this function go to Ali, on stack overflow
    # https://stackoverflow.com/a/25535723
    # needed a quick workaround because the tf invert_permutation function is MASSIVE and hard to convert to jax
    s = jnp.empty(p.size, p.dtype)
    s = s.at[p].set(jnp.arange(p.size))
    return s


def batch_invert_permutation(permutations):
    """Returns batched `jnp.invert_permutation` for every row in `permutations`."""
    return jnp.apply_along_axis(invert_permutation,-1,permutations)
    # perm = jax.lax.convert_element_type(permutations, jnp.float32)
    # dim = int(perm.shape[-1])
    # size = jax.lax.convert_element_type(jnp.shape(perm)[0], jnp.float32)
    # delta = jax.lax.convert_element_type(jnp.shape(perm)[-1], jnp.float32)
    # rg = jnp.arange(0, size * delta, delta, dtype=jnp.float32)
    # rg = jnp.expand_dims(rg, 1)
    # rg = jnp.tile(rg, [1, dim])
    # perm = jnp.add(perm, rg)
    # flat = jnp.reshape(perm, [-1])
    # perm = invert_permutation(jax.lax.convert_element_type(flat, jnp.int32))
    # perm = jnp.reshape(perm, [-1, dim])
    # return jnp.subtract(perm, jax.lax.convert_element_type(rg, jnp.int32))


def batch_gather(values, indices):
    """Returns batched `jnp.gather` for every row in the ijnput."""
    return jnp.take_along_axis(values, indices, axis=-1)
    # jnp.apply_along_axis(, -1, values)
    # idx = jnp.expand_dims(indices, -1)
    # size = jnp.shape(indices)[0]
    # rg = jnp.arange(size, dtype=jnp.int32)
    # rg = jnp.expand_dims(rg, -1)
    # rg = jnp.tile(rg, [1, int(indices.shape[-1])])
    # rg = jnp.expand_dims(rg, -1)
    # gidx = jnp.concatenate([rg, idx], -1)
    # return values[gidx]
    # return jnp.gather_nd(values, gidx)


def one_hot(length, index):
    """Return an nd array of given `length` filled with 0s and a 1 at `index`."""
    return jax.nn.one_hot(index,length)


# def reduce_prod(x, axis, name=None):
#     """Efficient reduce product over axis.
#
#     Uses jnp.cumprod and jnp.gather_nd as a workaround to the poor performance of calculating jnp.reduce_prod's gradient on CPU.
#     """
#     cp = jnp.cumprod(jnp.flip(x,axis=axis), axis)
#     size = jnp.shape(cp)[0]
#     idx1 = jnp.arange(jax.lax.convert_element_type(size, jnp.float32), dtype=jnp.float32)
#     idx2 = jnp.zeros([size], jnp.float32)
#     indices = jnp.stack([idx1, idx2], 1)
#     return cp[jax.lax.convert_element_type(indices, jnp.int32)]
#     # return jnp.gather_nd(cp, jax.lax.convert_element_type(indices, jnp.int32))
